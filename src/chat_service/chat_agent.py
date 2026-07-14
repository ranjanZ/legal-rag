"""
Chat Agent for RAG System with Planning, Memory, and Multi-Step Reasoning.
Orchestrates queries using retrieval service and Ollama-based LLM.
"""

import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import time

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.retrieval_service import RetrievalService
from src.config import INDEX_DIR

# Configure logging for chat service
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/workspace/src/chat_service/chat_service.log')
    ]
)
logger = logging.getLogger(__name__)


# Initialize LLM as specified
llm = ChatOllama(
    model="llama3.2",  
    temperature=0,
    num_ctx=2048,        
    num_thread=4
)


class ConversationMemory:
    """
    Manages conversation history and context for the chat agent.
    Supports short-term memory (current session) and can be extended for long-term.
    """
    
    def __init__(self, max_history_length: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_history_length: Maximum number of message pairs to retain
        """
        self.max_history_length = max_history_length
        self.messages: List[BaseMessage] = []
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        
    def add_message(self, message: BaseMessage):
        """Add a message to the conversation history."""
        self.messages.append(message)
        
        # Trim if exceeds max length (keep most recent)
        if len(self.messages) > self.max_history_length * 2:  # *2 for user+assistant pairs
            # Keep system message if present, then last max_history_length pairs
            start_idx = 0
            if self.messages and isinstance(self.messages[0], SystemMessage):
                start_idx = 1
            self.messages = self.messages[:start_idx] + self.messages[-(self.max_history_length * 2):]
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in the conversation history."""
        return self.messages.copy()
    
    def clear(self):
        """Clear the conversation history."""
        self.messages = []
        self.session_id = str(uuid.uuid4())
    
    def get_context_summary(self) -> str:
        """Generate a brief summary of the conversation context."""
        if not self.messages:
            return "No previous conversation context."
        
        # Extract key topics from recent exchanges
        recent_messages = self.messages[-6:]  # Last 3 pairs
        context_parts = []
        
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"User asked about: {msg.content[:100]}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant provided information about: {msg.content[:100]}")
        
        return "\n".join(context_parts)


class QueryPlan:
    """
    Represents a plan for answering complex queries that may require multiple steps.
    """
    
    def __init__(self, query: str):
        self.query = query
        self.steps: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []
        self.current_step = 0
        self.status = "created"  # created, in_progress, completed, failed
        
    def add_step(self, step_type: str, description: str, 
                 sub_query: Optional[str] = None,
                 category: Optional[str] = None) -> int:
        """
        Add a step to the plan.
        
        Args:
            step_type: Type of step ('retrieve', 'reason', 'synthesize', 'verify')
            description: Description of what this step does
            sub_query: Specific query for retrieval steps
            category: Category to search in
            
        Returns:
            Step index
        """
        step = {
            'index': len(self.steps),
            'type': step_type,
            'description': description,
            'sub_query': sub_query or self.query,
            'category': category,
            'status': 'pending',
            'result': None,
            'error': None
        }
        self.steps.append(step)
        return len(self.steps) - 1
    
    def mark_step_complete(self, step_index: int, result: Any):
        """Mark a step as completed with its result."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = 'completed'
            self.steps[step_index]['result'] = result
            self.results.append(result)
    
    def mark_step_failed(self, step_index: int, error: str):
        """Mark a step as failed."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = 'failed'
            self.steps[step_index]['error'] = error
    
    def get_next_pending_step(self) -> Optional[Dict[str, Any]]:
        """Get the next pending step."""
        for step in self.steps:
            if step['status'] == 'pending':
                return step
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary representation."""
        return {
            'query': self.query,
            'steps': self.steps,
            'current_step': self.current_step,
            'status': self.status
        }


class ChatAgent:
    """
    Advanced chat agent with planning, memory, and multi-step reasoning capabilities.
    
    The agent analyzes query complexity and decides whether to:
    1. Answer directly with single retrieval
    2. Create a multi-step plan for complex queries requiring multiple information sources
    3. Use reasoning to synthesize information from multiple chunks
    """
    
    def __init__(self, index_dir: Path = None):
        """
        Initialize the chat agent.
        
        Args:
            index_dir: Path to the index directory for retrieval
        """
        self.retrieval_service = RetrievalService(index_dir or INDEX_DIR)
        self.memory = ConversationMemory(max_history_length=10)
        self.current_plan: Optional[QueryPlan] = None
        
        # Prompts for different agent functions
        self.planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query planner for a legal document retrieval system.
Your task is to analyze the user's query and determine if it requires:
1. A simple single-step retrieval
2. A multi-step plan with multiple retrievals and reasoning

Consider:
- Does the query ask about multiple distinct concepts?
- Does it require comparing information from different documents?
- Does it need sequential reasoning (if X then Y)?
- Does it ask for calculations or aggregations?

Respond in JSON format with:
{{
    "complexity": "simple" | "moderate" | "complex",
    "requires_planning": true/false,
    "steps": [
        {{
            "step_number": 1,
            "type": "retrieve" | "reason" | "synthesize" | "verify",
            "description": "What this step does",
            "sub_query": "Specific query for this step if retrieval",
            "reason": "Why this step is needed"
        }}
    ],
    "categories_to_search": ["list", "of", "categories"],
    "reasoning": "Explanation of your analysis"
}}"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{query}")
        ])
        
        self.answer_synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful legal assistant answering questions based on retrieved documents.
Your task is to provide accurate, comprehensive answers using the retrieved chunks.

IMPORTANT REQUIREMENTS:
1. Answer the question clearly and concisely
2. CITE YOUR SOURCES: For each piece of information, cite the chunk using [Chunk ID: {chunk_id}, File: {file_name}]
3. If chunks contradict each other, acknowledge the contradiction
4. If information is insufficient, state what's missing
5. Use quotes sparingly and only for critical definitions

Format your answer naturally but include citations like this:
"According to the agreement [Chunk ID: abc123, File: contract.txt], the fee structure is..."

Retrieved Chunks:
{context}

Previous Conversation Context:
{chat_context}

Question: {question}"""),
            ("human", "Please provide a comprehensive answer with citations.")
        ])
        
        self.refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are reviewing an answer to ensure it's complete and accurate.
Check if:
1. All parts of the question were answered
2. Citations are properly included
3. The answer is coherent and well-structured
4. Any contradictions or gaps are acknowledged

If improvements are needed, provide a refined version.
If the answer is satisfactory, return it as-is.

Original Question: {original_question}
Current Answer: {current_answer}
Retrieved Context: {context}"""),
            ("human", "Refine the answer if necessary.")
        ])
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a query and create a plan if needed.
        
        Args:
            query: User's query
            
        Returns:
            Analysis results including complexity and plan
        """
        chat_history = self.memory.get_messages()
        
        response = self.planning_prompt.format_messages(
            query=query,
            chat_history=chat_history
        )
        
        llm_response = llm.invoke(response)
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle potential markdown formatting)
            content = llm_response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            return analysis
        except (json.JSONDecodeError, Exception) as e:
            # Fallback to simple plan if parsing fails
            return {
                "complexity": "moderate",
                "requires_planning": False,
                "steps": [{
                    "step_number": 1,
                    "type": "retrieve",
                    "description": "Retrieve relevant chunks",
                    "sub_query": query,
                    "reason": "Direct retrieval should suffice"
                }],
                "categories_to_search": [],
                "reasoning": f"Failed to parse complex plan: {str(e)}. Using simple retrieval."
            }
    
    def _execute_retrieval_step(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a retrieval step.
        
        Args:
            step: Step definition from the plan
            
        Returns:
            List of retrieved chunks
        """
        sub_query = step.get('sub_query', self.current_plan.query)
        categories = step.get('category')
        
        if categories and isinstance(categories, str):
            categories = [categories]
        elif not categories:
            categories = self.retrieval_service.discover_indexes()
        
        # Perform hybrid search
        results = self.retrieval_service.hybrid_search(
            query=sub_query,
            categories=categories,
            top_k=5  # Get more results for complex queries
        )
        
        return results
    
    def _format_chunks_for_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks for use in LLM context."""
        if not chunks:
            return "No relevant chunks retrieved."
        
        formatted = []
        for chunk in chunks:
            # Ensure chunk_id exists, generate one if missing
            chunk_id = chunk.get('chunk_id')
            if chunk_id is None:
                # Try to generate from available fields
                file_name = chunk.get('file_name', 'unknown')
                text_preview = chunk.get('text', '')[:50].replace(' ', '_')
                chunk_id = f"{file_name}_{hash(text_preview) % 10000}"
                chunk['chunk_id'] = chunk_id
                logger.warning(f"Generated missing chunk_id: {chunk_id}")
            
            chunk_info = (
                f"[Chunk ID: {chunk_id}]\n"
                f"File: {chunk.get('file_name', 'Unknown')}\n"
                f"Category: {chunk.get('category', 'Unknown')}\n"
                f"Score: {chunk.get('score', 0):.4f}\n"
                f"Content: {chunk.get('text', '')}\n"
            )
            formatted.append(chunk_info)
        
        return "\n---\n".join(formatted)
    
    def _synthesize_answer(self, question: str, chunks: List[Dict[str, Any]], 
                          chat_context: str = "") -> str:
        """
        Synthesize an answer from retrieved chunks.
        
        Args:
            question: Original question
            chunks: Retrieved chunks
            chat_context: Previous conversation context
            
        Returns:
            Synthesized answer with citations
        """
        context = self._format_chunks_for_context(chunks)
        
        prompt = self.answer_synthesis_prompt.format_messages(
            context=context,
            chat_context=chat_context or "No previous context.",
            question=question
        )
        
        response = llm.invoke(prompt)
        return response.content
    
    def _refine_answer(self, question: str, answer: str, 
                      chunks: List[Dict[str, Any]]) -> str:
        """
        Refine an answer to ensure completeness and accuracy.
        
        Args:
            question: Original question
            answer: Current answer
            chunks: Retrieved chunks
            
        Returns:
            Refined answer
        """
        context = self._format_chunks_for_context(chunks)
        
        prompt = self.refinement_prompt.format_messages(
            original_question=question,
            current_answer=answer,
            context=context
        )
        
        response = llm.invoke(prompt)
        return response.content
    
    def _execute_simple_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a simple single-step query.
        
        Args:
            query: User's query
            
        Returns:
            Response dictionary with answer and metadata
        """
        logger.info(f"Executing simple query: {query[:100]}...")
        start_time = time.time()
        
        # Discover available categories
        categories = self.retrieval_service.discover_indexes()
        logger.info(f"Searching categories: {categories}")
        
        # Perform retrieval
        retrieval_start = time.time()
        chunks = self.retrieval_service.hybrid_search(
            query=query,
            categories=categories,
            top_k=5
        )
        retrieval_time = time.time() - retrieval_start
        logger.info(f"Retrieval completed in {retrieval_time:.2f}s, found {len(chunks)} chunks")
        
        # Get conversation context
        chat_context = self.memory.get_context_summary()
        
        # Synthesize answer
        synthesis_start = time.time()
        answer = self._synthesize_answer(query, chunks, chat_context)
        synthesis_time = time.time() - synthesis_start
        logger.info(f"Answer synthesis completed in {synthesis_time:.2f}s")
        
        # Refine answer
        refinement_start = time.time()
        refined_answer = self._refine_answer(query, answer, chunks)
        refinement_time = time.time() - refinement_start
        logger.info(f"Answer refinement completed in {refinement_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Simple query completed in {total_time:.2f}s")
        
        return {
            'answer': refined_answer,
            'chunks_used': chunks,
            'plan': None,
            'complexity': 'simple',
            'timing': {
                'retrieval': retrieval_time,
                'synthesis': synthesis_time,
                'refinement': refinement_time,
                'total': total_time
            }
        }
    
    def _execute_complex_query(self, query: str, 
                              analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complex multi-step query.
        
        Args:
            query: User's query
            analysis: Query complexity analysis
            
        Returns:
            Response dictionary with answer and metadata
        """
        logger.info(f"Executing complex query: {query[:100]}...")
        start_time = time.time()
        
        # Create execution plan
        self.current_plan = QueryPlan(query)
        
        # Add steps from analysis
        for step_def in analysis.get('steps', []):
            self.current_plan.add_step(
                step_type=step_def.get('type', 'retrieve'),
                description=step_def.get('description', ''),
                sub_query=step_def.get('sub_query', query),
                category=step_def.get('category')
            )
        
        self.current_plan.status = "in_progress"
        logger.info(f"Created plan with {len(self.current_plan.steps)} steps")
        
        # Execute each step
        all_chunks = []
        step_results = []
        step_timings = []
        
        for step in self.current_plan.steps:
            step_index = step['index']
            step_start = time.time()
            logger.info(f"Executing step {step_index + 1}: {step['type']} - {step['description'][:50]}...")
            
            try:
                if step['type'] == 'retrieve':
                    # Execute retrieval
                    chunks = self._execute_retrieval_step(step)
                    self.current_plan.mark_step_complete(step_index, chunks)
                    all_chunks.extend(chunks)
                    step_duration = time.time() - step_start
                    step_results.append({
                        'step': step_index,
                        'type': 'retrieval',
                        'chunks_count': len(chunks),
                        'duration': step_duration
                    })
                    step_timings.append({'step': step_index, 'type': 'retrieval', 'duration': step_duration})
                    logger.info(f"Step {step_index + 1} completed in {step_duration:.2f}s, retrieved {len(chunks)} chunks")
                    
                elif step['type'] == 'reason':
                    # Perform intermediate reasoning
                    reasoning_prompt = f"""Based on the following retrieved information, perform intermediate reasoning:
                    
{self._format_chunks_for_context(all_chunks)}

Task: {step['description']}
Sub-query: {step.get('sub_query', query)}

Provide your reasoning:"""
                    
                    reasoning_response = llm.invoke(reasoning_prompt)
                    self.current_plan.mark_step_complete(step_index, reasoning_response.content)
                    step_duration = time.time() - step_start
                    step_results.append({
                        'step': step_index,
                        'type': 'reasoning',
                        'summary': reasoning_response.content[:200],
                        'duration': step_duration
                    })
                    step_timings.append({'step': step_index, 'type': 'reasoning', 'duration': step_duration})
                    logger.info(f"Step {step_index + 1} reasoning completed in {step_duration:.2f}s")
                    
                elif step['type'] == 'synthesize':
                    # Intermediate synthesis
                    synthesis_prompt = f"""Synthesize the following information:
                    
{self._format_chunks_for_context(all_chunks)}

Previous reasoning:
{chr(10).join([str(r) for r in step_results])}

Task: {step['description']}

Provide synthesized insights:"""
                    
                    synthesis_response = llm.invoke(synthesis_prompt)
                    self.current_plan.mark_step_complete(step_index, synthesis_response.content)
                    step_duration = time.time() - step_start
                    step_results.append({
                        'step': step_index,
                        'type': 'synthesis',
                        'summary': synthesis_response.content[:200],
                        'duration': step_duration
                    })
                    step_timings.append({'step': step_index, 'type': 'synthesis', 'duration': step_duration})
                    logger.info(f"Step {step_index + 1} synthesis completed in {step_duration:.2f}s")
                    
                elif step['type'] == 'verify':
                    # Verification step
                    verify_prompt = f"""Verify the completeness and accuracy of the gathered information:
                    
Question: {query}
Gathered Information:
{self._format_chunks_for_context(all_chunks)}

Previous Steps:
{json.dumps(step_results, indent=2)}

Identify any gaps or inconsistencies:"""
                    
                    verify_response = llm.invoke(verify_prompt)
                    self.current_plan.mark_step_complete(step_index, verify_response.content)
                    step_duration = time.time() - step_start
                    step_results.append({
                        'step': step_index,
                        'type': 'verification',
                        'findings': verify_response.content[:200],
                        'duration': step_duration
                    })
                    step_timings.append({'step': step_index, 'type': 'verification', 'duration': step_duration})
                    logger.info(f"Step {step_index + 1} verification completed in {step_duration:.2f}s")
                    
            except Exception as e:
                logger.error(f"Step {step_index} failed: {str(e)}")
                self.current_plan.mark_step_failed(step_index, str(e))
                step_duration = time.time() - step_start
                step_results.append({
                    'step': step_index,
                    'type': 'error',
                    'error': str(e)
                })
        
        self.current_plan.status = "completed"
        
        # Final synthesis using all gathered information
        chat_context = self.memory.get_context_summary()
        synthesis_start = time.time()
        final_answer = self._synthesize_answer(query, all_chunks, chat_context)
        synthesis_time = time.time() - synthesis_start
        
        # Final refinement
        refinement_start = time.time()
        refined_answer = self._refine_answer(query, final_answer, all_chunks)
        refinement_time = time.time() - refinement_start
        
        total_time = time.time() - start_time
        logger.info(f"Complex query completed in {total_time:.2f}s (synthesis: {synthesis_time:.2f}s, refinement: {refinement_time:.2f}s)")
        
        return {
            'answer': refined_answer,
            'chunks_used': all_chunks,
            'plan': self.current_plan.to_dict(),
            'step_results': step_results,
            'step_timings': step_timings,
            'complexity': analysis.get('complexity', 'complex'),
            'timing': {
                'steps': step_timings,
                'synthesis': synthesis_time,
                'refinement': refinement_time,
                'total': total_time
            }
        }
    
    def chat(self, query: str) -> Dict[str, Any]:
        """
        Main chat method - processes user query and returns response.
        
        Args:
            query: User's query
            
        Returns:
            Dictionary containing answer, citations, and metadata
        """
        # Add user message to memory
        self.memory.add_message(HumanMessage(content=query))
        
        # Analyze query complexity
        analysis = self._analyze_query_complexity(query)
        
        # Execute based on complexity
        if analysis.get('requires_planning', False) and len(analysis.get('steps', [])) > 1:
            # Complex multi-step query
            response_data = self._execute_complex_query(query, analysis)
        else:
            # Simple query
            response_data = self._execute_simple_query(query)
        
        # Add assistant response to memory
        self.memory.add_message(AIMessage(content=response_data['answer']))
        
        # Format response
        response = {
            'answer': response_data['answer'],
            'citations': self._extract_citations(response_data['answer']),
            'chunks_used': [
                {
                    'chunk_id': chunk.get('chunk_id'),
                    'file_name': chunk.get('file_name'),
                    'category': chunk.get('category'),
                    'score': chunk.get('score'),
                    'text_preview': chunk.get('text', '')[:200]
                }
                for chunk in response_data.get('chunks_used', [])
            ],
            'plan': response_data.get('plan'),
            'step_results': response_data.get('step_results'),
            'timing': response_data.get('timing', {}),
            'complexity': response_data.get('complexity', 'simple'),
            'session_id': self.memory.session_id
        }
        
        logger.info(f"Query completed. Complexity: {response['complexity']}, Chunks: {len(response['chunks_used'])}, Citations: {len(response['citations'])}")
        
        return response
    
    def _extract_citations(self, answer: str) -> List[Dict[str, str]]:
        """Extract citation information from the answer."""
        import re
        
        # Pattern to match citations like [Chunk ID: xxx, File: yyy]
        pattern = r'\[Chunk ID:\s*([^\]]+),\s*File:\s*([^\]]+)\]'
        matches = re.findall(pattern, answer)
        
        citations = []
        for chunk_id, file_name in matches:
            citations.append({
                'chunk_id': chunk_id.strip(),
                'file_name': file_name.strip()
            })
        
        return citations
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        history = []
        for msg in self.memory.get_messages():
            if isinstance(msg, HumanMessage):
                history.append({'role': 'user', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                history.append({'role': 'assistant', 'content': msg.content})
        return history
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
        self.current_plan = None
    
    def get_available_categories(self) -> List[str]:
        """Get list of available index categories."""
        return self.retrieval_service.discover_indexes()


# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    agent = ChatAgent()
    
    print("Available categories:", agent.get_available_categories())
    print("\n" + "="*80 + "\n")
    
    # Example queries
    test_queries = [
        "What is a non-disclosure agreement?",
        "What are the key differences between the contracts in terms of confidentiality clauses?",
        "Calculate the total fees mentioned across all documents and explain the payment structure."
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}\n")
        
        response = agent.chat(query)
        
        print("Answer:")
        print(response['answer'])
        print(f"\nComplexity: {response['complexity']}")
        print(f"Citations: {len(response['citations'])}")
        
        if response['plan']:
            print("\nExecution Plan:")
            print(json.dumps(response['plan'], indent=2))
        
        print("\nChunks Used:")
        for chunk in response['chunks_used'][:3]:  # Show first 3
            print(f"  - {chunk['chunk_id']} from {chunk['file_name']} (score: {chunk['score']:.4f})")
        
        print("\n" + "-"*80)
