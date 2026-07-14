import os
import sys
import time
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- Suppress Noisy Third-Party Logs ---
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# --- Import Production Retrieval Service ---
# NOTE: Adjust this import path to match your exact project structure 
# (e.g., from src.retrieval.retrieval_service import RetrievalService)
from src.retrieval_service import RetrievalService

# Setup Logging
LOG_DIR = os.path.join(os.path.dirname(__file__))
LOG_FILE = os.path.join(LOG_DIR, "chat_service.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("src.chat_service.chat_agent")

# --- Configuration ---
#LLM_MODEL = "llama3.2"
LLM_MODEL="qwen2.5:1.5b-instruct-q4_K_M"
TOP_K = 3

# Simple Greeting Keywords
GREETINGS = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]


class ConversationMemory:
    """Manages conversation history for a session."""
    def __init__(self, session_id: str, max_history: int = 10):
        self.session_id = session_id
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []  # List of {role, content}

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context_string(self) -> str:
        if not self.history:
            return "No previous context."
        context_lines = []
        for msg in self.history:
            role_name = "User" if msg["role"] == "human" else "Assistant"
            context_lines.append(f"{role_name}: {msg['content']}")
        return "\n".join(context_lines)

    def clear(self):
        self.history = []


class ChatAgent:
    """Main Chat Agent - Orchestrates Memory, RetrievalService, and LLM."""
    
    def __init__(self, session_id: str = None, index_dir: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.memory = ConversationMemory(self.session_id)
        
        # Initialize the Production RetrievalService
        self.retriever = RetrievalService(index_dir=Path(index_dir) if index_dir else None)
        
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0,
            num_ctx=2048,
            num_thread=4
        )
        logger.info(f"ChatAgent initialized with model {LLM_MODEL} and Production RetrievalService")

    def _is_greeting(self, query: str) -> bool:
        q_lower = query.lower().strip()
        return any(q_lower == g or q_lower.startswith(g) for g in GREETINGS)

    def _generate_greeting_response(self, query: str) -> Dict[str, Any]:
        """Return instant greeting without LLM."""
        responses = {
            "hi": "Hello! How can I assist you with legal documents today?",
            "hello": "Hello! Feel free to ask about contracts, clauses, or legal terms.",
            "hey": "Hey there! What do you need help with?",
            "good morning": "Good morning! Ready to analyze some legal data?",
            "good afternoon": "Good afternoon! How can I help?",
            "good evening": "Good evening! What's on your mind?"
        }
        q_lower = query.lower().strip()
        response_text = "Hello! How can I assist you today?"
        
        for key, val in responses.items():
            if q_lower.startswith(key):
                response_text = val
                break
                
        return {
            "answer": response_text,
            "sources": [],
            "steps": [{"step": "greeting_detection", "time": 0.001, "details": "Instant greeting response"}],
            "total_time": 0.001
        }

    def _synthesize_answer(self, query: str, chunks: List[Dict], chat_context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        
        context_text = ""
        for i, chunk in enumerate(chunks):
            # Adapted to keys returned by RetrievalService
            chunk_id = chunk.get('chunk_id', f"C{i+1}")
            file_name = chunk.get('file_name', chunk.get('category', 'Unknown'))
            text = chunk['text']
            context_text += f"[{chunk_id}] Source: {file_name}\nContent: {text}\n\n"

        system_prompt = """You are a helpful legal assistant. 
Answer the user's question based ONLY on the provided context below.
- If the context contains the answer, provide it clearly.
- Cite your sources using the format [Chunk ID] at the end of relevant sentences.
- If the context does NOT contain the answer, state clearly: "I don't have enough information in the provided documents to answer this."
- Keep answers concise.
"""
        
        user_prompt = f"""
Context:
{context_text}

Previous Conversation:
{chat_context}

User Question: {query}

Answer:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return response.content
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return "Error generating response. Please try again."

    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat entry point."""
        start_total = time.time()
        steps_log = []
        
        # 1. Check for Greeting (Instant - NO LLM)
        if self._is_greeting(query):
            logger.info("Detected greeting. Returning instant response.")
            resp = self._generate_greeting_response(query)
            self.memory.add_message("human", query)
            self.memory.add_message("ai", resp['answer'])
            return resp

        # 2. Standard Flow: Retrieve -> Synthesize
        logger.info(f"Processing query: {query}")
        
        # Step A: Retrieval via Production RetrievalService (Hybrid Search)
        t_start = time.time()
        chunks = self.retriever.search(query=query, top_k=TOP_K, search_type='hybrid')
        t_retrieve = time.time() - t_start
        steps_log.append({"step": "retrieval", "time": t_retrieve, "chunks_found": len(chunks)})
        
        chat_context = self.memory.get_context_string()
        
        # Step B: Synthesis
        t_start = time.time()
        if not chunks:
            answer = "I couldn't find specific information in the indexed documents to answer that. Could you rephrase or provide more details?"
            steps_log.append({"step": "no_retrieval", "time": 0.0, "details": "No relevant chunks found"})
        else:
            answer = self._synthesize_answer(query, chunks, chat_context)
            t_synthesize = time.time() - t_start
            steps_log.append({"step": "synthesis", "time": t_synthesize, "model": LLM_MODEL})
        
        total_time = time.time() - start_total
        steps_log.append({"step": "total", "time": total_time})
        
        # Update Memory
        self.memory.add_message("human", query)
        self.memory.add_message("ai", answer)
        
        # Format Sources for UI (adapting to RetrievalService output keys)
        sources = []
        for c in chunks:
            sources.append({
                "chunk_id": c.get('chunk_id'),
                "document_id": c.get('document_id'),
                "source": c.get('file_name', c.get('category', 'Unknown')),
                "score": c.get('score'),
                "text": c.get('text')
            })

        return {
            "answer": answer,
            "sources": sources,
            "steps": steps_log,
            "total_time": total_time
        }


if __name__ == "__main__":
    import json

    print("="*70)
    print("Starting Chat Service Tests (Using Production RetrievalService)")
    print("="*70)

    try:
        # 1. Test Production RetrievalService directly
        print("\n" + "="*70)
        print("TEST 1: Production RetrievalService")
        print("="*70)
        
        retriever = RetrievalService()
        available_indexes = retriever.discover_indexes()
        print(f"Discovered indexes: {available_indexes}")
        
        if not available_indexes:
            print("\n[WARNING] No indexes found!")
            print("Please ensure you have run the ingestion pipeline first.")
            print("The chat agent will correctly fall back to 'no information found'.\n")
        else:
            query = "What is the formula for calculating the per Transaction Inquiry advertising fee?"
            print(f"\nQuery: '{query}'")
            results = retriever.search(query=query, top_k=3, search_type='hybrid')
            print(f"Retrieved {len(results)} chunks:")
            for r in results:
                print(f"  - [{r.get('chunk_id')}] Score: {r.get('score'):.4f} | File: {r.get('file_name')}")
                print(f"    Text: {r.get('text')[:100]}...")

        # 2. Test ChatAgent
        print("\n" + "="*70)
        print("TEST 2: ChatAgent")
        print("="*70)
        agent = ChatAgent(session_id="test_session_prod_123")
        
        # Test A: Greeting (should be instant, no LLM call)
        print("\n--- Test A: Greeting (Fast-path) ---")
        resp_greeting = agent.chat("Hello there!")
        print(f"Answer: {resp_greeting['answer']}")
        print(f"Total Time: {resp_greeting['total_time']:.4f}s")

        # Test B: Standard Query (Retrieval + LLM Synthesis)
        print("\n--- Test B: Standard Query (Retrieval + Synthesis) ---")
        print("NOTE: This requires Ollama to be running with 'llama3.2' pulled.")
        test_query = "What are the termination rules?"
        resp_query = agent.chat(test_query)
        print(f"Answer: {resp_query['answer']}")
        print(f"Total Time: {resp_query['total_time']:.4f}s")
        print(f"Steps: {json.dumps(resp_query['steps'], indent=2)}")
        print(f"Sources cited: {len(resp_query['sources'])}")

        # Test C: Follow-up Query (Tests Conversation Memory)
        print("\n--- Test C: Follow-up Query (Conversation Memory) ---")
        resp_followup = agent.chat("And what about confidentiality?")
        print(f"Answer: {resp_followup['answer']}")
        print(f"Total Time: {resp_followup['total_time']:.4f}s")

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        print(f"\n[Error] Test failed: {e}")
    
    print("\n" + "="*70)
    print("Tests Completed")
    print("="*70)