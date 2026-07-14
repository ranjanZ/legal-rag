import os
import sys
import time
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sentence_transformers import SentenceTransformer
import numpy as np

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
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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


class ChunkRetriever:
    """Handles embedding and retrieval from the corpus."""
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus_lite")
        self.embedder = None
        self.chunks = []
        self.embeddings = None
        self._load_data()

    def _load_data(self):
        """Load chunks from JSON/Text files in the data directory."""
        logger.info(f"Loading data from {self.data_dir}")
        import json
        import glob
        
        all_chunks = []
        if os.path.exists(self.data_dir):
            for file_path in glob.glob(os.path.join(self.data_dir, "**", "*.json"), recursive=True):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                chunk_text = item.get('text') or item.get('content') or str(item)
                                if chunk_text:
                                    all_chunks.append({
                                        "id": item.get('id', str(uuid.uuid4())[:8]),
                                        "text": chunk_text,
                                        "source": os.path.basename(file_path),
                                        "category": os.path.basename(os.path.dirname(file_path))
                                    })
                        elif isinstance(data, dict):
                             pass 
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
        
        if not all_chunks:
            logger.warning("No chunks loaded from data directory. Retrieval will return empty results.")
        
        self.chunks = all_chunks
        logger.info(f"Loaded {len(self.chunks)} chunks.")

    def _ensure_embeddings(self):
        if self.embeddings is not None or not self.chunks:
            return
        
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        texts = [c['text'] for c in self.chunks]
        logger.info("Computing embeddings...")
        self.embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        logger.info("Embeddings computed.")

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks for a query."""
        if not self.chunks:
            return []
        
        self._ensure_embeddings()
        
        start_time = time.time()
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        sims = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(sims)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if sims[idx] > 0.1:
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(sims[idx])
                if 'chunk_id' not in chunk:
                    chunk['chunk_id'] = chunk.get('id', f"chunk_{idx}")
                results.append(chunk)
        
        elapsed = time.time() - start_time
        logger.info(f"Retrieval completed in {elapsed:.2f}s, found {len(results)} chunks")
        return results


class ChatAgent:
    """Main Chat Agent - Simplified for speed."""
    
    def __init__(self, session_id: str = None, data_dir: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.memory = ConversationMemory(self.session_id)
        self.retriever = ChunkRetriever(data_dir)
        
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0,
            num_ctx=2048,
            num_thread=4
        )
        logger.info(f"ChatAgent initialized with model {LLM_MODEL}")

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
            chunk_id = chunk.get('chunk_id', f"C{i+1}")
            source = chunk.get('source', 'Unknown')
            text = chunk['text']
            context_text += f"[{chunk_id}] Source: {source}\nContent: {text}\n\n"

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
        
        # Step A: Retrieval
        t_start = time.time()
        chunks = self.retriever.retrieve(query)
        t_retrieve = time.time() - t_start
        steps_log.append({"step": "retrieval", "time": t_retrieve, "chunks_found": len(chunks)})
        
        chat_context = self.memory.get_context_string()
        
        # Step B: Synthesis
        t_start = time.time()
        if not chunks:
            answer = "I couldn't find specific information in the documents to answer that. Could you rephrase or provide more details?"
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
        
        # Format Sources for UI
        sources = []
        for c in chunks:
            sources.append({
                "chunk_id": c.get('chunk_id'),
                "source": c.get('source'),
                "score": c.get('score'),
                "text": c.get('text')
            })

        return {
            "answer": answer,
            "sources": sources,
            "steps": steps_log,
            "total_time": total_time
        }
