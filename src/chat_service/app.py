"""
Streamlit Chat UI for RAG System.
Provides an interactive chat interface for querying legal documents.
"""

import streamlit as st
import json
from typing import List, Dict, Any
from pathlib import Path

# Import the chat agent
from src.chat_service.chat_agent import ChatAgent


# Page configuration
st.set_page_config(
    page_title="Legal Document Chat Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #e3f2fd;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
    .citation-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9em;
    }
    .chunk-info {
        background-color: #e8f5e9;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.3rem;
        font-size: 0.85em;
    }
    .plan-step {
        background-color: #f3e5f5;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
    }
    .st-expander {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_agent():
    """Initialize or retrieve the chat agent from session state."""
    if 'chat_agent' not in st.session_state:
        st.session_state.chat_agent = ChatAgent()
    return st.session_state.chat_agent


def render_citation(citation: Dict[str, str]) -> str:
    """Render a citation in a formatted way."""
    return f"📄 **Chunk:** `{citation.get('chunk_id', 'N/A')}` | **File:** {citation.get('file_name', 'Unknown')}"


def render_chunk(chunk: Dict[str, Any]) -> str:
    """Render chunk information."""
    score = chunk.get('score', 0)
    score_color = "green" if score > 0.7 else "orange" if score > 0.4 else "red"
    
    return f"""
    <div class="chunk-info">
        <strong>📍 Source:</strong> {chunk.get('file_name', 'Unknown')}<br>
        <strong>Category:</strong> {chunk.get('category', 'N/A')}<br>
        <strong>Chunk ID:</strong> {chunk.get('chunk_id', 'N/A')}<br>
        <strong>Relevance Score:</strong> <span style="color: {score_color};">{score:.4f}</span><br>
        <details>
            <summary>View Content Preview</summary>
            <p style="font-size: 0.85em; margin-top: 0.5rem;">{chunk.get('text_preview', '')}...</p>
        </details>
    </div>
    """


def render_plan(plan: Dict[str, Any]):
    """Render the execution plan for complex queries."""
    if not plan:
        return
    
    st.markdown("### 📋 Execution Plan")
    
    complexity = plan.get('complexity', 'unknown')
    complexity_emoji = {"simple": "🟢", "moderate": "🟡", "complex": "🔴"}.get(complexity, "⚪")
    st.markdown(f"**Complexity:** {complexity_emoji} {complexity.capitalize()}")
    
    steps = plan.get('steps', [])
    if steps:
        st.markdown("**Steps:**")
        for step in steps:
            step_type = step.get('type', 'unknown')
            step_emoji = {
                'retrieve': '🔍',
                'reason': '🤔',
                'synthesize': '🧩',
                'verify': '✅'
            }.get(step_type, '📌')
            
            status = step.get('status', 'pending')
            status_emoji = {"completed": "✅", "in_progress": "⏳", "pending": "⏸️", "failed": "❌"}.get(status, "")
            
            with st.expander(f"{step_emoji} Step {step.get('index', 0) + 1}: {step.get('description', 'No description')} {status_emoji}"):
                st.markdown(f"**Type:** {step_type}")
                st.markdown(f"**Status:** {status}")
                if step.get('sub_query'):
                    st.markdown(f"**Sub-query:** {step.get('sub_query')}")
                if step.get('result'):
                    result = step.get('result')
                    if isinstance(result, list):
                        st.markdown(f"**Result:** {len(result)} chunks retrieved")
                    elif isinstance(result, str):
                        st.markdown(f"**Result:** {result[:200]}...")
                if step.get('error'):
                    st.error(f"Error: {step.get('error')}")


def main():
    """Main Streamlit application."""
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Initialize agent
        agent = initialize_agent()
        
        # Show available categories
        st.markdown("### 📚 Available Corpora")
        categories = agent.get_available_categories()
        if categories:
            for cat in categories:
                st.markdown(f"✓ {cat}")
        else:
            st.warning("No indexes found. Please run ingestion first.")
        
        st.divider()
        
        # Session controls
        st.markdown("### 💬 Session Controls")
        
        if st.button("🗑️ Clear Conversation History", use_container_width=True):
            agent.clear_memory()
            st.session_state.messages = []
            st.rerun()
        
        # Show session info
        if 'chat_agent' in st.session_state:
            st.info(f"Session ID: `{agent.memory.session_id[:8]}...`")
        
        st.divider()
        
        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        This chat assistant helps you query legal documents using advanced RAG (Retrieval-Augmented Generation).
        
        **Features:**
        - 🔍 Hybrid search (BM25 + Semantic)
        - 🧠 Multi-step reasoning for complex queries
        - 📝 Automatic citation of sources
        - 💾 Conversation memory
        
        **Model:** Llama 3.2 via Ollama
        """)
    
    # Main chat area
    st.title("⚖️ Legal Document Chat Assistant")
    st.markdown("Ask questions about legal contracts, agreements, and documents. The assistant will find relevant information and cite its sources.")
    
    # Initialize message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show additional info for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                
                # Show citations
                if metadata.get('citations'):
                    with st.expander(f"📚 Citations ({len(metadata['citations'])})"):
                        for citation in metadata['citations']:
                            st.markdown(render_citation(citation))
                
                # Show chunks used
                if metadata.get('chunks_used'):
                    with st.expander(f"📄 Source Chunks ({len(metadata['chunks_used'])})"):
                        for chunk in metadata['chunks_used']:
                            st.markdown(render_chunk(chunk), unsafe_allow_html=True)
                
                # Show execution plan for complex queries
                if metadata.get('plan'):
                    render_plan(metadata['plan'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about legal documents..."):
        # Add user message to display
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from agent
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your query and searching documents..."):
                agent = initialize_agent()
                
                try:
                    response = agent.chat(prompt)
                    
                    # Display answer
                    st.markdown(response['answer'])
                    
                    # Store metadata for expandable sections
                    metadata = {
                        'citations': response.get('citations', []),
                        'chunks_used': response.get('chunks_used', []),
                        'plan': response.get('plan'),
                        'complexity': response.get('complexity', 'simple')
                    }
                    
                    # Add assistant message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response['answer'],
                        "metadata": metadata
                    })
                    
                    # Show quick stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sources Used", len(response.get('chunks_used', [])))
                    with col2:
                        st.metric("Citations", len(response.get('citations', [])))
                    with col3:
                        complexity = response.get('complexity', 'simple')
                        st.metric("Query Complexity", complexity.capitalize())
                    
                except Exception as e:
                    st.error(f"❌ Error processing your query: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"I encountered an error while processing your query: {str(e)}. Please try rephrasing your question."
                    })
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        Powered by RAG (Retrieval-Augmented Generation) • Llama 3.2 • Hybrid Search
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
