import streamlit as st
import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chat_service.chat_agent import ChatAgent

# Page config
st.set_page_config(
    page_title="Legal RAG Chat",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = ChatAgent()
    st.session_state.messages = []
    st.session_state.last_response = None

st.title("⚖️ Legal Document Chat Assistant")
st.caption("Powered by Ollama + Retrieval Augmented Generation")

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about legal documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                start_time = time.time()
                try:
                    response = st.session_state.agent.chat(prompt)
                    elapsed = time.time() - start_time
                    
                    st.markdown(response["answer"])
                    
                    # Store response for tabs
                    st.session_state.last_response = response
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    
                    # Show timing info
                    st.caption(f"⏱️ Completed in {elapsed:.2f}s")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

with col2:
    st.subheader("📊 Quick Stats")
    st.metric("Messages", len(st.session_state.messages))
    if st.session_state.last_response:
        st.metric("Last Response Time", f"{st.session_state.last_response['total_time']:.2f}s")
        st.metric("Chunks Found", len(st.session_state.last_response['sources']))

# Tabs for details
if st.session_state.last_response:
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "⏱️ Performance", "📄 Sources"])
    
    with tab1:
        st.markdown("### Conversation History")
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.info(f"**You:** {msg['content']}")
            else:
                st.success(f"**Assistant:** {msg['content']}")
    
    with tab2:
        st.markdown("### ⏱️ Performance Timing")
        resp = st.session_state.last_response
        
        # Total time
        st.metric("Total Time", f"{resp['total_time']:.3f}s")
        
        # Step-by-step breakdown
        if resp.get('steps'):
            st.markdown("#### Step Breakdown")
            for i, step in enumerate(resp['steps']):
                step_name = step.get('step', f'Step {i}')
                step_time = step.get('time', 0)
                
                if step_name != 'total':
                    cols = st.columns([2, 1, 3])
                    cols[0].markdown(f"**{step_name}**")
                    cols[1].metric("", f"{step_time:.3f}s")
                    if 'details' in step:
                        cols[2].caption(step['details'])
                    elif 'chunks_found' in step:
                        cols[2].caption(f"Found {step['chunks_found']} chunks")
                    elif 'model' in step:
                        cols[2].caption(f"Model: {step['model']}")
            
            # Visual timeline
            st.markdown("#### Timeline Visualization")
            steps_for_viz = [s for s in resp['steps'] if s.get('step') != 'total']
            if steps_for_viz:
                total = sum(s.get('time', 0) for s in steps_for_viz)
                if total > 0:
                    for step in steps_for_viz:
                        pct = (step.get('time', 0) / total) * 100
                        st.progress(pct / 100)
                        st.caption(f"{step.get('step')}: {step.get('time', 0):.3f}s ({pct:.1f}%)")
    
    with tab3:
        st.markdown("### 📄 Source Chunks")
        sources = resp.get('sources', [])
        
        if not sources:
            st.info("No source chunks retrieved for this query.")
        else:
            for i, chunk in enumerate(sources):
                with st.expander(f"Chunk {i+1}: {chunk.get('chunk_id', 'N/A')} (Score: {chunk.get('score', 0):.3f})"):
                    st.markdown(f"**Source:** {chunk.get('source', 'Unknown')}")
                    st.markdown(f"**Category:** {chunk.get('category', 'N/A')}")
                    st.markdown("---")
                    st.text(chunk.get('text', ''))

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.markdown("**Model:** llama3.2")
    st.markdown("**Embedding:** all-MiniLM-L6-v2")
    st.markdown("**Top K:** 3")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_response = None
        st.session_state.agent = ChatAgent()
        st.rerun()
    
    st.markdown("---")
    st.caption("Built with Streamlit + LangChain + Ollama")
