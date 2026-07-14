import streamlit as st
import sys
import os
import time

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
    st.session_state.pending_prompt = ""  # Used to capture button clicks

st.title("⚖️ Legal Document Chat Assistant")
st.caption("Powered by Ollama + Retrieval Augmented Generation")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "⏱️ Performance", "📄 Sources"])

with tab1:
    st.markdown("### 💬 Chat")
    
    # 1. Show Suggested Questions ONLY if the chat is empty
    if len(st.session_state.messages) == 0:
        st.markdown("##### 💡 Suggested Questions (Click to ask)")
        sq_col1, sq_col2 = st.columns(2)
        with sq_col1:
            if st.button("What are the termination rules?", use_container_width=True, key="sq_1"):
                st.session_state.pending_prompt = "What are the termination rules?"
            if st.button("How long is the confidentiality period?", use_container_width=True, key="sq_2"):
                st.session_state.pending_prompt = "How long is the confidentiality period?"
        with sq_col2:
            if st.button("Who owns the intellectual property?", use_container_width=True, key="sq_3"):
                st.session_state.pending_prompt = "Who owns the intellectual property?"
            if st.button("What are the payment terms?", use_container_width=True, key="sq_4"):
                st.session_state.pending_prompt = "What are the payment terms?"
        st.markdown("---")

    # 2. Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 3. Always show the chat input box at the bottom
    user_input = st.chat_input("Ask about legal documents...")

    # 4. Determine the final prompt (from typing OR from a suggested question button)
    prompt = user_input
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""  # Clear it after capturing

    # 5. Process the prompt if one exists
    if prompt:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                start_time = time.time()
                try:
                    response = st.session_state.agent.chat(prompt)
                    elapsed = time.time() - start_time
                    
                    st.markdown(response["answer"])
                    
                    # Store response for other tabs
                    st.session_state.last_response = response
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    
                    st.caption(f"⏱️ Completed in {elapsed:.2f}s")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

with tab2:
    st.markdown("### ⏱️ Performance Timing")
    if st.session_state.last_response:
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
                    # FIX: Added label and label_visibility to prevent empty label warning
                    cols[1].metric(label="Duration", value=f"{step_time:.3f}s", label_visibility="collapsed")
                    
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
    else:
        st.info("No query has been run yet. Performance stats will appear here after your first question.")

with tab3:
    st.markdown("### 📄 Source Chunks")
    if st.session_state.last_response:
        sources = st.session_state.last_response.get('sources', [])
        
        if not sources:
            st.info("No source chunks retrieved for this query.")
        else:
            for i, chunk in enumerate(sources):
                with st.expander(f"Chunk {i+1}: {chunk.get('chunk_id', 'N/A')} (Score: {chunk.get('score', 0):.3f})"):
                    st.markdown(f"**Source:** {chunk.get('source', 'Unknown')}")
                    st.markdown(f"**Category:** {chunk.get('category', 'N/A')}")
                    if chunk.get('document_id'):
                        st.markdown(f"**Document ID:** {chunk.get('document_id')}")
                    st.markdown("---")
                    st.text(chunk.get('text', ''))
    else:
        st.info("No query has been run yet. Sources will appear here after your first question.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.markdown("**Model:** llama3.2")
    st.markdown("**Embedding:** all-MiniLM-L6-v2")
    st.markdown("**Top K:** 3")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response = None
        st.session_state.pending_prompt = ""
        # Re-initialize agent to clear its internal conversation memory as well
        st.session_state.agent = ChatAgent()
        st.rerun()
    
    st.markdown("---")
    st.caption("Built with Streamlit + LangChain + Ollama")