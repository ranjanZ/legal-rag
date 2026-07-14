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
    st.session_state.pending_prompt = ""

st.title("⚖️ Legal Document Chat Assistant")
st.caption("Powered by Ollama + Retrieval Augmented Generation")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "⏱️ Performance", "📄 Sources"])

with tab1:
    st.markdown("### 💬 Chat")
    
    # 1. Show Suggested Questions ONLY if chat is empty AND no prompt is currently processing
    if len(st.session_state.messages) == 0 and not st.session_state.pending_prompt:
        st.markdown("##### 💡 Suggested Questions (Click to ask)")
        sq_col1, sq_col2 = st.columns(2)
        with sq_col1:
            if st.button("How are profits and losses shared in the ACCELERATED TECHNOLOGIES joint venture?", use_container_width=True, key="sq_1"):
                st.session_state.pending_prompt = "How are profits and losses shared in the ACCELERATED TECHNOLOGIES joint venture?"
                st.rerun()  # Force immediate rerun to process the click
            if st.button("Are there any warranty disclaimers in the NSK supplier agreement?", use_container_width=True, key="sq_2"):
                st.session_state.pending_prompt = "Are there any warranty disclaimers in the NSK supplier agreement?"
                st.rerun()
        with sq_col2:
            if st.button("What confidentiality obligations are specified in the ABILITY INC agreement?", use_container_width=True, key="sq_3"):
                st.session_state.pending_prompt = "What confidentiality obligations are specified in the ABILITY INC agreement?"
                st.rerun()
            if st.button("How are profits and losses shared in the ACCELERATED TECHNOLOGIES joint venture?", use_container_width=True, key="sq_4"):
                st.session_state.pending_prompt = "How are profits and losses shared in the ACCELERATED TECHNOLOGIES joint venture?"
                st.rerun()
        st.markdown("---")

    # 2. Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Chat input (Streamlit natively keeps this sticky at the bottom of the viewport)
    user_input = st.chat_input("Ask about legal documents...")

    # 4. Determine the final prompt (from typing OR from a suggested question button)
    prompt = user_input or st.session_state.pending_prompt
    
    if prompt:
        # Clear pending prompt immediately so it doesn't trigger again
        if st.session_state.pending_prompt:
            st.session_state.pending_prompt = ""
            
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # FIX: Render user message immediately so it shows up while the assistant is "thinking"
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
                    
                    # Store response for other tabs
                    st.session_state.last_response = response
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    
                    st.caption(f"⏱️ Completed in {elapsed:.2f}s")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

    # Add slight bottom padding so the chat input doesn't overlap the last message on short pages
    st.markdown("<br><br><br>", unsafe_allow_html=True)


with tab2:
    st.markdown("### ⏱️ Performance Timing")
    if st.session_state.last_response:
        resp = st.session_state.last_response
        st.metric("Total Time", f"{resp['total_time']:.3f}s")
        
        if resp.get('steps'):
            st.markdown("#### Step Breakdown")
            for i, step in enumerate(resp['steps']):
                step_name = step.get('step', f'Step {i}')
                step_time = step.get('time', 0)
                
                if step_name != 'total':
                    cols = st.columns([2, 1, 3])
                    cols[0].markdown(f"**{step_name}**")
                    cols[1].metric(label="Duration", value=f"{step_time:.3f}s", label_visibility="collapsed")
                    
                    if 'details' in step:
                        cols[2].caption(step['details'])
                    elif 'chunks_found' in step:
                        cols[2].caption(f"Found {step['chunks_found']} chunks")
                    elif 'model' in step:
                        cols[2].caption(f"Model: {step['model']}")
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
    st.markdown("**Model:** qwen2.5:1.5b-instruct-q4_K_M")
    st.markdown("**Embedding:** all-MiniLM-L6-v2")
    st.markdown("**Top K:** 3")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response = None
        st.session_state.pending_prompt = ""
        st.session_state.agent = ChatAgent()
        st.rerun()
    
    st.markdown("---")
    st.caption("Built with Streamlit + LangChain + Ollama")