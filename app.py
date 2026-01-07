import streamlit as st
import os
import sys
import torch

# Ensure src is in python path
sys.path.append(os.path.abspath("src"))

try:
    from src.rag_pipeline import get_retriever, get_llm, get_rag_chain, VECTOR_STORE_PATH
except ImportError as e:
    st.error(f"Error importing RAG modules: {e}")
    st.stop()

st.set_page_config(page_title="CrediTrust RAG Chat", page_icon="🏦", layout="wide")

st.title("🏦 CrediTrust Customer Complaint Assistant")
st.markdown("Ask questions about customer complaints and view the sources used to generate the answer.")

@st.cache_resource
def load_rag_pipeline():
    if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
        return None, "Vector store not found. Please run the build script first."
    
    retriever = get_retriever(VECTOR_STORE_PATH)
    
    # Determine optimal device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print(f"Attempting to load LLM on {device}...")
        llm = get_llm(device=device)
        rag_chain = get_rag_chain(retriever, llm)
        status_msg = f"RAG System Ready ({device.upper()})"
        print(status_msg)
        return rag_chain, status_msg
    except Exception as e:
        print(f"Failed to load on {device}: {e}")
        if device == "cuda":
            try:
                print("Falling back to CPU...")
                llm = get_llm(device="cpu")
                rag_chain = get_rag_chain(retriever, llm)
                status_msg = "RAG System Ready (CPU - Fallback)"
                print(status_msg)
                return rag_chain, status_msg
            except Exception as inner_e:
                 return None, f"Failed to initialize RAG pipeline: {str(inner_e)}"
        else:
            return None, f"Failed to initialize RAG pipeline: {str(e)}"

# Initialize pipeline
with st.spinner("Initializing RAG Pipeline... This may take a moment."):
    rag_chain, status = load_rag_pipeline()

if rag_chain is None:
    st.error(status)
    st.stop()
else:
    # Small toast or sidebar info for status
    st.sidebar.success(status)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                st.markdown(message["sources"])

# Input for new question
if prompt := st.chat_input("What would you like to know about the complaints?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources_text = ""
        
        with st.spinner("Thinking..."):
            try:
                result = rag_chain.invoke(prompt)
                answer = result["answer"]
                
                # Cleanup answer format
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
                
                full_response = answer
                message_placeholder.markdown(full_response)
                
                # Process sources
                docs = result.get("docs", [])
                if docs:
                    formatted_sources = []
                    for i, doc in enumerate(docs):
                        metadata = doc.metadata
                        content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        formatted_sources.append(f"**Source {i+1}** (ID: {metadata.get('complaint_id', 'N/A')})\n>{content}")
                    sources_text = "\n\n".join(formatted_sources)
                    
            except Exception as e:
                full_response = f"An error occurred: {str(e)}"
                message_placeholder.error(full_response)
        
        # Display sources in an expander below the answer
        if sources_text:
            with st.expander("View Sources"):
                st.markdown(sources_text)
                
    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": sources_text
    })

# Clear chat button
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()
