import gradio as gr
from src.rag_pipeline import RAGPipeline
import os

# Initialize RAG Pipeline
# Ensure vector store exists or handle gracefully
VECTOR_STORE_PATH = "vector_store"

rag = None
if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
    try:
        rag = RAGPipeline(VECTOR_STORE_PATH)
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
else:
    print("Vector store not found. Please run src/build_vector_store.py first.")

def chat_function(message, history, product_category):
    if not rag:
        return "System not initialized. Please build the vector store first."
    
    answer, docs = rag.query(message, product_filter=product_category)
    
    # Format sources for display
    sources_text = "\n\n**Sources:**\n"
    for i, doc in enumerate(docs):
        sources_text += f"{i+1}. Product: {doc.metadata.get('product')}, Issue: {doc.metadata.get('issue')}\n"
    
    return answer + sources_text

# Gradio UI
with gr.Blocks(title="CrediTrust Complaint Chatbot") as demo:
    gr.Markdown("# CrediTrust Financial Complaint Chatbot")
    gr.Markdown("Ask questions about customer complaints to get insights.")
    
    with gr.Row():
        product_dropdown = gr.Dropdown(
            choices=["All", "Credit card", "Personal loan", "Savings account", "Money transfers"],
            value="All",
            label="Filter by Product"
        )
    
    chatbot = gr.ChatInterface(
        fn=chat_function,
        additional_inputs=[product_dropdown],
        title="Chat with Complaints Data"
    )

if __name__ == "__main__":
    demo.launch()
