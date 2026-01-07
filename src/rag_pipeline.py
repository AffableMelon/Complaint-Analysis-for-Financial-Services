import os
from typing import List, Optional
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Default configuration
VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Qwen2.5-1.5B-Instruct is efficient and powerful.
# google/flan-t5-base
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 

def get_retriever(vector_store_path: str = VECTOR_STORE_PATH, k: int = 5, device: str = None):
    """
    Loads the Chroma vector store and returns a retriever.
    """
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Vector store not found at {vector_store_path}")

    model_kwargs = {}
    if device:
        model_kwargs["device"] = device

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)
    
    # Initialize Chroma from existing directory
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embeddings
    )
    
    return vector_store.as_retriever(search_kwargs={"k": k})

def format_docs(docs: List[Document]) -> str:
    """
    Formats the retrieved documents into a single string for the prompt context.
    """
    formatted_docs = []
    for doc in docs:
        source_info = f"[Source: Complaint ID {doc.metadata.get('complaint_id', 'N/A')}]"
        content = doc.page_content.replace("\n", " ")
        formatted_docs.append(f"{source_info} {content}")
    
    return "\n\n".join(formatted_docs)

def get_llm(model_id: str = LLM_MODEL_ID, device: str = None):
    """
    Initializes and returns the LLM pipeline.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if we should use a specific task for the model
    task = "text-generation"
    if "t5" in model_id:
        task = "text2text-generation"
        
    model_kwargs = {"temperature": 0.1, "do_sample": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        
    pipe = pipeline(
        task=task,
        model=model_id,
        device=0 if device == "cuda" else -1,
        max_new_tokens=512,
        model_kwargs=model_kwargs
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def get_rag_chain(retriever, llm):
    """
    Constructs the RAG chain.
    """
    
    template = """
    You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
    Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, 
    state that you don't have enough information.

    Context: 
    {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate.from_template(template)

    def retrieve_with_docs(question):
        docs = retriever.invoke(question)
        return {"docs": docs, "context": format_docs(docs), "question": question}

    from langchain_core.runnables import RunnableLambda

    rag_chain_with_source = (
        RunnableLambda(retrieve_with_docs)
        | RunnablePassthrough.assign(
            answer=(
                PromptTemplate.from_template(template)
                | llm
                | StrOutputParser()
            )
        )
    )
    
    return rag_chain_with_source

def query_rag(question: str, rag_chain):
    """
    Runs the RAG chain for a single question.
    """
    result = rag_chain.invoke(question)
    answer = result["answer"]
    # Clean up the answer if it includes the prompt
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return answer

def get_representative_questions():
    """
    Returns a list of representative questions for evaluation.
    """
    return [
        "What are the common complaints about student loans?",
        "How does the company handle credit card disputes?",
        "What are the issues related to mortgage payments?",
        "Are there complaints about account closing difficulties?",
        "What do customers say about identity theft?",
        "What are the complaints regarding debt collection?",
        "How are credit reporting errors described?"
    ]

def evaluate_rag(rag_chain, questions, verbose=False):
    """
    Evaluates the RAG pipeline on a set of questions.
    """
    results = []
    for q in questions:
        try:
            result = rag_chain.invoke(q)
            
            # Clean up the answer
            answer_text = result["answer"]
            if "Answer:" in answer_text:
                answer_text = answer_text.split("Answer:")[-1].strip()

            # extract retrieved sources for display
            sources = []
            if "docs" in result:
                for idx, doc in enumerate(result["docs"][:2]): # show top 2
                     sources.append(f"Source {idx+1} (ID: {doc.metadata.get('complaint_id', 'N/A')}): {doc.page_content[:100]}...")
            
            # If verbose (script mode), include context in answer print
            if verbose:
                print(f"Question: {q}")
                print(f"Answer: {answer_text}")
                print("-" * 50)

            item = {
                "Question": q,
                "Generated Answer": answer_text,
                "Retrieved Sources": " | ".join(sources),
            }
            results.append(item)
        except Exception as e:
            results.append({
                "Question": q,
                "Generated Answer": f"Error: {e}",
                "Retrieved Sources": ""
            })
            
    return results

if __name__ == "__main__":
    # Test block
    try:
        retriever = get_retriever()
        llm = get_llm()
        rag_chain = get_rag_chain(retriever, llm)
        
        # Run Evaluation
        questions = get_representative_questions()
        results = evaluate_rag(rag_chain, questions, verbose=True)
        
        # Simple print of results
        # import pandas as pd
        # df = pd.DataFrame(results)
        # print("\nEvaluation Results:")
        # print(df.to_string())
        
    except Exception as e:
        print(f"Error initializing or running RAG: {e}")
