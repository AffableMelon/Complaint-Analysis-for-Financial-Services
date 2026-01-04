import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def load_processed_data(filepath):
    """Loads the processed dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def create_chunks(df, chunk_size=500, chunk_overlap=50):
    """Splits narratives into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    documents = []
    for _, row in df.iterrows():
        text = row['cleaned_narrative']
        if not isinstance(text, str) or not text.strip():
            continue
            
        # Create metadata dictionary
        metadata = {
            "complaint_id": str(row.get('Complaint ID', '')),
            "product": row.get('normalized_product', 'Unknown'),
            "issue": row.get('Issue', 'Unknown'),
            "company": row.get('Company', 'Unknown'),
            "date_received": str(row.get('Date received', ''))
        }
        
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
            
    return documents

def build_vector_store(documents, persist_directory, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Embeds documents and stores them in ChromaDB."""
    print(f"Initializing embedding model: {embedding_model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    print(f"Creating vector store in {persist_directory}...")
    # Chroma automatically persists if persist_directory is provided
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Vector store created and persisted.")
    return vector_store

if __name__ == "__main__":
    PROCESSED_DATA_PATH = "data/processed/filtered_complaints.csv"
    VECTOR_STORE_PATH = "vector_store"
    
    # Sampling for Task 2 (as per instructions)
    SAMPLE_SIZE = 10000
    
    print("Loading processed data...")
    try:
        df = load_processed_data(PROCESSED_DATA_PATH)
        
        # Stratified sampling
        print(f"Sampling {SAMPLE_SIZE} records...")
        if len(df) > SAMPLE_SIZE:
            df = df.groupby('normalized_product', group_keys=False).apply(lambda x: x.sample(min(len(x), int(SAMPLE_SIZE / 5))))
        
        print("Chunking text...")
        documents = create_chunks(df)
        print(f"Created {len(documents)} chunks.")
        
        build_vector_store(documents, VECTOR_STORE_PATH)
        
    except FileNotFoundError:
        print("Processed data not found. Run src/eda_and_preprocessing.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")
