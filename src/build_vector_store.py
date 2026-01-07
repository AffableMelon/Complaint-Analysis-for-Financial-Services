import pandas as pd
import os
from sklearn.model_selection import train_test_split
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
    # Stratified sampling to 10K-15K complaints with proportional product representation
    TARGET_SAMPLE_SIZE = 12500 
    
    # Chunking Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    print("Loading processed data...")
    try:
        df = load_processed_data(PROCESSED_DATA_PATH)
        
        # Stratified sampling - Proportional
        if len(df) > TARGET_SAMPLE_SIZE:
            print(f"performing proportional stratified sampling to reduce to {TARGET_SAMPLE_SIZE} records...")
            # Use train_test_split to get a stratified subset
            df, _ = train_test_split(
                df, 
                train_size=TARGET_SAMPLE_SIZE, 
                stratify=df['normalized_product'], 
                random_state=42
            )
            print(f"Sampled dataset columns: {df['normalized_product'].value_counts(normalize=True)}")
        
        print(f"Chunking text (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
        documents = create_chunks(df, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"Created {len(documents)} chunks.")
        
        build_vector_store(documents, VECTOR_STORE_PATH)
        
    except FileNotFoundError:
        print("Processed data not found. Run src/eda_and_preprocessing.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")
