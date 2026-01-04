# RAG-Powered Complaint Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot to analyze and answer questions about customer complaints for CrediTrust Financial.

## Project Structure

- `data/`: Contains raw and processed data.
- `vector_store/`: Stores the persisted ChromaDB index.
- `notebooks/`: Jupyter notebooks for EDA and experimentation.
- `src/`: Source code for preprocessing, embedding, and the RAG application.
- `app.py`: Gradio interface for the chatbot.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the raw data in `data/raw/`.

3. Run preprocessing:
   ```bash
   python src/eda_and_preprocessing.py
   ```

4. Build the vector store:
   ```bash
   python src/build_vector_store.py
   ```

5. Run the application:
   ```bash
   python app.py
   ```
