import pandas as pd
import os

def load_data(filepath, chunksize=None):
    """Loads the dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, chunksize=chunksize, low_memory=False)

def filter_data(df):
    """Filters the dataset for specific products and non-empty narratives."""
    target_products = [
        "Credit card",
        "Credit card or prepaid card", # Often grouped
        "Personal loan", # Often grouped as 'Payday loan, title loan, or personal loan'
        "Savings account", # Often grouped as 'Checking or savings account'
        "Money transfers", # Often grouped as 'Money transfer, virtual currency, or money service'
        "Money transfer, virtual currency, or money service",
        "Payday loan, title loan, or personal loan",
        "Checking or savings account"
    ]
    
    # Note: The exact product names in CFPB dataset might vary slightly over years.
    # We will filter based on the provided list in the prompt, but being broad for safety.
    # The prompt specifies: Credit Cards, Personal Loans, Savings Accounts, Money Transfers.
    
    # Let's stick to a broader filter first or exact matches if we knew the exact strings.
    # For now, I'll implement a check based on the prompt's categories.
    
    # Standardizing product names for the project
    product_map = {
        "Credit card": "Credit card",
        "Credit card or prepaid card": "Credit card",
        "Prepaid card": "Credit card", # Optional inclusion
        "Payday loan, title loan, or personal loan": "Personal loan",
        "Personal loan": "Personal loan",
        "Checking or savings account": "Savings account",
        "Savings account": "Savings account",
        "Money transfer, virtual currency, or money service": "Money transfers",
        "Money transfers": "Money transfers"
    }
    
    df['normalized_product'] = df['Product'].map(product_map)
    df = df.dropna(subset=['normalized_product'])
    
    # Filter for non-empty narratives
    # The column name for narrative is usually 'Consumer complaint narrative'
    if 'Consumer complaint narrative' in df.columns:
        df = df.dropna(subset=['Consumer complaint narrative'])
    else:
        print("Warning: 'Consumer complaint narrative' column not found.")
        
    return df

def clean_text(text):
    """Cleans the text narrative."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove boilerplate (example)
    text = text.replace("xxxx", "") # CFPB redactions often use XXXX
    
    # Basic whitespace cleanup
    text = " ".join(text.split())
    
    return text

def preprocess_data(input_path, output_path):
    """Main function to run the preprocessing pipeline."""
    print(f"Processing data from {input_path}...")
    
    try:
        # Use chunking to avoid OOM errors
        chunk_size = 50000
        first_chunk = True
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        for chunk in load_data(input_path, chunksize=chunk_size):
            # Filter
            chunk = filter_data(chunk)
            
            if chunk.empty:
                continue
                
            # Clean narratives
            chunk['cleaned_narrative'] = chunk['Consumer complaint narrative'].apply(clean_text)
            
            # Save
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            chunk.to_csv(output_path, index=False, mode=mode, header=header)
            first_chunk = False
            
        print(f"Saved processed data to {output_path}")
        print("Done.")

    except FileNotFoundError:
        print("Data file not found. Please ensure data is in data/raw/")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define paths
    RAW_DATA_PATH = "data/raw/complaints.csv" # Assumed name
    PROCESSED_DATA_PATH = "data/processed/filtered_complaints.csv"
    
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
