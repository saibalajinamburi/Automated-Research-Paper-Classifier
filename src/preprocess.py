import pandas as pd
import numpy as np
import logging
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(input_path: str = "data/raw/arxiv_data.csv", output_dir: str = "data/processed/"):
    logging.info(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Drop rows with missing abstracts or categories
    df = df.dropna(subset=['abstract', 'category'])
    
    # 1. Combine Title and Abstract for maximum context
    df['text'] = df['title'] + ". " + df['abstract']
    
    # 2. Encode Labels (Categories)
    logging.info("Encoding categories...")
    le = LabelEncoder()
    y = le.fit_transform(df['category'])
    
    # Save the label mapping so we can decode predictions later in the API
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "classes.npy"), le.classes_)

    # 3. Generate Transformer Embeddings
    logging.info("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    # This downloads the model the first time, then caches it locally
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    logging.info("Generating dense embeddings. This might take a moment...")
    # encode() automatically uses a GPU if one is available
    X = model.encode(df['text'].tolist(), show_progress_bar=True)
    
    # 4. Save processed data
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    logging.info(f"Saved features (X) and labels (y) to {output_dir}")

if __name__ == "__main__":
    process_data()
