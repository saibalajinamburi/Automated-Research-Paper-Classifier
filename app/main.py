from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import logging

logging.basicConfig(level=logging.INFO)

# 1. Initialize FastAPI
app = FastAPI(title="ArXiv Paper Classifier API", version="1.0")

# 2. Define the expected input data structure
class PaperInput(BaseModel):
    text: str

# Global variables to hold our models
embedding_model = None
classifier_model = None
classes = None

@app.on_event("startup")
def load_models():
    global embedding_model, classifier_model, classes
    logging.info("Starting up API and loading models...")

    # Load the semantic embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load the category names saved during preprocessing
    try:
        classes = np.load("data/processed/classes.npy", allow_pickle=True)
    except FileNotFoundError:
        logging.error("Could not find classes.npy.")
        raise RuntimeError("Missing label encoder classes.")

    # Load the trained LightGBM model directly from disk (no MLflow dependency at runtime)
    try:
        classifier_model = joblib.load("models/classifier.pkl")
        logging.info("All models loaded successfully!")
    except FileNotFoundError:
        logging.error("Could not find models/classifier.pkl.")
        raise RuntimeError("Missing trained model file.")

@app.post("/predict")
def predict_category(paper: PaperInput):
    if not paper.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # 1. Convert raw text into a 384-dimensional vector embedding
    vector = embedding_model.encode([paper.text])

    # 2. Predict the category using the LightGBM model
    prediction_idx = classifier_model.predict(vector)[0]

    # 3. Map the numerical index back to the human-readable category name
    predicted_category = classes[prediction_idx]

    return {
        "input_text_preview": paper.text[:50] + "...",
        "predicted_category": predicted_category
    }

@app.get("/")
def health_check():
    return {"status": "API is live and models are loaded."}
