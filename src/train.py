import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_dir: str = "data/processed/"):
    logging.info("Loading processed tensor data...")
    
    try:
        X = np.load(os.path.join(data_dir, "X.npy"))
        y = np.load(os.path.join(data_dir, "y.npy"))
    except FileNotFoundError:
        logging.error("Could not find X.npy or y.npy. Ensure Step 3 (preprocess.py) completed successfully.")
        return

    # Split the 10,000 papers: 80% for training, 20% for testing
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up MLflow to use SQLite as the backend (modern standard, avoids file store corruption)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("arxiv_quantum_classifier")

    with mlflow.start_run():
        logging.info("Initializing LightGBM Training...")
        
        # We define the hyperparameter dictionary so MLflow can log it
        params = {
            "objective": "multiclass",
            "num_class": len(np.unique(y)),
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "random_state": 42,
            # GPU acceleration: uncomment if CUDA is confirmed working
            # "device": "gpu"
        }
        
        mlflow.log_params(params)

        # Train the brain
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # Test the brain
        logging.info("Generating predictions on the test set...")
        y_pred = model.predict(X_test)

        # Calculate how well it did
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        logging.info(f"Training Complete! Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")

        # Log our success metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Save the actual trained model to MLflow's artifact registry
        mlflow.lightgbm.log_model(model, "model")
        logging.info("Model and metrics successfully logged to MLflow.")

if __name__ == "__main__":
    train_model()
