# Official NVIDIA PyTorch image: comes with CUDA 12.1 + PyTorch 2.5.1 pre-installed
# This is the industry standard for production GPU-accelerated AI containers
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /project

# Copy and install our app-specific dependencies
# NOTE: torch is already bundled in the base image, so we don't reinstall it
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ ./app/

# Copy the processed ML data (embeddings and label classes)
COPY data/processed/ ./data/processed/

# Copy the exported trained model (no MLflow dependency at inference time)
COPY models/ ./models/

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
