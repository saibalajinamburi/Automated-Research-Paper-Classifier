# Automated Research Paper Classifier

An end-to-end, industry-grade MLOps pipeline that automatically ingests raw research papers from the arXiv API, generates semantic embeddings using a Hugging Face Transformer, trains a LightGBM classifier, tracks experiments with MLflow, and deploys the model as a GPU-accelerated REST API inside a Docker container.

---

## Architecture

```
arXiv API → data_ingestion.py → arxiv_data.csv
                                      ↓
                              preprocess.py (all-MiniLM-L6-v2)
                                      ↓
                             X.npy + y.npy (384-dim embeddings)
                                      ↓
                              train.py (LightGBM + MLflow)
                                      ↓
                             models/classifier.pkl
                                      ↓
                 FastAPI (app/main.py) → Docker Container (GPU)
```

## Project Structure

```
arxiv_classifier/
├── .github/workflows/ci_cd.yml   # GitHub Actions CI pipeline
├── data/
│   ├── raw/                       # Raw CSV from arXiv API
│   └── processed/                 # Numpy embeddings (X.npy, y.npy)
├── src/
│   ├── data_ingestion.py          # Paginated arXiv API scraper + cleaner
│   ├── preprocess.py              # Sentence Transformer embedding generation
│   ├── train.py                   # LightGBM training + MLflow tracking
│   └── predict.py
├── app/
│   ├── main.py                    # FastAPI inference server
│   └── requirements.txt
├── models/
│   └── classifier.pkl             # Exported trained model
├── Dockerfile                     # GPU-enabled NVIDIA PyTorch container
├── .gitignore
└── README.md
```

## Results

| Metric | Score |
|---|---|
| **Accuracy** | 75.20% |
| **F1 Score (weighted)** | 66.60% |
| **Training Data** | ~500 real arXiv papers (quantum domain) |
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim) |
| **Classifier** | LightGBM (GBDT) |

## Tech Stack

- **Data**: arXiv Atom API, Pandas, Python `xml.etree`
- **ML**: Sentence Transformers (Hugging Face), LightGBM, scikit-learn
- **MLOps**: MLflow (experiment tracking, model registry)
- **API**: FastAPI, Uvicorn, Pydantic
- **Containerization**: Docker (NVIDIA PyTorch CUDA base image)
- **CI/CD**: GitHub Actions

## Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r app/requirements.txt
```

### 2. Run the pipeline
```bash
python src/data_ingestion.py    # Fetch & clean data
python src/preprocess.py        # Generate embeddings
python src/train.py             # Train model + log to MLflow
```

### 3. Start API locally
```bash
uvicorn app.main:app --reload
```

### 4. Run with Docker (GPU)
```bash
docker build -t arxiv_classifier:latest .
docker run --gpus all -p 8000:8000 arxiv_classifier:latest
```

### 5. View MLflow Dashboard
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open: http://127.0.0.1:5000

### 6. Test the API
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "We propose a new deep learning approach for quantum state classification."}'
```

## Author

**Saibalaji Namburi** — saibalajinamburi@gmail.com
