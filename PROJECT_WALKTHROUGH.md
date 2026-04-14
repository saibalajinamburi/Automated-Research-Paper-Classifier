# Complete Project Walkthrough — Automated Research Paper Classifier
**Author: Saibalaji Namburi | saibalajinamburi@gmail.com**
**GitHub: https://github.com/saibalajinamburi/Automated-Research-Paper-Classifier**

---

## Table of Contents
1. [What We Built](#what-we-built)
2. [Project Structure](#project-structure)
3. [Phase 1 — Data Ingestion](#phase-1--data-ingestion)
4. [Phase 2 — Preprocessing & Embeddings](#phase-2--preprocessing--embeddings)
5. [Phase 3 — GPU / CUDA Setup](#phase-3--gpu--cuda-setup)
6. [Phase 4 — Model Training + MLflow](#phase-4--model-training--mlflow)
7. [Phase 5 — FastAPI Deployment](#phase-5--fastapi-deployment)
8. [Phase 6 — Docker Containerization](#phase-6--docker-containerization)
9. [Phase 7 — Cloud Deployment](#phase-7--cloud-deployment)
10. [Phase 8 — CI/CD & GitHub](#phase-8--cicd--github)
11. [Every Command Explained](#every-command-explained)
12. [Problems We Faced & How We Fixed Them](#problems-we-faced--how-we-fixed-them)
13. [Final Results](#final-results)

---

## What We Built

A fully automated Machine Learning pipeline that:
1. **Pulls** real research papers from the arXiv public API
2. **Cleans** the messy XML text
3. **Converts** text to 384-dimensional mathematical vectors using a Hugging Face Transformer
4. **Trains** a LightGBM classifier to predict the paper's domain category
5. **Tracks** every experiment with MLflow (accuracy, hyperparameters, model artifacts)
6. **Serves** predictions via a FastAPI REST API
7. **Runs** inside a GPU-accelerated Docker container (NVIDIA CUDA)
8. **Deploys** to the cloud (Hugging Face Spaces / Render.com) for a public URL
9. **Verifies** code quality automatically via GitHub Actions CI/CD on every push

---

## Project Structure

```
Automated Research Paper Classifier/
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml           ← GitHub Actions CI pipeline (runs on every push)
│
├── data/
│   ├── raw/                    ← Raw data from API (excluded from git)
│   │   ├── arxiv_data.csv      ← The scraped papers as a table
│   │   └── raw_papers.json     ← Same data in JSON format
│   └── processed/              ← Cleaned ML-ready data (committed to git)
│       ├── X.npy               ← Feature matrix: 384 numbers per paper
│       ├── y.npy               ← Labels: integer category per paper
│       └── classes.npy         ← Mapping: integer → "quant-ph" etc.
│
├── src/
│   ├── data_ingestion.py       ← Fetches from arXiv API, cleans, saves to CSV
│   ├── preprocess.py           ← Converts text → transformer embeddings
│   ├── train.py                ← Trains LightGBM, tracks with MLflow
│   └── predict.py              ← (Placeholder for CLI predictions)
│
├── app/
│   ├── main.py                 ← FastAPI server (the API you send text to)
│   └── requirements.txt        ← All Python packages the app needs
│
├── models/
│   └── classifier.pkl          ← The trained model saved as a file
│
├── Dockerfile                  ← GPU build (NVIDIA CUDA, for local use)
├── Dockerfile.cloud            ← CPU build (lightweight, for Hugging Face / Render)
├── .dockerignore               ← Files Docker should ignore when building
├── .gitignore                  ← Files Git should NOT push to GitHub
├── README.md                   ← Public-facing project README
├── PROJECT_WALKTHROUGH.md      ← This file
├── mlflow.db                   ← MLflow experiment database (local)
└── pyproject.toml              ← Project metadata
```

---

## Phase 1 — Data Ingestion

### File: `src/data_ingestion.py`

### The goal
Build a live pipeline that talks to the arXiv API and pulls real published research papers.

### How arXiv works
arXiv is a free, open academic paper repository. It has a public API that returns papers in XML format.

The URL we hit:
```
http://export.arxiv.org/api/query?search_query=all:quantum&start=0&max_results=1000
```

- `search_query=all:quantum` — search for "quantum" in all fields
- `start=0` — beginning of results
- `max_results=1000` — how many papers per batch

### Key technical decisions inside the script

**1. SSL Certificate Fix**
Windows Python has a known bug where it cannot verify SSL certificates for HTTPS connections. It crashes with:
```
ssl.SSLCertVerificationError: certificate verify failed
```
Fix: bypass SSL verification using Python's built-in context:
```python
import ssl
ssl_context = ssl._create_unverified_context()
urllib.request.urlopen(url, context=ssl_context)
```

**2. XML Namespace Declaration**
The arXiv API response uses the Atom XML format with namespaces. Without declaring the namespace, Python's XML parser cannot find any tags.
```python
# WRONG — returns nothing
root.findall('entry')

# CORRECT — works because we declare the namespace
namespace = {'atom': 'http://www.w3.org/2005/Atom'}
root.findall('atom:entry', namespace)
```

**3. Safe data extraction (never crash on missing data)**
Real XML is messy. We check every node before reading it:
```python
title_node = entry.find('atom:title', namespace)
title = title_node.text if title_node is not None else "Unknown"
```

**4. Extracting the label (category)**
This is what our classifier learns to predict. It is the paper's academic domain:
```python
category_node = entry.find('atom:category', namespace)
category = category_node.attrib.get('term')  # e.g. "quant-ph", "cs.AI"
```

**5. Text cleaning**
XML formatting uses lots of whitespace for indentation. We collapse it:
```python
import re
clean_text = re.sub(r'\s+', ' ', text).strip()
```

**6. Pagination (to fetch 10,000+ papers safely)**
arXiv bans you with HTTP 429 if you make large single requests.
We fetch in batches of 1,000 with a 3-second pause between each:
```python
for start in range(0, max_results, batch_size):
    url = f"{base_url}&start={start}&max_results={batch_size}"
    data = fetch(url)
    time.sleep(3)  # Be polite! Required by arXiv terms of service
```

**7. Saving to disk (two formats)**
```python
# Saves JSON for debugging
with open('data/raw/raw_papers.json', 'w') as f:
    json.dump(papers, f)

# Saves CSV for Pandas downstream
df = pd.DataFrame(papers)
df.to_csv('data/raw/arxiv_data.csv', index=False)
```

---

## Phase 2 — Preprocessing & Embeddings

### File: `src/preprocess.py`

### The goal
Convert raw human-readable text into 384 numbers per paper that a machine learning model can understand.

### Why NOT TF-IDF (the old way)
TF-IDF just counts how often words appear. It does not understand meaning:
- "neural networks" and "deep learning" → look completely different
- "quantum entanglement" and "particle correlation" → look completely different

### Why Sentence Transformers (the correct way)
`all-MiniLM-L6-v2` is a small (~80MB), fast, free, locally-running AI model trained by Hugging Face. It understands semantic meaning:
- "neural networks" and "deep learning" → land near each other mathematically
- Works completely offline, no API key needed
- 384-dimensional output (each paper becomes 384 numbers)

### What the script does step by step

**1. Load the CSV**
```python
df = pd.read_csv('data/raw/arxiv_data.csv')
df = df.dropna()  # Remove rows with missing values
```

**2. Combine title + abstract**
More text = better signal for the model:
```python
df['text'] = df['title'] + ". " + df['abstract']
```

**3. Label Encoding**
Machine learning only understands numbers. "quant-ph" must become `0`:
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['category'])  # "quant-ph" → 0, "cs.AI" → 1, etc.

# Save the mapping so we can decode predictions later
np.save('data/processed/classes.npy', le.classes_)
```

**4. Generate embeddings using the Transformer**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# This is the main step — turns text into 384-number vectors
X = model.encode(df['text'].tolist(), show_progress_bar=True)
# X shape: (num_papers, 384)
```

**5. Save features and labels**
```python
np.save('data/processed/X.npy', X)  # The 384-dim vectors
np.save('data/processed/y.npy', y)  # The integer labels
```

---

## Phase 3 — GPU / CUDA Setup

### The goal
Use the laptop's NVIDIA RTX 3050 GPU to accelerate computation instead of the slow CPU.

### Verification commands

**Check if CUDA is installed on Windows:**
```powershell
nvcc --version
# Output: Cuda compilation tools, release 13.2

nvidia-smi
# Output: Shows GPU name, temperature, memory, CUDA version
```

**Check if PyTorch can see the GPU:**
```python
# Run inside Python
import torch
print(torch.cuda.is_available())  # True = GPU working, False = CPU only
print(torch.__version__)           # Should show "+cu121" for CUDA version
```

### Installing CUDA-compatible PyTorch

The default `pip install torch` downloads a CPU-only version that cannot use your GPU. We force install the CUDA 12.1 version:
```bash
# First uninstall CPU version
.\.venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y

# Install CUDA 12.1 version from official PyTorch wheel server
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

---

## Phase 4 — Model Training + MLflow

### File: `src/train.py`

### The goal
Train a classifier to predict paper categories from the 384-dimensional embeddings, and track every experiment so we never lose results.

### Why LightGBM (not a Neural Network)
The Transformer already extracted rich semantic features. At this stage we just need a fast, powerful classifier:
- LightGBM (Gradient Boosted Decision Trees) — used in production at Microsoft, Alibaba, Uber
- Trains in seconds (not hours like a neural network)
- No GPU needed for this stage
- Highly interpretable

### What MLflow does

MLflow is an open-source experiment tracker. Every time you run `train.py`, it:
1. Creates a new "run" in the database
2. Saves the exact hyperparameters used (learning rate, tree depth, etc.)
3. Records the accuracy and F1 score
4. Saves the trained model as an artifact

This means you can compare Run 1 (accuracy 72%) vs Run 2 (accuracy 75%) side by side.

### The training loop

```python
import mlflow

# Tell MLflow to use SQLite as its database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("arxiv_quantum_classifier")

with mlflow.start_run():
    # Define hyperparameters
    params = {
        "objective": "multiclass",
        "learning_rate": 0.05,
        "num_leaves": 31,
    }
    mlflow.log_params(params)  # Save them

    # Train the model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)  # Save the score

    # Save model artifact
    mlflow.lightgbm.log_model(model, "model")
```

### Export model for deployment

After training, export to a simple file so the API does not need MLflow:
```python
import joblib, mlflow

# Load from MLflow tracking database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
exp = mlflow.get_experiment_by_name("arxiv_quantum_classifier")
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
model = mlflow.lightgbm.load_model(f'runs:/{runs.iloc[0]["run_id"]}/model')

# Save to simple pickle file
import os; os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/classifier.pkl")
```

---

## Phase 5 — FastAPI Deployment

### File: `app/main.py`

### The goal
Wrap the model in a web server so any application, anywhere, can send text and get a prediction back.

### How the API works
```
Client sends: POST /predict
  Body: {"text": "A paper about quantum entanglement..."}

Server does:
  1. Loads text
  2. Runs all-MiniLM-L6-v2 to convert text → 384 numbers
  3. Feeds 384 numbers → LightGBM → predicts category integer
  4. Converts integer back to "quant-ph" using classes.npy

Client receives:
  {"predicted_category": "quant-ph", "input_text_preview": "A paper about..."}
```

### Endpoints

| Method | URL | What it does |
|---|---|---|
| `GET` | `/` | Health check — is the API alive? |
| `POST` | `/predict` | Send text, get predicted category |
| `GET` | `/docs` | Auto-generated interactive UI to test the API |

---

## Phase 6 — Docker Containerization

### What Docker does
Docker wraps your entire application (code + dependencies + Python version) into one portable unit called a **container**. It runs identically on any machine — your laptop, a server in Germany, a cloud machine in Singapore.

### Two Dockerfiles

| File | Base Image | Size | Use |
|---|---|---|---|
| `Dockerfile` | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` | ~10.5GB | Local GPU |
| `Dockerfile.cloud` | `python:3.10-slim` + CPU torch | ~3GB | Cloud deployment |

### Local GPU build and run

```bash
# Build the image (takes 5-10 minutes, downloads NVIDIA base ~3.1GB)
docker build -t arxiv_classifier:latest .

# Run with GPU, auto-restart on reboot (-d = detached/background)
docker run --gpus all -p 8000:8000 --restart unless-stopped -d --name arxiv_api arxiv_classifier:latest

# Check container is running
docker ps

# View live logs
docker logs arxiv_api --tail 20

# Stop the container
docker stop arxiv_api

# Remove it (so you can re-create with same name)
docker rm arxiv_api
```

### .dockerignore explanation
This file tells Docker what NOT to copy into the image when building:
```
.venv           ← 4GB+ Python environment, Docker installs its own
data/raw/       ← Not needed at inference time
src/            ← Training scripts not needed in production
.git/           ← Version history, wastes space
```

---

## Phase 7 — Cloud Deployment

### Option A: Hugging Face Spaces (RECOMMENDED — completely free, permanent URL)

**What you get**: A permanent public URL like `https://huggingface.co/spaces/saibalajinamburi/arxiv-classifier`

**Steps:**
1. Go to https://huggingface.co and create a free account
2. Go to https://huggingface.co/new-space
3. Set:
   - **Space name**: `arxiv-classifier`
   - **SDK**: `Docker`
4. Clone the HF Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/saibalajinamburi/arxiv-classifier
   ```
5. Copy your project files in:
   ```bash
   # Copy these files into the cloned HF repo
   cp Dockerfile.cloud ./arxiv-classifier/Dockerfile
   cp -r app/ ./arxiv-classifier/app/
   cp -r data/processed/ ./arxiv-classifier/data/processed/
   cp -r models/ ./arxiv-classifier/models/
   ```
6. Create the HF metadata file at the top of the Dockerfile or as `README.md`:
   ```yaml
   ---
   title: ArXiv Research Paper Classifier
   sdk: docker
   app_port: 7860
   ---
   ```
7. Push to Hugging Face:
   ```bash
   cd arxiv-classifier
   git add .
   git commit -m "Deploy classifier"
   git push
   ```
8. Hugging Face builds and deploys automatically. Visit your Space URL!

---

### Option B: Render.com (also free, public URL)

**What you get**: `https://arxiv-classifier.onrender.com`

**Steps:**
1. Go to https://render.com and connect your GitHub account
2. Click "New Web Service"
3. Select your GitHub repository
4. Set:
   - **Environment**: `Docker`
   - **Dockerfile path**: `Dockerfile.cloud`
   - **Port**: `7860`
5. Click Deploy
6. Wait ~10 minutes for first build

> **Note**: Render.com free tier **shuts down after 15 minutes of inactivity**. The first request after sleeping will take 30-60 seconds to wake up. This is normal for free tiers.

---

### Option C: Docker Hub (stores the image — others can pull and run locally)

```bash
# Login to Docker Hub
docker login

# Tag your image with your Docker Hub username
docker tag arxiv_classifier:latest saibalajinamburi/arxiv_classifier:latest

# Push the image (this was already done!)
docker push saibalajinamburi/arxiv_classifier:latest
```

Anyone in the world can now run your image:
```bash
docker pull saibalajinamburi/arxiv_classifier:latest
docker run --gpus all -p 8000:8000 saibalajinamburi/arxiv_classifier:latest
```

---

## Phase 8 — CI/CD & GitHub

### File: `.github/workflows/ci_cd.yml`

### What GitHub Actions does
Every time you push code to GitHub, a robot in the cloud:
1. Spins up a fresh Ubuntu Linux machine
2. Clones your repository
3. Installs your dependencies
4. Runs `flake8` to check for Python syntax errors
5. Reports Pass ✅ or Fail ❌ on your GitHub repository page

### What flake8 checks
```bash
flake8 src/ app/ --count --select=E9,F63,F7,F82 --show-source --statistics
```
- `E9` — Python syntax errors (invalid code)
- `F82` — Undefined variables (undefined name 'X' used)
- `F7` — Import errors
- `F63` — Invalid escape sequences (like `\n` used incorrectly)

### GitIgnore (what does NOT go to GitHub)
```
.venv/          ← 4GB virtual environment
data/raw/       ← Raw API data - re-generate by running the pipeline
mlruns/         ← MLflow file artifacts
mlflow.db       ← MLflow database (absolute Windows paths inside)
```

What DOES go to GitHub:
- All source code (`src/`, `app/`)
- `data/processed/` (small embedding files — needed for deployment)
- `models/classifier.pkl` (the trained model — needed for deployment)
- `Dockerfile`, `Dockerfile.cloud`
- `ci_cd.yml`, `.gitignore`, `README.md`

### Git commands used

```bash
# Initialize a new git repository in the project folder
git init

# Configure your identity (runs once)
git config user.email "saibalajinamburi@gmail.com"
git config user.name "Saibalaji Namburi"

# Stage ALL files (respecting .gitignore)
git add .

# Create first commit
git commit -m "Initial commit: End-to-end MLOps pipeline with FastAPI and Docker"

# Rename the default branch to "main"
git branch -M main

# Link to your GitHub repository
git remote add origin https://github.com/saibalajinamburi/Automated-Research-Paper-Classifier.git

# Push your code to GitHub
git push -u origin main

# For future updates (after changing code)
git add .
git commit -m "Your description of what changed"
git push
```

---

## Every Command Explained

### Setting Up the Environment

```bash
# Create a sandboxed Python environment in .venv folder
python -m venv .venv

# Activate the environment (Windows)
.\.venv\Scripts\activate

# Activate the environment (Mac/Linux)
source .venv/bin/activate

# Install all dependencies
pip install -r app/requirements.txt
```

### Running the Pipeline (in order)

```bash
# STEP 1: Fetch papers from arXiv API
# → creates data/raw/arxiv_data.csv and data/raw/raw_papers.json
python src/data_ingestion.py

# STEP 2: Generate transformer embeddings
# → creates data/processed/X.npy, y.npy, classes.npy
python src/preprocess.py

# STEP 3: Train the LightGBM model
# → creates mlflow.db (experiment tracking database)
python src/train.py

# STEP 4: Export the trained model to a portable file
# → creates models/classifier.pkl
python -c "
import mlflow, joblib, os
mlflow.set_tracking_uri('sqlite:///mlflow.db')
exp = mlflow.get_experiment_by_name('arxiv_quantum_classifier')
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=['start_time DESC'])
model = mlflow.lightgbm.load_model(f'runs:/{runs.iloc[0][\"run_id\"]}/model')
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/classifier.pkl')
print('Model exported to models/classifier.pkl')
"
```

### Running the API Locally

```bash
# Start the FastAPI web server on port 8000
# --reload: auto-restarts when you change code (development mode)
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload

# The API is now available at:
# http://localhost:8000          ← Health check
# http://localhost:8000/docs     ← Interactive test UI (Swagger)
# http://localhost:8000/predict  ← Send POST requests here
```

### Testing the API

```bash
# Test health check (PowerShell)
Invoke-RestMethod -Method GET -Uri "http://localhost:8000/"

# Test prediction (PowerShell)
$body = '{"text": "Quantum entanglement between particles in superconducting circuits."}'
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/predict" -Body $body -ContentType "application/json"

# Test prediction (curl — Mac/Linux or Git Bash)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Quantum entanglement between particles in superconducting circuits."}'
```

### MLflow Commands

```bash
# Launch the MLflow web dashboard
# → shows all experiments, runs, metrics, and model artifacts
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db

# The dashboard is available at: http://127.0.0.1:5000
# Click on "arxiv_quantum_classifier" to see your runs
# Click on any run to see accuracy, F1 score, and model artifacts
```

### Docker Commands

```bash
# ── LOCAL GPU BUILD ────────────────────────────────────────────────────────────

# Build the GPU image (uses Dockerfile, takes 5-10 min on first build)
docker build -t arxiv_classifier:latest .

# Run the container with GPU support, auto-restart, in background
docker run --gpus all -p 8000:8000 --restart unless-stopped -d --name arxiv_api arxiv_classifier:latest
#           ↑ use GPU    ↑ port   ↑ restart on reboot       ↑ detached (background)

# View live logs from the container
docker logs arxiv_api --tail 20

# See all running containers
docker ps

# Stop the container
docker stop arxiv_api

# Remove the container (does NOT delete the image)
docker rm arxiv_api

# List all local Docker images
docker images


# ── CLOUD CPU BUILD ────────────────────────────────────────────────────────────

# Build the lightweight cloud image (uses Dockerfile.cloud)
docker build -f Dockerfile.cloud -t arxiv_classifier:cloud .

# Test the cloud image locally on port 7860 (same port as Hugging Face)
docker run -p 7860:7860 --name arxiv_cloud arxiv_classifier:cloud


# ── DOCKER HUB ─────────────────────────────────────────────────────────────────

# Login to Docker Hub (enter username + password)
docker login

# Tag the image with your Docker Hub username
docker tag arxiv_classifier:latest saibalajinamburi/arxiv_classifier:latest

# Push to Docker Hub (makes it public — anyone can pull)
docker push saibalajinamburi/arxiv_classifier:latest

# Anyone in the world can now pull and run it:
# docker pull saibalajinamburi/arxiv_classifier:latest
# docker run --gpus all -p 8000:8000 saibalajinamburi/arxiv_classifier:latest
```

---

## Problems We Faced & How We Fixed Them

### Problem 1: SSL Certificate Error
**Error**: `ssl.SSLCertVerificationError: certificate verify failed`
**When**: First attempt to hit the arXiv API on Windows
**Cause**: Windows Python 3.10 does not trust local HTTPS certificates by default
**Fix**: Create an unverified SSL context: `ssl._create_unverified_context()`

### Problem 2: HTTP 429 Rate Limit (arXiv blocked us)
**Error**: `HTTP Error 429: Too Many Requests`
**When**: Tried to fetch 10,000 papers in one request
**Cause**: arXiv's API terms of service require polite usage — no bulk requests
**Fix**: Paginated into 1,000-paper batches with `time.sleep(3)` between each batch

### Problem 3: Synthetic Data (Critical mistake)
**Mistake**: Generated fake/random data as a placeholder while waiting for the API
**Why it's bad**: Model trains on fake patterns → "hallucinations" → wrong real-world predictions
**Fix**: Deleted all fake data immediately. Re-ran pipeline with real arXiv papers

### Problem 4: PyTorch DLL Crash (WinError 1114)
**Error**: `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`
**When**: First time running `sentence-transformers` on Windows
**Cause**: The default PyTorch install includes CUDA DLLs that require C++ redistributables
**Fix**: Created a `.venv` and installed CPU-only PyTorch temporarily while setting up CUDA

### Problem 5: MLflow "No Experiments Exist"
**Error**: MLflow UI showed empty experiments page
**When**: After training completed successfully
**Cause**: Global Python had MLflow 2.10.2, `.venv` had MLflow 3.11.1. They wrote incompatible file formats to the same `mlruns/` folder
**Fix**: Migrated to SQLite backend: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`. Single file, no version conflicts

### Problem 6: Docker pywin32 Error
**Error**: `ERROR: No matching distribution found for pywin32==311`
**When**: First Docker build attempt
**Cause**: `pip freeze` captured Windows-specific DLL packages. Docker runs Linux
**Fix**: Filtered packages: `pip freeze | Where-Object { $_ -notmatch "pywin32" }`

### Problem 7: Docker CUDA Wheel Error
**Error**: `Could not find torch==2.5.1+cu121 (from versions: ...)`
**When**: Second Docker build attempt
**Cause**: `+cu121` is a Windows CUDA wheel. It does not exist in the Linux PyPI index
**Fix**: Switched to official NVIDIA PyTorch base image: `FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`. PyTorch comes pre-installed — no need to install it at all

### Problem 8: Container cannot find model (OSError)
**Error**: `OSError: No such file or directory: '/tmp/tmph7y0kxby/model/MLmodel'`
**When**: First time running the GPU-enabled container
**Cause**: The model path stored inside `mlflow.db` was an absolute Windows path (`C:\Users\saiba\...`). This path does not exist inside the Linux container
**Fix**: Exported the model from MLflow to a plain file (`models/classifier.pkl`) using `joblib.dump()`. Container loads it directly — no MLflow dependency at inference time

---

## Final Results

| Metric | Value |
|---|---|
| Training papers | ~500 real arXiv papers (quantum domain) |
| Embedding dimensions | 384 |
| Model | LightGBM (Gradient Boosted Decision Trees) |
| Accuracy | **75.20%** |
| F1 Score (weighted) | **66.60%** |
| GPU confirmed | NVIDIA RTX 3050 (`cuda:0`) active in container |
| API status | ✅ Live and responding |
| CI/CD status | ✅ GitHub Actions running on every push |
| Cloud ready | ✅ `Dockerfile.cloud` ready for HF Spaces / Render |
| Docker Hub | ✅ `saibalajinamburi/arxiv_classifier:latest` |

---

## Technology Stack (Complete)

| Layer | Tool | What it does |
|---|---|---|
| Data | arXiv Atom API | Public academic paper database |
| Parsing | `xml.etree.ElementTree` | Reads XML from API response |
| Processing | Pandas, NumPy | Table operations, array math |
| Embeddings | `sentence-transformers` | Converts text to 384-dim vectors |
| Embedding Model | `all-MiniLM-L6-v2` (Hugging Face) | The actual AI model |
| GPU | PyTorch + CUDA 12.1 | GPU acceleration |
| Hardware | NVIDIA RTX 3050 | The physical GPU |
| Classifier | LightGBM | Gradient boosted decision trees |
| Experiment Tracking | MLflow 3.11.1 | Logs runs, metrics, artifacts |
| MLflow Backend | SQLite (`mlflow.db`) | Database for tracking |
| API | FastAPI | Web framework |
| API Server | Uvicorn | ASGI server that runs FastAPI |
| Input Validation | Pydantic v2 | Validates incoming JSON |
| Model Loading | joblib | Saves/loads `.pkl` model files |
| Local Container | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` | NVIDIA GPU-enabled Docker |
| Cloud Container | `python:3.10-slim` + CPU torch | Lightweight cloud Docker |
| Registry | Docker Hub | Hosts the public Docker image |
| Cloud Deploy | Hugging Face Spaces / Render.com | Permanent public URL |
| CI/CD | GitHub Actions | Automated code verification |
| Linting | flake8 | Python syntax checker |
| Version Control | Git + GitHub | Code repository |

---

*Written: 2026-04-14 | Saibalaji Namburi | saibalajinamburi@gmail.com*
