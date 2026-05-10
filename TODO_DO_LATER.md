# TODO — Remaining Tasks (Do Later)
**Project: Automated Research Paper Classifier**
Last updated: 2026-04-14

---

## ⬜ 1. Finish Docker Hub Push

The 10.52GB image upload was stopped early because of slow internet (2 MB/s).

**What to do:**
Connect to faster Wi-Fi or ethernet, then run:
```bash
cd "c:\Users\saiba\Documents\Automated Research Paper Classifier"
docker push saibalajinamburi/arxiv_classifier:latest
```

**How to verify it finished:**
Go to https://hub.docker.com/r/saibalajinamburi/arxiv_classifier
It should show the image with the `latest` tag and a file size.

---

## ⬜ 2. Deploy to Hugging Face Spaces (Public Live URL)

This gives a permanent public URL that anyone can open in their browser.
URL will be: `https://huggingface.co/spaces/saibalajinamburi/arxiv-classifier`

**Step-by-step:**

### Step A — Create a Hugging Face account (if you don't have one)
Go to https://huggingface.co/join and sign up (free)

### Step B — Install the HF CLI tool
```bash
.\.venv\Scripts\python.exe -m pip install huggingface_hub
.\.venv\Scripts\python.exe -m huggingface_hub login
# Paste your HF token from https://huggingface.co/settings/tokens
```

### Step C — Create a new Space
Go to https://huggingface.co/new-space and fill in:
- **Owner**: saibalajinamburi
- **Space name**: arxiv-classifier
- **SDK**: Docker
- **Visibility**: Public

### Step D — Clone the Space repo locally
```bash
# In a separate folder from your main project:
git clone https://huggingface.co/spaces/saibalajinamburi/arxiv-classifier
cd arxiv-classifier
```

### Step E — Copy the project files into the Space repo
```powershell
# Run from your main project folder:
Copy-Item "Dockerfile.cloud" "C:\Users\saiba\[path-to-cloned-space]\Dockerfile"
Copy-Item -Recurse "app\" "C:\Users\saiba\[path-to-cloned-space]\app\"
Copy-Item -Recurse "data\processed\" "C:\Users\saiba\[path-to-cloned-space]\data\processed\"
Copy-Item -Recurse "models\" "C:\Users\saiba\[path-to-cloned-space]\models\"
```

### Step F — Add the HF metadata to the Dockerfile
The first lines of the Dockerfile in the Space must be:
```
---
title: ArXiv Research Paper Classifier
sdk: docker
app_port: 7860
---
```
(Already set up in `Dockerfile.cloud` and `README_HF.md` in the project)

### Step G — Push to Hugging Face
```bash
cd [path-to-cloned-space]
git add .
git commit -m "Initial deployment"
git push
```
Hugging Face will automatically build and deploy. Takes 5-10 minutes.
Watch progress at: https://huggingface.co/spaces/saibalajinamburi/arxiv-classifier

---

## ⬜ 3. (Optional) Improve the Model — More Data, More Categories

Right now the model was trained only on `all:quantum` papers, so it mostly predicts `quant-ph`.

**To make it properly multi-category:**

In `src/data_ingestion.py`, change the last few lines to fetch from multiple domains:
```python
# Replace the single pipeline run with multiple:
queries = ["all:machine+learning", "all:quantum", "all:neuroscience", "all:biology", "all:economics"]

all_papers = []
for query in queries:
    pipeline = ArxivDataPipeline(query=query, max_results=2000)
    papers = pipeline.run()
    all_papers.extend(papers)

# Then save all_papers to CSV as before
```
Then re-run the full pipeline: `preprocess.py` → `train.py` → export model → push to GitHub

---

## ⬜ 4. (Optional) Add a Simple Web UI (Gradio)

Instead of using the raw FastAPI `/docs` page, add a friendly Gradio interface.

Install:
```bash
.\.venv\Scripts\python.exe -m pip install gradio
```

Create a file `app/gradio_ui.py`:
```python
import gradio as gr
import requests

def classify(text):
    r = requests.post("http://localhost:8000/predict", json={"text": text})
    return r.json()["predicted_category"]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Paste paper abstract here"),
    outputs=gr.Text(label="Predicted Category"),
    title="ArXiv Research Paper Classifier",
    description="Powered by all-MiniLM-L6-v2 + LightGBM"
)

demo.launch()
```

Run it: `.\.venv\Scripts\python.exe app/gradio_ui.py`

---

## ✅ Already Done (for reference)

- [x] arXiv data ingestion pipeline (`src/data_ingestion.py`)
- [x] Sentence Transformer embeddings (`src/preprocess.py`)
- [x] CUDA GPU setup — RTX 3050 verified (`cuda:0`)
- [x] LightGBM training — Accuracy 75.20%, F1 66.60%
- [x] MLflow experiment tracking (SQLite backend, UI working)
- [x] FastAPI inference server (`app/main.py`)
- [x] GPU Docker container — NVIDIA base image, confirmed running
- [x] Full walkthrough document (`PROJECT_WALKTHROUGH.md`)
- [x] GitHub repository pushed — https://github.com/saibalajinamburi/Automated-Research-Paper-Classifier
- [x] GitHub Actions CI/CD pipeline (runs `flake8` on every push)
- [x] `Dockerfile.cloud` created for cloud deployment (CPU, lightweight)
- [x] Docker Hub tag created (`saibalajinamburi/arxiv_classifier:latest`)
