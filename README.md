# Medical RAG Chatbot

A fully offline, production-ready **Retrieval-Augmented Generation (RAG)** chatbot that answers medical questions using the MedQuAD dataset. Every query is tracked as an MLflow experiment run for performance monitoring and analysis.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [MLflow Tracking](#mlflow-tracking)
- [What Gets Logged](#what-gets-logged)

---

## Overview

This chatbot retrieves answers **strictly** from a local database of 16,412 verified medical Q&A pairs (MedQuAD), sourced from the NIH, CDC, GARD, and other trusted health authorities. No internet connection is required for answering questions — the only network call is the optional MLflow logging to a remote tracking server.

> **Important:** This tool is for informational purposes only. Always consult a qualified healthcare professional for medical advice.

---

## Features

| Feature | Details |
|---|---|
| Fully offline answering | No internet, no API calls, no LLM inference costs |
| Semantic vector search | FAISS inner-product search over sentence embeddings |
| Local embeddings | `all-MiniLM-L6-v2` via SentenceTransformers — runs on CPU |
| Persistent index | FAISS index built once and reused on restart |
| Streamlit UI | Clean chat interface with source badges and document expander |
| MLflow tracking | Every query logged as a run with params, metrics, and artifacts |
| Graceful degradation | MLflow errors never crash the chatbot |

---

## How It Works

```
User Query
    │
    ▼
Embed query with SentenceTransformer (all-MiniLM-L6-v2)
    │
    ▼
FAISS cosine similarity search over 16,412 indexed documents
    │
    ▼
Filter results by minimum similarity score (default: 0.25)
    │
    ▼
Format structured answer from top-K retrieved documents
    │
    ▼
Log query + response + metrics to MLflow
    │
    ▼
Display answer in Streamlit UI
```

The system uses **cosine similarity** (implemented as inner product after L2 normalization) to find the most semantically relevant Q&A pairs for any given question.

---

## Project Structure

```
RAG/
├── app.py                  # Streamlit web UI
├── rag_chatbot.py          # Core chatbot: embeddings, FAISS search, answer formatting
├── mlflow_config.py        # MLflow tracking URI and experiment name
├── mlflow_tracker.py       # Logs each query as an MLflow run
├── rag_chat.py             # Early prototype script (Ollama + ChromaDB)
├── requirements.txt        # Python dependencies
│
├── medquad.csv/
│   └── medquad.csv         # MedQuAD dataset (16,412 medical Q&A pairs)
│
├── faiss_index.bin         # FAISS index (auto-generated on first run)
└── medquad_df.pkl          # Pickled DataFrame (auto-generated on first run)
```

> `faiss_index.bin` and `medquad_df.pkl` are created automatically the first time the chatbot runs. Subsequent starts load them from disk, making startup significantly faster.

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector index | `faiss-cpu` — IndexFlatIP (exact search) |
| Data handling | `pandas`, `numpy` |
| Web UI | `streamlit` |
| Experiment tracking | `mlflow` |
| Language | Python 3.10+ |

---

## Dataset

**MedQuAD** (Medical Question Answer Dataset)

| Property | Value |
|---|---|
| Total Q&A pairs | 16,412 |
| Unique medical topics | 5,127 |
| Columns | `question`, `answer`, `source`, `focus_area` |
| Sources | NIH, CDC, GARD, GHR, MedlinePlus, NHLBI, NIDDK, NIHSeniorHealth, NINDS |
| File path | `medquad.csv/medquad.csv` |

---

## Installation

### 1. Clone / download the project

```bash
cd Desktop/ML_projects/RAG
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First-time setup downloads the `all-MiniLM-L6-v2` model (~90 MB) automatically.

---

## Usage

### Streamlit Web App (recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Sidebar controls:**

| Control | Description |
|---|---|
| Top-K slider | Number of documents retrieved per query (1–10, default 5) |
| Min similarity slider | Minimum score to accept a result (0.0–1.0, default 0.25) |
| MLflow status | Live connection indicator + link to MLflow UI |
| Sample questions | One-click example queries |
| Clear Chat | Reset the conversation |

### CLI (command line)

```bash
python rag_chatbot.py
```

Type your question and press Enter. Type `quit` to exit.

---

## Configuration

All key settings live at the top of `rag_chatbot.py`:

```python
EMBED_MODEL = "all-MiniLM-L6-v2"   # embedding model (local, no API key needed)
TOP_K       = 5                      # default documents retrieved per query
MIN_SCORE   = 0.25                   # default minimum cosine similarity threshold
```

MLflow settings are in `mlflow_config.py`:

```python
mlflow.set_tracking_uri("http://103.49.125.28:8501/mlflow/")
mlflow.set_experiment("MedRAG-Chatbot")
```

---

## MLflow Tracking

Every chat query is automatically logged as an MLflow run to the remote tracking server at `http://103.49.125.28:8501/mlflow/`.

**To view tracked runs:**

1. Open `http://103.49.125.28:8501/mlflow/` in your browser
2. Select the **MedRAG-Chatbot** experiment
3. Click any run to see its parameters, metrics, and artifacts

If the MLflow server is unreachable, the chatbot continues to work normally — a warning is printed to the console and the run is silently skipped.

---

## What Gets Logged

### Parameters (settings used for the run)

| Parameter | Example Value | Description |
|---|---|---|
| `top_k` | `5` | Documents retrieved |
| `min_score` | `0.25` | Minimum similarity threshold |
| `embed_model` | `all-MiniLM-L6-v2` | Embedding model name |

### Metrics (measured during the run)

| Metric | Example Value | Description |
|---|---|---|
| `num_docs_retrieved` | `4` | How many docs passed the score filter |
| `top_score` | `0.72` | Similarity score of the best match |
| `avg_score` | `0.61` | Average similarity across retrieved docs |
| `min_score_result` | `0.51` | Lowest similarity among retrieved docs |
| `answer_found` | `1` | `1` = answer found, `0` = no match |
| `answer_length` | `843` | Character count of the answer |
| `embedding_latency` | `0.043` | Seconds to embed the query |
| `search_latency` | `0.002` | Seconds for FAISS search |
| `total_latency` | `0.045` | Total seconds for the full pipeline |

### Artifacts (files saved per run)

| File | Contents |
|---|---|
| `query.txt` | Raw user question |
| `response.txt` | Full formatted answer |
| `retrieved_docs.json` | All retrieved documents with scores |
