# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Medical RAG Chatbot is a fully offline, production-ready Retrieval-Augmented Generation system that answers medical questions using the MedQuAD dataset (16,412 verified Q&A pairs). It uses semantic vector search with FAISS and local embeddings, with automatic MLflow experiment tracking for each query.

**Key characteristic:** Fully offline operation—no API calls, no internet, no LLM inference costs.

## Architecture

### Core Pipeline (rag_chatbot.py)

The chatbot follows a linear flow:
1. **Load & Embed**: MedQuAD CSV → SentenceTransformers (all-MiniLM-L6-v2, ~90 MB model)
2. **Index**: FAISS IndexFlatIP (inner-product search after L2 normalization = cosine similarity)
3. **Retrieve**: For each query, embed and find top-K documents passing min_score threshold
4. **Format**: Build structured markdown answer from retrieved docs (NO external LLM used)

**Key classes:**
- `MedRAGChatbot`: Main class handling embeddings, FAISS search, and answer retrieval
  - `setup()`: Load or build index (creates faiss_index.bin + medquad_df.pkl on first run)
  - `retrieve(query)`: Returns (RetrievedDoc[], embedding_latency, search_latency)
  - `chat(query)`: Returns ChatResponse with answer, sources, and timing metadata
- `ChatResponse`: Dataclass containing answer, sources, retrieved_docs, found flag, and latencies
- `RetrievedDoc`: Individual document with question, answer, source, focus_area, score

**Index persistence:**
- `faiss_index.bin` (25 MB): FAISS index built once, reused on startup
- `medquad_df.pkl` (22 MB): Pickled DataFrame for O(1) document lookup

### Web UI (app.py)

Streamlit application with:
- **Sidebar**: Sample questions, Clear Chat button
- **Chat history**: Renders user/assistant messages, shows expander with retrieved documents
- **Session state management**: Tracks messages, metadata (sources, docs), MLflow run count

**Critical detail:** `bot.setup()` is cached via `@st.cache_resource` to avoid rebuilding the index on every page reload.

### MLflow Integration (mlflow_tracker.py + mlflow_config.py)

**Tracking URI:** `http://103.49.125.28:8501/mlflow/`
**Experiment:** `MedRAG-Chatbot`

Each query logs:
- **Parameters**: top_k, min_score, embed_model
- **Metrics**: num_docs_retrieved, top_score, avg_score, min_score_result, answer_found, answer_length, embedding_latency, search_latency, total_latency
- **Artifacts**: query.txt, response.txt, retrieved_docs.json (all created in temp directory)

**Graceful degradation**: All MLflow calls wrapped in try/except—server unreachability never crashes the chatbot.

## Common Development Tasks

### Run the Chatbot

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in browser.

**CLI mode:**
```bash
python rag_chatbot.py
```
Type questions interactively; type `quit` to exit.

### Adjust Configuration

All key settings at the top of `rag_chatbot.py`:
```python
EMBED_MODEL = "all-MiniLM-L6-v2"  # embedding model (run locally)
TOP_K = 5                          # default documents retrieved
MIN_SCORE = 0.25                   # cosine similarity threshold
```

MLflow settings in `mlflow_config.py`:
```python
mlflow.set_tracking_uri("http://103.49.125.28:8501/mlflow/")
mlflow.set_experiment("MedRAG-Chatbot")
```

### Rebuild the FAISS Index

Delete both files and rerun:
```bash
rm faiss_index.bin medquad_df.pkl
python rag_chatbot.py  # or streamlit run app.py
```
First run will rebuild and persist the index (~2-5 minutes depending on hardware).

### Debug a Single Query

```python
from rag_chatbot import MedRAGChatbot

bot = MedRAGChatbot()
bot.setup()
response = bot.chat("What is diabetes?")
print(f"Found: {response.found}")
print(f"Answer:\n{response.answer}")
print(f"Sources: {response.sources}")
for doc in response.retrieved_docs:
    print(f"  - {doc.focus_area} ({doc.score:.3f}) from {doc.source}")
```

## Project Structure

```
medical-rag-chatbot/
├── app.py                    # Streamlit web UI
├── rag_chatbot.py            # Core RAG: embeddings, FAISS search, answer formatting
├── mlflow_config.py          # MLflow URI and experiment setup
├── mlflow_tracker.py         # Logs each query as MLflow run
├── rag_chat.py               # Prototype (Ollama + ChromaDB, not used)
├── requirements.txt          # Dependencies
├── medquad.csv/
│   └── medquad.csv           # 16,412 medical Q&A pairs (sources: NIH, CDC, GARD, etc.)
├── faiss_index.bin           # Auto-generated FAISS index
├── medquad_df.pkl            # Auto-generated pickled DataFrame
└── README.md                 # Full documentation
```

## Dependencies

```
pandas          # Data handling and CSV loading
numpy           # Numerical operations for embeddings
faiss-cpu       # Vector search (IndexFlatIP for cosine similarity)
sentence-transformers  # Embedding model (all-MiniLM-L6-v2)
streamlit       # Web UI
mlflow          # Experiment tracking
```

Install with: `pip install -r requirements.txt`

## Dataset

**MedQuAD** (Medical Question Answer Dataset)
- **16,412** Q&A pairs
- **5,127** unique medical topics
- **Columns**: question, answer, source, focus_area
- **Sources**: NIH, CDC, GARD, GHR, MedlinePlus, NHLBI, NIDDK, NIHSeniorHealth, NINDS
- **Location**: `medquad.csv/medquad.csv`

## Important Implementation Notes

### Answer Formatting (rag_chatbot.py, lines 56-124)

- Offline formatting—no LLM, pure string manipulation
- Detects and preserves list-like patterns (dashes, bullets)
- Limits additional context to 2 supporting documents from same focus area
- Always includes medical disclaimer

### Similarity Scoring

- Uses L2-normalized embeddings with FAISS IndexFlatIP
- Inner product of normalized vectors = cosine similarity (range 0.0–1.0)
- Default threshold: MIN_SCORE = 0.25 (tunable in app sidebar)
- Documents with score < threshold are filtered out

### MLflow Error Handling

All MLflow logging is wrapped in try/except (mlflow_tracker.py, line 31–110). If the tracking server is unreachable or MLflow fails for any reason:
- A warning is printed to console
- The chatbot continues normally
- The query/response still completes and displays to user

### Streamlit Caching

The chatbot is loaded once per session with `@st.cache_resource`:
```python
@st.cache_resource
def load_chatbot():
    bot = MedRAGChatbot()
    bot.setup()
    return bot
```

This prevents re-initializing embeddings/index on every interaction. To force reload, use the Streamlit rerun button or clear cache manually.

## Testing & Debugging

### Test a Query Retrieval

```python
from rag_chatbot import MedRAGChatbot

bot = MedRAGChatbot(top_k=3, min_score=0.2)
bot.setup()

docs, emb_lat, search_lat = bot.retrieve("heart disease symptoms")
print(f"Embedding: {emb_lat:.4f}s, Search: {search_lat:.4f}s")
for doc in docs:
    print(f"  Score: {doc.score:.3f} | {doc.focus_area} | {doc.source}")
```

### Check MLflow Connection

```python
import urllib.request
try:
    urllib.request.urlopen("http://103.49.125.28:8501/mlflow/", timeout=2)
    print("MLflow server reachable")
except Exception as e:
    print(f"MLflow unreachable: {e}")
```

### Verify Index Integrity

```python
from rag_chatbot import MedRAGChatbot
import os

bot = MedRAGChatbot()
if os.path.exists("faiss_index.bin") and os.path.exists("medquad_df.pkl"):
    bot.load_index()
    print(f"Index loaded: {bot.index.ntotal} documents")
else:
    print("Index files not found")
```

## Performance Characteristics

- **Embedding latency**: ~40–50 ms (CPU, SentenceTransformers)
- **FAISS search latency**: ~1–2 ms (16,412 documents)
- **Total response time**: ~50–60 ms (query to answer)
- **Memory footprint**: ~2.5 GB (embeddings + index in RAM when loaded)
- **First-run overhead**: ~2–5 minutes (building and normalizing 16,412 embeddings)

## Disclaimer

This tool is for **informational purposes only**. The chatbot always includes a disclaimer in responses. Users should always consult qualified healthcare professionals for medical advice.
