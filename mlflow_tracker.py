"""
MLflow Tracker — Medical RAG Chatbot
======================================
Logs each chat query as an MLflow run with params, metrics, and text artifacts.
All calls are wrapped in try/except so a missing/unreachable server never
crashes the chatbot.
"""

import json
import tempfile
import os

import mlflow

# Apply tracking URI and experiment from shared config
import mlflow_config  # noqa: F401  — side-effects only (set_tracking_uri + set_experiment)


def log_run(query: str, response, top_k: int, min_score: float, embedder) -> None:
    """
    Log a single RAG query/response pair as an MLflow run.

    Parameters
    ----------
    query      : raw user query string
    response   : ChatResponse dataclass instance from rag_chatbot.py
    top_k      : number of documents retrieved (setting)
    min_score  : minimum similarity threshold (setting)
    embedder   : SentenceTransformer instance (used to derive model name)
    """
    try:
        # Derive embed model name from the SentenceTransformer object
        embed_model_name = getattr(
            embedder, "_model_card_vars", {}
        ).get("model_name", "") or str(getattr(embedder, "_modules", {}).get("0", ""))
        # Fallback: check the first module name or use a default
        if not embed_model_name:
            try:
                embed_model_name = embedder[0].auto_model.config._name_or_path
            except Exception:
                embed_model_name = "all-MiniLM-L6-v2"

        docs = response.retrieved_docs
        scores = [d.score for d in docs]

        num_docs      = len(docs)
        top_score     = float(scores[0])  if scores else 0.0
        avg_score     = float(sum(scores) / len(scores)) if scores else 0.0
        min_score_res = float(min(scores)) if scores else 0.0
        answer_found  = 1 if response.found else 0
        answer_length = len(response.answer)

        run_name = query[:40].strip().replace("\n", " ")

        with mlflow.start_run(run_name=run_name):

            # ── Parameters ────────────────────────────────────────────────────
            mlflow.log_param("top_k",       top_k)
            mlflow.log_param("min_score",   min_score)
            mlflow.log_param("embed_model", embed_model_name)

            # Log index size if available on the response (optional)
            index_size = getattr(response, "index_size", None)
            if index_size is not None:
                mlflow.log_param("index_size", index_size)

            # ── Metrics ───────────────────────────────────────────────────────
            mlflow.log_metric("num_docs_retrieved",  num_docs)
            mlflow.log_metric("top_score",           top_score)
            mlflow.log_metric("avg_score",           avg_score)
            mlflow.log_metric("min_score_result",    min_score_res)
            mlflow.log_metric("answer_found",        answer_found)
            mlflow.log_metric("answer_length",       answer_length)
            mlflow.log_metric("embedding_latency",   response.embedding_latency)
            mlflow.log_metric("search_latency",      response.search_latency)
            mlflow.log_metric("total_latency",       response.total_latency)

            # ── Artifacts ─────────────────────────────────────────────────────
            with tempfile.TemporaryDirectory() as tmpdir:
                # query.txt
                query_path = os.path.join(tmpdir, "query.txt")
                with open(query_path, "w", encoding="utf-8") as f:
                    f.write(query)
                mlflow.log_artifact(query_path)

                # response.txt
                response_path = os.path.join(tmpdir, "response.txt")
                with open(response_path, "w", encoding="utf-8") as f:
                    f.write(response.answer)
                mlflow.log_artifact(response_path)

                # retrieved_docs.json
                docs_data = [
                    {
                        "rank":       i + 1,
                        "score":      d.score,
                        "focus_area": d.focus_area,
                        "source":     d.source,
                        "question":   d.question,
                        "answer":     d.answer[:800],  # truncate very long answers
                    }
                    for i, d in enumerate(docs)
                ]
                docs_path = os.path.join(tmpdir, "retrieved_docs.json")
                with open(docs_path, "w", encoding="utf-8") as f:
                    json.dump(docs_data, f, indent=2, ensure_ascii=False)
                mlflow.log_artifact(docs_path)

    except Exception as exc:
        print(f"[MLflow] Warning: could not log run — {exc}")
