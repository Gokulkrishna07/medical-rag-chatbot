"""
Medical RAG Chatbot — Fully Offline
=====================================
Pipeline:
  1. Load MedQuAD CSV from local storage
  2. Embed questions using SentenceTransformers (runs locally)
  3. Index embeddings with FAISS (in-memory / on-disk)
  4. Retrieve top-k relevant documents for a query
  5. Format a structured answer from retrieved context only
     — NO internet, NO API calls, NO external models
"""

import os
import pickle
import re
from dataclasses import dataclass, field

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
CSV_PATH   = "medquad.csv/medquad.csv"
INDEX_PATH = "faiss_index.bin"
DF_PATH    = "medquad_df.pkl"

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL       = "all-MiniLM-L6-v2"  # fully local sentence embeddings
TOP_K             = 5                    # documents to retrieve
MIN_SCORE         = 0.25                 # minimum similarity to accept a result


@dataclass
class RetrievedDoc:
    question:   str
    answer:     str
    source:     str
    focus_area: str
    score:      float


@dataclass
class ChatResponse:
    answer:        str
    sources:       list[str]       = field(default_factory=list)
    retrieved_docs: list[RetrievedDoc] = field(default_factory=list)
    found:         bool            = True


# ── Answer formatter (offline, no LLM) ────────────────────────────────────────

def _clean(text: str) -> str:
    """Remove duplicate whitespace and tidy up text."""
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def format_answer(query: str, docs: list[RetrievedDoc]) -> str:
    """
    Build a structured, readable answer purely from retrieved documents.
    No external model or API is used.
    """
    if not docs:
        return (
            "The answer to your question is **not available** in the local "
            "medical database. Please consult a qualified healthcare professional."
        )

    primary = docs[0]
    topic   = primary.focus_area or "Medical Information"

    lines: list[str] = []

    # ── Header ─────────────────────────────────────────────────────────────────
    lines.append(f"### {topic}")
    lines.append("")

    # ── Primary answer ─────────────────────────────────────────────────────────
    primary_answer = _clean(primary.answer)

    # If the answer contains list-like patterns (dash / bullet), preserve them
    if re.search(r"(?m)^\s*[-•]", primary_answer) or " - " in primary_answer:
        # Split on " - " style inline lists into proper bullet points
        parts = re.split(r"\s{1,3}-\s{1,3}", primary_answer)
        if len(parts) > 2:
            lines.append(parts[0])          # intro sentence
            lines.append("")
            for part in parts[1:]:
                part = part.strip()
                if part:
                    lines.append(f"- {part}")
        else:
            lines.append(primary_answer)
    else:
        lines.append(primary_answer)

    # ── Supporting context from additional docs ─────────────────────────────────
    extras = [
        d for d in docs[1:]
        if d.focus_area == primary.focus_area and d.answer != primary.answer
    ]
    if extras:
        lines.append("")
        lines.append("**Additional Information:**")
        for doc in extras[:2]:
            snippet = _clean(doc.answer)[:400]
            if not snippet.endswith("."):
                snippet += "..."
            lines.append(f"> {snippet}")

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(
        "_This information is retrieved from local medical documents. "
        "Always consult a qualified healthcare professional for medical advice._"
    )

    return "\n".join(lines)


# ── Main chatbot class ─────────────────────────────────────────────────────────

class MedRAGChatbot:
    """
    Fully offline Retrieval-Augmented Generation chatbot.
    Uses only local embeddings (SentenceTransformers) and FAISS vector search.
    """

    def __init__(
        self,
        csv_path:   str = CSV_PATH,
        index_path: str = INDEX_PATH,
        df_path:    str = DF_PATH,
        embed_model: str = EMBED_MODEL,
        top_k:      int = TOP_K,
        min_score:  float = MIN_SCORE,
    ):
        self.csv_path   = csv_path
        self.index_path = index_path
        self.df_path    = df_path
        self.top_k      = top_k
        self.min_score  = min_score

        print("[1/3] Loading local embedding model (offline)...")
        self.embedder = SentenceTransformer(embed_model)

        self.index: faiss.Index | None = None
        self.df:    pd.DataFrame | None = None

    # ── Index management ───────────────────────────────────────────────────────

    def build_index(self) -> None:
        """Embed all documents and persist the FAISS index to disk."""
        print("[2/3] Building FAISS index from local CSV...")

        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
        self.df = df

        texts = (
            df["question"].fillna("") + " " + df["focus_area"].fillna("")
        ).tolist()

        print(f"      Encoding {len(texts):,} documents locally...")
        embeddings = self.embedder.encode(
            texts, show_progress_bar=True, batch_size=256, convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(embeddings)          # cosine similarity via inner product

        dim        = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, self.index_path)
        with open(self.df_path, "wb") as f:
            pickle.dump(self.df, f)

        print(f"      Index saved → {self.index_path}")

    def load_index(self) -> None:
        print("[2/3] Loading FAISS index from local disk...")
        self.index = faiss.read_index(self.index_path)
        with open(self.df_path, "rb") as f:
            self.df = pickle.load(f)

    def setup(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.df_path):
            self.load_index()
        else:
            self.build_index()
        print(f"[3/3] Ready — {self.index.ntotal:,} documents indexed (offline).\n")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        query_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, self.top_k)

        docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or float(score) < self.min_score:
                continue
            row = self.df.iloc[idx]
            docs.append(
                RetrievedDoc(
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    source=str(row.get("source", "")),
                    focus_area=str(row.get("focus_area", "")),
                    score=float(score),
                )
            )
        return docs

    # ── Public interface ───────────────────────────────────────────────────────

    def chat(self, query: str) -> ChatResponse:
        """
        Retrieve relevant documents and return a structured offline answer.

        Returns a ChatResponse with:
          - answer        : formatted text answer
          - sources       : deduplicated source labels
          - retrieved_docs: raw retrieved RetrievedDoc objects
          - found         : False if no relevant docs were found
        """
        docs = self.retrieve(query)

        if not docs:
            return ChatResponse(
                answer=(
                    "The answer to your question is **not available** in the local "
                    "medical database. Please consult a qualified healthcare professional."
                ),
                sources=[],
                retrieved_docs=[],
                found=False,
            )

        answer = format_answer(query, docs)

        sources = list(
            dict.fromkeys(
                f"{d.focus_area} ({d.source})" for d in docs
            )
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieved_docs=docs,
            found=True,
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("   Medical RAG Chatbot — Fully Offline (MedQuAD)")
    print("=" * 60)

    bot = MedRAGChatbot()
    bot.setup()

    print("Ask a medical question below. Type 'quit' to exit.\n")
    print("-" * 60)

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        response = bot.chat(query)

        print(f"\nBot:\n{response.answer}")
        if response.sources:
            print(f"\nSources: {' | '.join(response.sources[:3])}")
        print("-" * 60)


if __name__ == "__main__":
    main()
