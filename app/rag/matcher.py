"""
Matcher: the query side of RAG.

    CV.pdf -> text -> strip PII -> embed -> nearest jobs in the store

This is retrieval. Generation (the LLM explaining each match) comes after,
and is built on top of what this returns.
"""

import sys
from pathlib import Path

from app.rag.cv import extract_text, strip_pii
from app.rag.embeddings import Embedder
from app.rag.vectorstore import VectorStore


def retrieve(cv_text: str, store: VectorStore, top_k: int = 5) -> list[dict]:
    """Nearest jobs for a CV, best first. Pure retrieval, no LLM."""
    return store.search(cv_text, top_k=top_k)


if __name__ == "__main__":
    # usage: uv run python -m app.rag.matcher path/to/cv.pdf ["Full Name"]
    pdf = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else None

    cv_text = strip_pii(extract_text(pdf), name)

    store = VectorStore(Embedder())

    def show(label: str, query: str) -> None:
        print(f"\n== {label} ({len(query)} chars) ==")
        print(f"{'score':<8} {'id':<9} {'title':<36} company")
        for m in retrieve(query, store, top_k=10):
            print(f"{m['score']:<8} {m['job_id']:<9} {m['title']:<36} {m['company']}")

    # Experiment: is the whole-CV vector a blur? Compare against the summary only.
    show("whole CV", cv_text)
    show("profile summary only", cv_text[:500])

    store.close()
