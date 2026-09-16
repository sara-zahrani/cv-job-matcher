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

# Below this cosine score the top match is not a real match. Chosen from
# data/eval: focused CVs score 0.40+, off-topic CVs top out around 0.21-0.24.
# Re-check if the embedding model changes; the scale is model-specific.
MIN_SCORE = 0.25


def retrieve(cv_text: str, store: VectorStore, top_k: int = 5) -> list[dict]:
    """Nearest jobs for a CV, best first. Pure retrieval, no LLM."""
    return store.search(cv_text, top_k=top_k)


def has_strong_match(matches: list[dict]) -> bool:
    return bool(matches) and matches[0]["score"] >= MIN_SCORE


def load_cv(path: Path, name: str | None = None) -> str:
    """PDF or plain text in, PII-stripped text out."""
    raw = extract_text(path) if path.suffix.lower() == ".pdf" else path.read_text()
    return strip_pii(raw, name)


if __name__ == "__main__":
    # usage: uv run python -m app.rag.matcher path/to/cv.(pdf|txt) ["Full Name"]
    cv_text = load_cv(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)

    store = VectorStore(Embedder())
    matches = retrieve(cv_text, store, top_k=5)
    store.close()

    if not has_strong_match(matches):
        print(f"No strong matches (best score {matches[0]['score']:.3f} < {MIN_SCORE}). Showing nearest anyway:\n")

    print(f"{'score':<8} {'id':<9} {'title':<36} company")
    for m in matches:
        print(f"{m['score']:<8} {m['job_id']:<9} {m['title']:<36} {m['company']}")
