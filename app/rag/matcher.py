"""
Matcher: the query side of RAG.

    CV.pdf -> text -> strip PII -> embed -> nearest jobs in the store

This is retrieval. Generation (the LLM explaining each match) comes after,
and is built on top of what this returns.
"""

import sys
import time
from pathlib import Path

from app.rag.cv import extract_text, strip_pii
from app.rag.embeddings import Embedder
from app.rag.llm import LLMClient
from app.rag.prompts import SYSTEM, build_match_prompt
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


def match(cv_text: str, store: VectorStore, llm: LLMClient, top_k: int = 5) -> dict:
    """The full RAG query: retrieve, then generate. If nothing retrieved is a
    real match, skip the LLM: there is nothing worth explaining."""
    t0 = time.perf_counter()
    matches = retrieve(cv_text, store, top_k=top_k)
    strong = has_strong_match(matches)
    print(f"[retrieve] {len(matches)} jobs, best score {matches[0]['score']:.3f}, "
          f"{time.perf_counter() - t0:.1f}s", file=sys.stderr)

    assessment = None
    if strong:
        prompt = build_match_prompt(cv_text, matches)
        print(f"[generate] asking {llm.model} ({len(prompt)} chars of prompt)...", file=sys.stderr)
        t0 = time.perf_counter()
        assessment = llm.generate(SYSTEM, prompt)
        print(f"[generate] done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    return {"strong_match": strong, "matches": matches, "assessment": assessment}


if __name__ == "__main__":
    # usage: uv run python -m app.rag.matcher path/to/cv.(pdf|txt) ["Full Name"]
    cv_text = load_cv(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)

    store = VectorStore(Embedder())
    result = match(cv_text, store, LLMClient())
    store.close()

    print(f"{'score':<8} {'id':<9} {'title':<36} company")
    for m in result["matches"]:
        print(f"{m['score']:<8} {m['job_id']:<9} {m['title']:<36} {m['company']}")

    if result["strong_match"]:
        print(f"\n{result['assessment']}")
    else:
        print(f"\nNo strong matches (best score below {MIN_SCORE}); LLM not called.")
