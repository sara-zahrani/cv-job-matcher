"""
Retrieval evaluation.

Runs every CV in data/eval/ through the matcher two ways, whole text and
the first 500 characters, and prints the top matches side by side. No
LLM involved: this looks only at what the vector search returns.

    uv run python -m app.rag.eval_retrieval
"""

from pathlib import Path

from app.rag.embeddings import Embedder
from app.rag.matcher import retrieve
from app.rag.vectorstore import VectorStore

EVAL_DIR = Path("data/eval")
TOP_K = 5
SUMMARY_CHARS = 500


def main() -> None:
    store = VectorStore(Embedder())

    for path in sorted(EVAL_DIR.glob("*.txt")):
        text = path.read_text()
        whole = retrieve(text, store, top_k=TOP_K)
        summary = retrieve(text[:SUMMARY_CHARS], store, top_k=TOP_K)

        print(f"\n{'=' * 100}\n{path.stem}  ({len(text)} chars)\n")
        print(f"  {'WHOLE TEXT':<48}   {'FIRST 500 CHARS':<48}")
        for w, s in zip(whole, summary):
            left = f"{w['score']:.3f}  {w['title'][:40]}"
            right = f"{s['score']:.3f}  {s['title'][:40]}"
            print(f"  {left:<48}   {right:<48}")

    store.close()


if __name__ == "__main__":
    main()
