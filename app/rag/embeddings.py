"""
Embeddings.

Turns text into a vector: a list of numbers that places the text on a
"map of meaning". Texts about similar things land close together. This is
the one operation both sides of RAG share: jobs are embedded at index time,
the CV is embedded at query time, with the SAME model, or the coordinates
are not comparable.
"""

import math

from openai import OpenAI

from app.config import settings


class Embedder:
    """Calls the embedding model through OpenRouter's OpenAI-compatible API."""

    def __init__(self, model: str = settings.embed_model) -> None:
        settings.require_api_key()
        self.model = model
        self.client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per input text, in the same order. Batched in one call."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Angle between two vectors, as a number from -1 (opposite) to 1 (same).
    Ignores length, so a long CV and a short job can still be 'close'."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    texts = [
        "Data engineer building Airflow pipelines and Delta Lake tables in Python",
        "Machine learning engineer deploying PyTorch models with Kubernetes",
        "Pastry chef specialising in French desserts and wedding cakes",
    ]
    embedder = Embedder()
    vectors = embedder.embed(texts)

    print(f"model: {embedder.model}")
    print(f"each vector has {len(vectors[0])} numbers; first five of text 1: "
          f"{[round(x, 3) for x in vectors[0][:5]]}\n")

    labels = ["data eng", "ML eng", "pastry"]
    for i in range(3):
        for j in range(i + 1, 3):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"{labels[i]:<9} vs {labels[j]:<9} similarity = {sim:.3f}")
