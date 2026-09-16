"""
Vector store.

Qdrant, in local mode: no server, a folder on disk read by the client
in-process. It holds one vector per job plus a small payload, and answers
one question fast: which stored vectors are nearest to this one?

It is a searchable shadow of the Delta table, never the source of truth.
Delete the folder and it is rebuilt from the table.
"""

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.rag.embeddings import Embedder

STORE_PATH = Path("data/qdrant")
COLLECTION = "jobs"


def job_to_text(job: dict) -> str:
    """The text the model reads for a posting. Title and skills carry a lot
    of meaning in few words, so they go in alongside the description."""
    # Read back from Delta via pandas, "skills" is a numpy array, not a list.
    # list() makes it a list either way; the None check avoids asking numpy
    # a true/false question it refuses to answer.
    raw = job.get("skills")
    skills = ", ".join(list(raw) if raw is not None else [])
    return f"{job['title']} at {job['company']}. Skills: {skills}. {job['description']}"


def point_id(job_id: str) -> str:
    """Qdrant ids must be integers or UUIDs. uuid5 turns 'job-001' into the
    same UUID every time, so re-runs address the same point."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, job_id))


class VectorStore:
    def __init__(self, embedder: Embedder, path: Path = STORE_PATH) -> None:
        self.embedder = embedder
        self.client = QdrantClient(path=str(path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(COLLECTION):
            return
        # The vector size must match the embedding model. Embed one short
        # string to learn it rather than hard-coding 2048.
        size = len(self.embedder.embed(["probe"])[0])
        self.client.create_collection(
            COLLECTION, vectors_config=VectorParams(size=size, distance=Distance.COSINE)
        )

    def existing_ids(self) -> set[str]:
        """Job ids already stored, read back from payloads."""
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                COLLECTION, limit=256, offset=offset, with_payload=["job_id"], with_vectors=False
            )
            ids.update(p.payload["job_id"] for p in points)
            if offset is None:
                return ids

    def add_jobs(self, jobs: list[dict]) -> int:
        """Embed and store postings not already present. Returns how many were added."""
        known = self.existing_ids()
        new = [j for j in jobs if j["id"] not in known]
        if not new:
            return 0

        vectors = self.embedder.embed([job_to_text(j) for j in new])
        self.client.upsert(
            COLLECTION,
            points=[
                PointStruct(
                    id=point_id(j["id"]),
                    vector=v,
                    payload={
                        "job_id": j["id"],
                        "title": j["title"],
                        "company": j["company"],
                        "location": j["location"],
                    },
                )
                for j, v in zip(new, vectors)
            ],
        )
        return len(new)

    def count(self) -> int:
        return self.client.count(COLLECTION).count

    def close(self) -> None:
        """Flush to disk. Call this rather than relying on garbage collection:
        local-mode Qdrant cannot clean up once Python is already shutting down."""
        self.client.close()


if __name__ == "__main__":
    from app.storage.lakehouse import read_jobs

    jobs = read_jobs().to_dict(orient="records")  # DataFrame rows -> list of dicts
    print(f"read {len(jobs)} jobs from the Delta table")
    print(f"sample text to embed:\n  {job_to_text(jobs[0])[:160]}...\n")

    store = VectorStore(Embedder())
    added = store.add_jobs(jobs)
    print(f"added {added} new vectors; store now holds {store.count()}")
    store.close()
