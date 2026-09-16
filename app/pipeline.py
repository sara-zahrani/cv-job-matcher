"""
Ingestion pipeline: source -> broker -> quality gate -> lakehouse.

Runs the three "rooms" together. Re-running with the same source is safe:
ids already in the table are treated as duplicates and skipped.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.broker import Broker
from app.ingestion.producer import TOPIC, load_jobs, produce
from app.quality.consumer import validate_stream
from app.quality.gate import QualityGate
from app.rag.embeddings import Embedder
from app.rag.vectorstore import VectorStore
from app.storage.lakehouse import existing_ids, read_jobs, write_jobs

SOURCE_PATH = Path("data/jobs.json")
QUARANTINE_DIR = Path("data/quarantine")


def write_quarantine(rejected: list[dict], directory: Path = QUARANTINE_DIR) -> Path | None:
    """One JSON file per run, timestamped, so runs don't overwrite each other."""
    if not rejected:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{stamp}.json"
    path.write_text(json.dumps(rejected, indent=2, ensure_ascii=False))
    return path


async def run(source: Path = SOURCE_PATH) -> dict:
    broker = Broker()
    broker.create_topic(TOPIC)

    # Seed the gate with what storage already holds: this is what makes
    # a second run a no-op instead of a duplicate.
    gate = QualityGate(known_ids=existing_ids())

    jobs = load_jobs(source)
    _, (clean, rejected, skipped) = await asyncio.gather(
        produce(broker, jobs),
        validate_stream(broker, gate),
    )

    version = write_jobs(clean) if clean else None
    quarantine_file = write_quarantine(rejected)

    # Index from the table, not from the clean list: the table is the truth,
    # and this also picks up anything stored earlier but not yet embedded.
    store = VectorStore(Embedder())
    embedded = store.add_jobs(read_jobs().to_dict(orient="records"))
    indexed = store.count()
    store.close()

    return {
        "jobs read from jobs.json": len(jobs),
        "jobs passed the gate, written to delta table": len(clean),
        "jobs skipped, already in delta table": len(skipped),
        "jobs rejected, written to quarantine": len(rejected),
        "delta table version after write (None = no write)": version,
        "quarantine file": str(quarantine_file) if quarantine_file else None,
        "jobs embedded into vector store this run": embedded,
        "vectors in store, total": indexed,
    }


if __name__ == "__main__":
    summary = asyncio.run(run())
    for key, value in summary.items():
        print(f"{key:<52} {value}")
