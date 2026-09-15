"""
Job producer.

Reads job postings from a source and publishes each one, unchanged, to the
broker. Today the source is a JSON file. Later it can be an API. Nothing
downstream will know the difference, because all they see is the topic.
"""

import asyncio
import json
from pathlib import Path

from app.ingestion.broker import Broker

TOPIC = "jobs_raw"


def load_jobs(path: Path) -> list[dict]:
    """Read the JSON file and return the list of postings as dictionaries."""
    # Path.read_text opens, reads, and closes the file in one call.
    # json.loads turns the text into Python lists and dicts.
    return json.loads(path.read_text())


async def produce(broker: Broker, jobs: list[dict]) -> int:
    """Publish every posting to the topic. Returns how many were published."""
    for job in jobs:
        await broker.publish(TOPIC, job)
    return len(jobs)


if __name__ == "__main__":

    async def demo() -> None:
        broker = Broker()
        broker.create_topic(TOPIC)

        jobs = load_jobs(Path("data/jobs.json"))
        count = await produce(broker, jobs)
        print(f"published {count} jobs, waiting in queue: {broker.size(TOPIC)}")

        # Peek at the first one to see what a message looks like on the wire.
        first = await broker.consume(TOPIC)
        print("first message:", first["id"], "-", first["title"], "@", first["company"])

    asyncio.run(demo())
