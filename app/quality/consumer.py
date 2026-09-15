"""
Validation consumer.

Takes postings off the broker one at a time, runs each through the quality
gate, and sorts them into two piles: clean and quarantined. Stops when it
sees the end-of-stream sentinel.
"""

import asyncio
from pathlib import Path

from app.ingestion.broker import Broker
from app.ingestion.producer import END_OF_STREAM, TOPIC, load_jobs, produce
from app.quality.gate import QualityGate


async def validate_stream(broker: Broker, gate: QualityGate) -> tuple[list[dict], list[dict]]:
    """Consume until end-of-stream. Returns (clean, quarantined)."""
    clean: list[dict] = []
    quarantined: list[dict] = []

    while True:
        job = await broker.consume(TOPIC)
        if job is END_OF_STREAM:
            break

        reasons = gate.check(job)
        if reasons:
            # Keep the original posting and attach why it was rejected.
            quarantined.append({"job": job, "reasons": reasons})
        else:
            clean.append(job)

    return clean, quarantined


if __name__ == "__main__":

    async def demo() -> None:
        broker = Broker()
        broker.create_topic(TOPIC)
        gate = QualityGate()

        jobs = load_jobs(Path("data/jobs.json"))

        # Producer and consumer run at the same time, like the broker demo.
        # gather returns each coroutine's result in order; produce() returns a
        # count we don't need here, so it goes to _.
        _, (clean, quarantined) = await asyncio.gather(
            produce(broker, jobs),
            validate_stream(broker, gate),
        )

        print(f"clean: {len(clean)}   quarantined: {len(quarantined)}")
        for item in quarantined:
            print(f"  {item['job'].get('id', '?'):<8} {item['reasons']}")

    asyncio.run(demo())
