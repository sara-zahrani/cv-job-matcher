"""
In-memory message broker.

Stands in for Kafka. A broker has named topics; each topic is a queue.
Producers publish to a topic and move on. Consumers take from a topic
when they are ready. Neither side knows about the other.
"""

import asyncio


class Broker:
    def __init__(self) -> None:
        # One queue per topic. asyncio.Queue is a queue that async tasks
        # can wait on without blocking the whole program.
        self.topics: dict[str, asyncio.Queue] = {}

    def create_topic(self, name: str) -> None:
        self.topics[name] = asyncio.Queue()

    async def publish(self, topic: str, message: dict) -> None:
        # put() adds to the end of the queue. Awaiting it lets other tasks
        # run in the meantime.
        await self.topics[topic].put(message)

    async def consume(self, topic: str) -> dict:
        # get() waits until something is in the queue, then removes and
        # returns the oldest item. If the queue is empty, this task sleeps
        # and the event loop runs something else.
        return await self.topics[topic].get()


    def size(self, topic: str) -> int:
        """How many messages are waiting in a topic right now."""
        return self.topics[topic].qsize()


if __name__ == "__main__":

    async def fast_producer(broker: Broker) -> None:
        # Publishes one message every 0.1 s. Five messages in half a second.
        for i in range(1, 6):
            await broker.publish("jobs_raw", {"id": f"job-{i:03d}"})
            print(f"  produced job-{i:03d}   waiting in queue: {broker.size('jobs_raw')}")
            await asyncio.sleep(0.1)

    async def slow_consumer(broker: Broker) -> None:
        # Takes one message every 0.5 s. Five times slower than the producer.
        for _ in range(5):
            message = await broker.consume("jobs_raw")
            await asyncio.sleep(0.5)  # pretend this is real work
            print(f"consumed {message['id']}   waiting in queue: {broker.size('jobs_raw')}")

    async def demo() -> None:
        broker = Broker()
        broker.create_topic("jobs_raw")

        # gather() runs both coroutines at the same time on the one event
        # loop and waits until both finish. While one is sleeping, the
        # other runs. This is the whole trick.
        await asyncio.gather(fast_producer(broker), slow_consumer(broker))

    # asyncio.run starts the event loop, runs demo() until it finishes,
    # then shuts the loop down. This is the one place the loop is started.
    asyncio.run(demo())
