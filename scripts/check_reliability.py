"""Check the real Compose stack on isolated queues and temporary containers.

Start and train the normal stack first; export host-side DB settings as for the
live tests. This script does not stop the normal app/worker or delete DB rows.
"""

import argparse
import asyncio
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aio_pika
import httpx
from sqlalchemy import select

from mlops.config import Settings
from mlops.db import task_results
from mlops.db.session import create_db_engine
from mlops.messaging import TaskMessage
from mlops.producer.publisher import QueueConfig, RabbitPublisher


async def docker(*args: str) -> str:
    result = await asyncio.to_thread(
        subprocess.run, ["docker", *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


async def wait_for(check):
    async with asyncio.timeout(60):
        while True:
            if result := await check():
                return result
            await asyncio.sleep(0.05)


async def verify(args):
    token = uuid4().hex[:12]
    queue = QueueConfig(f"verify_{token}", 3)
    api, worker = f"mlops-verify-api-{token}", f"mlops-verify-worker-{token}"
    common = ["-e", f"QUEUE_NAME={queue.name}", "-e", "QUEUE_MAX_LENGTH=3"]
    common += ["-e", "INFERENCE_DELAY_SECONDS=3", "-e", "PREFETCH_COUNT=1"]
    tasks = [TaskMessage.new(f"Reliability check {token} {i}") for i in range(3)]
    engine = create_db_engine()
    checks = {"timestamp_utc": datetime.now(UTC).isoformat(), "queue": queue.name}
    succeeded = False

    async def rows(ids):
        async with engine.connect() as connection:
            return (
                (
                    await connection.execute(
                        select(task_results).where(task_results.c.task_id.in_(ids))
                    )
                )
                .mappings()
                .all()
            )

    async def worker_healthy():
        try:
            return (
                await docker(
                    "exec",
                    worker,
                    "python",
                    "-c",
                    "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status)",
                )
                == "204"
            )
        except subprocess.CalledProcessError:
            return False

    try:
        await docker(
            "compose",
            "run",
            "--detach",
            "--no-deps",
            "--name",
            api,
            "--publish",
            "127.0.0.1::8000",
            *common,
            "app",
        )
        address = await docker("port", api, "8000/tcp")
        base_url = "http://127.0.0.1:" + address.rsplit(":", 1)[1]
        async with httpx.AsyncClient(base_url=base_url, trust_env=False, timeout=5) as client:

            async def api_healthy():
                try:
                    return (await client.get("/healthz")).status_code == 200
                except httpx.TransportError:
                    return False

            await wait_for(api_healthy)
            async with RabbitPublisher(args.rabbitmq_url, queue) as publisher:
                for task in tasks:
                    await publisher.publish_once(task)
                rejected_text = f"Rejected reliability check {token}"
                started = time.monotonic()
                response = await client.post("/api/v1/tasks", json={"text": rejected_text})
                latency = time.monotonic() - started
                assert response.status_code == 503, response.text
                assert response.headers["Retry-After"] == "3"
                assert latency < 1, f"Overload response took {latency:.3f}s"
                assert not await rows([task.task_id for task in tasks])
                checks["api_overload"] = {
                    "status": response.status_code,
                    "seconds": latency,
                    "retry_after": response.headers["Retry-After"],
                    "ready": await publisher.ready_depth(),
                    "pending_rows": 0,
                }

                await docker(
                    "compose", "run", "--detach", "--no-deps", "--name", worker, *common, "worker"
                )
                await wait_for(worker_healthy)
                started = time.monotonic()
                assert await worker_healthy()
                checks["health_during_inference"] = {
                    "status": 204,
                    "seconds_including_docker_exec": time.monotonic() - started,
                }
                # A 3-second inference is in flight with two more tasks queued.
                assert await publisher.ready_depth() == 2
                started = time.monotonic()
                await docker("stop", "--time", "30", worker)
                shutdown = time.monotonic() - started
                exit_code = await docker("inspect", "--format", "{{.State.ExitCode}}", worker)
                completed = await rows([task.task_id for task in tasks])
                assert exit_code == "0" and len(completed) == 1
                assert await publisher.ready_depth() == 2
                checks["graceful_sigterm"] = {
                    "seconds": shutdown,
                    "exit_code": int(exit_code),
                    "completed_rows": len(completed),
                    "remaining_ready": 2,
                }
                await docker("start", worker)
                await wait_for(worker_healthy)

                async def all_completed():
                    return len(await rows([task.task_id for task in tasks])) == 3

                await wait_for(all_completed)
                # The marker proves the sequential worker passed the duplicate.
                marker = TaskMessage.new(f"Idempotency marker {token}")
                await publisher.publish_once(tasks[0])
                await publisher.publish_once(marker)

                async def marker_completed():
                    return bool(await rows([marker.task_id]))

                await wait_for(marker_completed)
                assert len(await rows([tasks[0].task_id])) == 1
                checks["idempotency"] = {"publishes": 2, "result_rows": 1}

                connection = await aio_pika.connect(args.rabbitmq_url)
                async with connection:
                    channel = await connection.channel()
                    poison_id = str(uuid4())
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            b"invalid json",
                            message_id=poison_id,
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        ),
                        routing_key=queue.name,
                        mandatory=True,
                    )
                    dlq = await channel.get_queue(queue.name + ".dlq")

                    async def poison_arrived():
                        return await dlq.get(fail=False)

                    poison = await wait_for(poison_arrived)
                    assert poison.message_id == poison_id
                    assert poison.body == b"invalid json" and poison.headers["attempt"] == 3
                    await poison.ack()
                    checks["poison_dlq"] = {"attempts": 3, "body_and_id_retained": True}

                async with engine.connect() as connection:
                    rejected = (
                        await connection.execute(
                            select(task_results.c.task_id).where(
                                task_results.c.text == rejected_text
                            )
                        )
                    ).all()
                assert not rejected
                checks["api_overload"]["rejected_result_rows"] = 0
                checks["accepted_tasks_completed"] = 3
                succeeded = True
    finally:
        # Only this run's randomly named containers are removed. Volumes and
        # normal services remain intact, and failed test queues are preserved.
        if succeeded:
            await docker("stop", "--time", "30", worker)
        for name in (worker, api):
            await asyncio.to_thread(
                subprocess.run, ["docker", "rm", "--force", name], capture_output=True, check=False
            )
        await engine.dispose()
        if succeeded:
            connection = await aio_pika.connect(args.rabbitmq_url)
            async with connection:
                channel = await connection.channel()
                for name in (queue.name, queue.name + ".retry", queue.name + ".dlq"):
                    state = (await channel.declare_queue(name, passive=True)).declaration_result
                    if state.message_count or state.consumer_count:
                        raise RuntimeError(f"Verification queue is not empty or unused: {name}")
                    # Quorum queues reject if-unused. Only this run publishes
                    # here, and both temporary containers have already stopped.
                    await channel.queue_delete(name, if_empty=False, if_unused=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rabbitmq-url", default=Settings().rabbitmq_url)
    parser.add_argument("--output", type=Path, default=Path("metrics/reliability.json"))
    asyncio.run(verify(parser.parse_args()))
