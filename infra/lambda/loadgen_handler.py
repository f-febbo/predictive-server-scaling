"""Load generator: replay the trace into the queues, one minute at a time.

Runs once a minute on an EventBridge schedule. Each invocation looks up how
many arrivals the trace recorded for the current replay minute, divides by the
scale-down factor, and sends that many messages to every arm's queue.

Time is deliberately not compressed. Replaying a day in an hour would leave
instance boot time — fixed at two to three minutes by physics — spanning over
an hour of trace time, which is a completely different regime from the one the
simulator studied. Scaling the message volume down instead shrinks the fleet to
something cheap while preserving the ratio of boot delay to demand-change
timescale, which is the only thing that makes the live run comparable.

Every arm receives identical messages from the same invocation, so the two
scaling strategies are compared against the same load rather than against two
independent samples of it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pathlib
import uuid

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sqs = boto3.client("sqs")

HERE = pathlib.Path(__file__).parent
SECONDS_PER_MINUTE = 60.0
SQS_BATCH_LIMIT = 10

with open(HERE / "replay_trace.json") as handle:
    TRACE = json.load(handle)["values"]


def handler(event, context):  # noqa: ARG001 - Lambda signature
    replay_start = int(os.environ["REPLAY_START_EPOCH"])
    divisor = float(os.environ["ARRIVAL_DIVISOR"])
    queue_urls = [url for url in os.environ["QUEUE_URLS"].split(",") if url]

    minute = int(
        (dt.datetime.now(dt.timezone.utc).timestamp() - replay_start) // SECONDS_PER_MINUTE
    )
    if minute < 0 or minute >= len(TRACE):
        logger.info("Replay minute %s is outside the trace; sending nothing.", minute)
        return {"status": "outside_replay_window", "minute": minute}

    count = int(round(TRACE[minute] / divisor))
    if count <= 0:
        return {"status": "ok", "minute": minute, "sent": 0}

    sent = {url: _send(url, count, minute) for url in queue_urls}
    logger.info("minute=%s count=%s sent=%s", minute, count, sent)

    return {"status": "ok", "minute": minute, "sent": sent}


def _send(queue_url: str, count: int, minute: int) -> int:
    """Send `count` messages, batched to SQS's ten-per-request limit.

    Messages are sent as fast as the batches go out rather than spread across
    the minute. At this volume the difference is a few seconds of sub-minute
    burstiness, which the queue absorbs; spreading them would mean holding the
    Lambda open for the full minute and paying for the wait.
    """
    delivered = 0

    for start in range(0, count, SQS_BATCH_LIMIT):
        batch = [
            {
                "Id": str(index),
                "MessageBody": json.dumps({"minute": minute, "id": str(uuid.uuid4())}),
            }
            for index in range(start, min(start + SQS_BATCH_LIMIT, count))
        ]

        try:
            response = sqs.send_message_batch(QueueUrl=queue_url, Entries=batch)
            delivered += len(response.get("Successful", []))
            for failure in response.get("Failed", []):
                logger.warning("Failed to enqueue %s: %s", failure.get("Id"), failure.get("Message"))
        except Exception:
            # A failed batch loses that minute's messages rather than the run.
            # The scaler's recent-level correction reads the real arrival
            # metric, so it will see the shortfall and size accordingly.
            logger.exception("send_message_batch failed for %s", queue_url)

    return delivered
