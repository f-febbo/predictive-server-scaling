"""Shutdown watchdog: force every fleet to zero once the experiment is over.

The scaler already returns its own fleet to the floor when the replay ends, but
that still leaves one instance per arm running indefinitely, and it only covers
the arm the scaler drives. This is the backstop: some hours after the replay
finishes, both Auto Scaling groups are pinned to zero — min, max, and desired —
which stops all compute cost regardless of what any scaling policy wants.

It is deliberately not a teardown. Destroying the stack stays a manual,
deliberate act via `destroy.sh`; this only stops the meter on the expensive
part. The two failure modes it defends against are forgetting the run exists
and a scaling policy misbehaving after the load has stopped.

Setting max_size to zero matters as much as desired_capacity: native Predictive
Scaling and target tracking both act on their own schedule, and either would
happily raise capacity again from a floor of zero.
"""

from __future__ import annotations

import datetime as dt
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

autoscaling = boto3.client("autoscaling")

SECONDS_PER_HOUR = 3600.0


def handler(event, context):  # noqa: ARG001 - Lambda signature
    replay_start = int(os.environ["REPLAY_START_EPOCH"])
    shutdown_after_hours = float(os.environ["SHUTDOWN_AFTER_HOURS"])
    group_names = [name for name in os.environ["ASG_NAMES"].split(",") if name]

    now = dt.datetime.now(dt.timezone.utc).timestamp()
    if not _should_shut_down(now, replay_start, shutdown_after_hours):
        remaining = shutdown_after_hours - (now - replay_start) / SECONDS_PER_HOUR
        logger.info("Experiment still running; %.1f hours until shutdown.", remaining)
        return {"status": "running", "hours_remaining": round(remaining, 2)}

    stopped = [name for name in group_names if _pin_to_zero(name)]
    logger.warning("Shutdown window reached; pinned to zero: %s", stopped)

    return {"status": "shut_down", "groups": stopped}


def _should_shut_down(
    now_epoch: float, replay_start_epoch: int, shutdown_after_hours: float
) -> bool:
    elapsed_hours = (now_epoch - replay_start_epoch) / SECONDS_PER_HOUR
    return elapsed_hours >= shutdown_after_hours


def _pin_to_zero(group_name: str) -> bool:
    """Force one group to zero capacity. Idempotent."""
    try:
        autoscaling.update_auto_scaling_group(
            AutoScalingGroupName=group_name,
            MinSize=0,
            MaxSize=0,
            DesiredCapacity=0,
        )
        return True
    except Exception:
        # One group failing must not stop the others being shut down.
        logger.exception("Could not pin %s to zero.", group_name)
        return False
