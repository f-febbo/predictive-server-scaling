"""Predictive scaler: forecast the load, size the fleet, set desired capacity.

Runs once a minute on an EventBridge schedule. Each invocation:

  1. works out where the replay has reached,
  2. reads the forecast that the offline model produced for this minute,
  3. corrects it by how the observed arrival rate compares with the trace,
  4. converts the corrected forecast to an instance count via Little's Law,
  5. calls SetDesiredCapacity, and publishes what it did to CloudWatch.

Why the forecast is a lookup table rather than a model evaluated here
--------------------------------------------------------------------
The values in `forecast_table.json` are the LightGBM quantile model's genuine
out-of-sample output, produced by the Phase 4 rolling-origin backtest: the
forecast stored for minute `m` was computed only from data at or before `m`.
Since the live run replays exactly that historical period, evaluating the model
again here would recompute numbers we already have.

Shipping LightGBM itself would drag numpy and scipy into the deployment package
for roughly 60MB and no additional insight at this timescale. In a system
replaying live traffic rather than a known trace, this handler would call the
model instead of indexing it — the surrounding control loop is identical either
way, and the control loop is what this deployment exists to demonstrate.

The step that is genuinely live is (3): the recent-level correction reads real
CloudWatch metrics, so the scaler responds to what the queue is actually
receiving rather than running open-loop off a script.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import pathlib

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

autoscaling = boto3.client("autoscaling")
cloudwatch = boto3.client("cloudwatch")

HERE = pathlib.Path(__file__).parent
SECONDS_PER_MINUTE = 60.0

# Bounds on the recent-level correction. Unclipped, a near-idle comparison
# window turns one quiet minute into an enormous capacity request.
RATIO_BOUNDS = (0.5, 2.0)
CORRECTION_WINDOW_MINUTES = 10


def _load(name: str) -> list[float]:
    with open(HERE / name) as handle:
        return json.load(handle)["values"]


FORECAST = _load("forecast_table.json")
TRACE = _load("replay_trace.json")


def handler(event, context):  # noqa: ARG001 - Lambda signature
    config = _config()
    minute = _replay_minute(config["replay_start_epoch"])

    if minute >= len(FORECAST):
        # The replay has finished. Scale back to the floor rather than leaving
        # the fleet frozen at its last decision: the load generator has stopped,
        # so those instances would poll an empty queue and bill indefinitely.
        # This is a cost guard, not a teardown -- the stack still has to be
        # destroyed explicitly.
        _scale_to_floor(config)
        return {"status": "replay_finished", "minute": minute}

    if minute < 0:
        # Applied slightly before the replay clock starts; the fleet is already
        # at its floor, so there is nothing to do.
        logger.info("Replay has not started yet (minute %s).", minute)
        return {"status": "before_replay", "minute": minute}

    predicted = FORECAST[minute]
    if predicted is None or not math.isfinite(predicted):
        logger.info("No forecast for minute %s; leaving capacity alone.", minute)
        return {"status": "no_forecast", "minute": minute}

    correction = _recent_level_correction(config, minute)
    corrected = predicted * correction

    desired = _instances_for(
        arrivals_per_minute=corrected / config["arrival_divisor"],
        service_seconds=config["service_seconds"],
        target_utilization=config["target_utilization"],
    )
    desired = max(config["min_size"], min(config["max_size"], desired))

    autoscaling.set_desired_capacity(
        AutoScalingGroupName=config["asg_name"],
        DesiredCapacity=desired,
        HonorCooldown=False,
    )

    logger.info(
        "minute=%s forecast=%.1f correction=%.2f corrected=%.1f desired=%s",
        minute,
        predicted,
        correction,
        corrected,
        desired,
    )
    _publish(config, minute, predicted, corrected, desired)

    return {
        "status": "ok",
        "minute": minute,
        "forecast": predicted,
        "correction": correction,
        "desired": desired,
    }


def _scale_to_floor(config: dict) -> None:
    """Return the fleet to its minimum size.

    Used once the replay is over. Without this the fleet stays wherever the
    last forecast put it — possibly a dozen instances — polling a queue that
    nothing is feeding, for as long as the stack exists.
    """
    try:
        autoscaling.set_desired_capacity(
            AutoScalingGroupName=config["asg_name"],
            DesiredCapacity=config["min_size"],
            HonorCooldown=False,
        )
        logger.info("Replay finished; scaled to floor of %s.", config["min_size"])
    except Exception:
        logger.exception("Could not scale down after the replay finished.")


def _config() -> dict:
    return {
        "asg_name": os.environ["ASG_NAME"],
        "queue_name": os.environ["QUEUE_NAME"],
        "namespace": os.environ["METRIC_NAMESPACE"],
        "replay_start_epoch": int(os.environ["REPLAY_START_EPOCH"]),
        "service_seconds": float(os.environ["SERVICE_SECONDS"]),
        "target_utilization": float(os.environ["TARGET_UTILIZATION"]),
        "arrival_divisor": float(os.environ["ARRIVAL_DIVISOR"]),
        "min_size": int(os.environ["MIN_SIZE"]),
        "max_size": int(os.environ["MAX_SIZE"]),
    }


def _replay_minute(replay_start_epoch: int) -> int:
    """How many whole minutes into the replay we are.

    Time is not compressed: one wall-clock minute is one trace minute. Instance
    boot time is fixed at two to three minutes by physics, so compressing the
    replay would inflate the boot delay in trace terms and change the regime
    being demonstrated.
    """
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    return int((now - replay_start_epoch) // SECONDS_PER_MINUTE)


def _instances_for(
    arrivals_per_minute: float, service_seconds: float, target_utilization: float
) -> int:
    """Little's Law: L = lambda * W.

    Serving `lambda` messages per second, each occupying a worker for
    `service_seconds`, needs `lambda * service_seconds` workers busy at all
    times. Dividing by the utilisation target buys headroom.

    Deliberately identical to `src.policies.capacity.instances_for_arrival_rate`
    so that the deployed fleet is sized by the same arithmetic the simulator
    studied. A test asserts the two agree.
    """
    if arrivals_per_minute <= 0:
        return 0

    rate_per_second = arrivals_per_minute / SECONDS_PER_MINUTE
    return math.ceil(rate_per_second * service_seconds / target_utilization)


def _recent_level_correction(config: dict, minute: int) -> float:
    """How the observed arrival rate compares with the trace's expectation.

    This is the live half of the scaler. If the replay is running hot or cold
    relative to the script — a lost load-generator invocation, throttling, an
    SQS hiccup — the forecast is re-levelled to match what the queue is really
    receiving rather than what it was supposed to receive.
    """
    window_start = max(0, minute - CORRECTION_WINDOW_MINUTES)
    expected = sum(TRACE[window_start:minute]) / config["arrival_divisor"]
    if expected <= 0:
        return 1.0

    observed = _observed_arrivals(config, minutes=minute - window_start)
    if observed is None:
        return 1.0

    low, high = RATIO_BOUNDS
    return max(low, min(high, observed / expected))


def _observed_arrivals(config: dict, minutes: int) -> float | None:
    """Messages actually enqueued over the last `minutes`, from CloudWatch."""
    if minutes <= 0:
        return None

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(minutes=minutes)

    try:
        response = cloudwatch.get_metric_statistics(
            Namespace="AWS/SQS",
            MetricName="NumberOfMessagesSent",
            Dimensions=[{"Name": "QueueName", "Value": config["queue_name"]}],
            StartTime=start,
            EndTime=end,
            Period=60,
            Statistics=["Sum"],
        )
    except Exception:
        # A metrics outage must not take the scaler down; fall back to the
        # uncorrected forecast.
        logger.exception("Could not read arrival metrics; skipping correction.")
        return None

    points = response.get("Datapoints", [])
    if not points:
        return None
    return sum(point["Sum"] for point in points)


def _publish(
    config: dict, minute: int, forecast: float, corrected: float, desired: int
) -> None:
    """Publish what the scaler saw and decided, so the dashboard can show it."""
    common = {"Dimensions": [{"Name": "AutoScalingGroupName", "Value": config["asg_name"]}]}
    try:
        cloudwatch.put_metric_data(
            Namespace=config["namespace"],
            MetricData=[
                {"MetricName": "ForecastArrivalsPerMinute", "Value": forecast, **common},
                {"MetricName": "CorrectedArrivalsPerMinute", "Value": corrected, **common},
                {"MetricName": "DesiredInstances", "Value": float(desired), **common},
                {"MetricName": "ReplayMinute", "Value": float(minute), **common},
            ],
        )
    except Exception:
        logger.exception("Could not publish scaler metrics.")
