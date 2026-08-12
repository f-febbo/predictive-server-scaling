"""Tests for the deployed Lambda handlers.

The most important one asserts that the capacity arithmetic running in AWS is
the same arithmetic the simulator studied. The handler cannot import from
`src/` — it ships as a standalone zip — so the formula is written out twice,
and a drift between the two copies would mean the live fleet is sized by
different rules than every result in the README. That is exactly the kind of
divergence nobody notices until the numbers stop matching.
"""

import datetime as dt
import importlib
import os
import sys
from pathlib import Path

import pytest

LAMBDA_DIR = Path(__file__).resolve().parents[1] / "infra" / "lambda"

# boto3 clients are created at import time, as AWS recommends for connection
# reuse, and creating one resolves region and credentials. Pin dummy static
# credentials so the suite does not depend on whatever profile happens to be
# configured on the machine running it — otherwise these tests pass or fail
# based on someone's ~/.aws/config, which has nothing to do with the code.
# No API call is ever made; every AWS call in these tests is patched out.
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ.pop("AWS_PROFILE", None)

sys.path.insert(0, str(LAMBDA_DIR))

scaler = importlib.import_module("scaler_handler")
loadgen = importlib.import_module("loadgen_handler")

from src.policies.capacity import instances_for_arrival_rate  # noqa: E402


# --- the deployed maths matches the simulated maths -------------------------


@pytest.mark.parametrize("arrivals_per_minute", [0, 1, 7, 30, 60, 119, 120, 137, 500])
@pytest.mark.parametrize("service_seconds", [10.0, 30.0, 60.0])
@pytest.mark.parametrize("utilization", [0.5, 0.8, 1.0])
def test_lambda_capacity_matches_the_simulator(
    arrivals_per_minute, service_seconds, utilization
):
    deployed = scaler._instances_for(
        arrivals_per_minute=arrivals_per_minute,
        service_seconds=service_seconds,
        target_utilization=utilization,
    )
    simulated = instances_for_arrival_rate(
        arrivals_per_minute / 60.0, service_seconds, utilization
    )

    assert deployed == simulated


def test_capacity_is_a_hand_checkable_number():
    # 120 arrivals/min is 2/s; at 30s each that is 60 instances.
    assert scaler._instances_for(120, service_seconds=30.0, target_utilization=1.0) == 60


def test_capacity_rounds_up():
    assert scaler._instances_for(1, service_seconds=30.0, target_utilization=1.0) == 1


def test_no_arrivals_needs_no_instances():
    assert scaler._instances_for(0, service_seconds=30.0, target_utilization=1.0) == 0


# --- replay clock -----------------------------------------------------------


def test_replay_minute_counts_whole_minutes_since_the_start():
    now = dt.datetime.now(dt.timezone.utc).timestamp()

    assert scaler._replay_minute(int(now)) == 0
    assert scaler._replay_minute(int(now - 599)) == 9
    assert scaler._replay_minute(int(now - 600)) == 10


def test_the_fleet_is_returned_to_its_floor_once_the_replay_ends(monkeypatch):
    # Otherwise the fleet stays wherever the last forecast put it, polling a
    # queue nothing is feeding, billing until someone notices. This is a cost
    # guard, so it is worth a test of its own.
    calls = []
    monkeypatch.setattr(
        scaler.autoscaling,
        "set_desired_capacity",
        lambda **kwargs: calls.append(kwargs),
    )
    # A replay start far enough in the past that the trace has run out.
    past = int(dt.datetime.now(dt.timezone.utc).timestamp()) - (len(scaler.TRACE) + 10) * 60
    monkeypatch.setenv("REPLAY_START_EPOCH", str(past))
    _set_required_env(monkeypatch)

    result = scaler.handler({}, None)

    assert result["status"] == "replay_finished"
    assert len(calls) == 1
    assert calls[0]["DesiredCapacity"] == 1


def test_nothing_is_scaled_before_the_replay_starts(monkeypatch):
    calls = []
    monkeypatch.setattr(
        scaler.autoscaling,
        "set_desired_capacity",
        lambda **kwargs: calls.append(kwargs),
    )
    future = int(dt.datetime.now(dt.timezone.utc).timestamp()) + 600
    monkeypatch.setenv("REPLAY_START_EPOCH", str(future))
    _set_required_env(monkeypatch)

    result = scaler.handler({}, None)

    assert result["status"] == "before_replay"
    assert calls == []


def _set_required_env(monkeypatch) -> None:
    for key, value in {
        "ASG_NAME": "test-asg",
        "QUEUE_NAME": "test-queue",
        "METRIC_NAMESPACE": "Test",
        "SERVICE_SECONDS": "30",
        "TARGET_UTILIZATION": "0.8",
        "ARRIVAL_DIVISOR": "5",
        "MIN_SIZE": "1",
        "MAX_SIZE": "30",
    }.items():
        monkeypatch.setenv(key, value)


def test_replay_minute_is_negative_before_the_start():
    # Guards the "outside the replay window" branch: a stack applied a few
    # minutes before the clock starts must not index the trace at -1 and
    # silently read from the end of it.
    now = dt.datetime.now(dt.timezone.utc).timestamp()

    assert scaler._replay_minute(int(now + 300)) < 0


# --- recent-level correction ------------------------------------------------


def test_correction_is_neutral_when_observations_match_the_trace(monkeypatch):
    config = _config()
    expected = sum(scaler.TRACE[0:10]) / config["arrival_divisor"]
    monkeypatch.setattr(scaler, "_observed_arrivals", lambda *_, **__: expected)

    assert scaler._recent_level_correction(config, minute=10) == pytest.approx(1.0)


def test_correction_scales_up_when_more_is_arriving_than_expected(monkeypatch):
    config = _config()
    expected = sum(scaler.TRACE[0:10]) / config["arrival_divisor"]
    monkeypatch.setattr(scaler, "_observed_arrivals", lambda *_, **__: expected * 1.5)

    assert scaler._recent_level_correction(config, minute=10) == pytest.approx(1.5)


def test_correction_is_clipped_at_both_ends(monkeypatch):
    # Unclipped, a near-idle comparison window turns one quiet minute into an
    # enormous capacity request.
    config = _config()
    low, high = scaler.RATIO_BOUNDS

    monkeypatch.setattr(scaler, "_observed_arrivals", lambda *_, **__: 1e9)
    assert scaler._recent_level_correction(config, minute=10) == high

    monkeypatch.setattr(scaler, "_observed_arrivals", lambda *_, **__: 0.0)
    assert scaler._recent_level_correction(config, minute=10) == low


def test_correction_falls_back_to_neutral_when_metrics_are_unavailable(monkeypatch):
    # A CloudWatch outage must not take the scaler down or spike the fleet.
    config = _config()
    monkeypatch.setattr(scaler, "_observed_arrivals", lambda *_, **__: None)

    assert scaler._recent_level_correction(config, minute=10) == 1.0


def test_correction_is_neutral_at_the_very_start_of_the_replay():
    # There is no history to compare against in minute zero.
    assert scaler._recent_level_correction(_config(), minute=0) == 1.0


# --- shipped payload --------------------------------------------------------


def test_the_trace_and_forecast_tables_are_the_same_length():
    # A mismatch would have the generator sending one minute of the trace while
    # the scaler provisions for another.
    assert len(scaler.TRACE) == len(scaler.FORECAST)
    assert len(loadgen.TRACE) == len(scaler.TRACE)


def test_the_payload_covers_the_full_replay_window():
    assert len(scaler.TRACE) == 2880  # 48 hours at one-minute resolution


def test_trace_values_are_non_negative():
    assert all(value >= 0 for value in scaler.TRACE)


def test_forecast_values_are_non_negative_where_present():
    assert all(value is None or value >= 0 for value in scaler.FORECAST)


def _config() -> dict:
    return {
        "asg_name": "test-asg",
        "queue_name": "test-queue",
        "namespace": "Test",
        "replay_start_epoch": 0,
        "service_seconds": 30.0,
        "target_utilization": 0.8,
        "arrival_divisor": 5.0,
        "min_size": 1,
        "max_size": 30,
    }


# --- shutdown watchdog ------------------------------------------------------

watchdog = importlib.import_module("watchdog_handler")

HOUR = 3600


def test_watchdog_holds_off_during_the_experiment():
    start = 1_000_000

    assert not watchdog._should_shut_down(start, start, shutdown_after_hours=50)
    assert not watchdog._should_shut_down(start + 47 * HOUR, start, 50)
    assert not watchdog._should_shut_down(start + 49.9 * HOUR, start, 50)


def test_watchdog_fires_once_the_window_closes():
    start = 1_000_000

    assert watchdog._should_shut_down(start + 50 * HOUR, start, 50)
    assert watchdog._should_shut_down(start + 200 * HOUR, start, 50)


def test_watchdog_fires_exactly_at_the_boundary():
    start = 1_000_000

    assert watchdog._should_shut_down(start + 50 * HOUR, start, shutdown_after_hours=50)


def test_watchdog_pins_every_group_to_zero(monkeypatch):
    # max_size matters as much as desired_capacity: predictive scaling and
    # target tracking would each happily raise capacity again from a floor of
    # zero if the ceiling were left in place.
    calls = []
    monkeypatch.setattr(
        watchdog.autoscaling,
        "update_auto_scaling_group",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setenv("REPLAY_START_EPOCH", "0")
    monkeypatch.setenv("SHUTDOWN_AFTER_HOURS", "50")
    monkeypatch.setenv("ASG_NAMES", "arm-a,arm-b")

    result = watchdog.handler({}, None)

    assert result["status"] == "shut_down"
    assert len(calls) == 2
    for call in calls:
        assert call["MinSize"] == 0
        assert call["MaxSize"] == 0
        assert call["DesiredCapacity"] == 0


def test_watchdog_reports_time_remaining_while_running(monkeypatch):
    monkeypatch.setenv(
        "REPLAY_START_EPOCH",
        str(int(dt.datetime.now(dt.timezone.utc).timestamp())),
    )
    monkeypatch.setenv("SHUTDOWN_AFTER_HOURS", "50")
    monkeypatch.setenv("ASG_NAMES", "arm-a")

    result = watchdog.handler({}, None)

    assert result["status"] == "running"
    assert 49 < result["hours_remaining"] <= 50


def test_one_failing_group_does_not_block_the_others(monkeypatch):
    def flaky(**kwargs):
        if kwargs["AutoScalingGroupName"] == "arm-a":
            raise RuntimeError("throttled")

    monkeypatch.setattr(watchdog.autoscaling, "update_auto_scaling_group", flaky)
    monkeypatch.setenv("REPLAY_START_EPOCH", "0")
    monkeypatch.setenv("SHUTDOWN_AFTER_HOURS", "50")
    monkeypatch.setenv("ASG_NAMES", "arm-a,arm-b")

    result = watchdog.handler({}, None)

    assert result["groups"] == ["arm-b"]
