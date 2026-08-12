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
# reuse. They need a region even when no call is made.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
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
