"""Tests for the predictive scaling policy.

The policy consumes a forecast that was produced under rolling-origin
discipline: the value stored at minute `m` was computed from data at or before
minute `m`, and predicts minute `m + horizon`. Reading it at simulation minute
`m` is therefore legitimate, and the tests below pin that indexing exactly —
an off-by-one here would hand the policy real future data and quietly
invalidate the whole comparison.
"""

import numpy as np
import pandas as pd
import pytest

from src.policies.arrival_rate import ArrivalRateTargetPolicy
from src.policies.forecast_policy import ForecastPolicy, forecast_array
from src.sim.observation import Observation

SERVICE_SECONDS = 30.0


def observation(t: float = 600.0, **overrides) -> Observation:
    defaults = dict(
        t=t,
        queue_depth=0,
        oldest_message_age=0.0,
        in_service_instances=40,
        pending_instances=0,
        recent_arrivals=(60,) * 60,
        completions_since_last_tick=60,
    )
    return Observation(**{**defaults, **overrides})


# --- forecast to capacity ---------------------------------------------------


def test_the_forecast_is_converted_to_capacity_by_littles_law():
    # A forecast of 120 arrivals in a minute is 2/s; at 30s each that needs 60
    # instances.
    forecasts = np.full(100, 120.0)
    policy = ForecastPolicy(forecasts, service_seconds=SERVICE_SECONDS)

    assert policy.decide(600.0, observation()) == 60


def test_the_safety_margin_multiplies_the_requirement():
    forecasts = np.full(100, 120.0)
    policy = ForecastPolicy(
        forecasts, service_seconds=SERVICE_SECONDS, safety_margin=1.25
    )

    assert policy.decide(600.0, observation()) == 75


def test_headroom_is_applied_through_the_utilisation_target():
    forecasts = np.full(100, 120.0)
    policy = ForecastPolicy(
        forecasts, service_seconds=SERVICE_SECONDS, target_utilization=0.8
    )

    assert policy.decide(600.0, observation()) == 75


def test_a_zero_forecast_asks_for_the_floor_only():
    policy = ForecastPolicy(
        np.zeros(100), service_seconds=SERVICE_SECONDS, min_instances=5
    )

    assert policy.decide(600.0, observation()) == 5


# --- indexing ---------------------------------------------------------------


def test_the_policy_reads_the_forecast_for_the_current_minute():
    # Each entry is its own minute index scaled up, so a misread is visible.
    forecasts = np.arange(100, dtype=float) * 60.0
    policy = ForecastPolicy(forecasts, service_seconds=SERVICE_SECONDS)

    # At t=600s we are in minute 10, whose entry is 600 arrivals = 10/s,
    # needing 300 instances.
    assert policy.decide(600.0, observation(t=600.0)) == 300


def test_reading_past_the_end_of_the_forecast_falls_back_to_the_floor():
    policy = ForecastPolicy(
        np.full(5, 120.0), service_seconds=SERVICE_SECONDS, min_instances=7
    )

    assert policy.decide(60_000.0, observation(t=60_000.0)) == 7


def test_a_missing_forecast_falls_back_rather_than_crashing():
    # Early minutes have no forecast because the model needs history.
    forecasts = np.full(100, np.nan)
    policy = ForecastPolicy(
        forecasts, service_seconds=SERVICE_SECONDS, min_instances=12
    )

    assert policy.decide(600.0, observation()) == 12


# --- composition with a reactive floor --------------------------------------


def test_the_composed_policy_takes_the_larger_of_forecast_and_reactive():
    # Predictive sets the minimum; target tracking handles whatever the
    # forecast missed. A burst that was not predicted still gets capacity.
    forecasts = np.full(100, 60.0)  # 1/s -> 30 instances
    reactive = ArrivalRateTargetPolicy(
        service_seconds=SERVICE_SECONDS, target_utilization=1.0, window_minutes=5
    )
    policy = ForecastPolicy(
        forecasts, service_seconds=SERVICE_SECONDS, reactive_floor=reactive
    )

    # Observed arrivals are double the forecast, so the reactive term wins.
    busy = observation(recent_arrivals=(120,) * 60)
    assert policy.decide(600.0, busy) == 60


def test_the_forecast_wins_when_it_is_ahead_of_observed_load():
    # The whole point: demand that has not arrived yet is already provisioned.
    forecasts = np.full(100, 240.0)  # 4/s -> 120 instances
    reactive = ArrivalRateTargetPolicy(
        service_seconds=SERVICE_SECONDS, target_utilization=1.0, window_minutes=5
    )
    policy = ForecastPolicy(
        forecasts, service_seconds=SERVICE_SECONDS, reactive_floor=reactive
    )

    quiet = observation(recent_arrivals=(60,) * 60)
    assert policy.decide(600.0, quiet) == 120


def test_the_pure_policy_ignores_the_queue_entirely():
    # Worth pinning: the pure variant is deliberately blind to the backlog, so
    # the experiment can separate what prediction contributes from what
    # reactive correction contributes.
    forecasts = np.full(100, 60.0)
    policy = ForecastPolicy(forecasts, service_seconds=SERVICE_SECONDS)

    calm = policy.decide(600.0, observation(queue_depth=0))
    swamped = policy.decide(600.0, observation(queue_depth=100_000))

    assert calm == swamped == 30


# --- interface --------------------------------------------------------------


def test_the_decision_is_a_plain_integer():
    policy = ForecastPolicy(np.full(100, 120.0), service_seconds=SERVICE_SECONDS)

    assert type(policy.decide(600.0, observation())) is int


def test_the_policy_reports_a_name():
    pure = ForecastPolicy(np.full(10, 1.0), service_seconds=SERVICE_SECONDS)
    composed = ForecastPolicy(
        np.full(10, 1.0),
        service_seconds=SERVICE_SECONDS,
        reactive_floor=ArrivalRateTargetPolicy(SERVICE_SECONDS, 1.0, 5),
    )

    assert pure.name == "forecast"
    assert composed.name == "forecast_composed"


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        ForecastPolicy(np.full(10, 1.0), service_seconds=0.0)
    with pytest.raises(ValueError):
        ForecastPolicy(np.full(10, 1.0), service_seconds=30.0, safety_margin=0.0)


# --- aligning a forecast series to the trace --------------------------------


def test_forecast_array_aligns_to_the_trace_minutes():
    trace_index = pd.date_range("2024-01-01", periods=10, freq="1min")
    forecasts = pd.Series(
        [7.0, 8.0], index=pd.to_datetime(["2024-01-01 00:03", "2024-01-01 00:05"])
    )

    aligned = forecast_array(forecasts, trace_index)

    assert len(aligned) == 10
    assert aligned[3] == 7.0
    assert aligned[5] == 8.0
    assert np.isnan(aligned[0])


def test_forecast_array_ignores_timestamps_outside_the_trace():
    trace_index = pd.date_range("2024-01-01", periods=5, freq="1min")
    forecasts = pd.Series(
        [1.0, 2.0], index=pd.to_datetime(["2024-01-01 00:02", "2024-06-01 00:00"])
    )

    aligned = forecast_array(forecasts, trace_index)

    assert len(aligned) == 5
    assert aligned[2] == 1.0
