"""Tests for the arrival-series summary statistics.

These numbers are the evidence for the project's central claim that the traffic
is genuinely bursty rather than smooth, so each one is pinned against an input
whose answer can be worked out by hand.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.data.stats import (
    autocorrelation_at_lags,
    index_of_dispersion,
    summarize_arrivals,
)


def series_from(values, freq: str = "1min") -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq=freq, name="ts")
    return pd.Series(values, index=index, name="arrivals")


# --- index of dispersion ----------------------------------------------------


def test_index_of_dispersion_is_zero_for_a_constant_series():
    # No variance at all: perfectly smooth traffic.
    assert index_of_dispersion(series_from([7] * 10)) == 0.0


def test_index_of_dispersion_matches_a_hand_computed_value():
    # [1, 2, 3]: mean 2, sample variance 1, so the ratio is 0.5.
    assert index_of_dispersion(series_from([1, 2, 3])) == pytest.approx(0.5)


def test_index_of_dispersion_is_about_one_for_a_poisson_series():
    # The Fano factor of a Poisson process is 1 by construction. This is the
    # reference point the real trace gets compared against: materially above 1
    # means the arrivals are burstier than Poisson.
    rng = np.random.default_rng(0)
    poisson = series_from(rng.poisson(lam=50, size=20_000))

    assert index_of_dispersion(poisson) == pytest.approx(1.0, abs=0.1)


def test_index_of_dispersion_exceeds_one_for_bursty_traffic():
    # Alternating quiet and busy minutes: same mean, far more spread.
    bursty = series_from([0, 100] * 100)

    assert index_of_dispersion(bursty) > 1.0


def test_index_of_dispersion_is_undefined_for_an_all_zero_series():
    # Mean of zero would divide by zero; report NaN rather than raise so a
    # summary over a dead window still returns.
    assert np.isnan(index_of_dispersion(series_from([0, 0, 0])))


# --- autocorrelation --------------------------------------------------------


def test_autocorrelation_of_a_linear_ramp_is_one_at_lag_one():
    assert autocorrelation_at_lags(series_from([1, 2, 3, 4]), {"1min": 1})[
        "1min"
    ] == pytest.approx(1.0)


def test_autocorrelation_of_an_alternating_series_flips_sign_with_lag():
    alternating = series_from([1, -1] * 50)

    result = autocorrelation_at_lags(alternating, {"lag1": 1, "lag2": 2})

    assert result["lag1"] == pytest.approx(-1.0)
    assert result["lag2"] == pytest.approx(1.0)


def test_autocorrelation_detects_daily_seasonality():
    # One week of hourly data with a repeating daily shape: correlation at a
    # 24-hour lag should be near perfect.
    daily_shape = list(range(24))
    weekly = series_from(daily_shape * 7, freq="1h")

    result = autocorrelation_at_lags(weekly, {"1d": 24})

    assert result["1d"] == pytest.approx(1.0)


def test_autocorrelation_of_a_constant_series_is_nan_without_warning():
    # A flat series has zero variance, so the correlation denominator is zero.
    # Report NaN deliberately rather than letting numpy emit a divide warning
    # and hand back a nan that looks accidental.
    constant = series_from([5] * 100)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = autocorrelation_at_lags(constant, {"1min": 1})

    assert np.isnan(result["1min"])


def test_autocorrelation_is_nan_when_the_lag_exceeds_the_series_length():
    # Asking for a one-week lag on one day of data must not silently return a
    # number computed from a handful of overlapping points.
    result = autocorrelation_at_lags(series_from([1, 2, 3]), {"1w": 10_080})

    assert np.isnan(result["1w"])


# --- summary ----------------------------------------------------------------


def test_summary_reports_hand_checkable_location_statistics():
    # 0..100 inclusive: mean 50, median 50, and the 99th percentile lands
    # exactly on 99 because 0.99 * 100 is a whole index.
    summary = summarize_arrivals(series_from(list(range(101))))

    assert summary["mean"] == pytest.approx(50.0)
    assert summary["median"] == pytest.approx(50.0)
    assert summary["p99"] == pytest.approx(99.0)
    assert summary["max"] == 100
    assert summary["min"] == 0


def test_summary_reports_the_observation_count_and_time_span():
    summary = summarize_arrivals(series_from([1] * 60))

    assert summary["n_observations"] == 60
    assert summary["start"] == pd.Timestamp("2024-01-01 00:00")
    assert summary["end"] == pd.Timestamp("2024-01-01 00:59")


def test_summary_includes_dispersion_and_the_four_required_lags():
    summary = summarize_arrivals(series_from(list(range(101))))

    assert "index_of_dispersion" in summary
    for lag in ("1min", "1hour", "1day", "1week"):
        assert lag in summary["autocorrelation"]


def test_summary_totals_conserve_the_input():
    values = [3, 0, 7, 2]

    summary = summarize_arrivals(series_from(values))

    assert summary["total_arrivals"] == sum(values)
