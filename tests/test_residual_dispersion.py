"""Tests for seasonally-adjusted burstiness.

The plain index of dispersion counts the predictable daily and weekly swing as
"burstiness", which overstates how hard the traffic actually is to anticipate.
These statistics strip the seasonal shape out first, so what remains is the
part a seasonal forecaster genuinely cannot see coming.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.stats import residual_index_of_dispersion, seasonal_expectation


def hourly_shape_series(weeks: int, shape: list[float]) -> pd.Series:
    """A 1-minute series whose rate depends only on the hour of the week."""
    periods = weeks * 7 * 24 * 60
    index = pd.date_range("2024-01-01", periods=periods, freq="1min", name="ts")
    hour_of_week = index.dayofweek * 24 + index.hour
    return pd.Series([shape[h] for h in hour_of_week], index=index, name="arrivals")


def test_seasonal_expectation_recovers_a_purely_seasonal_series():
    # Every week is identical, so the fitted seasonal shape reproduces the
    # series exactly and nothing is left over.
    shape = [float(h % 24) * 3 for h in range(168)]
    series = hourly_shape_series(weeks=3, shape=shape)

    expectation = seasonal_expectation(series)

    np.testing.assert_allclose(expectation.to_numpy(), series.to_numpy())


def test_seasonal_expectation_is_aligned_to_the_input_index():
    series = hourly_shape_series(weeks=2, shape=[10.0] * 168)

    expectation = seasonal_expectation(series)

    assert expectation.index.equals(series.index)


def test_seasonal_expectation_preserves_the_overall_mean():
    rng = np.random.default_rng(0)
    series = hourly_shape_series(weeks=2, shape=list(rng.uniform(5, 100, size=168)))

    assert seasonal_expectation(series).mean() == pytest.approx(series.mean())


def test_residual_dispersion_is_zero_for_a_perfectly_seasonal_series():
    # No surprise at all: a scaler that knew the weekly shape would never be
    # caught out.
    shape = [float(h % 24) * 3 + 1 for h in range(168)]
    series = hourly_shape_series(weeks=3, shape=shape)

    assert residual_index_of_dispersion(series) == pytest.approx(0.0, abs=1e-9)


def test_residual_dispersion_is_about_one_for_poisson_noise_on_a_seasonal_mean():
    # Poisson fluctuation around a known time-varying rate is the reference
    # case: unpredictable, but only as unpredictable as pure randomness.
    rng = np.random.default_rng(1)
    shape = [50.0 + 40.0 * np.sin(h / 168 * 2 * np.pi) for h in range(168)]
    seasonal = hourly_shape_series(weeks=4, shape=shape)
    noisy = pd.Series(
        rng.poisson(seasonal.to_numpy()), index=seasonal.index, name="arrivals"
    )

    assert residual_index_of_dispersion(noisy) == pytest.approx(1.0, abs=0.15)


def test_residual_dispersion_is_below_the_raw_index_when_seasonality_is_strong():
    # This is the whole point of the statistic: the raw index credits the
    # predictable daily swing as burstiness, the residual one does not.
    from src.data.stats import index_of_dispersion

    rng = np.random.default_rng(2)
    shape = [50.0 + 45.0 * np.sin(h / 24 * 2 * np.pi) for h in range(168)]
    seasonal = hourly_shape_series(weeks=4, shape=shape)
    noisy = pd.Series(
        rng.poisson(seasonal.to_numpy()), index=seasonal.index, name="arrivals"
    )

    assert residual_index_of_dispersion(noisy) < index_of_dispersion(noisy)


def test_residual_dispersion_detects_bursts_the_seasonal_shape_cannot_explain():
    # Same seasonal mean, but with sporadic spikes that no weekly profile
    # anticipates. This must score well above the Poisson reference of 1.
    rng = np.random.default_rng(3)
    seasonal = hourly_shape_series(weeks=4, shape=[50.0] * 168)
    values = rng.poisson(seasonal.to_numpy()).astype(float)
    spike_positions = rng.choice(len(values), size=len(values) // 100, replace=False)
    values[spike_positions] += 400.0
    bursty = pd.Series(values, index=seasonal.index, name="arrivals")

    assert residual_index_of_dispersion(bursty) > 5.0
