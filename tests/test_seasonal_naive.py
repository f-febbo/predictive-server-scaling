"""Tests for the seasonal-naive forecasters.

These are the baselines the LightGBM model has to beat. On strongly periodic
traffic they are far better than people expect, and if the gradient-boosted
model cannot improve on them that is a finding worth reporting rather than a
bug worth hiding.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecast.features import WEEK_MINUTES
from src.forecast.seasonal_naive import SeasonalNaive, SeasonalNaiveAdjusted

HORIZON = 15


def periodic(weeks: int, amplitude: float = 50.0, level: float = 100.0) -> pd.Series:
    """A perfectly repeating weekly series — no noise, no drift."""
    minutes = weeks * WEEK_MINUTES
    index = pd.date_range("2024-01-01", periods=minutes, freq="1min", name="ts")
    phase = np.arange(minutes) % WEEK_MINUTES
    values = level + amplitude * np.sin(2 * np.pi * phase / WEEK_MINUTES)
    return pd.Series(values, index=index, name="arrivals")


def noisy(weeks: int, seed: int = 0) -> pd.Series:
    base = periodic(weeks)
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, 10, size=len(base))


# --- point forecast ---------------------------------------------------------


def test_the_point_forecast_is_the_value_one_week_before_the_target():
    # Predicting t+15 from decision time t means reading the value at
    # t + 15 - one week, which is comfortably in the past.
    values = periodic(3)
    model = SeasonalNaive(horizon_minutes=HORIZON)
    model.fit(values)

    forecast = model.predict(values, quantile=0.5)
    position = 2 * WEEK_MINUTES + 500

    expected = values.iloc[position + HORIZON - WEEK_MINUTES]
    assert forecast.iloc[position] == pytest.approx(expected, abs=1e-9)


def test_forecast_is_exact_on_a_perfectly_periodic_series():
    values = periodic(3)
    model = SeasonalNaive(horizon_minutes=HORIZON)
    model.fit(values)

    forecast = model.predict(values, quantile=0.5)
    target = values.shift(-HORIZON)

    usable = forecast.notna() & target.notna()
    np.testing.assert_allclose(
        forecast[usable].to_numpy(), target[usable].to_numpy(), atol=1e-6
    )


def test_the_first_week_has_no_forecast():
    # There is nothing a week before the first week.
    values = periodic(2)
    model = SeasonalNaive(horizon_minutes=HORIZON)
    model.fit(values)

    forecast = model.predict(values, quantile=0.5)

    assert forecast.iloc[: WEEK_MINUTES - HORIZON].isna().all()
    assert forecast.iloc[WEEK_MINUTES:].notna().all()


# --- quantiles --------------------------------------------------------------


def test_higher_quantiles_forecast_higher():
    # Sweeping the quantile is what traces the cost/SLO frontier, so the
    # ordering has to be strict.
    values = noisy(4)
    model = SeasonalNaive(horizon_minutes=HORIZON)
    model.fit(values)

    low = model.predict(values, quantile=0.1)
    mid = model.predict(values, quantile=0.5)
    high = model.predict(values, quantile=0.9)

    usable = low.notna()
    assert (low[usable] <= mid[usable]).all()
    assert (mid[usable] <= high[usable]).all()
    assert (low[usable] < high[usable]).any()


def test_quantiles_collapse_to_the_point_forecast_without_noise():
    # With zero residual spread there is no uncertainty to price in.
    values = periodic(3)
    model = SeasonalNaive(horizon_minutes=HORIZON)
    model.fit(values)

    low = model.predict(values, quantile=0.05)
    high = model.predict(values, quantile=0.95)

    usable = low.notna()
    np.testing.assert_allclose(
        low[usable].to_numpy(), high[usable].to_numpy(), atol=1e-6
    )


def test_a_high_quantile_covers_most_of_the_actuals():
    values = noisy(5, seed=3)
    model = SeasonalNaive(horizon_minutes=HORIZON)
    model.fit(values)

    forecast = model.predict(values, quantile=0.9)
    target = values.shift(-HORIZON)

    usable = forecast.notna() & target.notna()
    covered = (forecast[usable] >= target[usable]).mean()
    assert 0.82 < covered < 0.97


def test_forecasts_are_never_negative():
    # Negative capacity is meaningless, and a low quantile on a quiet period
    # would otherwise produce one.
    values = noisy(3, seed=7).clip(lower=0)
    model = SeasonalNaive(horizon_minutes=HORIZON)
    model.fit(values)

    forecast = model.predict(values, quantile=0.01)

    assert (forecast.dropna() >= 0).all()


def test_predicting_before_fitting_is_refused():
    model = SeasonalNaive(horizon_minutes=HORIZON)

    with pytest.raises(RuntimeError, match="fit"):
        model.predict(periodic(2), quantile=0.5)


# --- recent-level adjustment ------------------------------------------------


def test_the_adjusted_model_scales_last_week_by_the_recent_level():
    # Last week's shape, re-levelled by how the last hour compares with the
    # same hour a week ago. If today is running 50% hot, tomorrow's forecast
    # should be too.
    base = periodic(3)
    lifted = base.copy()
    # Raise the most recent day by half.
    lifted.iloc[-1440:] = lifted.iloc[-1440:] * 1.5

    plain = SeasonalNaive(horizon_minutes=HORIZON)
    plain.fit(base)
    adjusted = SeasonalNaiveAdjusted(horizon_minutes=HORIZON)
    adjusted.fit(base)

    position = len(base) - 100
    plain_forecast = plain.predict(lifted, quantile=0.5).iloc[position]
    adjusted_forecast = adjusted.predict(lifted, quantile=0.5).iloc[position]

    assert adjusted_forecast > plain_forecast * 1.2


def test_the_adjustment_is_neutral_when_the_level_is_unchanged():
    values = periodic(3)
    plain = SeasonalNaive(horizon_minutes=HORIZON)
    plain.fit(values)
    adjusted = SeasonalNaiveAdjusted(horizon_minutes=HORIZON)
    adjusted.fit(values)

    position = len(values) - 100
    assert adjusted.predict(values, quantile=0.5).iloc[position] == pytest.approx(
        plain.predict(values, quantile=0.5).iloc[position], rel=0.02
    )


def test_the_adjustment_ratio_is_clipped():
    # An unclipped ratio explodes when last week's same hour was near zero,
    # turning one quiet minute into an enormous capacity request.
    values = periodic(3).copy()
    values.iloc[-70:-10] = 0.001  # a near-dead recent hour

    model = SeasonalNaiveAdjusted(horizon_minutes=HORIZON, ratio_bounds=(0.5, 2.0))
    model.fit(periodic(3))

    forecast = model.predict(values, quantile=0.5)
    reference = SeasonalNaive(horizon_minutes=HORIZON)
    reference.fit(periodic(3))
    plain = reference.predict(values, quantile=0.5)

    position = len(values) - 5
    assert forecast.iloc[position] >= plain.iloc[position] * 0.5 - 1e-6


def test_both_models_report_a_name():
    assert SeasonalNaive(horizon_minutes=HORIZON).name == "seasonal_naive"
    assert (
        SeasonalNaiveAdjusted(horizon_minutes=HORIZON).name == "seasonal_naive_adjusted"
    )
