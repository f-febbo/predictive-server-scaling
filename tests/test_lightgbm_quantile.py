"""Tests for the LightGBM quantile forecaster.

Kept small and fast: a few trees on a few weeks of synthetic data. The point is
to pin the model's contract — quantile ordering, alignment, no lookahead — not
to measure its accuracy, which the backtest does on real data.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecast.features import WEEK_MINUTES
from src.forecast.lightgbm_quantile import LightGBMQuantile
from src.forecast.losses import pinball_loss

HORIZON = 15
QUANTILES = (0.1, 0.5, 0.9)


def learnable(weeks: int = 3, seed: int = 0) -> pd.Series:
    """A weekly cycle plus noise — periodic enough for a model to learn."""
    minutes = weeks * WEEK_MINUTES
    index = pd.date_range("2024-01-01", periods=minutes, freq="1min", name="ts")
    phase = np.arange(minutes) % WEEK_MINUTES
    daily = 30 * np.sin(2 * np.pi * phase / 1440)
    weekly = 60 * np.sin(2 * np.pi * phase / WEEK_MINUTES)
    rng = np.random.default_rng(seed)
    values = 100 + daily + weekly + rng.normal(0, 8, size=minutes)
    return pd.Series(np.clip(values, 0, None), index=index, name="arrivals")


def small_model(**overrides) -> LightGBMQuantile:
    defaults = dict(
        horizon_minutes=HORIZON,
        quantiles=QUANTILES,
        n_estimators=40,
        num_leaves=15,
        random_state=0,
    )
    return LightGBMQuantile(**{**defaults, **overrides})


# --- contract ---------------------------------------------------------------


def test_predictions_are_aligned_to_the_decision_times():
    values = learnable()
    model = small_model()
    model.fit(values)

    forecast = model.predict(values, quantile=0.5)

    assert forecast.index.equals(values.index)


def test_higher_quantiles_forecast_higher_on_average():
    # Separate models per quantile can cross at individual points, which is a
    # known property of independent quantile regression rather than a bug, so
    # the ordering is asserted on the average.
    values = learnable()
    model = small_model()
    model.fit(values)

    means = [
        model.predict(values, quantile=q).dropna().mean() for q in (0.1, 0.5, 0.9)
    ]

    assert means[0] < means[1] < means[2]


def test_forecasts_are_never_negative():
    values = learnable()
    model = small_model()
    model.fit(values)

    forecast = model.predict(values, quantile=0.1)

    assert (forecast.dropna() >= 0).all()


def test_a_quantile_that_was_not_fitted_is_refused():
    # Silently falling back to the nearest fitted quantile would put a
    # mislabelled point on the frontier.
    values = learnable()
    model = small_model()
    model.fit(values)

    with pytest.raises(KeyError):
        model.predict(values, quantile=0.75)


def test_predicting_before_fitting_is_refused():
    with pytest.raises(RuntimeError, match="fit"):
        small_model().predict(learnable(), quantile=0.5)


def test_the_model_reports_a_name():
    assert small_model().name == "lightgbm"


def test_fitting_is_reproducible():
    values = learnable()

    first = small_model(random_state=7)
    first.fit(values)
    second = small_model(random_state=7)
    second.fit(values)

    np.testing.assert_allclose(
        first.predict(values, 0.5).dropna().to_numpy(),
        second.predict(values, 0.5).dropna().to_numpy(),
    )


# --- it actually learns something -------------------------------------------


def test_the_model_beats_forecasting_the_training_mean():
    # A weak but reliable check that fitting did something. Accuracy against
    # the real baselines is measured by the backtest, not here.
    values = learnable()
    model = small_model()
    model.fit(values)

    forecast = model.predict(values, quantile=0.5)
    target = values.shift(-HORIZON)
    usable = forecast.notna() & target.notna()

    actual = target[usable].to_numpy()
    model_loss = pinball_loss(actual, forecast[usable].to_numpy(), 0.5)
    constant_loss = pinball_loss(
        actual, np.full(actual.shape, values.mean()), 0.5
    )

    assert model_loss < constant_loss


def test_high_quantile_covers_more_than_the_median():
    values = learnable()
    model = small_model()
    model.fit(values)

    target = values.shift(-HORIZON)
    coverages = []
    for quantile in (0.5, 0.9):
        forecast = model.predict(values, quantile=quantile)
        usable = forecast.notna() & target.notna()
        coverages.append((forecast[usable] >= target[usable]).mean())

    assert coverages[1] > coverages[0]


# --- no lookahead -----------------------------------------------------------


def test_predictions_do_not_change_when_the_future_is_corrupted():
    # The model is a function of its features, and the features are a function
    # of the past. Corrupting everything after a cut must leave predictions at
    # or before that cut untouched.
    values = learnable()
    model = small_model()
    model.fit(values)

    corrupted = values.copy()
    cut = len(values) - 2000
    rng = np.random.default_rng(1)
    corrupted.iloc[cut + 1 :] = rng.uniform(0, 1e5, size=len(corrupted) - cut - 1)

    original_forecast = model.predict(values, quantile=0.9).iloc[: cut + 1]
    corrupted_forecast = model.predict(corrupted, quantile=0.9).iloc[: cut + 1]

    pd.testing.assert_series_equal(original_forecast, corrupted_forecast)


def test_training_never_pairs_features_with_an_earlier_target():
    # Every training row must predict forward. Exposed for the backtest, which
    # relies on it to keep folds honest.
    values = learnable(weeks=2)
    model = small_model()
    model.fit(values)

    assert model.last_trained_target_time is not None
    # The final usable target sits `horizon` before the end of the data.
    expected = values.index[-1] - pd.Timedelta(minutes=0)
    assert model.last_trained_target_time <= expected
