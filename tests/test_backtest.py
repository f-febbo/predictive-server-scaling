"""Tests for rolling-origin backtesting.

This is where lookahead leakage would be easiest to introduce and hardest to
notice, because a leaking backtest produces beautiful numbers rather than an
error. So the discipline is asserted directly: a spy model records exactly what
it was trained on and when, and the tests check those boundaries arithmetically
rather than trusting the loop.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecast.backtest import rolling_origin_backtest

HORIZON = 15
DAY = 1440


def series(days: int, start: str = "2024-01-01") -> pd.Series:
    minutes = days * DAY
    index = pd.date_range(start, periods=minutes, freq="1min", name="ts")
    values = 100 + 20 * np.sin(2 * np.pi * np.arange(minutes) / DAY)
    return pd.Series(values, index=index, name="arrivals")


class SpyModel:
    """Records what it was fitted on; forecasts a constant."""

    name = "spy"
    fit_calls: list[pd.Series] = []
    predict_calls: list[pd.Series] = []

    def __init__(self, constant: float = 42.0):
        self.constant = constant

    def fit(self, train: pd.Series) -> None:
        SpyModel.fit_calls.append(train)

    def predict(self, data: pd.Series, quantile: float) -> pd.Series:
        SpyModel.predict_calls.append(data)
        return pd.Series(self.constant * quantile, index=data.index)


def spy_factory():
    SpyModel.fit_calls = []
    SpyModel.predict_calls = []
    return lambda: SpyModel()


# --- fold structure ---------------------------------------------------------


def test_one_retrain_per_fold():
    # 28 days of data, 14-day initial window, retrained weekly: two folds.
    factory = spy_factory()

    rolling_origin_backtest(
        series(28),
        factory,
        horizon_minutes=HORIZON,
        quantiles=(0.5,),
        initial_train_days=14,
        retrain_days=7,
    )

    assert len(SpyModel.fit_calls) == 2


def test_each_fold_trains_only_on_data_before_its_origin():
    # The core guarantee. Fold k is trained on everything up to its origin and
    # nothing after it.
    factory = spy_factory()
    data = series(28)

    rolling_origin_backtest(
        data,
        factory,
        horizon_minutes=HORIZON,
        quantiles=(0.5,),
        initial_train_days=14,
        retrain_days=7,
    )

    origins = [data.index[0] + pd.Timedelta(days=d) for d in (14, 21)]
    for train_series, origin in zip(SpyModel.fit_calls, origins):
        assert train_series.index[-1] < origin


def test_training_windows_grow_as_the_origin_advances():
    # Each retrain sees everything up to that point, not a sliding window.
    factory = spy_factory()

    rolling_origin_backtest(
        series(28),
        factory,
        horizon_minutes=HORIZON,
        quantiles=(0.5,),
        initial_train_days=14,
        retrain_days=7,
    )

    assert len(SpyModel.fit_calls[1]) > len(SpyModel.fit_calls[0])


def test_a_fold_is_never_shown_data_beyond_its_own_test_window():
    # Predict is handed only the history it needs, so it is structurally
    # incapable of reading past the end of the fold it is scoring.
    factory = spy_factory()
    data = series(28)

    rolling_origin_backtest(
        data,
        factory,
        horizon_minutes=HORIZON,
        quantiles=(0.5,),
        initial_train_days=14,
        retrain_days=7,
    )

    fold_ends = [data.index[0] + pd.Timedelta(days=d) for d in (21, 28)]
    for predict_series, fold_end in zip(SpyModel.predict_calls, fold_ends):
        assert predict_series.index[-1] < fold_end


def test_forecasts_start_at_the_first_origin():
    data = series(28)

    result = rolling_origin_backtest(
        data,
        spy_factory(),
        horizon_minutes=HORIZON,
        quantiles=(0.5,),
        initial_train_days=14,
        retrain_days=7,
    )

    assert result.forecasts.index[0] == data.index[0] + pd.Timedelta(days=14)


def test_folds_tile_the_evaluation_period_without_gaps_or_overlap():
    data = series(28)

    result = rolling_origin_backtest(
        data,
        spy_factory(),
        horizon_minutes=HORIZON,
        quantiles=(0.5,),
        initial_train_days=14,
        retrain_days=7,
    )

    expected = data.index[data.index >= data.index[0] + pd.Timedelta(days=14)]
    assert result.forecasts.index.equals(expected)
    assert result.forecasts.index.is_unique


def test_a_partial_final_fold_is_still_evaluated():
    # 25 days with a 14-day window and weekly retraining leaves a 4-day tail;
    # discarding it would silently shrink the evaluation set.
    data = series(25)

    result = rolling_origin_backtest(
        data,
        spy_factory(),
        horizon_minutes=HORIZON,
        quantiles=(0.5,),
        initial_train_days=14,
        retrain_days=7,
    )

    assert result.forecasts.index[-1] == data.index[-1]


# --- outputs ----------------------------------------------------------------


def test_one_forecast_column_per_quantile():
    result = rolling_origin_backtest(
        series(28),
        spy_factory(),
        horizon_minutes=HORIZON,
        quantiles=(0.5, 0.9),
        initial_train_days=14,
        retrain_days=7,
    )

    assert list(result.forecasts.columns) == [0.5, 0.9]


def test_the_actual_series_is_the_value_one_horizon_ahead():
    data = series(28)

    result = rolling_origin_backtest(
        data,
        spy_factory(),
        horizon_minutes=HORIZON,
        quantiles=(0.5,),
        initial_train_days=14,
        retrain_days=7,
    )

    decision_time = result.forecasts.index[100]
    expected = data.loc[decision_time + pd.Timedelta(minutes=HORIZON)]
    assert result.actual.loc[decision_time] == pytest.approx(expected)


def test_metrics_are_reported_per_quantile():
    result = rolling_origin_backtest(
        series(28),
        spy_factory(),
        horizon_minutes=HORIZON,
        quantiles=(0.5, 0.9),
        initial_train_days=14,
        retrain_days=7,
    )

    metrics = result.metrics()

    assert set(metrics["quantile"]) == {0.5, 0.9}
    for column in ("mae", "pinball_loss", "coverage", "n"):
        assert column in metrics.columns


def test_coverage_rises_with_the_quantile_for_a_sane_model():
    # The spy forecasts `42 * q`, so a higher quantile really is a higher
    # forecast and must cover more.
    result = rolling_origin_backtest(
        series(28),
        spy_factory(),
        horizon_minutes=HORIZON,
        quantiles=(0.1, 0.9),
        initial_train_days=14,
        retrain_days=7,
    )

    metrics = result.metrics().set_index("quantile")
    assert metrics.loc[0.9, "coverage"] >= metrics.loc[0.1, "coverage"]


def test_the_model_name_is_recorded():
    result = rolling_origin_backtest(
        series(28),
        spy_factory(),
        horizon_minutes=HORIZON,
        quantiles=(0.5,),
        initial_train_days=14,
        retrain_days=7,
    )

    assert result.model_name == "spy"


# --- validation -------------------------------------------------------------


def test_too_little_data_for_even_one_fold_is_rejected():
    with pytest.raises(ValueError, match="initial_train_days"):
        rolling_origin_backtest(
            series(10),
            spy_factory(),
            horizon_minutes=HORIZON,
            quantiles=(0.5,),
            initial_train_days=14,
            retrain_days=7,
        )


def test_a_non_positive_retrain_interval_is_rejected():
    with pytest.raises(ValueError):
        rolling_origin_backtest(
            series(28),
            spy_factory(),
            horizon_minutes=HORIZON,
            quantiles=(0.5,),
            initial_train_days=14,
            retrain_days=0,
        )
