"""Rolling-origin backtesting.

Walk forward through time. Train on everything up to an origin, forecast the
window that follows, advance the origin, retrain. Never let a training window
contain data from after the target it is being asked to predict.

A leaking backtest does not raise an error — it produces excellent numbers,
which is exactly what makes it dangerous. Two structural defences are used here
rather than relying on care:

1. Each fold's model is fitted on `series[:origin]`, so there is no later data
   in the object it was given.
2. Each fold's `predict` is handed only `series[:fold_end]`, so it cannot read
   past the window it is scoring even if its features tried to.

A fresh model is built per fold via the factory, so nothing carries over from
one origin to the next.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from src.forecast.losses import coverage, mean_absolute_error, pinball_loss

MINUTES_PER_DAY = 1440


@dataclass
class BacktestResult:
    """Out-of-sample forecasts and the actuals they are scored against.

    Attributes:
        forecasts: One column per quantile, indexed by decision time. The value
            at `t` is the forecast for `t + horizon`.
        actual: What actually arrived at `t + horizon`.
        model_name: Which forecaster produced this.
        horizon_minutes: How far ahead each forecast reached.
        fold_origins: Where each retrain happened, for plotting and audit.
    """

    forecasts: pd.DataFrame
    actual: pd.Series
    model_name: str
    horizon_minutes: int
    fold_origins: list[pd.Timestamp]

    def metrics(self) -> pd.DataFrame:
        """Per-quantile accuracy, calibration, and sample size.

        Coverage is the one to read first: a model can post a respectable
        pinball loss while being systematically miscalibrated, and a p90
        forecast that only covers 70% of actuals will under-provision every
        time it matters.
        """
        rows = []
        for quantile in self.forecasts.columns:
            predicted = self.forecasts[quantile]
            usable = predicted.notna() & self.actual.notna()
            actual_values = self.actual[usable].to_numpy()
            predicted_values = predicted[usable].to_numpy()

            rows.append(
                {
                    "model": self.model_name,
                    "quantile": quantile,
                    "mae": mean_absolute_error(actual_values, predicted_values),
                    "pinball_loss": pinball_loss(
                        actual_values, predicted_values, quantile
                    ),
                    "coverage": coverage(actual_values, predicted_values),
                    "n": int(usable.sum()),
                }
            )

        return pd.DataFrame(rows)


def rolling_origin_backtest(
    series: pd.Series,
    model_factory: Callable[[], object],
    horizon_minutes: int,
    quantiles: Sequence[float],
    initial_train_days: int = 28,
    retrain_days: int = 7,
) -> BacktestResult:
    """Walk forward through `series`, retraining periodically.

    Args:
        series: Arrivals per minute.
        model_factory: Builds a fresh, unfitted model. Called once per fold so
            that no state survives across origins.
        horizon_minutes: How far ahead to forecast.
        quantiles: Quantiles to evaluate.
        initial_train_days: History required before the first forecast.
        retrain_days: How often to refit. Shorter means the model tracks drift
            more closely at proportionally more compute.

    Returns:
        Out-of-sample forecasts covering everything after the first origin.
    """
    if retrain_days <= 0:
        raise ValueError(f"retrain_days must be positive, got {retrain_days}")
    if horizon_minutes <= 0:
        raise ValueError(f"horizon_minutes must be positive, got {horizon_minutes}")

    start = series.index[0]
    first_origin = start + pd.Timedelta(days=initial_train_days)
    if first_origin >= series.index[-1]:
        raise ValueError(
            f"initial_train_days={initial_train_days} leaves no data to evaluate; "
            f"the series covers {(series.index[-1] - start).days} days"
        )

    forecasts = pd.DataFrame(
        index=series.index[series.index >= first_origin], columns=list(quantiles), dtype=float
    )
    fold_origins: list[pd.Timestamp] = []
    model_name = "unknown"

    origin = first_origin
    while origin < series.index[-1]:
        fold_end = min(origin + pd.Timedelta(days=retrain_days), series.index[-1] + pd.Timedelta(minutes=1))

        # Train strictly before the origin.
        train = series[series.index < origin]
        model = model_factory()
        model_name = getattr(model, "name", "unknown")
        model.fit(train)
        fold_origins.append(origin)

        # Predict with only the history this fold is allowed to see. The slice
        # includes pre-origin history because the features need lags, but stops
        # at the fold boundary.
        visible = series[series.index < fold_end]
        in_fold = (visible.index >= origin) & (visible.index < fold_end)

        for quantile in quantiles:
            predicted = model.predict(visible, quantile)
            forecasts.loc[visible.index[in_fold], quantile] = predicted[in_fold].to_numpy()

        origin = fold_end

    actual = series.shift(-horizon_minutes).reindex(forecasts.index)

    return BacktestResult(
        forecasts=forecasts,
        actual=actual,
        model_name=model_name,
        horizon_minutes=horizon_minutes,
        fold_origins=fold_origins,
    )
