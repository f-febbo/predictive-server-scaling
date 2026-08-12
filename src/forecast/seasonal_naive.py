"""Seasonal-naive forecasters: last week, same time.

On traffic with a strong weekly cycle these are much harder to beat than they
look, and they cost nothing to run. They are here as the bar the LightGBM model
has to clear — if it cannot, that goes in the README, because a gradient
boosting model that loses to a one-line baseline is a more useful thing to know
than a model that wins by an unexamined margin.

Quantiles come from the empirical distribution of past errors rather than from
a distributional assumption. Arrival counts are heteroscedastic — a busy minute
has a much larger absolute error than a quiet one — so residuals are
standardised by the square root of the prediction before their quantiles are
taken. That is the Poisson-like scaling the trace roughly follows (its
seasonally-adjusted dispersion is about 3.2, versus 1.0 for pure Poisson), and
it keeps the interval sensible at both ends of the daily cycle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast.features import WEEK_MINUTES


class SeasonalNaive:
    """Predict the value observed one season earlier.

    Args:
        horizon_minutes: How far ahead to forecast.
        season_minutes: Length of the repeating cycle. A week by default,
            which captures both the daily shape and the weekday/weekend split.
    """

    name = "seasonal_naive"

    def __init__(
        self, horizon_minutes: int = 15, season_minutes: int = WEEK_MINUTES
    ):
        if horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if season_minutes <= horizon_minutes:
            raise ValueError("season_minutes must exceed horizon_minutes")

        self.horizon_minutes = horizon_minutes
        self.season_minutes = season_minutes
        self._residual_quantiles: pd.Series | None = None

    def fit(self, train: pd.Series) -> None:
        """Learn the spread of this model's own errors on the training data."""
        point = self._point_forecast(train)
        target = train.shift(-self.horizon_minutes)

        usable = point.notna() & target.notna()
        self._residuals = _standardised_residuals(
            target[usable].to_numpy(), point[usable].to_numpy()
        )
        self._residual_quantiles = pd.Series(dtype=float)  # marks the model fitted

    def predict(self, series: pd.Series, quantile: float) -> pd.Series:
        """Forecast `horizon` minutes ahead of every decision time in `series`.

        Returns:
            A series aligned to `series.index`; the value at `t` is the
            forecast for `t + horizon`. NaN where there is not yet a full
            season of history.
        """
        self._require_fitted()
        point = self._point_forecast(series)
        offset = float(np.quantile(self._residuals, quantile))
        return _apply_offset(point, offset)

    def _point_forecast(self, series: pd.Series) -> pd.Series:
        # The target is at t + horizon, and we want the value one season before
        # that, so we look back (season - horizon) from the decision time. That
        # is strictly in the past for any horizon shorter than the season.
        return series.shift(self.season_minutes - self.horizon_minutes)

    def _require_fitted(self) -> None:
        if self._residual_quantiles is None:
            raise RuntimeError("call fit before predict")


class SeasonalNaiveAdjusted(SeasonalNaive):
    """Last week's shape, re-levelled by how the recent hour compares.

    The plain seasonal naive assumes this week looks exactly like last week. If
    the whole day is running 20% hot — a busy period, a marketing push, a
    heatwave — it will be 20% short all day. Scaling last week's shape by the
    ratio of the last hour to the same hour a week ago fixes the level while
    keeping the shape.

    The ratio is clipped, because when last week's comparison hour was nearly
    idle the raw ratio explodes and turns one quiet minute into an enormous
    capacity request.
    """

    name = "seasonal_naive_adjusted"

    def __init__(
        self,
        horizon_minutes: int = 15,
        season_minutes: int = WEEK_MINUTES,
        level_window_minutes: int = 60,
        ratio_bounds: tuple[float, float] = (0.5, 2.0),
    ):
        super().__init__(horizon_minutes, season_minutes)
        self.level_window_minutes = level_window_minutes
        self.ratio_bounds = ratio_bounds

    def _point_forecast(self, series: pd.Series) -> pd.Series:
        base = super()._point_forecast(series)

        # Both windows end at t-1: the minute containing t has not finished.
        recent = series.shift(1).rolling(self.level_window_minutes).mean()
        same_hour_last_season = (
            series.shift(1 + self.season_minutes)
            .rolling(self.level_window_minutes)
            .mean()
        )

        low, high = self.ratio_bounds
        ratio = (recent / same_hour_last_season).clip(lower=low, upper=high)
        # Where the comparison is unavailable, fall back to the plain forecast
        # rather than dropping the prediction entirely.
        return base * ratio.fillna(1.0)


def _standardised_residuals(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Errors scaled so their spread is comparable across the daily cycle.

    Dividing by sqrt(prediction) reflects that count variance grows roughly in
    proportion to the mean. Without it, the quantiles would be dominated by the
    busy hours and would badly over-provision the quiet ones.
    """
    scale = np.sqrt(np.maximum(predicted, 1.0))
    return (actual - predicted) / scale


def _apply_offset(point: pd.Series, standardised_offset: float) -> pd.Series:
    """Undo the standardisation to turn a residual quantile back into counts."""
    scale = np.sqrt(np.maximum(point, 1.0))
    return (point + standardised_offset * scale).clip(lower=0.0)
