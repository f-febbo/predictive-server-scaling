"""LightGBM quantile regression on lag, calendar, and rolling features.

Gradient boosting on tabular lag features is the right tool for a series this
size. An LSTM or a transformer would be a credential rather than a decision:
on a few tens of thousands of rows of strongly seasonal tabular data they cost
far more to train and generally lose, and picking one would be exactly the kind
of unforced error this project is meant to avoid.

One model is trained per quantile, each minimising pinball loss at its own
alpha. That is the point — the quantile is the frontier's parameter, and a
single conditional-mean model could not produce the curve at all. Because the
models are independent they can occasionally cross at individual points; that
is a known property of independent quantile regression, not a bug, and it is
harmless here because the policy consumes one quantile at a time.
"""

from __future__ import annotations

from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.forecast.features import build_features, build_target


class LightGBMQuantile:
    """Per-quantile gradient boosted forecaster.

    Args:
        horizon_minutes: How far ahead to forecast.
        quantiles: Quantiles to fit. Requesting an unfitted quantile at
            predict time is an error rather than a silent fallback.
        n_estimators: Boosting rounds per quantile.
        num_leaves: Tree capacity.
        learning_rate: Boosting step size.
        min_child_samples: Minimum rows per leaf; guards against fitting
            individual minutes.
        random_state: Seed, so a reported result can be reproduced.
    """

    name = "lightgbm"

    def __init__(
        self,
        horizon_minutes: int = 15,
        quantiles: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99),
        n_estimators: int = 300,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        min_child_samples: int = 50,
        random_state: int = 0,
    ):
        if horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        for quantile in quantiles:
            if not 0.0 < quantile < 1.0:
                raise ValueError(f"quantile must be in (0, 1), got {quantile}")

        self.horizon_minutes = horizon_minutes
        self.quantiles = tuple(quantiles)
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.random_state = random_state

        self._models: dict[float, lgb.LGBMRegressor] = {}
        self._feature_names: list[str] | None = None
        self.last_trained_target_time: pd.Timestamp | None = None

    def fit(self, train: pd.Series) -> None:
        """Train one model per quantile.

        Rows whose target falls outside `train` are dropped, so a model never
        sees an outcome from beyond the data it was given. The backtest relies
        on this to keep its folds honest.
        """
        features = build_features(train)
        target = build_target(train, self.horizon_minutes)

        usable = target.notna()
        design = features[usable]
        outcomes = target[usable]

        if design.empty:
            raise ValueError("no usable training rows; series is shorter than horizon")

        self._feature_names = list(design.columns)
        self.last_trained_target_time = (
            outcomes.index[-1] + pd.Timedelta(minutes=self.horizon_minutes)
        )

        for quantile in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=self.n_estimators,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbose=-1,
            )
            model.fit(design, outcomes)
            self._models[quantile] = model

    def predict(self, series: pd.Series, quantile: float) -> pd.Series:
        """Forecast `horizon` minutes ahead of every decision time in `series`.

        Returns:
            A series aligned to `series.index`; the value at `t` forecasts
            `t + horizon`.
        """
        if not self._models:
            raise RuntimeError("call fit before predict")
        if quantile not in self._models:
            raise KeyError(
                f"quantile {quantile} was not fitted; available: "
                f"{sorted(self._models)}"
            )

        features = build_features(series)[self._feature_names]
        predictions = self._models[quantile].predict(features)

        return pd.Series(
            np.clip(predictions, 0.0, None), index=series.index, name="forecast"
        )

    def feature_importance(self, quantile: float) -> pd.Series:
        """Gain-based importance, for sanity-checking what the model leans on."""
        if quantile not in self._models:
            raise KeyError(f"quantile {quantile} was not fitted")

        model = self._models[quantile]
        return pd.Series(
            model.booster_.feature_importance(importance_type="gain"),
            index=self._feature_names,
        ).sort_values(ascending=False)
