"""Scoring forecasts under asymmetric costs.

The costs in this problem are wildly lopsided. Under-forecasting means the
fleet is too small, the queue builds, and the SLO breaks. Over-forecasting
means a few extra instances run for a few minutes. Treating those as equally
bad — which is exactly what MSE and MAPE do — optimises for the wrong thing and
produces a model tuned to be right on average rather than safe.

Pinball loss encodes the asymmetry directly, and the quantile is the dial. At
q=0.9 a shortfall costs nine times what an equivalent excess costs, so the
model that minimises it learns to sit above the mean. Sweeping q is what traces
the cost/SLO frontier rather than producing one arbitrary operating point.
"""

from __future__ import annotations

import numpy as np


def pinball_loss(
    actual: np.ndarray, forecast: np.ndarray, quantile: float
) -> float:
    """Mean pinball (quantile) loss.

    For each point, a shortfall is weighted by `quantile` and an excess by
    `1 - quantile`. The loss is minimised by the true `quantile`-th quantile of
    the conditional distribution.

    Args:
        actual: Observed values.
        forecast: Predicted values.
        quantile: Target quantile, strictly between 0 and 1.

    Returns:
        Mean loss over the sample.
    """
    actual, forecast = _validated(actual, forecast)
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    error = actual - forecast
    # Where the forecast fell short (error > 0) the weight is `quantile`;
    # where it overshot, `quantile - 1` applied to a negative error.
    loss = np.maximum(quantile * error, (quantile - 1.0) * error)
    return float(loss.mean())


def coverage(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Share of points where the forecast was at least the actual.

    The calibration check. A p90 forecast should cover roughly 90% of actuals;
    if it covers 60%, the model is quietly under-provisioning no matter how
    good its pinball loss looks. An exact match counts as covered, since
    forecasting precisely the demand that arrived is not a shortfall.
    """
    actual, forecast = _validated(actual, forecast)
    return float((forecast >= actual).mean())


def mean_absolute_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error.

    Reported for comparability with the forecasting literature, not used as a
    training objective — it is symmetric, which this problem is not.
    """
    actual, forecast = _validated(actual, forecast)
    return float(np.abs(actual - forecast).mean())


def _validated(
    actual: np.ndarray, forecast: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)

    if actual.shape != forecast.shape:
        raise ValueError(
            f"actual and forecast must have the same shape, got "
            f"{actual.shape} and {forecast.shape}"
        )
    if actual.size == 0:
        raise ValueError("cannot score an empty sample")

    return actual, forecast
