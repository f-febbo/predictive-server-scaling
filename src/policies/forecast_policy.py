"""Predictive scaling: provision for the load that is about to arrive.

The forecast is precomputed rather than evaluated inside the simulator, which
is both how a real deployment would do it (a scheduled Lambda writes a
prediction; the scaler reads it) and the only tractable way to run hundreds of
simulations. That moves the no-lookahead guarantee out of the simulator and
into the backtest, which is where `forecast.backtest` enforces it: the value
stored at minute `m` was computed only from data at or before `m`, and predicts
minute `m + horizon`.

The policy reads the entry for the *current* minute. That entry already looks
`horizon` minutes ahead, and the horizon is chosen as boot time plus a margin,
so instances requested now finish booting just before the demand lands. This
indexing is the one place a subtle off-by-one would hand the policy genuine
future knowledge, so it is pinned by test.

Composition with a reactive floor is the default in practice: the forecast sets
the minimum and target tracking handles whatever the forecast missed. Both the
composed and the pure variants are supported, because comparing them is what
separates the contribution of prediction from the contribution of correction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.policies.capacity import instances_for_arrival_rate
from src.sim.observation import Observation, ScalingPolicy

SECONDS_PER_MINUTE = 60.0


class ForecastPolicy:
    """Scale to a forecast of the arrival rate.

    Args:
        forecast_per_minute: Predicted arrivals for each trace minute, indexed
            by minute offset from the start of the trace. NaN where no forecast
            is available.
        service_seconds: Mean processing time per message.
        target_utilization: Fraction of capacity to run at.
        safety_margin: Multiplier on the converted capacity. Distinct from the
            quantile: the quantile prices uncertainty in the *demand*, while
            this prices uncertainty in the service time and the conversion.
        reactive_floor: Optional policy whose decision is treated as a lower
            bound. This is what catches bursts the forecast missed.
        min_instances: Absolute floor.
    """

    def __init__(
        self,
        forecast_per_minute: np.ndarray,
        service_seconds: float,
        target_utilization: float = 1.0,
        safety_margin: float = 1.0,
        reactive_floor: ScalingPolicy | None = None,
        min_instances: int = 0,
    ):
        if service_seconds <= 0:
            raise ValueError(f"service_seconds must be positive, got {service_seconds}")
        if safety_margin <= 0:
            raise ValueError(f"safety_margin must be positive, got {safety_margin}")
        if not 0.0 < target_utilization <= 1.0:
            raise ValueError(
                f"target_utilization must be in (0, 1], got {target_utilization}"
            )
        if min_instances < 0:
            raise ValueError(f"min_instances must be non-negative, got {min_instances}")

        self.forecast_per_minute = np.asarray(forecast_per_minute, dtype=np.float64)
        self.service_seconds = service_seconds
        self.target_utilization = target_utilization
        self.safety_margin = safety_margin
        self.reactive_floor = reactive_floor
        self.min_instances = int(min_instances)

    @property
    def name(self) -> str:
        return "forecast_composed" if self.reactive_floor else "forecast"

    def decide(self, t: float, obs: Observation) -> int:
        required = self._from_forecast(t)

        if self.reactive_floor is not None:
            required = max(required, self.reactive_floor.decide(t, obs))

        return int(max(self.min_instances, required))

    def _from_forecast(self, t: float) -> int:
        """Capacity implied by the forecast held for the current minute.

        Returns 0 when no forecast exists — at the very start of a run, or past
        the end of the forecast series — so the floor or the reactive term
        takes over rather than the policy failing.
        """
        minute = int(t // SECONDS_PER_MINUTE)
        if minute >= len(self.forecast_per_minute):
            return 0

        predicted = self.forecast_per_minute[minute]
        if not np.isfinite(predicted):
            return 0

        rate_per_second = predicted / SECONDS_PER_MINUTE
        required = instances_for_arrival_rate(
            rate_per_second, self.service_seconds, self.target_utilization
        )
        return math.ceil(required * self.safety_margin)

    def __repr__(self) -> str:
        return (
            f"ForecastPolicy(target_utilization={self.target_utilization}, "
            f"safety_margin={self.safety_margin}, "
            f"reactive_floor={self.reactive_floor!r}, "
            f"min_instances={self.min_instances})"
        )


def forecast_array(
    forecasts: pd.Series, trace_index: pd.DatetimeIndex
) -> np.ndarray:
    """Align a timestamped forecast series to trace-minute offsets.

    The simulator counts seconds from the start of the trace, so the forecast
    has to be flattened onto that same grid. Minutes with no forecast become
    NaN, which the policy treats as "no opinion" rather than "no demand".
    """
    aligned = forecasts.reindex(trace_index)
    return aligned.to_numpy(dtype=np.float64)
