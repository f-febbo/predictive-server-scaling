"""Target tracking on arrival rate — the strong reactive baseline.

This is the policy the predictive approach actually has to beat, and it is
built to be as good as a reactive policy can be. It scales on the *arrival
rate* rather than on queue depth, which matters: arrival rate is a direct
measurement of incoming load, whereas queue depth only reveals a capacity
shortfall after that shortfall has already accumulated.

It still lags, and that lag is the whole remaining gap. A burst is only visible
after it has been averaged over the observation window, then the decision waits
for the next scaler tick, and then the instances take the boot delay to arrive.
Prediction attacks exactly that chain — but there is nothing else wrong with
this policy, and pretending otherwise would make the comparison dishonest.

Two terms, and the fleet is sized to the larger:

1. Enough capacity for the observed arrival rate, plus headroom.
2. Enough capacity to clear whatever backlog exists within a drain window.

The second term matters more than it looks. Pure arrival-rate tracking
provisions exactly enough for incoming work and therefore has no surplus to
work off a backlog. Once it falls behind — which any lagging policy eventually
does — it would stay behind forever. Leaving that out would produce a baseline
that loses for a reason nobody would tolerate in production.
"""

from __future__ import annotations

import math

from src.policies.capacity import instances_for_arrival_rate
from src.sim.observation import Observation


class ArrivalRateTargetPolicy:
    """Scale to the recent arrival rate, with headroom and backlog recovery.

    Args:
        service_seconds: Mean processing time per message.
        target_utilization: Fraction of capacity to run at. Lower means more
            headroom, more cost, and more tolerance of a burst arriving
            between ticks.
        window_minutes: Minutes of arrival history to average. Short windows
            react faster but chase noise; long ones are steadier but lag more.
        backlog_drain_seconds: Time budget for clearing an existing backlog.
            None disables the term.
        min_instances: Floor the policy holds on its own.
    """

    name = "arrival_rate"

    def __init__(
        self,
        service_seconds: float,
        target_utilization: float = 0.8,
        window_minutes: int = 5,
        backlog_drain_seconds: float | None = 300.0,
        min_instances: int = 0,
    ):
        if service_seconds <= 0:
            raise ValueError(f"service_seconds must be positive, got {service_seconds}")
        if not 0.0 < target_utilization <= 1.0:
            raise ValueError(
                f"target_utilization must be in (0, 1], got {target_utilization}"
            )
        if window_minutes <= 0:
            raise ValueError(f"window_minutes must be positive, got {window_minutes}")
        if backlog_drain_seconds is not None and backlog_drain_seconds <= 0:
            raise ValueError("backlog_drain_seconds must be positive or None")
        if min_instances < 0:
            raise ValueError(f"min_instances must be non-negative, got {min_instances}")

        self.service_seconds = service_seconds
        self.target_utilization = target_utilization
        self.window_minutes = int(window_minutes)
        self.backlog_drain_seconds = backlog_drain_seconds
        self.min_instances = int(min_instances)

    def decide(self, t: float, obs: Observation) -> int:
        rate = obs.recent_arrival_rate(minutes=self.window_minutes)
        for_arrivals = instances_for_arrival_rate(
            rate, self.service_seconds, self.target_utilization
        )

        required = max(for_arrivals, self._instances_to_drain(obs.queue_depth))
        return int(max(self.min_instances, required))

    def _instances_to_drain(self, queue_depth: int) -> int:
        """Capacity needed to work off the current backlog within the budget.

        Each queued message occupies a worker for `service_seconds`, so
        clearing `queue_depth` of them inside `backlog_drain_seconds` takes
        that much work divided by the time allowed.
        """
        if self.backlog_drain_seconds is None or queue_depth <= 0:
            return 0

        work_seconds = queue_depth * self.service_seconds
        return math.ceil(work_seconds / self.backlog_drain_seconds)

    def __repr__(self) -> str:
        return (
            f"ArrivalRateTargetPolicy(target_utilization={self.target_utilization}, "
            f"window_minutes={self.window_minutes}, "
            f"backlog_drain_seconds={self.backlog_drain_seconds}, "
            f"min_instances={self.min_instances})"
        )
