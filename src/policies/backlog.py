"""Target tracking on backlog per instance — the approach AWS recommends for SQS.

The metric is ``queue_depth / in_service_instances`` and the target comes from
Little's Law: ``acceptable_latency / service_time`` messages per worker.

This is the headline reactive baseline, and it is also the policy the project's
thesis is about. Queue depth is the *integral* of capacity error: it only
becomes non-zero once the fleet is already too small, and it keeps growing for
as long as that remains true. By the time the metric crosses its target the
system has been underprovisioned for a while, and only then does it start
paying the instance boot delay on top. That lag is structural, not a tuning
mistake, which is why the policy is tuned honestly here rather than hobbled.
"""

from __future__ import annotations

import math

from src.policies.capacity import backlog_target_per_instance
from src.sim.observation import Observation


class BacklogPerInstancePolicy:
    """Scale so that no worker has more than `target` messages queued behind it.

    Args:
        acceptable_latency_s: Latency budget for the oldest message. This is
            the knob the sweep tunes; it sets the target via Little's Law.
        service_seconds: Mean processing time per message.
        min_instances: Floor the policy holds on its own. A real operator
            would configure one, so leaving it at zero would understate the
            baseline.
    """

    name = "backlog_per_instance"

    def __init__(
        self,
        acceptable_latency_s: float,
        service_seconds: float,
        min_instances: int = 0,
    ):
        if min_instances < 0:
            raise ValueError(f"min_instances must be non-negative, got {min_instances}")

        self.acceptable_latency_s = acceptable_latency_s
        self.service_seconds = service_seconds
        self.min_instances = int(min_instances)
        # Raises for non-positive inputs, so invalid configuration fails here
        # rather than silently at the first decision.
        self.target = backlog_target_per_instance(acceptable_latency_s, service_seconds)

    def decide(self, t: float, obs: Observation) -> int:
        required = math.ceil(obs.queue_depth / self.target)
        return int(max(self.min_instances, required))

    def __repr__(self) -> str:
        return (
            f"BacklogPerInstancePolicy(acceptable_latency_s="
            f"{self.acceptable_latency_s}, target={self.target:.2f}, "
            f"min_instances={self.min_instances})"
        )
