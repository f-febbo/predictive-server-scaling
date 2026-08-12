"""Static overprovisioning: pick a fleet size and never change it.

Not a serious operational proposal, but the reference every autoscaler has to
beat. Sweeping the level traces the frontier available with no scaling logic at
all, and any policy that fails to improve on that curve is not earning its
complexity.
"""

from __future__ import annotations

from src.sim.observation import Observation


class StaticPolicy:
    """Fixed capacity, regardless of what the queue is doing."""

    name = "static"

    def __init__(self, instances: int):
        if instances < 0:
            raise ValueError(f"instances must be non-negative, got {instances}")
        self.instances = int(instances)

    def decide(self, t: float, obs: Observation) -> int:
        return self.instances

    def __repr__(self) -> str:
        return f"StaticPolicy(instances={self.instances})"
