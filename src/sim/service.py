"""How long a worker takes to process one message.

Lognormal by default. Real request handlers are right-skewed: most messages
finish near the mean and a small tail takes much longer, holding a worker and
backing up everything behind it. A symmetric or constant service time would
understate that effect and make every scaling policy look better than it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ServiceTime:
    """Lognormal service time, parameterised by its mean and variability.

    Args:
        mean_seconds: Average time to process one message.
        cv: Coefficient of variation (standard deviation / mean). Zero gives a
            constant service time, which makes queue arithmetic exact and is
            used by the simulator's acceptance tests.
    """

    mean_seconds: float = 2.0
    cv: float = 0.5

    def __post_init__(self) -> None:
        if self.mean_seconds <= 0:
            raise ValueError(f"mean_seconds must be positive, got {self.mean_seconds}")
        if self.cv < 0:
            raise ValueError(f"cv must be non-negative, got {self.cv}")

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `size` service times in seconds."""
        if size == 0:
            return np.empty(0, dtype=np.float64)
        if self.cv == 0:
            return np.full(size, self.mean_seconds, dtype=np.float64)

        # Convert (mean, cv) to the underlying normal's parameters. For
        # X ~ LogNormal(mu, sigma): E[X] = exp(mu + sigma^2 / 2) and
        # CV = sqrt(exp(sigma^2) - 1).
        sigma_squared = math.log(1.0 + self.cv**2)
        mu = math.log(self.mean_seconds) - sigma_squared / 2.0
        return rng.lognormal(mean=mu, sigma=math.sqrt(sigma_squared), size=size)

    def throughput_per_instance(self) -> float:
        """Messages per second one worker sustains when continuously busy.

        This is the constant that converts a required throughput into a
        required instance count, so it is the heart of the Little's Law
        conversion the scaling policies perform.
        """
        return 1.0 / self.mean_seconds
