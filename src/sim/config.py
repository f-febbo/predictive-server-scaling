"""Simulator configuration.

Every parameter here is meant to be swept. `boot_time_s` in particular is the
one the whole project turns on: if instances were available instantly, a
reactive policy would have nothing to be late about and prediction would buy
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.sim.service import ServiceTime


@dataclass(frozen=True)
class SimConfig:
    """Parameters of one simulated worker fleet.

    Attributes:
        boot_time_s: Delay between requesting an instance and it being able to
            process work — instance launch plus application warmup.
        scaler_tick_s: How often the policy is consulted.
        scale_in_cooldown_s: Minimum gap between two scale-in actions, so a
            policy cannot thrash capacity down and up.
        service: Per-message service time distribution.
        initial_instances: Instances already in service at t=0.
        min_instances: Floor applied to every policy decision.
        max_instances: Ceiling applied to every policy decision.
        arrival_mode: Sub-minute placement of arrivals; see `sim.arrivals`.
        arrival_history_minutes: How many completed minutes of arrival counts
            an observation exposes.
        seed: Seed for arrival placement and service times.
        record_messages: Keep a per-message record of arrival, start, and
            completion times. Off by default because a full-trace run holds
            millions of messages.
    """

    boot_time_s: float = 180.0
    scaler_tick_s: float = 60.0
    scale_in_cooldown_s: float = 300.0
    service: ServiceTime = field(default_factory=ServiceTime)
    initial_instances: int = 1
    min_instances: int = 1
    max_instances: int = 500
    arrival_mode: str = "poisson"
    arrival_history_minutes: int = 60
    seed: int = 0
    record_messages: bool = False

    def __post_init__(self) -> None:
        if self.boot_time_s < 0:
            raise ValueError("boot_time_s must be non-negative")
        if self.scaler_tick_s <= 0:
            raise ValueError("scaler_tick_s must be positive")
        if self.scale_in_cooldown_s < 0:
            raise ValueError("scale_in_cooldown_s must be non-negative")
        if self.min_instances < 0:
            raise ValueError("min_instances must be non-negative")
        if self.max_instances < self.min_instances:
            raise ValueError("max_instances must be at least min_instances")
        if not self.min_instances <= self.initial_instances <= self.max_instances:
            raise ValueError(
                f"initial_instances ({self.initial_instances}) must lie within "
                f"[{self.min_instances}, {self.max_instances}]"
            )
