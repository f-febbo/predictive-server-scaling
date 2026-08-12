"""The standard experimental scenario shared by Phases 3 to 5.

Every policy is compared under identical fleet parameters. Defining them once,
here, is what makes the comparison a comparison rather than a collection of
unrelated runs.

Why a 30-second service time
----------------------------
Required capacity is arrival rate times service time, so the service time alone
decides how large the fleet is for a given trace. At a 2-second service time
this trace needs 2 to 9 instances, and the system crosses from catastrophically
overloaded to essentially perfect over a three-instance gap, because one
instance is a third of the fleet. A cost/SLO frontier swept across three
integer points is not a frontier: every difference between policies would be
integer rounding rather than decision quality.

At 30 seconds the usable band is roughly 37 to 84 instances, which gives real
resolution. It is also the more realistic model. Queue-backed asynchronous
workers — image processing, ETL chunks, report generation, model inference —
genuinely take tens of seconds. A 2-second task is a synchronous API call,
which would not sit behind a queue with a three-minute instance boot anyway.

The parameter remains swept-able; `SimConfig` still defaults to 2 seconds.
"""

from __future__ import annotations

from dataclasses import replace

from src.sim.config import SimConfig
from src.sim.service import ServiceTime

# Mean seconds to process one message.
SERVICE_SECONDS = 30.0

# Instance launch plus application warmup. The reason the project exists.
BOOT_TIME_SECONDS = 180.0

# The SLO: a message should not wait longer than this before being picked up.
# Age of the oldest message is the SLI, not queue depth. Depth is what we
# control; age is what users feel.
SLO_AGE_SECONDS = 60.0

# Metrics ignore the first hour so the startup transient — an arbitrary initial
# fleet size, and a policy that has not yet observed anything — does not
# contaminate the comparison.
WARMUP_SECONDS = 3600.0


def experiment_config(**overrides) -> SimConfig:
    """The shared fleet configuration, with optional per-experiment overrides."""
    base = SimConfig(
        service=ServiceTime(mean_seconds=SERVICE_SECONDS, cv=0.5),
        boot_time_s=BOOT_TIME_SECONDS,
        scaler_tick_s=60.0,
        scale_in_cooldown_s=300.0,
        initial_instances=40,
        min_instances=1,
        max_instances=500,
        arrival_mode="poisson",
        arrival_history_minutes=60,
        seed=0,
    )
    return replace(base, **overrides) if overrides else base
