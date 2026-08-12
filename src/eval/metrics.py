"""Scoring a simulated run: what it cost, and how well it served users.

These two numbers are the axes of every figure in the project. Reporting a
single blended score would hide the tradeoff that is the entire point — a
policy can always buy latency with money, and the interesting question is the
exchange rate.

The SLI is the age of the oldest message, not queue depth. Depth is what the
scaler controls; age is what a user waiting on a job actually experiences. A
deep queue drained quickly by a large fleet is fine; a shallow queue served by
nothing is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.sim.simulator import SimResult

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class RunMetrics:
    """Scored outcome of one run.

    Attributes:
        cost_instance_hours: Billed capacity, including instances that were
            booting and therefore not yet useful.
        slo_violation_fraction: Share of scored ticks whose oldest message was
            older than the threshold. The headline SLO number.
        slo_threshold_s: The threshold those violations were scored against.
        p50_age_s: Median age of the oldest message.
        p99_age_s: Tail age. A policy can hold a good median and still fail
            badly during the bursts that matter.
        max_age_s: Worst observed age.
        mean_instances: Average in-service fleet size across scored ticks.
        messages_arrived: Messages the trace delivered.
        messages_completed: Messages fully processed.
        unprocessed_at_end: Messages still queued or in flight when the run
            ended. A large value means the run never kept up.
        completion_fraction: Completed over arrived. Guards against a policy
            that looks cheap only because it did very little work.
    """

    cost_instance_hours: float
    slo_violation_fraction: float
    slo_threshold_s: float
    p50_age_s: float
    p99_age_s: float
    max_age_s: float
    mean_instances: float
    messages_arrived: int
    messages_completed: int
    unprocessed_at_end: int
    completion_fraction: float

    def as_row(self) -> dict:
        """Flat mapping for assembling a results table."""
        return {
            "cost_instance_hours": self.cost_instance_hours,
            "slo_violation_fraction": self.slo_violation_fraction,
            "slo_threshold_s": self.slo_threshold_s,
            "p50_age_s": self.p50_age_s,
            "p99_age_s": self.p99_age_s,
            "max_age_s": self.max_age_s,
            "mean_instances": self.mean_instances,
            "messages_arrived": self.messages_arrived,
            "messages_completed": self.messages_completed,
            "unprocessed_at_end": self.unprocessed_at_end,
            "completion_fraction": self.completion_fraction,
        }


def compute_metrics(
    result: SimResult,
    slo_threshold_s: float = 60.0,
    warmup_s: float = 3600.0,
) -> RunMetrics:
    """Score a run.

    Args:
        result: The simulated run.
        slo_threshold_s: A message older than this counts as a violation.
        warmup_s: Ticks before this time are excluded from the *quality*
            metrics. The start of a run reflects an arbitrary initial fleet
            size and a policy that has not observed anything yet, which says
            nothing about the policy itself. Cost is deliberately not
            discounted: a policy does not get free capacity for its first hour.

    Returns:
        The scored metrics.
    """
    scored = result.ticks[result.ticks["t"] >= warmup_s]
    if scored.empty:
        raise ValueError(
            f"warmup_s={warmup_s} excludes every tick; the run is only "
            f"{result.ticks['t'].max() if len(result.ticks) else 0}s long"
        )

    ages = scored["oldest_message_age"].to_numpy()
    unprocessed = result.messages_in_queue_at_end + result.messages_in_service_at_end

    return RunMetrics(
        cost_instance_hours=result.billed_instance_seconds / SECONDS_PER_HOUR,
        slo_violation_fraction=float((ages > slo_threshold_s).mean()),
        slo_threshold_s=slo_threshold_s,
        p50_age_s=float(np.percentile(ages, 50)),
        p99_age_s=float(np.percentile(ages, 99)),
        max_age_s=float(ages.max()),
        mean_instances=float(scored["in_service_instances"].mean()),
        messages_arrived=result.messages_arrived,
        messages_completed=result.messages_completed,
        unprocessed_at_end=unprocessed,
        completion_fraction=(
            result.messages_completed / result.messages_arrived
            if result.messages_arrived
            else 1.0
        ),
    )


def pareto_frontier(
    table: pd.DataFrame,
    objective: str = "slo_violation_fraction",
    cost: str = "cost_instance_hours",
) -> pd.DataFrame:
    """Keep only the points no other point beats on both cost and `objective`.

    A swept parameter usually produces configurations that are simply worse
    than another configuration of the *same* policy on both axes. Those are
    tuning failures rather than tradeoffs, and plotting them makes a policy
    look erratic instead of showing the curve it can actually achieve.

    Both axes are "lower is better". Ties in cost are broken by preferring the
    better objective, so an efficient point is never hidden behind an equally
    priced worse one.

    Returns:
        The efficient points, ordered by increasing cost.
    """
    for column in (objective, cost):
        if column not in table.columns:
            raise KeyError(f"column {column!r} is not in the table")

    ordered = table.sort_values([cost, objective]).reset_index(drop=True)

    keep = []
    best = float("inf")
    for position, value in enumerate(ordered[objective]):
        if value < best:
            keep.append(position)
            best = value

    return ordered.iloc[keep].reset_index(drop=True)
