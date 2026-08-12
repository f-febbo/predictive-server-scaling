"""Tests for the run metrics that form the cost/SLO frontier axes.

Both axes are load-bearing for every claim the project makes, so each is
pinned against a hand-built tick table rather than a simulator run.
"""

import pandas as pd
import pytest

from src.eval.metrics import compute_metrics, pareto_frontier
from src.sim.simulator import SimResult


def result_from(ages: list[float], t_start: float = 0.0, **overrides) -> SimResult:
    """A SimResult carrying a chosen sequence of per-tick message ages."""
    ticks = pd.DataFrame(
        {
            "t": [t_start + 60.0 * i for i in range(len(ages))],
            "queue_depth": [0] * len(ages),
            "oldest_message_age": ages,
            "in_service_instances": [10] * len(ages),
            "pending_instances": [0] * len(ages),
            "arrivals": [100] * len(ages),
            "completions": [100] * len(ages),
            "desired_instances": [10] * len(ages),
        }
    )
    defaults = dict(
        messages_arrived=1000,
        messages_completed=1000,
        messages_in_queue_at_end=0,
        messages_in_service_at_end=0,
        billed_instance_seconds=36_000.0,
    )
    return SimResult(ticks=ticks, **{**defaults, **overrides})


# --- cost -------------------------------------------------------------------


def test_cost_is_billed_instance_seconds_expressed_in_hours():
    result = result_from([0.0] * 10, billed_instance_seconds=7200.0)

    assert compute_metrics(result, warmup_s=0.0).cost_instance_hours == pytest.approx(2.0)


# --- SLO --------------------------------------------------------------------


def test_slo_violation_fraction_counts_ticks_above_the_threshold():
    # Four of ten ticks exceed a 60-second threshold.
    ages = [0.0, 10.0, 61.0, 100.0, 5.0, 500.0, 0.0, 0.0, 61.0, 0.0]

    metrics = compute_metrics(result_from(ages), slo_threshold_s=60.0, warmup_s=0.0)

    assert metrics.slo_violation_fraction == pytest.approx(0.4)


def test_a_tick_exactly_at_the_threshold_is_not_a_violation():
    metrics = compute_metrics(
        result_from([60.0, 60.0]), slo_threshold_s=60.0, warmup_s=0.0
    )

    assert metrics.slo_violation_fraction == 0.0


def test_a_perfectly_healthy_run_has_no_violations():
    metrics = compute_metrics(result_from([0.0] * 100), warmup_s=0.0)

    assert metrics.slo_violation_fraction == 0.0


def test_a_fully_broken_run_violates_everywhere():
    metrics = compute_metrics(
        result_from([9999.0] * 50), slo_threshold_s=60.0, warmup_s=0.0
    )

    assert metrics.slo_violation_fraction == 1.0


def test_age_percentiles_are_reported():
    ages = [float(value) for value in range(101)]

    metrics = compute_metrics(result_from(ages), warmup_s=0.0)

    assert metrics.p50_age_s == pytest.approx(50.0)
    assert metrics.p99_age_s == pytest.approx(99.0)
    assert metrics.max_age_s == pytest.approx(100.0)


# --- warmup exclusion -------------------------------------------------------


def test_warmup_ticks_are_excluded_from_the_metrics():
    # The first five minutes are a disaster, the rest is clean. Excluding the
    # startup transient must remove the disaster from the score.
    ages = [9999.0] * 5 + [0.0] * 10

    metrics = compute_metrics(result_from(ages), slo_threshold_s=60.0, warmup_s=300.0)

    assert metrics.slo_violation_fraction == 0.0
    assert metrics.max_age_s == 0.0


def test_warmup_boundary_tick_is_included():
    # warmup_s=300 excludes t < 300, so the tick at t=300 counts.
    ages = [9999.0] * 5 + [42.0]

    metrics = compute_metrics(result_from(ages), warmup_s=300.0)

    assert metrics.max_age_s == pytest.approx(42.0)


def test_cost_is_not_discounted_by_the_warmup_exclusion():
    # Warmup affects which ticks are scored, not what the run was billed. A
    # policy does not get free capacity for its first hour.
    result = result_from([0.0] * 100, billed_instance_seconds=36_000.0)

    metrics = compute_metrics(result, warmup_s=1800.0)

    assert metrics.cost_instance_hours == pytest.approx(10.0)


def test_excluding_every_tick_is_rejected():
    # Silently returning NaN metrics would put an empty point on the frontier.
    with pytest.raises(ValueError, match="warmup"):
        compute_metrics(result_from([0.0] * 5), warmup_s=100_000.0)


# --- throughput guards ------------------------------------------------------


def test_unprocessed_backlog_is_reported():
    # A policy that simply never processes anything would otherwise look cheap
    # and, once its queue stops growing, deceptively stable.
    result = result_from(
        [0.0] * 10,
        messages_arrived=1000,
        messages_completed=400,
        messages_in_queue_at_end=590,
        messages_in_service_at_end=10,
    )

    metrics = compute_metrics(result, warmup_s=0.0)

    assert metrics.unprocessed_at_end == 600
    assert metrics.completion_fraction == pytest.approx(0.4)


def test_completion_fraction_is_one_for_a_fully_drained_run():
    metrics = compute_metrics(result_from([0.0] * 10), warmup_s=0.0)

    assert metrics.completion_fraction == pytest.approx(1.0)


def test_mean_instance_count_is_reported():
    metrics = compute_metrics(result_from([0.0] * 10), warmup_s=0.0)

    assert metrics.mean_instances == pytest.approx(10.0)


def test_metrics_record_the_threshold_they_were_scored_against():
    # A frontier point is meaningless without knowing the SLO it assumed.
    metrics = compute_metrics(result_from([0.0] * 5), slo_threshold_s=30.0, warmup_s=0.0)

    assert metrics.slo_threshold_s == 30.0


# --- Pareto frontier --------------------------------------------------------


def frontier_table(rows: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["cost_instance_hours", "slo_violation_fraction"])


def test_frontier_drops_points_beaten_on_both_axes():
    # (20, 0.5) costs more than (10, 0.3) and is worse, so it is not a
    # tradeoff anyone would choose.
    table = frontier_table([(10.0, 0.3), (20.0, 0.5), (30.0, 0.1)])

    kept = pareto_frontier(table)

    assert list(kept["cost_instance_hours"]) == [10.0, 30.0]


def test_frontier_keeps_genuine_tradeoffs():
    table = frontier_table([(10.0, 0.5), (20.0, 0.3), (30.0, 0.1)])

    kept = pareto_frontier(table)

    assert len(kept) == 3


def test_frontier_is_ordered_by_increasing_cost():
    table = frontier_table([(30.0, 0.1), (10.0, 0.5), (20.0, 0.3)])

    kept = pareto_frontier(table)

    assert list(kept["cost_instance_hours"]) == [10.0, 20.0, 30.0]


def test_at_equal_cost_the_better_objective_survives():
    table = frontier_table([(10.0, 0.4), (10.0, 0.2)])

    kept = pareto_frontier(table)

    assert len(kept) == 1
    assert kept.loc[0, "slo_violation_fraction"] == 0.2


def test_frontier_can_score_a_different_objective_column():
    # The figure plots tail latency on the same axes, so the frontier must not
    # be hard-wired to the violation rate.
    table = pd.DataFrame(
        {
            "cost_instance_hours": [10.0, 20.0, 30.0],
            "slo_violation_fraction": [0.5, 0.5, 0.5],
            "p99_age_s": [90.0, 40.0, 80.0],
        }
    )

    kept = pareto_frontier(table, objective="p99_age_s")

    assert list(kept["cost_instance_hours"]) == [10.0, 20.0]


def test_frontier_rejects_a_missing_column():
    with pytest.raises(KeyError):
        pareto_frontier(frontier_table([(1.0, 0.1)]), objective="nope")
