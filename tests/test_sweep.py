"""Tests for the parameter sweep runner."""

import pandas as pd
import pytest

from src.eval.sweep import SweepSpec, run_sweep
from src.policies.static import StaticPolicy
from src.sim.config import SimConfig
from src.sim.service import ServiceTime


def trace(minutes: int = 120, rate: int = 60) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=minutes, freq="1min", name="ts")
    return pd.Series([rate] * minutes, index=index, name="arrivals")


def config() -> SimConfig:
    return SimConfig(
        service=ServiceTime(mean_seconds=30.0, cv=0.0),
        arrival_mode="uniform",
        initial_instances=30,
        max_instances=500,
        boot_time_s=180.0,
    )


def specs() -> list[SweepSpec]:
    return [
        SweepSpec(policy=StaticPolicy(n), label="static", params={"instances": n})
        for n in (20, 40, 60)
    ]


def test_sweep_returns_one_row_per_specification():
    table = run_sweep(trace(), specs(), config(), warmup_s=0.0)

    assert len(table) == 3


def test_swept_parameters_appear_as_columns():
    table = run_sweep(trace(), specs(), config(), warmup_s=0.0)

    assert "instances" in table.columns
    assert sorted(table["instances"]) == [20, 40, 60]


def test_policy_label_is_recorded():
    table = run_sweep(trace(), specs(), config(), warmup_s=0.0)

    assert set(table["policy"]) == {"static"}


def test_metric_columns_are_present():
    table = run_sweep(trace(), specs(), config(), warmup_s=0.0)

    for column in (
        "cost_instance_hours",
        "slo_violation_fraction",
        "p99_age_s",
        "completion_fraction",
    ):
        assert column in table.columns


def test_more_capacity_costs_more_and_violates_less():
    # The sweep must reproduce the basic tradeoff, or nothing downstream means
    # anything.
    table = run_sweep(trace(), specs(), config(), warmup_s=0.0).set_index("instances")

    assert table.loc[60, "cost_instance_hours"] > table.loc[20, "cost_instance_hours"]
    assert (
        table.loc[60, "slo_violation_fraction"]
        <= table.loc[20, "slo_violation_fraction"]
    )


def test_sweep_is_deterministic():
    first = run_sweep(trace(), specs(), config(), warmup_s=0.0)
    second = run_sweep(trace(), specs(), config(), warmup_s=0.0)

    pd.testing.assert_frame_equal(first, second)


def test_parallel_and_sequential_runs_agree():
    # Workers must not perturb the seeding or the ordering of results.
    sequential = run_sweep(trace(), specs(), config(), warmup_s=0.0, max_workers=1)
    parallel = run_sweep(trace(), specs(), config(), warmup_s=0.0, max_workers=2)

    pd.testing.assert_frame_equal(sequential, parallel)


def test_rows_stay_in_specification_order():
    table = run_sweep(trace(), specs(), config(), warmup_s=0.0, max_workers=2)

    assert list(table["instances"]) == [20, 40, 60]


def test_sweep_rejects_an_empty_specification_list():
    with pytest.raises(ValueError):
        run_sweep(trace(), [], config())
