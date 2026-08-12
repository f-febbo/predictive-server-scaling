"""Running a policy across a grid of its own parameters.

A single (cost, SLO) point says almost nothing: any policy can be made cheaper
or safer by turning one knob, so a lone measurement is really a statement about
the tuning, not the policy. Sweeping the knob traces the curve the policy can
actually reach, and comparing curves is the only comparison that means
anything.

Runs are independent, so they parallelise across processes. The seed lives in
the SimConfig rather than in the worker, so results do not depend on how many
cores happened to be available.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.eval.metrics import compute_metrics
from src.sim.config import SimConfig
from src.sim.observation import ScalingPolicy
from src.sim.simulator import simulate


@dataclass(frozen=True)
class SweepSpec:
    """One configuration to evaluate.

    Attributes:
        policy: The configured policy instance.
        label: Policy family name, used to group points into one curve.
        params: The swept knob values, recorded as columns so a frontier point
            can be traced back to the configuration that produced it.
    """

    policy: ScalingPolicy
    label: str
    params: dict[str, Any] = field(default_factory=dict)


def run_sweep(
    arrival_counts: pd.Series,
    specs: list[SweepSpec],
    config: SimConfig,
    slo_threshold_s: float = 60.0,
    warmup_s: float = 3600.0,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Evaluate every specification and collect the results into one table.

    Args:
        arrival_counts: Arrivals per minute.
        specs: Configurations to evaluate.
        config: Shared fleet parameters. Identical across every spec, which is
            what makes the comparison fair.
        slo_threshold_s: Age above which a tick counts as an SLO violation.
        warmup_s: Ticks before this are excluded from quality metrics.
        max_workers: Processes to use. 1 runs sequentially, which is easier to
            debug and what the tests use.

    Returns:
        One row per spec, in the order given, carrying the swept parameters
        alongside the scored metrics.
    """
    if not specs:
        raise ValueError("no specifications to sweep")

    arguments = [
        (arrival_counts, spec, config, slo_threshold_s, warmup_s) for spec in specs
    ]

    if max_workers == 1:
        rows = [_evaluate(argument) for argument in arguments]
    else:
        # "spawn" rather than the Linux default "fork": forking a process that
        # already has threads running is unsafe, and numpy/BLAS may have
        # started some.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as pool:
            # `map` preserves input order, so rows line up with specs
            # regardless of the order the workers finish in.
            rows = list(pool.map(_evaluate, arguments))

    return pd.DataFrame(rows)


def _evaluate(argument: tuple) -> dict:
    """Run one specification and flatten it into a results row.

    Takes a single packed argument so it can be used with `ProcessPoolExecutor.map`.
    """
    arrival_counts, spec, config, slo_threshold_s, warmup_s = argument

    result = simulate(arrival_counts, spec.policy, config)
    metrics = compute_metrics(
        result, slo_threshold_s=slo_threshold_s, warmup_s=warmup_s
    )

    return {"policy": spec.label, **spec.params, **metrics.as_row()}
