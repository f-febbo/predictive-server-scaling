"""Phase 3: sweep the three reactive baselines and plot their cost/SLO frontiers.

Run with:

    uv run python -m experiments.phase3_baselines

Produces:
    results/phase3_baselines.csv      every configuration evaluated
    results/phase3_frontier.png       the three frontiers on one figure
    results/phase3_summary.md         headline table

Every policy is swept across its own tuning knobs rather than evaluated at one
hand-picked setting. A single (cost, SLO) point describes a tuning, not a
policy; the curve is the policy. This is also what stops the reactive baselines
being strawmen — each one is shown at the best it can do, and the sweep is wide
enough to include settings that are clearly too aggressive and clearly too
timid.

Runs on the training split only. The held-out test period stays untouched until
Phase 4, so that the predictive policy cannot be tuned, even indirectly, on the
data it will be judged on.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.pipeline import load_processed
from src.eval.metrics import pareto_frontier
from src.eval.scenario import (
    SERVICE_SECONDS,
    SLO_AGE_SECONDS,
    WARMUP_SECONDS,
    experiment_config,
)
from src.eval.sweep import SweepSpec, run_sweep
from src.policies.arrival_rate import ArrivalRateTargetPolicy
from src.policies.backlog import BacklogPerInstancePolicy
from src.policies.static import StaticPolicy

RESULTS_DIR = Path("results")

# Categorical slots 1-3 of the validated palette. Assigned in fixed order and
# never cycled; the ordering is the colour-blindness safety mechanism.
POLICY_STYLE = {
    "static": {"color": "#2a78d6", "label": "Static overprovisioning", "marker": "o"},
    "backlog_per_instance": {
        "color": "#eb6834",
        "label": "Target tracking: backlog/instance",
        "marker": "s",
    },
    "arrival_rate": {
        "color": "#1baf7a",
        "label": "Target tracking: arrival rate",
        "marker": "^",
    },
}

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dcdbd6"


def build_specs() -> list[SweepSpec]:
    """The full grid of baseline configurations to evaluate."""
    specs: list[SweepSpec] = []

    # 1. Static overprovisioning. The trace needs roughly 34 instances on
    # average and 111 at its peak, so this brackets both.
    for instances in range(25, 135, 5):
        specs.append(
            SweepSpec(
                policy=StaticPolicy(instances),
                label="static",
                params={"knob": float(instances), "config": f"n={instances}"},
            )
        )

    # 2. Backlog per instance. The latency budget sets the target via Little's
    # Law; the floor is swept too, because a real operator would configure one
    # and omitting it would understate the baseline.
    # The floor range deliberately reaches past the trace's mean requirement
    # (~34 instances) and towards its peak (~111), so that if this policy can
    # only hit a tight SLO by degenerating into static overprovisioning, the
    # sweep is wide enough to show that rather than hide it.
    for latency in (15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 300.0, 600.0):
        for floor in (0, 20, 40, 60, 80):
            specs.append(
                SweepSpec(
                    policy=BacklogPerInstancePolicy(
                        acceptable_latency_s=latency,
                        service_seconds=SERVICE_SECONDS,
                        min_instances=floor,
                    ),
                    label="backlog_per_instance",
                    params={
                        "knob": latency,
                        "config": f"budget={latency:g}s, floor={floor}",
                    },
                )
            )

    # 3. Arrival rate tracking. Headroom and averaging window are the two knobs
    # that matter; the backlog drain term is held fixed so the sweep stays
    # legible.
    # Utilisation reaches down to 0.3 so the frontier is not truncated at its
    # safe end. This is the baseline the predictive policy has to beat in
    # Phase 4, and capping its headroom would hand that comparison an
    # advantage it did not earn.
    for utilization in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for window in (3, 5, 10, 15):
            specs.append(
                SweepSpec(
                    policy=ArrivalRateTargetPolicy(
                        service_seconds=SERVICE_SECONDS,
                        target_utilization=utilization,
                        window_minutes=window,
                        backlog_drain_seconds=300.0,
                    ),
                    label="arrival_rate",
                    params={
                        "knob": utilization,
                        "config": f"util={utilization:g}, window={window}m",
                    },
                )
            )

    return specs


def plot_frontiers(table: pd.DataFrame, path: Path) -> None:
    """Cost against each SLO measure, one panel per measure.

    Faint dots are every configuration evaluated; the line joins only the
    Pareto-efficient ones. Showing both matters: the line is what a policy can
    achieve when tuned well, and the scatter shows how much of its parameter
    space is wasted, which is a real operational cost of a fiddly policy.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)

    panels = [
        ("slo_violation_fraction", f"Ticks with oldest message > {SLO_AGE_SECONDS:g}s", True),
        ("p99_age_s", "p99 age of oldest message (s)", False),
    ]

    for axis, (column, ylabel, as_percent) in zip(axes, panels):
        axis.set_facecolor(SURFACE)
        # Clip to the efficient region. Badly tuned configurations run to
        # several times the cost of anything worth choosing, and letting them
        # set the axis would squeeze every curve that matters into the left
        # third of the panel.
        frontiers = [
            pareto_frontier(table[table["policy"] == policy], objective=column)[
                "cost_instance_hours"
            ]
            for policy in POLICY_STYLE
            if not table[table["policy"] == policy].empty
        ]
        axis.set_xlim(
            left=min(costs.min() for costs in frontiers) * 0.92,
            right=max(costs.max() for costs in frontiers) * 1.06,
        )

        for policy, style in POLICY_STYLE.items():
            points = table[table["policy"] == policy]
            if points.empty:
                continue

            scale = 100.0 if as_percent else 1.0
            axis.scatter(
                points["cost_instance_hours"],
                points[column] * scale,
                s=14,
                color=style["color"],
                alpha=0.22,
                linewidths=0,
                zorder=2,
            )

            frontier = pareto_frontier(points, objective=column)
            axis.plot(
                frontier["cost_instance_hours"],
                frontier[column] * scale,
                color=style["color"],
                marker=style["marker"],
                markersize=5,
                linewidth=2,
                label=style["label"],
                zorder=3,
            )

        axis.set_xlabel("Cost (instance-hours)", color=TEXT_SECONDARY, fontsize=10)
        axis.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
        axis.set_yscale("symlog", linthresh=1.0)
        axis.grid(True, color=GRID, linewidth=0.6, zorder=1)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(GRID)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    axes[0].set_title(
        "SLO violation rate vs cost", color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10
    )
    axes[1].set_title(
        "Tail latency vs cost", color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10
    )
    axes[0].legend(
        frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper right"
    )

    figure.suptitle(
        "Reactive baselines: cost/SLO frontiers  "
        f"(lower-left is better · {SERVICE_SECONDS:g}s service, "
        f"{experiment_config().boot_time_s:g}s boot)",
        color=TEXT_PRIMARY,
        fontsize=12.5,
        x=0.012,
        ha="left",
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def summarise(table: pd.DataFrame, path: Path) -> str:
    """Cheapest configuration of each policy that meets a range of SLO targets."""
    lines = [
        "# Phase 3 — reactive baseline frontiers",
        "",
        f"Training split, {SERVICE_SECONDS:g}s mean service time, "
        f"{experiment_config().boot_time_s:g}s boot delay, "
        f"{SLO_AGE_SECONDS:g}s SLO on age of oldest message, "
        f"{WARMUP_SECONDS / 3600:g}h warmup excluded.",
        "",
        "Cheapest configuration of each policy reaching a given SLO violation rate.",
        "A dash means no configuration in the sweep reached that target.",
        "",
        "| Target violation rate | Policy | Cost (inst-h) | Actual violation | p99 age | Configuration |",
        "|---|---|---|---|---|---|",
    ]

    for target in (0.10, 0.05, 0.01, 0.001, 0.0):
        for policy, style in POLICY_STYLE.items():
            candidates = table[
                (table["policy"] == policy)
                & (table["slo_violation_fraction"] <= target)
                # A policy that never drains is not a solution at any price.
                & (table["completion_fraction"] > 0.999)
            ]
            if candidates.empty:
                lines.append(
                    f"| ≤ {target:.1%} | {style['label']} | — | — | — | — |"
                )
                continue

            best = candidates.loc[candidates["cost_instance_hours"].idxmin()]
            lines.append(
                f"| ≤ {target:.1%} | {style['label']} | "
                f"{best['cost_instance_hours']:,.0f} | "
                f"{best['slo_violation_fraction']:.2%} | "
                f"{best['p99_age_s']:.1f}s | {best['config']} |"
            )
        lines.append("| | | | | | |")

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    train = load_processed("train")
    specs = build_specs()
    print(f"Sweeping {len(specs)} configurations over {len(train):,} minutes of trace…")

    table = run_sweep(
        train,
        specs,
        experiment_config(),
        slo_threshold_s=SLO_AGE_SECONDS,
        warmup_s=WARMUP_SECONDS,
    )

    table.to_csv(RESULTS_DIR / "phase3_baselines.csv", index=False)
    plot_frontiers(table, RESULTS_DIR / "phase3_frontier.png")
    print(summarise(table, RESULTS_DIR / "phase3_summary.md"))
    print(f"Wrote {RESULTS_DIR / 'phase3_baselines.csv'}")
    print(f"Wrote {RESULTS_DIR / 'phase3_frontier.png'}")


if __name__ == "__main__":
    main()
