"""Phase 4b: predictive vs reactive on the held-out period.

Run with:

    uv run python -m experiments.phase4_backtest    # writes the forecasts
    uv run python -m experiments.phase4_frontier

Produces:
    results/phase4_frontier.csv       every configuration evaluated
    results/phase4_frontier.png       the headline figure
    results/phase4_summary.md         headline table

The baselines are re-run here on the held-out period rather than reusing the
Phase 3 numbers, which were measured on training data. Comparing a predictive
policy scored on one period against a reactive policy scored on another would
be meaningless, however well tuned both were.

Sweeping the forecast quantile is what traces the predictive frontier. A single
quantile would give one point and no way to see where prediction actually wins.
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
from src.policies.forecast_policy import ForecastPolicy, forecast_array
from src.policies.static import StaticPolicy

RESULTS_DIR = Path("results")
PROCESSED_DIR = Path("data/processed")

FORECAST_MODELS = ("seasonal_naive", "seasonal_naive_adjusted", "lightgbm")
QUANTILES = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
UTILISATIONS = (0.7, 0.85, 1.0)

# The best-tuned reactive policy from Phase 3, used as the floor under the
# composed variant: prediction sets the minimum, target tracking handles the
# residual the forecast missed.
def reactive_floor() -> ArrivalRateTargetPolicy:
    return ArrivalRateTargetPolicy(
        service_seconds=SERVICE_SECONDS,
        target_utilization=0.8,
        window_minutes=10,
        backlog_drain_seconds=300.0,
    )


SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dcdbd6"

# Validated categorical slots, assigned in fixed order. Distinct markers give
# the secondary encoding the low-contrast slots require.
STYLE = {
    "static": ("#2a78d6", "o", "Static"),
    "backlog_per_instance": ("#eb6834", "s", "Reactive: backlog/instance"),
    "arrival_rate": ("#1baf7a", "^", "Reactive: arrival rate"),
    "forecast_lightgbm_composed": ("#eda100", "D", "Predictive: LightGBM + reactive"),
    "forecast_lightgbm_pure": ("#e87ba4", "v", "Predictive: LightGBM alone"),
    "forecast_seasonal_naive_composed": ("#2a78d6", "o", "Seasonal naive + reactive"),
    "forecast_seasonal_naive_adjusted_composed": (
        "#eb6834",
        "s",
        "Seasonal naive adj. + reactive",
    ),
}

HEADLINE_PANEL = (
    "static",
    "backlog_per_instance",
    "arrival_rate",
    "forecast_lightgbm_composed",
    "forecast_lightgbm_pure",
)
FORECASTER_PANEL = (
    "arrival_rate",
    "forecast_seasonal_naive_composed",
    "forecast_seasonal_naive_adjusted_composed",
    "forecast_lightgbm_composed",
)


def load_forecasts(name: str, trace_index: pd.DatetimeIndex) -> dict[float, "pd.Series"]:
    """Read persisted backtest forecasts, keyed by quantile."""
    path = PROCESSED_DIR / f"forecasts_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m experiments.phase4_backtest` first"
        )

    frame = pd.read_parquet(path)
    # Parquet round-trips column labels as strings.
    return {float(column): frame[column] for column in frame.columns}


def build_specs(trace_index: pd.DatetimeIndex) -> list[SweepSpec]:
    specs: list[SweepSpec] = []

    for instances in range(25, 135, 5):
        specs.append(
            SweepSpec(
                policy=StaticPolicy(instances),
                label="static",
                params={"knob": float(instances), "config": f"n={instances}"},
            )
        )

    for latency in (15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 300.0, 600.0):
        for floor in (0, 20, 40, 60, 80):
            specs.append(
                SweepSpec(
                    policy=BacklogPerInstancePolicy(
                        latency, SERVICE_SECONDS, min_instances=floor
                    ),
                    label="backlog_per_instance",
                    params={
                        "knob": latency,
                        "config": f"budget={latency:g}s, floor={floor}",
                    },
                )
            )

    for utilization in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for window in (3, 5, 10, 15):
            specs.append(
                SweepSpec(
                    policy=ArrivalRateTargetPolicy(
                        SERVICE_SECONDS, utilization, window, backlog_drain_seconds=300.0
                    ),
                    label="arrival_rate",
                    params={
                        "knob": utilization,
                        "config": f"util={utilization:g}, window={window}m",
                    },
                )
            )

    for model in FORECAST_MODELS:
        forecasts = load_forecasts(model, trace_index)
        for quantile in QUANTILES:
            if quantile not in forecasts:
                continue
            values = forecast_array(forecasts[quantile], trace_index)

            for utilization in UTILISATIONS:
                for variant, floor in (("pure", None), ("composed", reactive_floor())):
                    specs.append(
                        SweepSpec(
                            policy=ForecastPolicy(
                                values,
                                service_seconds=SERVICE_SECONDS,
                                target_utilization=utilization,
                                reactive_floor=floor,
                                min_instances=1,
                            ),
                            label=f"forecast_{model}_{variant}",
                            params={
                                "knob": quantile,
                                "config": f"q={quantile:g}, util={utilization:g}",
                            },
                        )
                    )

    return specs


def plot_frontiers(table: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor=SURFACE)

    panels = (
        (axes[0], HEADLINE_PANEL, "Predictive vs reactive"),
        (axes[1], FORECASTER_PANEL, "Does the forecaster matter?"),
    )

    for axis, families, title in panels:
        axis.set_facecolor(SURFACE)
        frontier_costs = []

        for family in families:
            points = table[table["policy"] == family]
            if points.empty:
                continue

            color, marker, label = STYLE[family]
            frontier = pareto_frontier(points)
            frontier_costs.append(frontier["cost_instance_hours"])

            axis.scatter(
                points["cost_instance_hours"],
                points["slo_violation_fraction"] * 100,
                s=12,
                color=color,
                alpha=0.2,
                linewidths=0,
                zorder=2,
            )
            axis.plot(
                frontier["cost_instance_hours"],
                frontier["slo_violation_fraction"] * 100,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=2,
                label=label,
                zorder=3,
            )

        if frontier_costs:
            axis.set_xlim(
                left=min(costs.min() for costs in frontier_costs) * 0.92,
                right=max(costs.max() for costs in frontier_costs) * 1.06,
            )

        # symlog so that exact zeros are plottable, but clipped at zero: a
        # violation rate is never negative, and letting the scale render its
        # negative half would throw away half the panel.
        axis.set_yscale("symlog", linthresh=0.05)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Cost (instance-hours)", color=TEXT_SECONDARY, fontsize=10)
        axis.set_ylabel(
            f"% of ticks with oldest message > {SLO_AGE_SECONDS:g}s",
            color=TEXT_SECONDARY,
            fontsize=10,
        )
        axis.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
        axis.grid(True, color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(GRID)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        axis.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY)

    figure.suptitle(
        "Cost/SLO frontiers on the held-out period  (lower-left is better)",
        color=TEXT_PRIMARY,
        fontsize=12.5,
        x=0.012,
        ha="left",
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def summarise(table: pd.DataFrame, path: Path) -> str:
    lines = [
        "# Phase 4 — predictive vs reactive on the held-out period",
        "",
        f"{SERVICE_SECONDS:g}s service, {experiment_config().boot_time_s:g}s boot, "
        f"{SLO_AGE_SECONDS:g}s SLO, {WARMUP_SECONDS / 3600:g}h warmup excluded.",
        "",
        "Cheapest configuration of each policy reaching a given SLO violation rate.",
        "",
        "| Target | Policy | Cost (inst-h) | Actual | p99 age | Configuration |",
        "|---|---|---|---|---|---|",
    ]

    families = list(dict.fromkeys(list(HEADLINE_PANEL) + list(FORECASTER_PANEL)))
    for target in (0.05, 0.01, 0.001, 0.0):
        for family in families:
            candidates = table[
                (table["policy"] == family)
                & (table["slo_violation_fraction"] <= target)
                & (table["completion_fraction"] > 0.999)
            ]
            label = STYLE[family][2]
            if candidates.empty:
                lines.append(f"| ≤ {target:.1%} | {label} | — | — | — | — |")
                continue

            best = candidates.loc[candidates["cost_instance_hours"].idxmin()]
            lines.append(
                f"| ≤ {target:.1%} | {label} | {best['cost_instance_hours']:,.0f} | "
                f"{best['slo_violation_fraction']:.2%} | {best['p99_age_s']:.1f}s | "
                f"{best['config']} |"
            )
        lines.append("| | | | | | |")

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    test = load_processed("test")
    print(f"Held-out period: {len(test):,} minutes from {test.index[0]}")

    specs = build_specs(test.index)
    print(f"Sweeping {len(specs)} configurations…")

    table = run_sweep(
        test,
        specs,
        experiment_config(),
        slo_threshold_s=SLO_AGE_SECONDS,
        warmup_s=WARMUP_SECONDS,
    )

    table.to_csv(RESULTS_DIR / "phase4_frontier.csv", index=False)
    plot_frontiers(table, RESULTS_DIR / "phase4_frontier.png")
    print(summarise(table, RESULTS_DIR / "phase4_summary.md"))
    print(f"Wrote {RESULTS_DIR / 'phase4_frontier.png'}")


if __name__ == "__main__":
    main()
