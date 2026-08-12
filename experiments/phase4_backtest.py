"""Phase 4a: backtest the three forecasters on the held-out period.

Run with:

    uv run python -m experiments.phase4_backtest

Produces:
    results/phase4_backtest_metrics.csv   MAE, pinball loss, coverage per quantile
    results/phase4_backtest.md            the same, formatted
    results/phase4_forecast_vs_actual.png predicted vs actual over a held-out week
    data/processed/forecasts_<model>.parquet  forecasts for the policy sweep

This is the first time the test split is read. Everything up to here — the
simulator, the baselines, their tuning — used training data only, so nothing
downstream has been fitted, even indirectly, to the period it is judged on.

Rolling origin: train on everything before the origin, forecast the week that
follows, advance, retrain. The first origin sits at the train/test boundary, so
every scored forecast is genuinely out of sample.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.pipeline import load_processed
from src.forecast.backtest import rolling_origin_backtest
from src.forecast.lightgbm_quantile import LightGBMQuantile
from src.forecast.seasonal_naive import SeasonalNaive, SeasonalNaiveAdjusted

RESULTS_DIR = Path("results")
PROCESSED_DIR = Path("data/processed")

# Horizon = boot time (180s) plus a margin, so instances requested now are warm
# just before the demand they were requested for arrives.
HORIZON_MINUTES = 15
QUANTILES = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
RETRAIN_DAYS = 7

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dcdbd6"
ACTUAL_COLOR = "#52514e"
MODEL_COLORS = {
    "seasonal_naive": "#2a78d6",
    "seasonal_naive_adjusted": "#eb6834",
    "lightgbm": "#1baf7a",
}


def model_factories() -> dict[str, callable]:
    """Fresh-model builders, in the order the spec asks them to be tried."""
    return {
        "seasonal_naive": lambda: SeasonalNaive(horizon_minutes=HORIZON_MINUTES),
        "seasonal_naive_adjusted": lambda: SeasonalNaiveAdjusted(
            horizon_minutes=HORIZON_MINUTES
        ),
        "lightgbm": lambda: LightGBMQuantile(
            horizon_minutes=HORIZON_MINUTES, quantiles=QUANTILES
        ),
    }


def plot_forecast_vs_actual(
    results: dict, full: pd.Series, path: Path, quantile: float = 0.9
) -> None:
    """A week of held-out actuals with each model's forecast over the top."""
    first_origin = min(result.forecasts.index[0] for result in results.values())
    window_start = first_origin
    window_end = window_start + pd.Timedelta(days=7)

    # Deliberately not sharex: the panels show different spans, and a shared
    # axis would make the second set_xlim silently retarget the first.
    figure, axes = plt.subplots(2, 1, figsize=(13, 7.5), facecolor=SURFACE)

    actual = full.loc[window_start:window_end]
    for axis in axes:
        axis.set_facecolor(SURFACE)
        axis.plot(
            actual.index,
            actual.to_numpy(),
            color=ACTUAL_COLOR,
            linewidth=0.7,
            alpha=0.55,
            label="Actual",
            zorder=2,
        )

    # Top: the whole week, to show seasonal tracking.
    # Bottom: one day, where the sub-hour behaviour is actually visible.
    day_start = window_start + pd.Timedelta(days=2)
    day_end = day_start + pd.Timedelta(days=1)

    for name, result in results.items():
        forecast = result.forecasts[quantile].loc[window_start:window_end]
        for axis in axes:
            axis.plot(
                forecast.index,
                forecast.to_numpy(),
                color=MODEL_COLORS[name],
                linewidth=1.3,
                label=name.replace("_", " "),
                zorder=3,
            )

    axes[0].set_xlim(window_start, window_end)
    axes[1].set_xlim(day_start, day_end)

    for axis in axes:
        axis.grid(True, color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(GRID)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        axis.set_ylabel("Arrivals per minute", color=TEXT_SECONDARY, fontsize=10)

    axes[0].set_title(
        f"Held-out week — q{quantile:g} forecast, {HORIZON_MINUTES} minutes ahead",
        color=TEXT_PRIMARY,
        fontsize=11,
        loc="left",
        pad=8,
    )
    axes[1].set_title(
        "One day, zoomed", color=TEXT_PRIMARY, fontsize=11, loc="left", pad=8
    )
    axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncol=4)

    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train = load_processed("train")
    test = load_processed("test")
    full = pd.concat([train, test]).sort_index()
    train_days = (test.index[0] - train.index[0]).days

    print(
        f"Training period: {train_days} days ending {test.index[0]}\n"
        f"Held-out period: {(full.index[-1] - test.index[0]).days} days "
        f"({len(test):,} minutes)\n"
    )

    results = {}
    metric_frames = []
    for name, factory in model_factories().items():
        print(f"Backtesting {name}…")
        result = rolling_origin_backtest(
            full,
            factory,
            horizon_minutes=HORIZON_MINUTES,
            quantiles=QUANTILES,
            initial_train_days=train_days,
            retrain_days=RETRAIN_DAYS,
        )
        results[name] = result
        metric_frames.append(result.metrics())

        # Persist for the policy sweep so it need not refit.
        result.forecasts.to_parquet(PROCESSED_DIR / f"forecasts_{name}.parquet")

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(RESULTS_DIR / "phase4_backtest_metrics.csv", index=False)

    report = _format_report(metrics)
    (RESULTS_DIR / "phase4_backtest.md").write_text(report, encoding="utf-8")
    print(report)

    plot_forecast_vs_actual(
        results, full, RESULTS_DIR / "phase4_forecast_vs_actual.png"
    )
    print(f"Wrote {RESULTS_DIR / 'phase4_forecast_vs_actual.png'}")


def _format_report(metrics: pd.DataFrame) -> str:
    lines = [
        "# Phase 4 — forecast backtest",
        "",
        f"Rolling origin, retrained every {RETRAIN_DAYS} days, "
        f"{HORIZON_MINUTES}-minute horizon, held-out period only.",
        "",
        "Coverage is the calibration check: a q0.9 forecast should sit above "
        "the actual about 90% of the time.",
        "",
        "| Model | Quantile | MAE | Pinball loss | Coverage | n |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['model']} | {row['quantile']:g} | {row['mae']:.2f} | "
            f"{row['pinball_loss']:.3f} | {row['coverage']:.1%} | {int(row['n']):,} |"
        )

    best = metrics.loc[metrics.groupby("quantile")["pinball_loss"].idxmin()]
    lines += ["", "## Best model by pinball loss at each quantile", ""]
    lines += ["| Quantile | Best model | Pinball loss |", "|---|---|---|"]
    for _, row in best.iterrows():
        lines.append(
            f"| {row['quantile']:g} | {row['model']} | {row['pinball_loss']:.3f} |"
        )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
