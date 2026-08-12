"""Exploratory analysis of the arrival trace.

Produces the Phase 1 figure and the summary statistics that establish the trace
is bursty enough for the benchmark to mean anything.

Reads the *training* series only. The holdout is not opened until Phase 4, so
nothing seen here can influence model or policy design.

Usage:
    uv run python experiments/eda.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.pipeline import load_processed
from src.data.stats import format_summary, summarize_arrivals
from src.eval.plots import plot_eda

RESULTS_DIR = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zoom-start",
        default=None,
        help="start of the minute-level detail panel, e.g. 2024-01-15",
    )
    parser.add_argument("--zoom-days", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = load_processed("train")

    summary = summarize_arrivals(train)
    report = format_summary(summary, title="Arrival trace: training window")
    print(report)

    stats_path = RESULTS_DIR / "phase1_arrival_stats.txt"
    stats_path.write_text(report + "\n")

    figure_path = plot_eda(
        train,
        RESULTS_DIR / "phase1_eda.png",
        zoom_start=args.zoom_start,
        zoom_days=args.zoom_days,
    )

    print(f"\nWrote {stats_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
