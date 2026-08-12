"""Download the NYC TLC trace and build the processed arrival series.

Usage:
    uv run python experiments/build_dataset.py [--start 2024-01] [--end 2024-02]
"""

from __future__ import annotations

import argparse

from src.data.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01", help="first month, YYYY-MM")
    parser.add_argument("--end", default="2024-02", help="last month, YYYY-MM")
    parser.add_argument("--freq", default="1min", help="resampling frequency")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="share of the timeline held out for the final evaluation",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-fetch the raw parquet even if it is already cached",
    )
    args = parser.parse_args()

    written = run_pipeline(
        start_month=args.start,
        end_month=args.end,
        freq=args.freq,
        test_fraction=args.test_fraction,
        force_download=args.force_download,
    )

    print("\nWrote:")
    for name, path in written.items():
        print(f"  {name:<6} {path}")
    print(
        "\nThe holdout (arrivals_test.parquet) stays closed until Phase 4. "
        "Everything before then reads the training series only."
    )


if __name__ == "__main__":
    main()
