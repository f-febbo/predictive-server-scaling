"""Export the trace slice and forecast table the Lambdas ship with.

Run with:

    uv run python -m experiments.export_infra_payload

Writes:
    infra/lambda/replay_trace.json    arrivals per minute, at original scale
    infra/lambda/forecast_table.json  LightGBM quantile forecast, same minutes

Both are written at the *original* scale. The scale-down factor lives in
Terraform (`arrival_divisor`) and is applied by the Lambdas at runtime, so the
fleet size can be retuned without regenerating the payload.

The forecast values are the genuine out-of-sample output of the Phase 4
rolling-origin backtest: the value stored for minute `m` was computed only from
data at or before `m`, and predicts `m + horizon`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.pipeline import load_processed

LAMBDA_DIR = Path("infra/lambda")
FORECAST_PARQUET = Path("data/processed/forecasts_lightgbm.parquet")

# 48 hours at one-minute resolution. Long enough to cover two full daily
# cycles, which is what makes the predictive advantage visible at all.
REPLAY_MINUTES = 2880
QUANTILE = 0.9


def main() -> None:
    LAMBDA_DIR.mkdir(parents=True, exist_ok=True)

    test = load_processed("test")
    if not FORECAST_PARQUET.exists():
        raise FileNotFoundError(
            f"{FORECAST_PARQUET} not found; run "
            "`python -m experiments.phase4_backtest` first"
        )

    forecasts = pd.read_parquet(FORECAST_PARQUET)
    forecasts.columns = [float(column) for column in forecasts.columns]
    if QUANTILE not in forecasts.columns:
        raise KeyError(f"quantile {QUANTILE} not in {sorted(forecasts.columns)}")

    # Both series must describe the same minutes, or the scaler would be
    # provisioning for one moment while the generator sends another.
    aligned = pd.DataFrame({"arrivals": test, "forecast": forecasts[QUANTILE]}).dropna(
        subset=["arrivals"]
    )
    window = aligned.iloc[:REPLAY_MINUTES]

    _write(
        LAMBDA_DIR / "replay_trace.json",
        values=[float(value) for value in window["arrivals"]],
        description="Arrivals per minute from the held-out split, original scale.",
        start=str(window.index[0]),
    )
    _write(
        LAMBDA_DIR / "forecast_table.json",
        values=[
            None if pd.isna(value) else float(value) for value in window["forecast"]
        ],
        description=(
            f"LightGBM q{QUANTILE} out-of-sample forecast; entry m predicts "
            "m + horizon."
        ),
        start=str(window.index[0]),
    )

    total = window["arrivals"].sum()
    print(
        f"Wrote {len(window):,} minutes starting {window.index[0]}\n"
        f"  arrivals: mean {window['arrivals'].mean():.1f}/min, "
        f"peak {window['arrivals'].max():.0f}/min, {total:,.0f} total\n"
        f"  forecast coverage: "
        f"{window['forecast'].notna().mean():.1%} of minutes"
    )


def _write(path: Path, values: list, description: str, start: str) -> None:
    path.write_text(
        json.dumps(
            {"description": description, "start": start, "values": values}, indent=1
        ),
        encoding="utf-8",
    )
    print(f"  {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
