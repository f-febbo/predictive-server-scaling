"""Assemble the raw TLC trip files into the processed arrival series.

Pipeline: download monthly parquet -> read pickup timestamps -> resample to a
gapless 1-minute count series -> split chronologically -> write to
`data/processed/`.

The test split happens here, at the edge of the data layer, so that everything
downstream receives an already-separated train set and the holdout is only ever
opened deliberately.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.download import download_months
from src.data.resample import resample_arrivals
from src.data.split import split_by_time

# Yellow and green TLC datasets name the same field differently.
PICKUP_COLUMNS = ("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime")

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def month_window(start_month: str, end_month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Half-open window covering whole months from `start_month` to `end_month`.

    ``("2024-01", "2024-02")`` yields ``2024-01-01`` to ``2024-03-01``, so the
    whole of February is included but no part of March.
    """
    start = pd.Timestamp(f"{start_month}-01")
    end = pd.Timestamp(f"{end_month}-01") + pd.offsets.MonthBegin(1)
    return start, end


def load_pickup_timestamps(path: Path | str) -> pd.Series:
    """Read just the pickup-timestamp column from one trip-records parquet.

    Only the one column is read: the full file carries twenty-odd fields we
    never use, and reading them all costs memory for nothing.
    """
    available = pd.read_parquet(path, engine="pyarrow").columns
    column = next((name for name in PICKUP_COLUMNS if name in available), None)
    if column is None:
        raise ValueError(
            f"no pickup timestamp column in {path}; looked for {PICKUP_COLUMNS}, "
            f"found {list(available)}"
        )

    frame = pd.read_parquet(path, columns=[column], engine="pyarrow")
    return pd.to_datetime(frame[column])


def build_arrival_series(
    paths: list[Path | str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    freq: str = "1min",
) -> pd.Series:
    """Combine trip files into one gapless arrival-count series over the window."""
    if not paths:
        raise ValueError("no input files given")

    pickups = pd.concat([load_pickup_timestamps(path) for path in paths], ignore_index=True)
    return resample_arrivals(pickups, freq=freq, start=start, end=end)


def run_pipeline(
    start_month: str = "2024-01",
    end_month: str = "2024-02",
    freq: str = "1min",
    test_fraction: float = 0.2,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    dataset: str = "yellow",
    force_download: bool = False,
) -> dict[str, Path]:
    """Download, resample, split, and persist the arrival series.

    Returns:
        Mapping of ``full``/``train``/``test`` to the parquet paths written.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset} trip records for {start_month}..{end_month}")
    paths = download_months(
        start_month, end_month, raw_dir, dataset=dataset, force=force_download
    )
    for path in paths:
        print(f"  {path} ({path.stat().st_size / 1e6:.0f} MB)")

    start, end = month_window(start_month, end_month)
    print(f"Resampling to {freq} arrival counts over [{start}, {end})")
    arrivals = build_arrival_series(paths, start=start, end=end, freq=freq)

    train, test = split_by_time(arrivals, test_fraction=test_fraction)
    print(
        f"Chronological split at {test.index.min()}: "
        f"{len(train):,} train minutes, {len(test):,} held-out minutes"
    )

    written = {
        "full": processed_dir / "arrivals.parquet",
        "train": processed_dir / "arrivals_train.parquet",
        "test": processed_dir / "arrivals_test.parquet",
    }
    arrivals.to_frame().to_parquet(written["full"])
    train.to_frame().to_parquet(written["train"])
    test.to_frame().to_parquet(written["test"])

    return written


def load_processed(
    which: str = "train", processed_dir: Path | str = DEFAULT_PROCESSED_DIR
) -> pd.Series:
    """Read back a persisted series.

    Args:
        which: One of ``full``, ``train``, or ``test``. The holdout (``test``)
            is not to be opened before Phase 4.
    """
    suffix = "" if which == "full" else f"_{which}"
    path = Path(processed_dir) / f"arrivals{suffix}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `make data` (or python -m experiments.build_dataset) first"
        )
    return pd.read_parquet(path)["arrivals"]
