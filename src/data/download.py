"""Fetch NYC TLC yellow taxi trip records.

Each trip's pickup timestamp is treated as one job arrival. The trace is used
instead of a synthetic arrival process on purpose: a sine wave plus noise is
trivially forecastable, which would make the whole predictive-versus-reactive
comparison vacuous. Real taxi demand has strong daily and weekly seasonality
*and* genuine irregular bursts, which is the structure the benchmark needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import requests

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# The download is one large streamed file; allow a generous read timeout but
# fail fast if the connection itself cannot be established.
CONNECT_TIMEOUT_S = 15
READ_TIMEOUT_S = 120
CHUNK_BYTES = 1 << 20


def tlc_url(year: int, month: int, dataset: str = "yellow") -> str:
    """Build the public parquet URL for one month of trip records."""
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    return f"{TLC_BASE_URL}/{dataset}_tripdata_{year:04d}-{month:02d}.parquet"


def months_range(start: str, end: str) -> list[tuple[int, int]]:
    """Enumerate ``(year, month)`` pairs from `start` to `end`, both inclusive.

    Args:
        start: First month as ``"YYYY-MM"``.
        end: Last month as ``"YYYY-MM"``.
    """
    start_year, start_month = _parse_month(start)
    end_year, end_month = _parse_month(end)

    first = start_year * 12 + (start_month - 1)
    last = end_year * 12 + (end_month - 1)
    if last < first:
        raise ValueError(f"end ({end}) is before start ({start})")

    return [(index // 12, index % 12 + 1) for index in range(first, last + 1)]


def download_month(
    year: int,
    month: int,
    dest_dir: Path | str,
    dataset: str = "yellow",
    force: bool = False,
    fetch: Callable[[str, Path], None] | None = None,
) -> Path:
    """Download one month of trip records, reusing an existing copy if present.

    Args:
        year: Four-digit year.
        month: Month, 1..12.
        dest_dir: Directory to write into; created if missing.
        dataset: TLC dataset name, e.g. ``"yellow"``.
        force: Re-download even when the file is already on disk.
        fetch: Injection point for the transfer itself, used by the tests.

    Returns:
        Path to the parquet file on disk.
    """
    fetch = fetch or _http_download
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    url = tlc_url(year, month, dataset)
    destination = dest_dir / url.rsplit("/", 1)[-1]

    if destination.exists() and not force:
        return destination

    # Download to a temporary name and rename only on success. A partially
    # written file left under the real name would look like a valid cache hit
    # to every later run.
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        fetch(url, partial)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    partial.replace(destination)
    return destination


def download_months(
    start: str,
    end: str,
    dest_dir: Path | str,
    dataset: str = "yellow",
    force: bool = False,
) -> list[Path]:
    """Download every month in ``[start, end]`` and return the file paths."""
    return [
        download_month(year, month, dest_dir, dataset=dataset, force=force)
        for year, month in months_range(start, end)
    ]


def _parse_month(value: str) -> tuple[int, int]:
    year, _, month = value.partition("-")
    if not month:
        raise ValueError(f"expected 'YYYY-MM', got {value!r}")
    return int(year), int(month)


def _http_download(url: str, dest: Path) -> None:
    """Stream a URL to disk without holding the whole file in memory."""
    with requests.get(
        url, stream=True, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
    ) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                handle.write(chunk)
