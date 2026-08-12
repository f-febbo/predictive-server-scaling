"""Turn a stream of event timestamps into a regular arrival-count series.

The simulator and every forecasting model downstream assume a *gapless* series
at a fixed frequency: bucket `i + 1` is always exactly one period after bucket
`i`. Pandas' default groupby-count silently omits empty buckets, which would
shift every lag feature in time. This module fills those gaps explicitly.
"""

from __future__ import annotations

import pandas as pd


def resample_arrivals(
    timestamps: pd.Series,
    freq: str = "1min",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Count events per `freq` bucket over the half-open window ``[start, end)``.

    Args:
        timestamps: Event times. Nulls are dropped; order does not matter.
        freq: Bucket width, as a pandas offset alias.
        start: Window start, inclusive. Defaults to the first event's bucket.
        end: Window end, *exclusive*. Defaults to one bucket past the last event.

    Returns:
        Integer counts named ``arrivals``, indexed by bucket start (``ts``),
        contiguous at `freq` with no missing buckets.
    """
    events = pd.to_datetime(pd.Series(timestamps, dtype="datetime64[ns]")).dropna()

    start, end = _resolve_window(events, freq, start, end)
    if end <= start:
        raise ValueError(f"end ({end}) must be strictly after start ({start})")

    # Half-open window: an event landing exactly on `end` belongs to the next
    # window, so the two windows can be concatenated without double counting.
    in_window = events[(events >= start) & (events < end)]

    buckets = pd.date_range(start=start, end=end, freq=freq, inclusive="left")
    counts = (
        in_window.dt.floor(freq)
        .value_counts()
        .reindex(buckets, fill_value=0)
        .astype("int64")
        .sort_index()
    )

    counts.index.name = "ts"
    counts.name = "arrivals"
    return counts


def _resolve_window(
    events: pd.Series,
    freq: str,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fill in whichever window bounds the caller left unspecified."""
    start = pd.Timestamp(start) if start is not None else None
    end = pd.Timestamp(end) if end is not None else None

    if (start is None or end is None) and events.empty:
        raise ValueError(
            "cannot infer the window from empty input; pass explicit start and end"
        )

    if start is None:
        start = events.min().floor(freq)
    if end is None:
        end = events.max().floor(freq) + pd.tseries.frequencies.to_offset(freq)

    return start, end
