"""Chronological train/test split.

A random split is invalid here. Adjacent minutes are highly correlated, so
scattering test rows among training rows lets the model interpolate between
neighbours it has already seen and reports a score it could never achieve in
production. The holdout must be a single contiguous block at the *end* of the
timeline, which is the only arrangement that matches how the model would
actually be used: fit on the past, predict the future.
"""

from __future__ import annotations

import pandas as pd


def split_by_time(
    series: pd.Series, test_fraction: float = 0.2
) -> tuple[pd.Series, pd.Series]:
    """Split a regular time series into an earlier train part and a later test part.

    The boundary is a *point in time* at `1 - test_fraction` of the total span,
    not a row position, and the boundary bucket itself belongs to the test set.

    Args:
        series: Time-indexed series with a regular, gapless index.
        test_fraction: Share of the timeline to hold out, strictly between 0 and 1.

    Returns:
        ``(train, test)``. Concatenating them reproduces the sorted input exactly.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    ordered = series.sort_index()
    if len(ordered) < 2:
        raise ValueError(f"need at least 2 observations to split, got {len(ordered)}")

    period = _regular_period(ordered.index)

    # The span runs to the *end* of the final bucket, not its start, so that a
    # 20% holdout of ten one-minute buckets is exactly the last two buckets.
    span_start = ordered.index[0]
    span_end = ordered.index[-1] + period
    boundary = span_start + (span_end - span_start) * (1.0 - test_fraction)

    train = ordered[ordered.index < boundary]
    test = ordered[ordered.index >= boundary]

    if train.empty or test.empty:
        raise ValueError(
            f"test_fraction={test_fraction} leaves an empty side for a series of "
            f"length {len(ordered)}"
        )

    return train, test


def _regular_period(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Return the index's sampling period, rejecting irregular indices.

    Splitting a gapped series on a fraction of its *span* would not correspond
    to a fraction of its *data*, so the two readings of "hold out 20%" would
    disagree. Rather than silently pick one, refuse the input.
    """
    gaps = index.to_series().diff().dropna().unique()
    if len(gaps) != 1:
        raise ValueError(
            "index must be regular and gapless; found "
            f"{len(gaps)} distinct sampling intervals"
        )
    return pd.Timedelta(gaps[0])
