"""Expand per-minute trace counts into individual arrival times.

The trace supplies a count per minute; the simulator needs individual events.
Something must therefore decide where inside each minute the arrivals land, and
that choice is a modelling decision, not an implementation detail.

Two modes are offered:

``poisson`` (default)
    Place the minute's arrivals uniformly at random within the minute. This is
    not an arbitrary choice: conditioned on observing N events in an interval,
    a Poisson process places those events exactly this way. It is the
    maximum-entropy placement consistent with the count we actually measured —
    we add no sub-minute structure the trace does not evidence, but we do keep
    the random clustering a real arrival stream has.

``uniform``
    Space the arrivals evenly. Deterministic and convenient for hand-checkable
    tests, but it is an unrealistically benign best case: perfectly regular
    inter-arrival gaps never produce the brief pile-ups that drive message age
    up between scaler ticks.

The default is ``poisson`` because the project's SLI is the age of the oldest
message, which is sensitive to exactly the sub-minute clustering that even
spacing would erase. Reporting queueing latency measured under evenly spaced
arrivals would flatter every policy and understate absolute message age.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_MINUTE = 60.0


def arrival_times_from_counts(
    counts: pd.Series,
    mode: str = "poisson",
    rng: np.random.Generator | None = None,
    bucket_seconds: float = SECONDS_PER_MINUTE,
) -> np.ndarray:
    """Expand bucketed counts into sorted arrival times.

    Args:
        counts: Arrivals per bucket, in trace order.
        mode: ``"poisson"`` for random placement, ``"uniform"`` for even spacing.
        rng: Required for ``"poisson"``; passing it explicitly keeps runs
            reproducible rather than silently seeding from entropy.
        bucket_seconds: Width of one trace bucket.

    Returns:
        Arrival times in seconds from the start of the trace, sorted ascending.
        Exactly ``counts.sum()`` of them.
    """
    if mode not in ("poisson", "uniform"):
        raise ValueError(f"mode must be 'poisson' or 'uniform', got {mode!r}")
    if mode == "poisson" and rng is None:
        raise ValueError("mode='poisson' requires an explicit rng for reproducibility")

    values = np.asarray(counts, dtype=np.int64)
    if (values < 0).any():
        raise ValueError("arrival counts must be non-negative")

    total = int(values.sum())
    if total == 0:
        return np.empty(0, dtype=np.float64)

    # Start of the bucket each arrival belongs to.
    bucket_starts = np.repeat(np.arange(len(values)), values) * bucket_seconds

    if mode == "uniform":
        offsets = _even_offsets(values, bucket_seconds)
    else:
        offsets = rng.uniform(0.0, bucket_seconds, size=total)

    times = bucket_starts + offsets
    # Random offsets are unordered within a bucket; sorting the whole array is
    # simpler than sorting per bucket and costs little.
    times.sort()
    return times


def _even_offsets(values: np.ndarray, bucket_seconds: float) -> np.ndarray:
    """Offsets that divide each bucket into equal slots, one arrival per slot.

    N arrivals sit at the centres of N equal sub-intervals, so they are
    symmetric within the bucket and never land exactly on its boundary.
    """
    nonzero = values[values > 0]
    # Position of each arrival within its own bucket: 0, 1, ... n-1.
    within = np.arange(nonzero.sum()) - np.repeat(
        np.cumsum(nonzero) - nonzero, nonzero
    )
    slots = np.repeat(nonzero, nonzero)
    return (within + 0.5) / slots * bucket_seconds
