"""Summary statistics that characterise the arrival process.

The project's benchmark is only meaningful if the traffic is genuinely bursty
and irregular. Smooth periodic traffic makes forecasting trivial and the
comparison against a reactive baseline vacuous. These statistics are the
evidence that the chosen trace clears that bar, so they are reported up front
rather than buried.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# Lags of interest for a 1-minute series, in periods.
DEFAULT_LAGS: dict[str, int] = {
    "1min": 1,
    "1hour": 60,
    "1day": 60 * 24,
    "1week": 60 * 24 * 7,
}


def index_of_dispersion(series: pd.Series) -> float:
    """Variance-to-mean ratio (the Fano factor) of the counts.

    A Poisson process has an index of exactly 1. Values materially above 1 mean
    the arrivals clump into bursts more than pure randomness would produce,
    which is precisely the regime where a lagging reactive scaler struggles and
    the project's thesis has something to prove.

    Returns NaN for an all-zero series, where the ratio is undefined.
    """
    mean = series.mean()
    if mean == 0:
        return float("nan")
    return float(series.var(ddof=1) / mean)


def seasonal_expectation(series: pd.Series) -> pd.Series:
    """The hour-of-week seasonal mean, broadcast back onto the series index.

    This is exactly what a seasonal-naive forecaster predicts, so the gap
    between it and the actual series is the part of the load that such a
    forecaster cannot anticipate.
    """
    hour_of_week = series.index.dayofweek * 24 + series.index.hour
    profile = series.groupby(hour_of_week).mean()
    return pd.Series(profile.loc[hour_of_week].to_numpy(), index=series.index)


def residual_index_of_dispersion(series: pd.Series) -> float:
    """Variance-to-mean ratio of what the weekly seasonal shape fails to explain.

    The plain `index_of_dispersion` counts the daily rush-hour swing as
    burstiness. That swing is entirely predictable, so crediting it overstates
    how hard the traffic is to scale for. Removing the seasonal mean first
    leaves the genuinely unanticipated fluctuation, measured on the same scale:
    1.0 is Poisson, higher means bursty beyond pure randomness.
    """
    mean = series.mean()
    if mean == 0:
        return float("nan")

    residual = series - seasonal_expectation(series)
    return float(residual.var(ddof=1) / mean)


def autocorrelation_at_lags(
    series: pd.Series, lags: dict[str, int] | None = None
) -> dict[str, float]:
    """Pearson correlation of the series with itself shifted by each lag.

    Args:
        series: The arrival-count series.
        lags: Mapping of label to lag in periods. Defaults to `DEFAULT_LAGS`.

    Returns:
        Mapping of the same labels to correlations. A lag too long for the
        series yields NaN rather than a value computed from a few overlapping
        points, which would look like a real seasonality signal but be noise.
    """
    lags = DEFAULT_LAGS if lags is None else lags

    results: dict[str, float] = {}
    for label, lag in lags.items():
        # Require a decent overlap, not merely a non-empty one: a correlation
        # from two surviving points is meaningless but looks authoritative.
        if lag >= len(series) - 1:
            results[label] = float("nan")
        else:
            results[label] = _correlation_with_lag(series, lag)
    return results


def _correlation_with_lag(series: pd.Series, lag: int) -> float:
    """Correlate a series against its own lagged copy.

    Either side having zero variance makes the correlation genuinely undefined
    rather than merely hard to compute, so return NaN up front instead of
    letting the division by a zero standard deviation raise a runtime warning.
    """
    current = series.iloc[lag:]
    lagged = series.iloc[:-lag]

    if current.std(ddof=1) == 0 or lagged.std(ddof=1) == 0:
        return float("nan")

    return float(np.corrcoef(current.to_numpy(), lagged.to_numpy())[0, 1])


def summarize_arrivals(
    series: pd.Series, lags: dict[str, int] | None = None
) -> dict:
    """Describe an arrival series: location, spread, burstiness, and memory.

    Units are arrivals per bucket, which for the project's 1-minute series is
    arrivals per minute.
    """
    return {
        "n_observations": int(len(series)),
        "start": series.index.min(),
        "end": series.index.max(),
        "total_arrivals": int(series.sum()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=1)),
        "min": int(series.min()),
        "p99": float(np.percentile(series, 99)),
        "max": int(series.max()),
        "index_of_dispersion": index_of_dispersion(series),
        "residual_index_of_dispersion": residual_index_of_dispersion(series),
        "autocorrelation": autocorrelation_at_lags(series, lags),
    }


def daily_profile(
    series: pd.Series,
    resolution: str = "minute",
    agg: Callable[[pd.core.groupby.SeriesGroupBy], pd.Series] | None = None,
) -> pd.Series:
    """Arrival rate by position within the day, averaged across days.

    Args:
        series: 1-minute arrival counts.
        resolution: ``"minute"`` for a 1440-point profile, ``"hour"`` for 24.
        agg: How to combine the values that share a position in the day.
            Defaults to the mean; pass a quantile to draw a spread band.

    Returns:
        Arrivals per minute at each position in the day, indexed by
        minute-of-day (0..1439) or hour-of-day (0..23).
    """
    index = series.index
    if resolution == "minute":
        position = index.hour * 60 + index.minute
        full_range = range(1440)
    elif resolution == "hour":
        position = index.hour
        full_range = range(24)
    else:
        raise ValueError(f"resolution must be 'minute' or 'hour', got {resolution!r}")

    agg = agg or (lambda group: group.mean())
    profile = agg(series.groupby(position))
    profile = profile.reindex(full_range, fill_value=0.0)
    profile.index.name = f"{resolution}_of_day"
    return profile


def weekly_profile(series: pd.Series) -> pd.Series:
    """Average arrival rate by hour of the week.

    Returns:
        168 points indexed by hour-of-week, where 0 is Monday 00:00. Values are
        arrivals per minute, so they are directly comparable with the daily
        profile rather than being 60x larger.
    """
    index = series.index
    hour_of_week = index.dayofweek * 24 + index.hour

    profile = series.groupby(hour_of_week).mean()
    profile = profile.reindex(range(168), fill_value=0.0)
    profile.index.name = "hour_of_week"
    return profile


def format_summary(summary: dict, title: str = "Arrival series summary") -> str:
    """Render a summary as plain text for the console and for `results/`."""
    lines = [
        title,
        "=" * len(title),
        f"window            : {summary['start']} .. {summary['end']}",
        f"observations      : {summary['n_observations']:,} minutes",
        f"total arrivals    : {summary['total_arrivals']:,}",
        "",
        "Arrival rate (arrivals per minute)",
        f"  mean            : {summary['mean']:.2f}",
        f"  median          : {summary['median']:.2f}",
        f"  std dev         : {summary['std']:.2f}",
        f"  min             : {summary['min']}",
        f"  p99             : {summary['p99']:.2f}",
        f"  max             : {summary['max']}",
        "",
        "Burstiness  (Poisson = 1.0; higher means burstier)",
        f"  raw index of dispersion      : {summary['index_of_dispersion']:.2f}",
        f"  seasonally-adjusted          : {summary['residual_index_of_dispersion']:.2f}",
        "  The raw figure counts the predictable daily swing as burstiness. The",
        "  adjusted figure removes the hour-of-week seasonal mean first, so it",
        "  reflects only what a seasonal forecaster cannot anticipate.",
        "",
        "Autocorrelation",
    ]
    for label, value in summary["autocorrelation"].items():
        shown = "n/a (lag exceeds series)" if np.isnan(value) else f"{value:+.3f}"
        lines.append(f"  lag {label:<6}      : {shown}")
    return "\n".join(lines)
