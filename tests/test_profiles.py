"""Tests for the seasonal profile computations used by the EDA figure.

These profiles are what show that the trace has the daily and weekly structure
the forecasting models will exploit, so they need to be right.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.stats import daily_profile, weekly_profile


def zero_series(days: int, start: str = "2024-01-01") -> pd.Series:
    """`days` days of zeroed 1-minute buckets starting on a Monday."""
    index = pd.date_range(start=start, periods=days * 24 * 60, freq="1min", name="ts")
    return pd.Series(0, index=index, name="arrivals")


# --- daily profile ----------------------------------------------------------


def test_daily_profile_averages_the_same_minute_across_days():
    # Midnight sees 10 arrivals on day one and 20 on day two -> mean 15.
    series = zero_series(2)
    series.iloc[0] = 10
    series.iloc[24 * 60] = 20

    profile = daily_profile(series)

    assert profile.loc[0] == 15.0


def test_daily_profile_covers_every_minute_of_the_day():
    profile = daily_profile(zero_series(1))

    assert len(profile) == 1440
    assert profile.index[0] == 0
    assert profile.index[-1] == 1439


def test_daily_profile_places_each_minute_at_its_minute_of_day_index():
    series = zero_series(1)
    series.iloc[90] = 7.0  # 01:30

    profile = daily_profile(series)

    assert profile.loc[90] == 7.0
    assert profile.loc[89] == 0.0


def test_daily_profile_can_be_aggregated_hourly():
    # Two arrivals inside hour 0, none elsewhere: the hourly profile averages
    # over the 60 minutes of that hour.
    series = zero_series(1)
    series.iloc[0] = 60.0

    profile = daily_profile(series, resolution="hour")

    assert len(profile) == 24
    assert profile.loc[0] == 1.0


def test_daily_profile_supports_quantile_aggregation():
    # The EDA figure draws a p10-p90 band around the mean daily shape to show
    # how much the same minute varies from day to day.
    series = zero_series(10)
    # Give minute 0 the values 0..9 across the ten days.
    for day in range(10):
        series.iloc[day * 24 * 60] = float(day)

    low = daily_profile(series, agg=lambda group: group.quantile(0.1))
    high = daily_profile(series, agg=lambda group: group.quantile(0.9))

    assert low.loc[0] == pytest.approx(0.9)
    assert high.loc[0] == pytest.approx(8.1)


def test_daily_profile_defaults_to_the_mean():
    series = zero_series(2)
    series.iloc[0] = 10
    series.iloc[24 * 60] = 20

    explicit = daily_profile(series, agg=lambda group: group.mean())

    assert explicit.loc[0] == daily_profile(series).loc[0]


# --- weekly profile ---------------------------------------------------------


def test_weekly_profile_covers_every_hour_of_the_week():
    profile = weekly_profile(zero_series(7))

    assert len(profile) == 168


def test_weekly_profile_indexes_monday_midnight_as_hour_zero():
    # 2024-01-01 is a Monday.
    series = zero_series(7, start="2024-01-01")
    series.iloc[0] = 60.0

    profile = weekly_profile(series)

    assert profile.loc[0] == 1.0


def test_weekly_profile_separates_days_of_the_week():
    # Tuesday 00:00 is hour 24 of the week, not hour 0.
    series = zero_series(7, start="2024-01-01")
    series.iloc[24 * 60] = 60.0

    profile = weekly_profile(series)

    assert profile.loc[24] == 1.0
    assert profile.loc[0] == 0.0


def test_weekly_profile_averages_across_repeated_weeks():
    # Two weeks: Monday midnight hour totals 60 then 120 -> mean rate 1.5/min.
    series = zero_series(14, start="2024-01-01")
    series.iloc[0] = 60.0
    series.iloc[7 * 24 * 60] = 120.0

    profile = weekly_profile(series)

    assert profile.loc[0] == 1.5


def test_profiles_preserve_the_overall_mean():
    # Averaging by season must not invent or destroy volume: the mean of the
    # daily profile equals the mean of the series itself.
    rng = np.random.default_rng(0)
    series = zero_series(7)
    series.iloc[:] = rng.poisson(30, size=len(series))

    assert daily_profile(series).mean() == series.mean()
