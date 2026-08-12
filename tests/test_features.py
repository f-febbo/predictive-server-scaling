"""Tests for forecast feature construction.

The central property is that a feature row indexed at time `t` depends only on
series values strictly before `t`. That is checked directly — by corrupting the
future and asserting the past does not move — rather than by reading the code
and trusting it.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecast.features import DAY_MINUTES, WEEK_MINUTES, build_features, build_target


def series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="1min", name="ts")
    return pd.Series(values, index=index, name="arrivals")


def ramp(minutes: int, start: str = "2024-01-01") -> pd.Series:
    return series([float(i) for i in range(minutes)], start=start)


# --- lags -------------------------------------------------------------------


def test_lag_one_is_the_previous_minute():
    # At decision time t the minute ending at t is the freshest complete
    # observation, so lag_1 is the value at t-1. Using the value at t would
    # mean reading a minute that has not finished.
    features = build_features(ramp(100))

    assert features.loc[features.index[50], "lag_1"] == 49.0


def test_lags_step_back_one_minute_at_a_time():
    features = build_features(ramp(100))
    row = features.loc[features.index[80]]

    assert row["lag_1"] == 79.0
    assert row["lag_2"] == 78.0
    assert row["lag_60"] == 20.0


def test_daily_and_weekly_lags_reach_the_right_distance_back():
    values = ramp(WEEK_MINUTES + 200)

    features = build_features(values)
    position = WEEK_MINUTES + 100
    row = features.loc[features.index[position]]

    assert row[f"lag_{DAY_MINUTES}"] == float(position - DAY_MINUTES)
    assert row[f"lag_{WEEK_MINUTES}"] == float(position - WEEK_MINUTES)


# --- rolling statistics -----------------------------------------------------


def test_rolling_mean_covers_the_window_before_the_decision_time():
    # At index 50 the five-minute mean averages minutes 45..49, not 46..50.
    features = build_features(ramp(100))

    assert features.loc[features.index[50], "roll_mean_5"] == pytest.approx(47.0)


def test_rolling_max_excludes_the_current_minute():
    values = series([1.0] * 20 + [999.0] + [1.0] * 20)

    features = build_features(values)

    # The spike is at index 20, so it is visible from index 21 onwards.
    assert features.loc[features.index[20], "roll_max_15"] == 1.0
    assert features.loc[features.index[21], "roll_max_15"] == 999.0


def test_rolling_std_is_zero_on_a_flat_series():
    features = build_features(series([7.0] * 100))

    assert features.loc[features.index[80], "roll_std_15"] == pytest.approx(0.0)


# --- calendar ---------------------------------------------------------------


def test_calendar_features_describe_the_decision_time_itself():
    # A clock is not lookahead: at time t we know what time it is.
    values = ramp(200, start="2024-01-03 14:30")  # a Wednesday

    features = build_features(values)
    row = features.loc[features.index[0]]

    assert row["hour"] == 14
    assert row["day_of_week"] == 2  # Monday is 0
    assert row["minute_of_day"] == 14 * 60 + 30
    assert row["is_weekend"] == 0


def test_weekend_flag_is_set_on_saturday():
    values = ramp(60, start="2024-01-06 09:00")  # a Saturday

    features = build_features(values)

    assert features.iloc[0]["is_weekend"] == 1


def test_holiday_flag_is_set_on_a_federal_holiday():
    # Traffic on a public holiday follows the weekend shape rather than the
    # weekday shape, and the model cannot learn that from lags alone.
    holiday = build_features(ramp(60, start="2024-07-04 09:00"))
    ordinary = build_features(ramp(60, start="2024-07-09 09:00"))

    assert holiday.iloc[0]["is_holiday"] == 1
    assert ordinary.iloc[0]["is_holiday"] == 0


# --- targets ----------------------------------------------------------------


def test_the_target_is_the_value_one_horizon_ahead():
    target = build_target(ramp(100), horizon_minutes=15)

    assert target.iloc[50] == 65.0


def test_the_target_runs_out_at_the_end_of_the_series():
    # The last `horizon` decision times have no observable outcome yet, so they
    # cannot be trained on.
    target = build_target(ramp(100), horizon_minutes=15)

    assert target.iloc[-15:].isna().all()
    assert target.iloc[:-15].notna().all()


# --- the leakage guarantee --------------------------------------------------


def test_features_do_not_change_when_the_future_is_corrupted():
    # The direct test of no lookahead: replace everything after a cut point
    # with garbage and confirm that every feature row at or before the cut is
    # bit-for-bit identical. If any feature reached forward, this would fail.
    original = ramp(WEEK_MINUTES + 500)
    corrupted = original.copy()
    cut = WEEK_MINUTES + 200
    rng = np.random.default_rng(0)
    corrupted.iloc[cut + 1 :] = rng.uniform(0, 1e6, size=len(corrupted) - cut - 1)

    from_original = build_features(original).iloc[: cut + 1]
    from_corrupted = build_features(corrupted).iloc[: cut + 1]

    pd.testing.assert_frame_equal(from_original, from_corrupted)


def test_corrupting_a_single_minute_leaves_every_earlier_row_untouched():
    # Sharper than the bulk-corruption test above. That one only proves rows
    # near the cut do not read past the cut; this proves no row reads *any*
    # later minute, however close. It is the strongest form of the guarantee
    # and the one that would catch an off-by-one or an accidentally centred
    # rolling window.
    original = ramp(WEEK_MINUTES + 500)
    spiked = original.copy()
    spike_at = WEEK_MINUTES + 300
    spiked.iloc[spike_at] = 1e9

    from_original = build_features(original).iloc[:spike_at]
    from_spiked = build_features(spiked).iloc[:spike_at]

    pd.testing.assert_frame_equal(from_original, from_spiked)


def test_a_feature_row_never_reads_its_own_minute():
    # A single spike must be invisible to the row at the same index.
    flat = series([1.0] * 200)
    spiked = flat.copy()
    spiked.iloc[100] = 10_000.0

    from_flat = build_features(flat).iloc[100]
    from_spiked = build_features(spiked).iloc[100]

    pd.testing.assert_series_equal(from_flat, from_spiked)


def test_feature_rows_are_indexed_by_decision_time():
    values = ramp(100)

    features = build_features(values)

    assert features.index.equals(values.index)


def test_early_rows_have_missing_lags_rather_than_invented_ones():
    # Before a week of history exists the weekly lag is genuinely unknown.
    # Filling it with a zero or a mean would teach the model a lie.
    features = build_features(ramp(100))

    assert features.iloc[0][f"lag_{WEEK_MINUTES}"] != features.iloc[0][f"lag_{WEEK_MINUTES}"]


def test_build_features_rejects_a_non_datetime_index():
    with pytest.raises(ValueError):
        build_features(pd.Series([1.0, 2.0, 3.0]))
