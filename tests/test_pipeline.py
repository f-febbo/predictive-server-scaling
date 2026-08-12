"""Tests for the end-to-end trace pipeline.

These use real parquet files written to a temp directory rather than mocks, so
they exercise the same read path the production run takes.
"""

import pandas as pd
import pytest

from src.data.pipeline import (
    build_arrival_series,
    load_pickup_timestamps,
    month_window,
)


def write_trips(path, timestamps, column="tpep_pickup_datetime"):
    frame = pd.DataFrame(
        {
            column: pd.to_datetime(timestamps),
            # A second column so the loader has to actually select one.
            "trip_distance": [1.0] * len(timestamps),
        }
    )
    frame.to_parquet(path)
    return path


# --- month window -----------------------------------------------------------


def test_month_window_spans_from_the_first_month_to_the_end_of_the_last():
    start, end = month_window("2024-01", "2024-02")

    assert start == pd.Timestamp("2024-01-01")
    assert end == pd.Timestamp("2024-03-01")


def test_month_window_for_a_single_month():
    start, end = month_window("2024-01", "2024-01")

    assert start == pd.Timestamp("2024-01-01")
    assert end == pd.Timestamp("2024-02-01")


def test_month_window_crosses_a_year_boundary():
    start, end = month_window("2023-12", "2024-01")

    assert start == pd.Timestamp("2023-12-01")
    assert end == pd.Timestamp("2024-02-01")


# --- loading ----------------------------------------------------------------


def test_loads_only_the_pickup_timestamp_column(tmp_path):
    path = write_trips(
        tmp_path / "trips.parquet", ["2024-01-01 00:00:00", "2024-01-01 00:01:00"]
    )

    pickups = load_pickup_timestamps(path)

    assert len(pickups) == 2
    assert pickups.iloc[0] == pd.Timestamp("2024-01-01 00:00:00")


def test_recognises_the_green_taxi_pickup_column(tmp_path):
    # Yellow and green cabs use different column prefixes for the same field.
    path = write_trips(
        tmp_path / "green.parquet",
        ["2024-01-01 00:00:00"],
        column="lpep_pickup_datetime",
    )

    pickups = load_pickup_timestamps(path)

    assert len(pickups) == 1


def test_raises_a_clear_error_when_no_pickup_column_exists(tmp_path):
    frame = pd.DataFrame({"some_other_column": [1, 2]})
    path = tmp_path / "wrong.parquet"
    frame.to_parquet(path)

    with pytest.raises(ValueError, match="pickup"):
        load_pickup_timestamps(path)


# --- assembly ---------------------------------------------------------------


def test_builds_a_single_series_across_multiple_files(tmp_path):
    # Two "months" of data must join into one continuous series, not two.
    first = write_trips(tmp_path / "a.parquet", ["2024-01-01 00:00:30"])
    second = write_trips(tmp_path / "b.parquet", ["2024-01-01 00:02:30"])

    arrivals = build_arrival_series(
        [first, second], start="2024-01-01 00:00", end="2024-01-01 00:04"
    )

    assert list(arrivals) == [1, 0, 1, 0]


def test_assembly_conserves_every_in_window_trip(tmp_path):
    timestamps = [f"2024-01-01 00:{m:02d}:15" for m in range(10)]
    path = write_trips(tmp_path / "trips.parquet", timestamps)

    arrivals = build_arrival_series(
        [path], start="2024-01-01 00:00", end="2024-01-01 00:10"
    )

    assert arrivals.sum() == len(timestamps)


def test_assembly_drops_trips_stamped_outside_the_requested_window(tmp_path):
    # TLC monthly files genuinely contain a handful of records from other
    # years; they must not stretch the series index.
    path = write_trips(
        tmp_path / "trips.parquet",
        ["2001-01-01 00:00:00", "2024-01-01 00:00:30", "2090-01-01 00:00:00"],
    )

    arrivals = build_arrival_series(
        [path], start="2024-01-01 00:00", end="2024-01-01 00:02"
    )

    assert list(arrivals) == [1, 0]
    assert arrivals.index.max() == pd.Timestamp("2024-01-01 00:01")


def test_assembly_rejects_an_empty_file_list():
    with pytest.raises(ValueError):
        build_arrival_series([], start="2024-01-01", end="2024-01-02")
