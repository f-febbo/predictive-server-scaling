"""Tests for the arrival-count resampler.

Every case here uses a hand-countable input so the expected output can be
verified by eye rather than by re-running the implementation.
"""

import pandas as pd
import pytest

from src.data.resample import resample_arrivals


def ts(*values: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values)))


def test_counts_arrivals_per_minute_bucket():
    # Two events in minute 00:00, none in 00:01, one in 00:02, none in 00:03.
    arrivals = resample_arrivals(
        ts("2024-01-01 00:00:10", "2024-01-01 00:00:59", "2024-01-01 00:02:30"),
        start="2024-01-01 00:00",
        end="2024-01-01 00:04",
    )

    assert list(arrivals) == [2, 0, 1, 0]


def test_empty_minutes_are_filled_with_zero_not_dropped():
    # A gap in the middle must survive as a zero, otherwise lag features
    # silently shift in time and every downstream forecast is wrong.
    arrivals = resample_arrivals(
        ts("2024-01-01 00:00:00", "2024-01-01 00:05:00"),
        start="2024-01-01 00:00",
        end="2024-01-01 00:06",
    )

    assert list(arrivals) == [1, 0, 0, 0, 0, 1]


def test_index_is_a_complete_contiguous_minute_range():
    arrivals = resample_arrivals(
        ts("2024-01-01 00:01:00"),
        start="2024-01-01 00:00",
        end="2024-01-01 00:03",
    )

    assert list(arrivals.index) == list(
        pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:02"])
    )


def test_end_bound_is_exclusive():
    # An event exactly at `end` belongs to the next window, not this one.
    arrivals = resample_arrivals(
        ts("2024-01-01 00:02:00"),
        start="2024-01-01 00:00",
        end="2024-01-01 00:02",
    )

    assert list(arrivals) == [0, 0]


def test_timestamps_outside_the_window_are_discarded():
    # Real TLC files contain a handful of records stamped years away from the
    # file's nominal month. Left in, they stretch the series index across empty
    # decades. This is a genuine defect in the source data, not a hypothetical.
    arrivals = resample_arrivals(
        ts(
            "2002-12-31 23:59:00",  # corrupt: far past
            "2024-01-01 00:00:30",  # good
            "2099-01-01 00:00:00",  # corrupt: far future
        ),
        start="2024-01-01 00:00",
        end="2024-01-01 00:02",
    )

    assert list(arrivals) == [1, 0]


def test_null_timestamps_are_discarded():
    arrivals = resample_arrivals(
        pd.Series(pd.to_datetime(["2024-01-01 00:00:30", None])),
        start="2024-01-01 00:00",
        end="2024-01-01 00:02",
    )

    assert list(arrivals) == [1, 0]


def test_unsorted_input_produces_time_ordered_output():
    arrivals = resample_arrivals(
        ts("2024-01-01 00:02:00", "2024-01-01 00:00:00", "2024-01-01 00:01:00"),
        start="2024-01-01 00:00",
        end="2024-01-01 00:03",
    )

    assert arrivals.index.is_monotonic_increasing
    assert list(arrivals) == [1, 1, 1]


def test_window_defaults_to_the_span_of_the_data():
    arrivals = resample_arrivals(
        ts("2024-01-01 00:00:30", "2024-01-01 00:02:10")
    )

    assert list(arrivals) == [1, 0, 1]


def test_result_is_integer_counts_named_arrivals():
    arrivals = resample_arrivals(
        ts("2024-01-01 00:00:30"), start="2024-01-01 00:00", end="2024-01-01 00:02"
    )

    assert arrivals.name == "arrivals"
    assert arrivals.dtype.kind == "i"
    assert arrivals.index.name == "ts"


def test_supports_non_minute_frequency():
    arrivals = resample_arrivals(
        ts("2024-01-01 00:00:30", "2024-01-01 00:04:00"),
        freq="5min",
        start="2024-01-01 00:00",
        end="2024-01-01 00:10",
    )

    assert list(arrivals) == [2, 0]


def test_all_input_events_inside_the_window_are_conserved():
    # The total count must equal the number of in-window events: no event
    # invented, none lost.
    events = ts(*[f"2024-01-01 00:{m:02d}:{s:02d}" for m in range(5) for s in (0, 30)])

    arrivals = resample_arrivals(events, start="2024-01-01 00:00", end="2024-01-01 00:05")

    assert arrivals.sum() == len(events)


def test_rejects_end_before_start():
    with pytest.raises(ValueError):
        resample_arrivals(
            ts("2024-01-01 00:00:30"),
            start="2024-01-01 01:00",
            end="2024-01-01 00:00",
        )


def test_empty_input_over_an_explicit_window_is_all_zeros():
    arrivals = resample_arrivals(
        pd.Series([], dtype="datetime64[ns]"),
        start="2024-01-01 00:00",
        end="2024-01-01 00:03",
    )

    assert list(arrivals) == [0, 0, 0]
