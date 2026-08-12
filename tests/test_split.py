"""Tests for the temporal train/test split.

The split must be chronological. A random split on a time series leaks the
future into training via neighbouring minutes and inflates every downstream
score, so these tests pin the boundary behaviour precisely.
"""

import pandas as pd
import pytest

from src.data.split import split_by_time


def minute_series(n: int, start: str = "2024-01-01 00:00") -> pd.Series:
    index = pd.date_range(start=start, periods=n, freq="1min", name="ts")
    return pd.Series(range(n), index=index, name="arrivals")


def test_holds_out_the_last_fraction_of_the_timeline():
    # 10 minutes, 20% held out -> the final 2 minutes are the test set.
    train, test = split_by_time(minute_series(10), test_fraction=0.2)

    assert len(train) == 8
    assert len(test) == 2
    assert list(test) == [8, 9]


def test_train_is_entirely_before_test():
    train, test = split_by_time(minute_series(100), test_fraction=0.2)

    assert train.index.max() < test.index.min()


def test_split_is_contiguous_and_lossless():
    # No row is dropped and none is duplicated across the boundary.
    original = minute_series(97)

    train, test = split_by_time(original, test_fraction=0.2)

    rejoined = pd.concat([train, test])
    pd.testing.assert_series_equal(rejoined, original)


def test_boundary_minute_belongs_to_test_not_train():
    train, test = split_by_time(minute_series(10), test_fraction=0.2)

    boundary = pd.Timestamp("2024-01-01 00:08")
    assert boundary in test.index
    assert boundary not in train.index


def test_split_is_deterministic():
    series = minute_series(1000)

    first_train, first_test = split_by_time(series, test_fraction=0.2)
    second_train, second_test = split_by_time(series, test_fraction=0.2)

    pd.testing.assert_series_equal(first_train, second_train)
    pd.testing.assert_series_equal(first_test, second_test)


def test_split_ignores_row_order_and_uses_timestamps():
    # Shuffling the rows must not change the split: the boundary is a point in
    # time, not a row position.
    series = minute_series(10)
    shuffled = series.sample(frac=1.0, random_state=0)

    train, test = split_by_time(shuffled, test_fraction=0.2)

    assert list(test) == [8, 9]
    assert list(train) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_fraction_is_configurable():
    train, test = split_by_time(minute_series(10), test_fraction=0.5)

    assert len(train) == 5
    assert len(test) == 5


def test_rejects_out_of_range_fractions():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            split_by_time(minute_series(10), test_fraction=bad)


def test_rejects_irregular_index():
    # A gapped series would make a fraction-of-span cut mean something
    # different from a fraction-of-data cut; refuse rather than guess.
    index = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:09"])
    irregular = pd.Series([1, 2, 3], index=index, name="arrivals")

    with pytest.raises(ValueError):
        split_by_time(irregular, test_fraction=0.2)


def test_rejects_a_series_too_short_to_split():
    with pytest.raises(ValueError):
        split_by_time(minute_series(1), test_fraction=0.2)
