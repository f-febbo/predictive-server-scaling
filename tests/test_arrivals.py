"""Tests for expanding per-minute trace counts into individual arrival times.

The trace gives a count per minute. The simulator needs individual events, so
something has to decide *where inside the minute* each arrival lands. That
choice is a modelling decision with real consequences for measured message age,
so both placement modes are pinned here.
"""

import numpy as np
import pandas as pd
import pytest

from src.sim.arrivals import arrival_times_from_counts

MINUTE = 60.0


def counts(values: list[int]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1min", name="ts")
    return pd.Series(values, index=index, name="arrivals")


# --- conservation -----------------------------------------------------------


def test_produces_exactly_as_many_arrivals_as_the_trace_counts():
    trace = counts([3, 0, 7, 2])

    times = arrival_times_from_counts(trace, rng=np.random.default_rng(0))

    assert len(times) == 12


def test_an_all_zero_trace_produces_no_arrivals():
    times = arrival_times_from_counts(counts([0, 0, 0]), rng=np.random.default_rng(0))

    assert len(times) == 0


def test_times_are_measured_in_seconds_from_the_start_of_the_trace():
    # A single arrival in the third minute must land in [120, 180).
    trace = counts([0, 0, 1])

    times = arrival_times_from_counts(trace, rng=np.random.default_rng(0))

    assert 2 * MINUTE <= times[0] < 3 * MINUTE


def test_every_arrival_lands_inside_its_own_minute():
    trace = counts([5, 5, 5, 5])

    times = arrival_times_from_counts(trace, rng=np.random.default_rng(0))

    for minute in range(4):
        in_minute = times[
            (times >= minute * MINUTE) & (times < (minute + 1) * MINUTE)
        ]
        assert len(in_minute) == 5


def test_output_is_sorted():
    trace = counts([4, 9, 2, 6])

    times = arrival_times_from_counts(trace, rng=np.random.default_rng(0))

    assert np.all(np.diff(times) >= 0)


# --- uniform (evenly spaced) placement --------------------------------------


def test_uniform_mode_spaces_arrivals_evenly_within_the_minute():
    # Two arrivals in one minute sit at the centres of its two halves.
    trace = counts([2])

    times = arrival_times_from_counts(trace, mode="uniform")

    np.testing.assert_allclose(times, [15.0, 45.0])


def test_uniform_mode_places_a_lone_arrival_at_the_minute_midpoint():
    times = arrival_times_from_counts(counts([1]), mode="uniform")

    np.testing.assert_allclose(times, [30.0])


def test_uniform_mode_is_deterministic_and_needs_no_rng():
    trace = counts([3, 5])

    first = arrival_times_from_counts(trace, mode="uniform")
    second = arrival_times_from_counts(trace, mode="uniform")

    np.testing.assert_array_equal(first, second)


# --- poisson (random) placement ---------------------------------------------


def test_poisson_mode_is_reproducible_for_a_given_seed():
    trace = counts([10, 10])

    first = arrival_times_from_counts(trace, rng=np.random.default_rng(42))
    second = arrival_times_from_counts(trace, rng=np.random.default_rng(42))

    np.testing.assert_array_equal(first, second)


def test_poisson_mode_differs_between_seeds():
    trace = counts([10, 10])

    first = arrival_times_from_counts(trace, rng=np.random.default_rng(1))
    second = arrival_times_from_counts(trace, rng=np.random.default_rng(2))

    assert not np.array_equal(first, second)


def test_poisson_mode_spreads_arrivals_across_the_whole_minute():
    # Conditioned on the count, a Poisson process places arrivals uniformly at
    # random within the interval, so a large sample should look flat.
    trace = counts([100_000])

    times = arrival_times_from_counts(trace, rng=np.random.default_rng(0))
    within_minute = times % MINUTE

    assert within_minute.mean() == pytest.approx(30.0, abs=0.5)
    assert within_minute.min() < 1.0
    assert within_minute.max() > 59.0


def test_poisson_mode_clusters_more_than_even_spacing():
    # The point of random placement: gaps between arrivals vary, so short
    # bursts occur. Evenly spaced arrivals have identical gaps by construction.
    trace = counts([1000])

    random_gaps = np.diff(arrival_times_from_counts(trace, rng=np.random.default_rng(0)))
    even_gaps = np.diff(arrival_times_from_counts(trace, mode="uniform"))

    assert random_gaps.std() > even_gaps.std()
    assert even_gaps.std() == pytest.approx(0.0, abs=1e-9)


# --- interface --------------------------------------------------------------


def test_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        arrival_times_from_counts(counts([1]), mode="magic")


def test_poisson_mode_requires_an_rng():
    # Silently seeding would make a "reproducible" run irreproducible.
    with pytest.raises(ValueError, match="rng"):
        arrival_times_from_counts(counts([1]), mode="poisson")


def test_rejects_negative_counts():
    with pytest.raises(ValueError):
        arrival_times_from_counts(counts([1, -2]), rng=np.random.default_rng(0))
