"""Tests for the worker service-time distribution."""

import numpy as np
import pytest

from src.sim.service import ServiceTime


def rng():
    return np.random.default_rng(0)


def test_deterministic_when_variability_is_zero():
    # A constant service time makes queueing behaviour exactly predictable,
    # which is what the simulator's arithmetic acceptance tests rely on.
    samples = ServiceTime(mean_seconds=2.0, cv=0.0).sample(5, rng())

    np.testing.assert_allclose(samples, [2.0] * 5)


def test_sample_mean_matches_the_configured_mean():
    samples = ServiceTime(mean_seconds=2.0, cv=0.5).sample(200_000, rng())

    assert samples.mean() == pytest.approx(2.0, rel=0.01)


def test_sample_variability_matches_the_configured_cv():
    samples = ServiceTime(mean_seconds=2.0, cv=0.75).sample(200_000, rng())

    assert samples.std() / samples.mean() == pytest.approx(0.75, rel=0.03)


def test_a_higher_cv_produces_a_longer_tail():
    # Right-skew is the reason lognormal is the default: a few slow messages
    # hold a worker far longer than the mean suggests.
    tight = ServiceTime(mean_seconds=2.0, cv=0.25).sample(100_000, rng())
    loose = ServiceTime(mean_seconds=2.0, cv=1.5).sample(100_000, rng())

    assert np.percentile(loose, 99) > np.percentile(tight, 99)


def test_service_times_are_strictly_positive():
    samples = ServiceTime(mean_seconds=2.0, cv=1.5).sample(100_000, rng())

    assert (samples > 0).all()


def test_sampling_is_reproducible_for_a_given_seed():
    first = ServiceTime(mean_seconds=2.0, cv=0.5).sample(100, np.random.default_rng(7))
    second = ServiceTime(mean_seconds=2.0, cv=0.5).sample(100, np.random.default_rng(7))

    np.testing.assert_array_equal(first, second)


def test_service_capacity_per_instance_is_the_reciprocal_of_the_mean():
    # One worker handling 2-second messages sustains 0.5 messages/second. This
    # conversion underpins the Little's Law capacity maths in the policies.
    assert ServiceTime(mean_seconds=2.0).throughput_per_instance() == pytest.approx(0.5)
    assert ServiceTime(mean_seconds=4.0).throughput_per_instance() == pytest.approx(0.25)


def test_rejects_a_non_positive_mean():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            ServiceTime(mean_seconds=bad)


def test_rejects_a_negative_cv():
    with pytest.raises(ValueError):
        ServiceTime(mean_seconds=2.0, cv=-0.1)


def test_sample_of_zero_size_returns_an_empty_array():
    assert len(ServiceTime().sample(0, rng())) == 0
