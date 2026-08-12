"""Tests for the Little's Law capacity conversions.

Every policy in the project turns some estimate of load into an instance count
through these two functions, so an error here would move all of them together
and stay invisible in a relative comparison.
"""

import pytest

from src.policies.capacity import backlog_target_per_instance, instances_for_arrival_rate


# --- arrival rate -> instances ----------------------------------------------


def test_instances_needed_is_arrival_rate_times_service_time():
    # 2 messages/s, each taking 30s, means 60 messages in flight at any moment
    # (Little's Law, L = lambda * W), so 60 workers are needed to keep up.
    assert instances_for_arrival_rate(2.0, service_seconds=30.0) == 60


def test_headroom_divides_by_the_target_utilisation():
    # Running at 80% utilisation needs 60 / 0.8 = 75 instances.
    assert (
        instances_for_arrival_rate(2.0, service_seconds=30.0, target_utilization=0.8)
        == 75
    )


def test_result_is_rounded_up_so_capacity_is_never_short():
    # 1.5 instances of demand needs 2 instances. Rounding down would leave the
    # fleet permanently and deliberately underprovisioned.
    assert instances_for_arrival_rate(0.05, service_seconds=30.0) == 2


def test_zero_arrival_rate_needs_no_instances():
    assert instances_for_arrival_rate(0.0, service_seconds=30.0) == 0


def test_a_longer_service_time_needs_proportionally_more_instances():
    slow = instances_for_arrival_rate(1.0, service_seconds=60.0)
    fast = instances_for_arrival_rate(1.0, service_seconds=30.0)

    assert slow == 2 * fast


def test_rejects_a_non_positive_service_time():
    with pytest.raises(ValueError):
        instances_for_arrival_rate(1.0, service_seconds=0.0)


def test_rejects_a_utilisation_outside_zero_to_one():
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            instances_for_arrival_rate(1.0, service_seconds=30.0, target_utilization=bad)


def test_rejects_a_negative_arrival_rate():
    with pytest.raises(ValueError):
        instances_for_arrival_rate(-1.0, service_seconds=30.0)


# --- acceptable latency -> backlog target ------------------------------------


def test_backlog_target_is_how_many_messages_fit_inside_the_latency_budget():
    # One worker clears a message every 30s. To keep the oldest message under
    # 60s, no worker may have more than 2 messages queued behind it. This is
    # the target AWS recommends tracking for SQS-backed fleets.
    assert backlog_target_per_instance(60.0, service_seconds=30.0) == pytest.approx(2.0)


def test_a_tighter_latency_budget_gives_a_smaller_backlog_target():
    tight = backlog_target_per_instance(30.0, service_seconds=30.0)
    loose = backlog_target_per_instance(300.0, service_seconds=30.0)

    assert tight == pytest.approx(1.0)
    assert loose == pytest.approx(10.0)
    assert tight < loose


def test_backlog_target_stays_fractional():
    # Rounding this to an integer would quietly coarsen the tuning sweep, and
    # a target below 1 is meaningful: it means scaling out before any worker
    # has even one message waiting.
    assert backlog_target_per_instance(15.0, service_seconds=30.0) == pytest.approx(0.5)


def test_backlog_target_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        backlog_target_per_instance(0.0, service_seconds=30.0)
    with pytest.raises(ValueError):
        backlog_target_per_instance(60.0, service_seconds=0.0)
