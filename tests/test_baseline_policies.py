"""Tests for the three reactive baselines.

The comparison in this project is only honest if the reactive baselines are
well built. A predictive policy beating a badly configured reactive one proves
nothing, and it is the first thing a reviewer will attack. So each policy here
is tested against hand-computed decisions, and each exposes the knobs the
sweep needs to tune it properly.
"""

import pytest

from src.policies.arrival_rate import ArrivalRateTargetPolicy
from src.policies.backlog import BacklogPerInstancePolicy
from src.policies.static import StaticPolicy
from src.sim.observation import Observation

SERVICE_SECONDS = 30.0


def observation(**overrides) -> Observation:
    defaults = dict(
        t=3600.0,
        queue_depth=0,
        oldest_message_age=0.0,
        in_service_instances=40,
        pending_instances=0,
        recent_arrivals=(60,) * 60,
        completions_since_last_tick=60,
    )
    return Observation(**{**defaults, **overrides})


# --- static overprovisioning ------------------------------------------------


def test_static_policy_always_returns_its_fixed_level():
    policy = StaticPolicy(instances=42)

    assert policy.decide(0.0, observation()) == 42
    assert policy.decide(9999.0, observation(queue_depth=100_000)) == 42


def test_static_policy_ignores_a_growing_backlog():
    # The reference frontier: capacity chosen in advance and never adjusted.
    policy = StaticPolicy(instances=10)

    assert policy.decide(0.0, observation(queue_depth=50_000)) == 10


# --- backlog per instance (the AWS SQS approach) ----------------------------


def test_backlog_policy_sizes_the_fleet_to_clear_the_queue_within_the_budget():
    # 100 queued messages with a target of 2 per instance needs 50 instances.
    # Target comes from Little's Law: 60s budget / 30s per message = 2.
    policy = BacklogPerInstancePolicy(
        acceptable_latency_s=60.0, service_seconds=SERVICE_SECONDS
    )

    assert policy.decide(0.0, observation(queue_depth=100)) == 50


def test_backlog_policy_rounds_up():
    policy = BacklogPerInstancePolicy(
        acceptable_latency_s=60.0, service_seconds=SERVICE_SECONDS
    )

    assert policy.decide(0.0, observation(queue_depth=101)) == 51


def test_a_tighter_latency_budget_demands_more_instances():
    tight = BacklogPerInstancePolicy(30.0, SERVICE_SECONDS).decide(
        0.0, observation(queue_depth=120)
    )
    loose = BacklogPerInstancePolicy(300.0, SERVICE_SECONDS).decide(
        0.0, observation(queue_depth=120)
    )

    assert tight == 120  # target 1.0 message per instance
    assert loose == 12  # target 10 messages per instance


def test_backlog_policy_asks_for_nothing_when_the_queue_is_empty():
    # This is the policy's defining weakness, not a bug: queue depth is the
    # integral of capacity error, so a fleet that is keeping up looks exactly
    # like a fleet that is not needed. The simulator's floor is what stops it
    # reaching zero, and a burst then arrives with no warm capacity waiting.
    policy = BacklogPerInstancePolicy(60.0, SERVICE_SECONDS)

    assert policy.decide(0.0, observation(queue_depth=0)) == 0


def test_backlog_policy_respects_its_own_floor_when_configured():
    # Tuning the baseline fairly means letting it hold a floor, which is what
    # a competent operator would configure.
    policy = BacklogPerInstancePolicy(60.0, SERVICE_SECONDS, min_instances=20)

    assert policy.decide(0.0, observation(queue_depth=0)) == 20


# --- arrival rate target tracking (the strong reactive baseline) ------------


def test_arrival_rate_policy_sizes_to_the_observed_rate():
    # 60 arrivals per minute is 1/s; at 30s each that is 30 instances.
    policy = ArrivalRateTargetPolicy(
        service_seconds=SERVICE_SECONDS, target_utilization=1.0, window_minutes=5
    )

    assert policy.decide(0.0, observation(recent_arrivals=(60,) * 10)) == 30


def test_headroom_adds_spare_capacity():
    # 30 instances of demand at 80% utilisation needs ceil(37.5) = 38.
    policy = ArrivalRateTargetPolicy(
        service_seconds=SERVICE_SECONDS, target_utilization=0.8, window_minutes=5
    )

    assert policy.decide(0.0, observation(recent_arrivals=(60,) * 10)) == 38


def test_only_the_configured_window_of_history_is_used():
    # A short window tracks the recent surge; a long one is still averaging in
    # the quiet period before it. This lag is exactly what the project claims
    # reactive scaling suffers from.
    history = (0,) * 55 + (120,) * 5

    short = ArrivalRateTargetPolicy(SERVICE_SECONDS, 1.0, window_minutes=5).decide(
        0.0, observation(recent_arrivals=history)
    )
    long = ArrivalRateTargetPolicy(SERVICE_SECONDS, 1.0, window_minutes=60).decide(
        0.0, observation(recent_arrivals=history)
    )

    assert short == 60
    assert long == 5


def test_arrival_rate_policy_also_clears_an_existing_backlog():
    # Pure arrival-rate tracking provisions exactly enough for incoming work
    # and so can never catch up after falling behind. Adding a drain term is
    # what makes this baseline genuinely competitive rather than a foil.
    # 600 queued messages, 30s each, drained within 300s, needs 60 instances.
    policy = ArrivalRateTargetPolicy(
        service_seconds=SERVICE_SECONDS,
        target_utilization=1.0,
        window_minutes=5,
        backlog_drain_seconds=300.0,
    )

    decision = policy.decide(
        0.0, observation(queue_depth=600, recent_arrivals=(60,) * 10)
    )

    assert decision == 60


def test_the_larger_of_the_rate_and_drain_requirements_wins():
    policy = ArrivalRateTargetPolicy(
        service_seconds=SERVICE_SECONDS,
        target_utilization=1.0,
        window_minutes=5,
        backlog_drain_seconds=300.0,
    )

    # A small backlog must not pull capacity below what the arrival rate needs.
    decision = policy.decide(
        0.0, observation(queue_depth=10, recent_arrivals=(60,) * 10)
    )

    assert decision == 30


def test_backlog_drain_can_be_disabled():
    policy = ArrivalRateTargetPolicy(
        service_seconds=SERVICE_SECONDS,
        target_utilization=1.0,
        window_minutes=5,
        backlog_drain_seconds=None,
    )

    decision = policy.decide(
        0.0, observation(queue_depth=100_000, recent_arrivals=(60,) * 10)
    )

    assert decision == 30


def test_arrival_rate_policy_handles_an_empty_history():
    # At the very start of a run there is nothing to average over.
    policy = ArrivalRateTargetPolicy(SERVICE_SECONDS, 1.0, window_minutes=5)

    assert policy.decide(0.0, observation(recent_arrivals=())) == 0


def test_arrival_rate_policy_respects_its_own_floor():
    policy = ArrivalRateTargetPolicy(
        SERVICE_SECONDS, 1.0, window_minutes=5, min_instances=15
    )

    assert policy.decide(0.0, observation(recent_arrivals=())) == 15


# --- shared interface -------------------------------------------------------


def test_every_policy_returns_a_plain_integer():
    # The simulator clamps and compares these; a numpy scalar or float would
    # propagate silently into the tick table and the results.
    policies = [
        StaticPolicy(10),
        BacklogPerInstancePolicy(60.0, SERVICE_SECONDS),
        ArrivalRateTargetPolicy(SERVICE_SECONDS, 0.8, 5),
    ]

    for policy in policies:
        decision = policy.decide(0.0, observation(queue_depth=37))
        assert type(decision) is int


def test_every_policy_has_a_name_for_the_results_table():
    assert StaticPolicy(10).name == "static"
    assert BacklogPerInstancePolicy(60.0, SERVICE_SECONDS).name == "backlog_per_instance"
    assert ArrivalRateTargetPolicy(SERVICE_SECONDS, 0.8, 5).name == "arrival_rate"


def test_policies_reject_invalid_configuration():
    with pytest.raises(ValueError):
        StaticPolicy(-1)
    with pytest.raises(ValueError):
        BacklogPerInstancePolicy(acceptable_latency_s=0.0, service_seconds=30.0)
    with pytest.raises(ValueError):
        ArrivalRateTargetPolicy(30.0, target_utilization=1.5, window_minutes=5)
    with pytest.raises(ValueError):
        ArrivalRateTargetPolicy(30.0, target_utilization=0.8, window_minutes=0)
