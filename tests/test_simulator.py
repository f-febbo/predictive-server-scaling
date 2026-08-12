"""Tests for the discrete-event simulator.

Covers the four acceptance properties from the spec — conservation, a bounded
queue under sufficient capacity, linear backlog growth under insufficient
capacity, and an honoured boot delay — plus the mechanics each of those rests
on. Wherever possible the configuration is made deterministic (constant service
time, evenly spaced arrivals) so the expected numbers can be derived by hand
rather than trusted from a previous run.
"""

import numpy as np
import pandas as pd
import pytest

from src.sim.config import SimConfig
from src.sim.observation import Observation
from src.sim.service import ServiceTime
from src.sim.simulator import simulate


# --- helpers ----------------------------------------------------------------


def counts(values: list[int]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1min", name="ts")
    return pd.Series(values, index=index, name="arrivals")


def flat_trace(rate_per_minute: int, minutes: int) -> pd.Series:
    return counts([rate_per_minute] * minutes)


class FixedPolicy:
    """Always asks for the same capacity."""

    def __init__(self, instances: int):
        self.instances = instances

    def decide(self, t: float, obs: Observation) -> int:
        return self.instances


class ScheduledPolicy:
    """Returns capacity according to a time -> instances schedule."""

    def __init__(self, schedule: dict[float, int], default: int):
        self.schedule = schedule
        self.default = default
        self.current = default

    def decide(self, t: float, obs: Observation) -> int:
        self.current = self.schedule.get(t, self.current)
        return self.current


class RecordingPolicy:
    """Wraps a policy and keeps every observation it was shown."""

    def __init__(self, inner):
        self.inner = inner
        self.seen: list[Observation] = []

    def decide(self, t: float, obs: Observation) -> int:
        self.seen.append(obs)
        return self.inner.decide(t, obs)


def exact_config(**overrides) -> SimConfig:
    """Configuration with all randomness removed, for hand-checkable tests."""
    defaults = dict(
        service=ServiceTime(mean_seconds=2.0, cv=0.0),
        arrival_mode="uniform",
        boot_time_s=180.0,
        scaler_tick_s=60.0,
        scale_in_cooldown_s=0.0,
        initial_instances=1,
        min_instances=0,
        max_instances=1000,
    )
    return SimConfig(**{**defaults, **overrides})


# --- (a) conservation -------------------------------------------------------


def test_every_message_is_accounted_for_exactly_once():
    trace = flat_trace(rate_per_minute=100, minutes=30)

    result = simulate(trace, FixedPolicy(3), exact_config())

    assert result.messages_arrived == 3000
    assert (
        result.messages_completed
        + result.messages_in_queue_at_end
        + result.messages_in_service_at_end
        == result.messages_arrived
    )


def test_conservation_holds_when_capacity_is_far_too_small():
    trace = flat_trace(rate_per_minute=200, minutes=20)

    result = simulate(trace, FixedPolicy(1), exact_config())

    assert result.messages_arrived == 4000
    assert (
        result.messages_completed
        + result.messages_in_queue_at_end
        + result.messages_in_service_at_end
        == 4000
    )
    assert result.messages_in_queue_at_end > 0  # genuinely backlogged


def test_conservation_holds_with_random_service_times_and_arrivals():
    trace = flat_trace(rate_per_minute=120, minutes=60)
    config = SimConfig(
        service=ServiceTime(mean_seconds=2.0, cv=1.0),
        arrival_mode="poisson",
        seed=7,
        initial_instances=4,
    )

    result = simulate(trace, FixedPolicy(5), config)

    assert result.messages_arrived == 7200
    assert (
        result.messages_completed
        + result.messages_in_queue_at_end
        + result.messages_in_service_at_end
        == 7200
    )


def test_no_message_is_processed_before_it_arrives():
    trace = flat_trace(rate_per_minute=60, minutes=10)

    result = simulate(trace, FixedPolicy(4), exact_config(record_messages=True))

    messages = result.messages
    assert (messages["started_at"] >= messages["arrived_at"]).all()
    assert (messages["completed_at"] > messages["started_at"]).all()


def test_messages_are_served_first_in_first_out():
    trace = flat_trace(rate_per_minute=120, minutes=10)

    result = simulate(trace, FixedPolicy(1), exact_config(record_messages=True))

    messages = result.messages.sort_values("arrived_at")
    assert messages["started_at"].is_monotonic_increasing


# --- (b) bounded queue below capacity ---------------------------------------


def test_queue_stays_bounded_when_capacity_exceeds_the_arrival_rate():
    # 60 arrivals/min = 1/s. Four workers at 2s each sustain 2/s, so the
    # service rate is double the arrival rate and nothing should accumulate.
    trace = flat_trace(rate_per_minute=60, minutes=60)

    result = simulate(trace, FixedPolicy(4), exact_config(initial_instances=4))

    assert result.ticks["queue_depth"].max() <= 5
    assert result.messages_in_queue_at_end == 0


def test_message_age_stays_low_when_capacity_is_sufficient():
    trace = flat_trace(rate_per_minute=60, minutes=60)

    result = simulate(trace, FixedPolicy(4), exact_config(initial_instances=4))

    assert result.ticks["oldest_message_age"].max() < 5.0


def test_queue_does_not_grow_at_exactly_balanced_load():
    # 30 arrivals/min = 0.5/s, one worker at 2s = 0.5/s. Perfectly balanced
    # with no randomness, so the queue should not run away.
    trace = flat_trace(rate_per_minute=30, minutes=120)

    result = simulate(trace, FixedPolicy(1), exact_config())

    assert result.ticks["queue_depth"].max() <= 2


# --- (c) linear backlog growth above capacity -------------------------------


def test_backlog_grows_at_the_arithmetically_expected_rate():
    # 120 arrivals/min = 2/s against one worker at 2s/message = 0.5/s.
    # The shortfall is 1.5 messages/s, so after 600s the backlog should be
    # about 900 messages.
    trace = flat_trace(rate_per_minute=120, minutes=10)

    result = simulate(trace, FixedPolicy(1), exact_config())

    assert result.messages_completed == pytest.approx(300, abs=2)
    assert result.messages_in_queue_at_end == pytest.approx(900, abs=3)


def test_backlog_growth_is_linear_over_time():
    # Sampling the backlog at three points should show equal increments.
    trace = flat_trace(rate_per_minute=120, minutes=30)

    result = simulate(trace, FixedPolicy(1), exact_config())
    ticks = result.ticks.set_index("t")["queue_depth"]

    first_third = ticks.loc[600] - ticks.loc[0]
    second_third = ticks.loc[1200] - ticks.loc[600]
    assert second_third == pytest.approx(first_third, rel=0.02)


def test_oldest_message_age_grows_when_underprovisioned():
    trace = flat_trace(rate_per_minute=120, minutes=30)

    result = simulate(trace, FixedPolicy(1), exact_config())
    ages = result.ticks["oldest_message_age"]

    assert ages.iloc[-1] > ages.iloc[len(ages) // 2] > ages.iloc[1]


# --- (d) boot delay ---------------------------------------------------------


def test_requested_capacity_is_unavailable_until_the_boot_delay_elapses():
    # Ask for 5 instances at t=0 with a 180s boot time. Until t=180 only the
    # single initial instance may be in service.
    trace = flat_trace(rate_per_minute=120, minutes=20)

    result = simulate(
        trace, FixedPolicy(5), exact_config(initial_instances=1, boot_time_s=180.0)
    )
    ticks = result.ticks.set_index("t")

    assert ticks.loc[0, "in_service_instances"] == 1
    assert ticks.loc[60, "in_service_instances"] == 1
    assert ticks.loc[120, "in_service_instances"] == 1
    assert ticks.loc[180, "in_service_instances"] == 5


def test_booting_instances_are_reported_as_pending():
    # Capacity that has been paid for but cannot yet do work must be visible,
    # otherwise a policy re-requests it every tick and massively overshoots.
    trace = flat_trace(rate_per_minute=120, minutes=20)

    result = simulate(trace, FixedPolicy(5), exact_config(initial_instances=1))
    ticks = result.ticks.set_index("t")

    assert ticks.loc[60, "pending_instances"] == 4
    assert ticks.loc[180, "pending_instances"] == 0


def test_a_longer_boot_delay_defers_relief_further():
    # This parameter is the entire reason the project exists, so its effect
    # must be unambiguous: slower boots mean a deeper backlog.
    trace = flat_trace(rate_per_minute=120, minutes=30)

    fast = simulate(trace, FixedPolicy(6), exact_config(boot_time_s=60.0))
    slow = simulate(trace, FixedPolicy(6), exact_config(boot_time_s=600.0))

    assert slow.ticks["queue_depth"].max() > fast.ticks["queue_depth"].max()
    # Total completions is deliberately *not* the discriminator: given enough
    # runway both fleets drain the same trace, and the slow one merely gets
    # there later. What the delay actually costs is accumulated message age,
    # which is the SLI.
    assert slow.ticks["oldest_message_age"].sum() > fast.ticks[
        "oldest_message_age"
    ].sum()


def test_capacity_requested_in_one_tick_is_not_requested_again_in_the_next():
    # in_service + pending is what the policy's request is compared against.
    trace = flat_trace(rate_per_minute=120, minutes=20)

    result = simulate(trace, FixedPolicy(5), exact_config(initial_instances=1))
    ticks = result.ticks.set_index("t")

    # Four booted instances at t=180, not sixteen from four repeated requests.
    assert ticks.loc[180, "in_service_instances"] == 5
    assert ticks.loc[300, "in_service_instances"] == 5


# --- scale-in ---------------------------------------------------------------


def test_scale_in_reduces_capacity():
    trace = flat_trace(rate_per_minute=10, minutes=30)
    policy = ScheduledPolicy({0.0: 5, 600.0: 2}, default=5)

    result = simulate(trace, policy, exact_config(initial_instances=5))
    ticks = result.ticks.set_index("t")

    assert ticks.loc[540, "in_service_instances"] == 5
    assert ticks.loc[660, "in_service_instances"] == 2


def test_scale_in_cooldown_prevents_repeated_shrinking():
    # With a 600s cooldown, a policy demanding a cut every tick may only get
    # one cut per cooldown window.
    trace = flat_trace(rate_per_minute=10, minutes=30)
    policy = ScheduledPolicy({0.0: 8, 120.0: 7, 180.0: 6, 240.0: 5}, default=8)

    result = simulate(
        trace,
        policy,
        exact_config(initial_instances=8, scale_in_cooldown_s=600.0),
    )
    ticks = result.ticks.set_index("t")

    # One reduction at t=120, then nothing further until the cooldown expires.
    assert ticks.loc[180, "in_service_instances"] == 7
    assert ticks.loc[300, "in_service_instances"] == 7


def test_scale_in_never_discards_a_message_being_processed():
    # Terminating a busy worker must drain it, not drop its message. This is
    # the conservation property under churn.
    trace = flat_trace(rate_per_minute=120, minutes=30)
    policy = ScheduledPolicy({0.0: 8, 300.0: 2, 600.0: 8, 900.0: 1}, default=8)

    result = simulate(trace, policy, exact_config(initial_instances=8))

    assert (
        result.messages_completed
        + result.messages_in_queue_at_end
        + result.messages_in_service_at_end
        == result.messages_arrived
    )


# --- capacity bounds --------------------------------------------------------


def test_desired_capacity_is_clamped_to_max_instances():
    trace = flat_trace(rate_per_minute=120, minutes=20)

    result = simulate(trace, FixedPolicy(500), exact_config(max_instances=6))

    assert result.ticks["in_service_instances"].max() == 6


def test_desired_capacity_is_clamped_to_min_instances():
    trace = flat_trace(rate_per_minute=10, minutes=20)

    result = simulate(
        trace, FixedPolicy(0), exact_config(min_instances=2, initial_instances=2)
    )

    assert result.ticks["in_service_instances"].min() == 2


# --- ticks and instrumentation ----------------------------------------------


def test_the_policy_is_consulted_on_the_configured_interval():
    trace = flat_trace(rate_per_minute=60, minutes=10)
    policy = RecordingPolicy(FixedPolicy(2))

    simulate(trace, policy, exact_config(scaler_tick_s=60.0))

    assert [obs.t for obs in policy.seen] == [float(m * 60) for m in range(10)]


def test_tick_records_carry_every_required_field():
    trace = flat_trace(rate_per_minute=60, minutes=10)

    result = simulate(trace, FixedPolicy(2), exact_config())

    for column in (
        "t",
        "queue_depth",
        "oldest_message_age",
        "in_service_instances",
        "pending_instances",
        "arrivals",
        "completions",
        "desired_instances",
    ):
        assert column in result.ticks.columns


def test_recorded_arrivals_cover_every_minute_before_the_last_tick():
    # Ticks run at 0..540s for a ten-minute trace, and each records the
    # arrivals since the previous tick. The trace's final minute therefore
    # falls *after* the last tick and is not represented in this column.
    # messages_arrived remains the authoritative total.
    trace = flat_trace(rate_per_minute=60, minutes=10)

    result = simulate(trace, FixedPolicy(4), exact_config())

    assert result.ticks["arrivals"].sum() == 540
    assert result.messages_arrived == 600


def test_billed_instance_seconds_include_booting_capacity():
    # A booting instance costs money before it does any work. Charging only
    # for in-service capacity would let a thrashing policy look free.
    trace = flat_trace(rate_per_minute=1, minutes=10)

    result = simulate(
        trace, FixedPolicy(3), exact_config(initial_instances=0, min_instances=0)
    )

    # Three instances requested at t=0 and paid for across the whole 600s run,
    # including their 180s boot.
    assert result.billed_instance_seconds == pytest.approx(3 * 600, rel=0.02)


# --- no lookahead -----------------------------------------------------------


def test_a_policy_only_ever_sees_arrivals_from_completed_past_minutes():
    # Build a trace that is silent then suddenly loud. A policy that could see
    # ahead would observe the loud minutes before they happen.
    quiet_then_loud = counts([0] * 10 + [500] * 10)
    policy = RecordingPolicy(FixedPolicy(2))

    simulate(quiet_then_loud, policy, exact_config())

    for observation in policy.seen:
        minutes_elapsed = int(observation.t // 60)
        # History may cover only minutes strictly before the current one.
        assert len(observation.recent_arrivals) <= minutes_elapsed
        if minutes_elapsed <= 10:
            assert sum(observation.recent_arrivals) == 0


def test_observed_history_matches_the_trace_up_to_the_current_minute():
    trace = counts([3, 9, 4, 7, 2, 8, 5, 1])
    policy = RecordingPolicy(FixedPolicy(2))

    simulate(trace, policy, exact_config())

    expected = list(trace)
    for observation in policy.seen:
        minutes_elapsed = int(observation.t // 60)
        assert list(observation.recent_arrivals) == expected[:minutes_elapsed]


def test_observed_state_matches_the_simulator_state_at_that_instant():
    trace = flat_trace(rate_per_minute=120, minutes=20)
    policy = RecordingPolicy(FixedPolicy(4))

    result = simulate(trace, policy, exact_config())

    ticks = result.ticks.set_index("t")
    for observation in policy.seen:
        assert observation.queue_depth == ticks.loc[observation.t, "queue_depth"]
        assert (
            observation.in_service_instances
            == ticks.loc[observation.t, "in_service_instances"]
        )


# --- reproducibility --------------------------------------------------------


def test_identical_seeds_produce_identical_runs():
    trace = flat_trace(rate_per_minute=120, minutes=30)
    config = SimConfig(seed=99, service=ServiceTime(2.0, cv=0.8))

    first = simulate(trace, FixedPolicy(4), config)
    second = simulate(trace, FixedPolicy(4), config)

    assert first.messages_completed == second.messages_completed
    pd.testing.assert_frame_equal(first.ticks, second.ticks)


def test_different_seeds_produce_different_runs():
    trace = flat_trace(rate_per_minute=120, minutes=30)

    first = simulate(trace, FixedPolicy(4), SimConfig(seed=1, service=ServiceTime(2.0, 0.8)))
    second = simulate(trace, FixedPolicy(4), SimConfig(seed=2, service=ServiceTime(2.0, 0.8)))

    assert first.messages_completed != second.messages_completed


# --- edge cases -------------------------------------------------------------


def test_an_empty_trace_runs_without_error():
    result = simulate(counts([0, 0, 0]), FixedPolicy(2), exact_config())

    assert result.messages_arrived == 0
    assert result.messages_completed == 0


def test_oldest_message_age_is_zero_when_the_queue_is_empty():
    result = simulate(counts([0, 0, 0]), FixedPolicy(2), exact_config())

    assert (result.ticks["oldest_message_age"] == 0).all()


def test_rejects_an_empty_trace_index():
    with pytest.raises(ValueError):
        simulate(pd.Series([], dtype="int64"), FixedPolicy(1), exact_config())
