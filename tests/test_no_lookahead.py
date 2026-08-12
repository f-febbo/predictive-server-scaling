"""Adversarial tests for the no-lookahead guarantee.

The other tests check that the observation *looks* right. These try to break
it: a policy that actively hunts for future data through everything it is
handed must come up empty. Lookahead leakage would silently invalidate every
comparison in the project, and it is the kind of bug that makes results look
better rather than obviously wrong, so it needs adversarial pressure rather
than a polite assertion.
"""

import dataclasses
import inspect

import pandas as pd
import pytest

from src.sim.config import SimConfig
from src.sim.observation import Observation
from src.sim.service import ServiceTime
from src.sim.simulator import simulate

SENTINEL = 999_999


def counts(values: list[int]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1min", name="ts")
    return pd.Series(values, index=index, name="arrivals")


def exact_config(**overrides) -> SimConfig:
    defaults = dict(
        service=ServiceTime(mean_seconds=2.0, cv=0.0),
        arrival_mode="uniform",
        max_instances=10_000,
    )
    return SimConfig(**{**defaults, **overrides})


def reachable_scalars(value, seen=None):
    """Every scalar reachable from `value` by walking dataclasses and containers.

    This is the search a determined policy would perform to find data it was
    not meant to have.
    """
    seen = seen if seen is not None else set()
    if id(value) in seen:
        return
    seen.add(id(value))

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from reachable_scalars(getattr(value, field.name), seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from reachable_scalars(key, seen)
            yield from reachable_scalars(item, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from reachable_scalars(item, seen)
    else:
        yield value


class ProbingPolicy:
    """Records every scalar it could reach from each observation."""

    def __init__(self, instances: int = 2):
        self.instances = instances
        self.observations: list[Observation] = []
        self.reachable: list[set] = []

    def decide(self, t: float, obs: Observation) -> int:
        self.observations.append(obs)
        self.reachable.append(
            {value for value in reachable_scalars(obs) if isinstance(value, (int, float))}
        )
        return self.instances


def test_a_probing_policy_cannot_reach_a_future_arrival_count():
    # The second half of the trace carries a value that never occurs in the
    # first half. If any of it reaches a policy before it happens, that value
    # will turn up in the reachable set.
    trace = counts([7] * 10 + [SENTINEL] * 10)
    policy = ProbingPolicy()

    simulate(trace, policy, exact_config())

    for observation, reachable in zip(policy.observations, policy.reachable):
        minute = int(observation.t // 60)
        if minute < 10:
            assert SENTINEL not in reachable, (
                f"future arrival count leaked into an observation at t={observation.t}"
            )


def test_the_policy_receives_nothing_but_the_time_and_the_observation():
    # A reference to the simulator, the config, or the trace would be an
    # escape hatch around every other guarantee here.
    trace = counts([5] * 5)
    captured: list[tuple] = []

    class ArgumentCapturingPolicy:
        def decide(self, *args, **kwargs):
            captured.append((args, kwargs))
            return 1

    simulate(trace, ArgumentCapturingPolicy(), exact_config())

    assert captured
    for args, kwargs in captured:
        assert kwargs == {}
        assert len(args) == 2
        assert isinstance(args[0], float)
        assert isinstance(args[1], Observation)


def test_the_observation_exposes_no_callables_to_fetch_more_data():
    # A method or bound callable on a field could lazily fetch future state.
    trace = counts([5] * 5)
    policy = ProbingPolicy()

    simulate(trace, policy, exact_config())

    for observation in policy.observations:
        for field in dataclasses.fields(observation):
            value = getattr(observation, field.name)
            assert not callable(value)
            assert not inspect.isgenerator(value)


def test_arrival_history_never_includes_the_minute_in_progress():
    # The minute containing `t` has not finished, so a real system could not
    # have its count yet. Off-by-one here would hand the policy a partial
    # measurement of the very burst it is meant to be surprised by.
    trace = counts(list(range(1, 21)))
    policy = ProbingPolicy()

    simulate(trace, policy, exact_config())

    for observation in policy.observations:
        minute = int(observation.t // 60)
        assert len(observation.recent_arrivals) == minute
        if minute > 0:
            # The most recent entry is the minute that just ended.
            assert observation.recent_arrivals[-1] == minute


def test_a_policy_cannot_influence_the_simulator_by_mutating_its_observation():
    trace = counts([60] * 10)

    class MutatingPolicy:
        def decide(self, t: float, obs: Observation) -> int:
            with pytest.raises(dataclasses.FrozenInstanceError):
                obs.queue_depth = -1
            return 2

    result = simulate(trace, MutatingPolicy(), exact_config())

    assert (result.ticks["queue_depth"] >= 0).all()


def test_capacity_cannot_be_conjured_before_the_boot_delay_even_by_a_greedy_policy():
    # The last defence: however early or aggressively a policy asks, physics
    # still applies and the instances are not usable until they have booted.
    trace = counts([600] * 20)

    class GreedyPolicy:
        def decide(self, t: float, obs: Observation) -> int:
            return 10_000

    result = simulate(
        trace, GreedyPolicy(), exact_config(initial_instances=1, boot_time_s=300.0)
    )
    ticks = result.ticks.set_index("t")

    for t in (0.0, 60.0, 120.0, 180.0, 240.0):
        assert ticks.loc[t, "in_service_instances"] == 1
    assert ticks.loc[300.0, "in_service_instances"] > 1
