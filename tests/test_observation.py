"""Tests for the observation a policy receives.

The project's credibility rests on policies being unable to see the future.
Convention is not enough — a comment saying "don't look ahead" is not a
control. The enforcement here is structural: the observation carries only
plain immutable values, so there is no object graph a policy could walk back
to the simulator, the arrival trace, or anything dated after `t`.
"""

import dataclasses

import pytest

from src.sim.observation import Observation

ALLOWED_FIELD_TYPES = (int, float, bool, str, type(None))


def an_observation(**overrides) -> Observation:
    defaults = dict(
        t=600.0,
        queue_depth=12,
        oldest_message_age=4.5,
        in_service_instances=3,
        pending_instances=1,
        recent_arrivals=(10, 12, 9),
        completions_since_last_tick=25,
    )
    return Observation(**{**defaults, **overrides})


def test_observation_is_immutable():
    # A policy must not be able to edit what it was shown and have that leak
    # into the simulator's state.
    observation = an_observation()

    with pytest.raises(dataclasses.FrozenInstanceError):
        observation.queue_depth = 999


def test_observation_carries_only_plain_immutable_values():
    # This is the structural no-lookahead guarantee. If a field ever held a
    # Series, an ndarray, or a reference to the simulator, a policy could
    # reach through it to data after `t`. Adding such a field breaks this test.
    observation = an_observation()

    for field in dataclasses.fields(observation):
        value = getattr(observation, field.name)
        if isinstance(value, tuple):
            assert all(isinstance(item, ALLOWED_FIELD_TYPES) for item in value), (
                f"field {field.name!r} contains non-primitive elements"
            )
        else:
            assert isinstance(value, ALLOWED_FIELD_TYPES), (
                f"field {field.name!r} is a {type(value).__name__}, which could "
                "expose data after t"
            )


def test_recent_arrivals_is_a_tuple_so_it_cannot_be_mutated_or_grown():
    observation = an_observation()

    with pytest.raises(AttributeError):
        observation.recent_arrivals.append(5)


def test_observation_rejects_unknown_fields():
    # Guards against a policy being handed extra state by accident.
    with pytest.raises(TypeError):
        Observation(
            t=0.0,
            queue_depth=0,
            oldest_message_age=0.0,
            in_service_instances=1,
            pending_instances=0,
            recent_arrivals=(),
            completions_since_last_tick=0,
            future_arrivals=(100, 200),
        )


def test_arrival_rate_helper_converts_recent_counts_to_per_second():
    # Policies need arrivals per second to apply Little's Law; doing the
    # conversion once here stops each policy reimplementing it.
    observation = an_observation(recent_arrivals=(60, 120, 60))

    assert observation.recent_arrival_rate(minutes=3) == pytest.approx(240 / 180)


def test_arrival_rate_helper_uses_only_the_most_recent_minutes():
    observation = an_observation(recent_arrivals=(0, 0, 120))

    assert observation.recent_arrival_rate(minutes=1) == pytest.approx(2.0)


def test_arrival_rate_helper_handles_a_shorter_history_than_requested():
    # Early in a run there is less history than the policy asks for; average
    # over what exists rather than dividing by a window that never happened.
    observation = an_observation(recent_arrivals=(120,))

    assert observation.recent_arrival_rate(minutes=10) == pytest.approx(2.0)


def test_arrival_rate_is_zero_when_there_is_no_history():
    observation = an_observation(recent_arrivals=())

    assert observation.recent_arrival_rate(minutes=5) == 0.0
