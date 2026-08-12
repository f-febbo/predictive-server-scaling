"""What a scaling policy is allowed to see, and the interface it implements.

Everything here exists to make lookahead structurally impossible rather than
merely discouraged. `Observation` is a frozen dataclass of plain values: no
Series, no ndarray, no back-reference to the simulator. A policy handed one of
these has no object graph to walk toward data dated after `t`, so it cannot
cheat even by accident — and a future field that would allow it fails the
test suite rather than quietly corrupting every result.

Each field also corresponds to something a real deployment could actually read
from CloudWatch, so a policy that works here is implementable for real:

    queue_depth                 ApproximateNumberOfMessagesVisible
    oldest_message_age          ApproximateAgeOfOldestMessage
    in_service_instances        GroupInServiceInstances
    pending_instances           GroupPendingInstances
    recent_arrivals             NumberOfMessagesSent, per minute
    completions_since_last_tick NumberOfMessagesDeleted
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True, slots=True)
class Observation:
    """A snapshot of what is knowable at time `t`.

    Attributes:
        t: Simulation time in seconds since the start of the trace.
        queue_depth: Messages waiting, not counting those being processed.
        oldest_message_age: Seconds the oldest waiting message has waited.
            Zero when the queue is empty. This is the SLI: it is what users
            feel, whereas queue depth is merely what we control.
        in_service_instances: Booted, running workers.
        pending_instances: Requested workers still inside their boot delay.
            Already being paid for, not yet doing any work.
        recent_arrivals: Per-minute arrival counts for the completed minutes
            up to `t`, oldest first.
        completions_since_last_tick: Messages processed since the last tick.
    """

    t: float
    queue_depth: int
    oldest_message_age: float
    in_service_instances: int
    pending_instances: int
    recent_arrivals: tuple[int, ...]
    completions_since_last_tick: int

    def recent_arrival_rate(self, minutes: int = 5) -> float:
        """Mean arrivals per second over the last `minutes` completed minutes.

        Averaging over fewer minutes tracks bursts faster but reacts to noise;
        over more it is steadier but lags. Policies choose their own window.

        When less history exists than requested, the average is taken over what
        is actually there. Dividing by a window that has not elapsed yet would
        understate the rate and under-provision at the start of a run.
        """
        if not self.recent_arrivals or minutes <= 0:
            return 0.0

        window = self.recent_arrivals[-minutes:]
        return sum(window) / (len(window) * SECONDS_PER_MINUTE)


@runtime_checkable
class ScalingPolicy(Protocol):
    """A swappable capacity decision rule.

    Implementations must be pure functions of `(t, obs)` plus their own
    internal state. They are never handed the simulator or the trace.
    """

    def decide(self, t: float, obs: Observation) -> int:
        """Return the desired total instance count at time `t`."""
        ...
