"""Discrete-event simulator for a queue-backed worker fleet.

All policy comparison happens here: offline, reproducible, and free. Iterating
against live AWS would be slow, expensive, and impossible to hold still enough
for a fair comparison.

The model:

    arrivals -> unbounded FIFO queue -> pool of workers, one message each

A scaler is consulted every `scaler_tick_s` and returns a desired instance
count. Capacity it requests does not arrive until `boot_time_s` later, which is
the delay the whole project is about. Capacity it releases is drained rather
than killed, so a message in flight is never lost.

Time is in seconds from the start of the trace. The run covers ``[0, end_time)``
where ``end_time`` is the trace length; work still in progress when that
boundary arrives is reported rather than discarded, so message conservation
holds exactly.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.sim.arrivals import SECONDS_PER_MINUTE, arrival_times_from_counts
from src.sim.config import SimConfig
from src.sim.observation import Observation, ScalingPolicy

# Ordering of events that land on the same instant. Completions free a worker
# before new work is considered, and the scaler observes last so it sees the
# state the instant has settled into.
_PRIORITY_COMPLETION = 0
_PRIORITY_READY = 1
_PRIORITY_ARRIVAL = 2
_PRIORITY_TICK = 3

TICK_COLUMNS = (
    "t",
    "queue_depth",
    "oldest_message_age",
    "in_service_instances",
    "pending_instances",
    "arrivals",
    "completions",
    "desired_instances",
)


@dataclass
class SimResult:
    """Outcome of one simulated run.

    Attributes:
        ticks: Per-tick instrumentation, one row per scaler consultation.
        messages_arrived: Messages the trace delivered.
        messages_completed: Messages fully processed before the run ended.
        messages_in_queue_at_end: Messages still waiting when the run ended.
        messages_in_service_at_end: Messages being processed when it ended.
        billed_instance_seconds: Integral of (in service + booting) over time.
            Booting instances are charged for: they cost money before they do
            any work, and ignoring that would make a thrashing policy look free.
        messages: Per-message timings, or None unless `record_messages` was set.
    """

    ticks: pd.DataFrame
    messages_arrived: int
    messages_completed: int
    messages_in_queue_at_end: int
    messages_in_service_at_end: int
    billed_instance_seconds: float
    messages: pd.DataFrame | None = None


def simulate(
    arrival_counts: pd.Series, policy: ScalingPolicy, config: SimConfig | None = None
) -> SimResult:
    """Replay a trace against a scaling policy.

    Args:
        arrival_counts: Arrivals per minute, in trace order.
        policy: Decision rule, consulted every `config.scaler_tick_s`.
        config: Fleet parameters; defaults to `SimConfig()`.

    Returns:
        A `SimResult` with per-tick instrumentation and run totals.
    """
    config = config or SimConfig()
    if len(arrival_counts) == 0:
        raise ValueError("arrival_counts is empty; nothing to simulate")

    rng = np.random.default_rng(config.seed)
    counts = [int(value) for value in arrival_counts]
    end_time = len(counts) * SECONDS_PER_MINUTE

    arrival_times = arrival_times_from_counts(
        arrival_counts,
        mode=config.arrival_mode,
        rng=rng if config.arrival_mode == "poisson" else None,
    )
    total_messages = len(arrival_times)
    # One service time per message, drawn up front so a run is reproducible
    # regardless of the order the policy happens to dispatch work in.
    service_times = config.service.sample(total_messages, rng)

    state = _RunState(config, counts, arrival_times, service_times, end_time)
    state.run(policy)

    return SimResult(
        ticks=pd.DataFrame(state.tick_rows, columns=list(TICK_COLUMNS)),
        messages_arrived=state.next_arrival,
        messages_completed=state.completed,
        messages_in_queue_at_end=len(state.queue),
        messages_in_service_at_end=state.busy,
        billed_instance_seconds=state.billed_instance_seconds,
        messages=state.message_frame(),
    )


class _RunState:
    """Mutable state of a single run.

    Kept in one object so the event handlers can share it without threading a
    dozen variables through every call.
    """

    def __init__(
        self,
        config: SimConfig,
        counts: list[int],
        arrival_times: np.ndarray,
        service_times: np.ndarray,
        end_time: float,
    ):
        self.config = config
        self.counts = counts
        self.arrival_times = arrival_times
        self.service_times = service_times
        self.end_time = end_time

        # Queue holds message indices; arrival_times[i] gives its enqueue time.
        self.queue: deque[int] = deque()
        self.completion_heap: list[tuple[float, int]] = []
        self.ready_heap: list[float] = []

        self.in_service = config.initial_instances
        self.pending = 0
        self.busy = 0
        # Busy workers that must exit when they finish their current message.
        self.terminating = 0

        self.next_arrival = 0
        self.completed = 0
        self.arrivals_since_tick = 0
        self.completions_since_tick = 0

        self.last_scale_in_time = -float("inf")
        self.billed_instance_seconds = 0.0
        self.last_billed_at = 0.0

        self.tick_rows: list[tuple] = []

        if config.record_messages:
            self.started_at = np.full(len(arrival_times), np.nan)
            self.completed_at = np.full(len(arrival_times), np.nan)
        else:
            self.started_at = None
            self.completed_at = None

    # --- main loop ----------------------------------------------------------

    def run(self, policy: ScalingPolicy) -> None:
        next_tick = 0.0

        while True:
            when, priority = self._next_event(next_tick)
            if when is None or when >= self.end_time:
                break

            if priority == _PRIORITY_COMPLETION:
                self._handle_completion(when)
            elif priority == _PRIORITY_READY:
                self._handle_instance_ready(when)
            elif priority == _PRIORITY_ARRIVAL:
                self._handle_arrival(when)
            else:
                self._handle_tick(when, policy)
                next_tick += self.config.scaler_tick_s

        self._accrue_cost(self.end_time)

    def _next_event(self, next_tick: float) -> tuple[float | None, int]:
        """Earliest pending event, ties broken by the priority constants."""
        candidates: list[tuple[float, int]] = []

        if self.next_arrival < len(self.arrival_times):
            candidates.append(
                (float(self.arrival_times[self.next_arrival]), _PRIORITY_ARRIVAL)
            )
        if self.completion_heap:
            candidates.append((self.completion_heap[0][0], _PRIORITY_COMPLETION))
        if self.ready_heap:
            candidates.append((self.ready_heap[0], _PRIORITY_READY))
        if next_tick < self.end_time:
            candidates.append((next_tick, _PRIORITY_TICK))

        if not candidates:
            return None, -1
        return min(candidates)

    # --- event handlers -----------------------------------------------------

    def _handle_arrival(self, t: float) -> None:
        index = self.next_arrival
        self.next_arrival += 1
        self.arrivals_since_tick += 1
        self.queue.append(index)
        self._dispatch(t)

    def _handle_completion(self, t: float) -> None:
        heapq.heappop(self.completion_heap)
        self.busy -= 1
        self.completed += 1
        self.completions_since_tick += 1

        if self.terminating > 0:
            # This worker was released during a scale-in and stayed only long
            # enough to finish its message.
            self._accrue_cost(t)
            self.terminating -= 1
            self.in_service -= 1

        self._dispatch(t)

    def _handle_instance_ready(self, t: float) -> None:
        heapq.heappop(self.ready_heap)
        # Cost is unchanged: a booting instance was already being charged for.
        self.pending -= 1
        self.in_service += 1
        self._dispatch(t)

    def _handle_tick(self, t: float, policy: ScalingPolicy) -> None:
        observation = self._observe(t)
        desired = policy.decide(t, observation)
        desired = _clamp(int(desired), self.config.min_instances, self.config.max_instances)

        self.tick_rows.append(
            (
                t,
                observation.queue_depth,
                observation.oldest_message_age,
                observation.in_service_instances,
                observation.pending_instances,
                self.arrivals_since_tick,
                self.completions_since_tick,
                desired,
            )
        )
        self.arrivals_since_tick = 0
        self.completions_since_tick = 0

        self._apply_capacity_decision(t, desired)

    # --- mechanics ----------------------------------------------------------

    def _dispatch(self, t: float) -> None:
        """Start as much queued work as there are idle workers for."""
        while self.queue and self.busy < self.in_service:
            index = self.queue.popleft()
            finish_at = t + float(self.service_times[index])
            self.busy += 1
            heapq.heappush(self.completion_heap, (finish_at, index))

            if self.started_at is not None:
                self.started_at[index] = t
                self.completed_at[index] = finish_at

    def _apply_capacity_decision(self, t: float, desired: int) -> None:
        total_capacity = self.in_service + self.pending

        if desired > total_capacity:
            self._launch(t, desired - total_capacity)
        elif desired < self.in_service:
            self._release(t, self.in_service - desired)

    def _launch(self, t: float, count: int) -> None:
        """Request `count` instances; they become useful after the boot delay."""
        self._accrue_cost(t)
        self.pending += count
        ready_at = t + self.config.boot_time_s
        for _ in range(count):
            heapq.heappush(self.ready_heap, ready_at)

    def _release(self, t: float, excess: int) -> None:
        """Give back up to `excess` instances, subject to the scale-in cooldown.

        Pending instances are deliberately not cancelled. A real launch that is
        already in flight generally completes and is billed for, and leaving it
        in place keeps the cost of an indecisive policy visible rather than
        letting it retract requests for free.
        """
        if t - self.last_scale_in_time < self.config.scale_in_cooldown_s:
            return

        self._accrue_cost(t)

        idle = self.in_service - self.busy
        removed_now = min(excess, idle)
        self.in_service -= removed_now

        # Whatever is left can only come from busy workers, which drain: they
        # finish the message in hand and then exit. Killing them would lose a
        # message and break conservation.
        draining = min(excess - removed_now, self.busy - self.terminating)
        self.terminating += draining

        if removed_now or draining:
            self.last_scale_in_time = t

    def _accrue_cost(self, t: float) -> None:
        """Charge for capacity held since the last accounting point."""
        self.billed_instance_seconds += (self.in_service + self.pending) * (
            t - self.last_billed_at
        )
        self.last_billed_at = t

    # --- observation --------------------------------------------------------

    def _observe(self, t: float) -> Observation:
        """Build the policy's view of the world at `t`.

        Only completed minutes are exposed. The minute containing `t` is still
        in progress, and a real system could not have its count yet.
        """
        minutes_elapsed = int(t // SECONDS_PER_MINUTE)
        history_start = max(0, minutes_elapsed - self.config.arrival_history_minutes)
        recent = tuple(self.counts[history_start:minutes_elapsed])

        if self.queue:
            oldest_age = t - float(self.arrival_times[self.queue[0]])
        else:
            oldest_age = 0.0

        return Observation(
            t=t,
            queue_depth=len(self.queue),
            oldest_message_age=oldest_age,
            in_service_instances=self.in_service,
            pending_instances=self.pending,
            recent_arrivals=recent,
            completions_since_last_tick=self.completions_since_tick,
        )

    def message_frame(self) -> pd.DataFrame | None:
        """Per-message timings for the messages that started processing."""
        if self.started_at is None:
            return None

        started = ~np.isnan(self.started_at)
        return pd.DataFrame(
            {
                "arrived_at": self.arrival_times[started],
                "started_at": self.started_at[started],
                "completed_at": self.completed_at[started],
            }
        )


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
