"""Turning an estimate of load into an instance count, via Little's Law.

Kept separate from the policies deliberately. Every policy in the project —
reactive and predictive alike — funnels through these two functions, so they
are the one place where the queueing arithmetic can be checked in isolation.
Inlining this into each policy would mean an error moved all of them together
and stayed invisible in a relative comparison.

Little's Law: L = lambda * W. The average number of items in a system is the
arrival rate times the average time each spends there. For a worker fleet, an
instance holds one message for `service_seconds`, so serving `lambda` messages
per second needs `lambda * service_seconds` instances busy at all times.
"""

from __future__ import annotations

import math


def instances_for_arrival_rate(
    arrival_rate_per_s: float,
    service_seconds: float,
    target_utilization: float = 1.0,
) -> int:
    """Instances needed to keep pace with an arrival rate.

    Args:
        arrival_rate_per_s: Messages arriving per second.
        service_seconds: Mean seconds to process one message.
        target_utilization: Fraction of capacity to run at. Below 1.0 this
            buys headroom: at exactly 1.0 the fleet has no slack, so any
            fluctuation immediately builds a queue.

    Returns:
        Instance count, always rounded up. Rounding down would leave the fleet
        deliberately short of the load it was just told about.
    """
    if arrival_rate_per_s < 0:
        raise ValueError(f"arrival rate must be non-negative, got {arrival_rate_per_s}")
    if service_seconds <= 0:
        raise ValueError(f"service_seconds must be positive, got {service_seconds}")
    if not 0.0 < target_utilization <= 1.0:
        raise ValueError(
            f"target_utilization must be in (0, 1], got {target_utilization}"
        )

    required = arrival_rate_per_s * service_seconds / target_utilization
    return math.ceil(required)


def backlog_target_per_instance(
    acceptable_latency_s: float, service_seconds: float
) -> float:
    """Messages per instance that still clear within the latency budget.

    This is the target for the backlog-per-instance tracking policy, which is
    the approach AWS recommends for SQS-backed Auto Scaling groups. If a worker
    clears one message every `service_seconds`, then holding no more than
    `acceptable_latency_s / service_seconds` messages behind each worker keeps
    the oldest message inside the budget.

    Deliberately fractional. A target below 1.0 is meaningful — it means
    scaling out before any single worker has even one message waiting — and
    rounding would coarsen the tuning sweep for no reason.
    """
    if acceptable_latency_s <= 0:
        raise ValueError(
            f"acceptable_latency_s must be positive, got {acceptable_latency_s}"
        )
    if service_seconds <= 0:
        raise ValueError(f"service_seconds must be positive, got {service_seconds}")

    return acceptable_latency_s / service_seconds
