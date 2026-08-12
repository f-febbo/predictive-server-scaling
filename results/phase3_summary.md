# Phase 3 — reactive baseline frontiers

Training split, 30s mean service time, 180s boot delay, 60s SLO on age of oldest message, 1h warmup excluded.

Cheapest configuration of each policy reaching a given SLO violation rate.
A dash means no configuration in the sweep reached that target.

| Target violation rate | Policy | Cost (inst-h) | Actual violation | p99 age | Configuration |
|---|---|---|---|---|---|
| ≤ 10.0% | Static overprovisioning | 69,120 | 9.06% | 2329.9s | n=60 |
| ≤ 10.0% | Target tracking: backlog/instance | 67,347 | 5.17% | 80.3s | budget=30s, floor=40 |
| ≤ 10.0% | Target tracking: arrival rate | 44,571 | 6.85% | 143.0s | util=0.9, window=15m |
| | | | | | |
| ≤ 5.0% | Static overprovisioning | 74,880 | 3.65% | 1047.7s | n=65 |
| ≤ 5.0% | Target tracking: backlog/instance | 69,784 | 4.40% | 97.7s | budget=90s, floor=60 |
| ≤ 5.0% | Target tracking: arrival rate | 44,718 | 3.68% | 102.8s | util=0.9, window=10m |
| | | | | | |
| ≤ 1.0% | Static overprovisioning | 86,400 | 0.67% | 7.5s | n=75 |
| ≤ 1.0% | Target tracking: backlog/instance | 70,234 | 0.60% | 53.8s | budget=45s, floor=60 |
| ≤ 1.0% | Target tracking: arrival rate | 50,251 | 0.82% | 54.5s | util=0.8, window=10m |
| | | | | | |
| ≤ 0.1% | Static overprovisioning | 103,680 | 0.06% | 0.0s | n=90 |
| ≤ 0.1% | Target tracking: backlog/instance | 72,351 | 0.02% | 24.8s | budget=15s, floor=60 |
| ≤ 0.1% | Target tracking: arrival rate | 66,631 | 0.05% | 8.5s | util=0.6, window=15m |
| | | | | | |
| ≤ 0.0% | Static overprovisioning | 109,440 | 0.00% | 0.0s | n=95 |
| ≤ 0.0% | Target tracking: backlog/instance | 92,321 | 0.00% | 0.0s | budget=15s, floor=80 |
| ≤ 0.0% | Target tracking: arrival rate | 132,775 | 0.00% | 0.0s | util=0.3, window=15m |
| | | | | | |
