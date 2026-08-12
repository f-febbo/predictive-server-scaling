# Phase 4 — predictive vs reactive on the held-out period

30s service, 180s boot, 60s SLO, 1h warmup excluded.

Cheapest configuration of each policy reaching a given SLO violation rate.

| Target | Policy | Cost (inst-h) | Actual | p99 age | Configuration |
|---|---|---|---|---|---|
| ≤ 5.0% | Static | 21,600 | 3.92% | 1989.5s | n=75 |
| ≤ 5.0% | Reactive: backlog/instance | 17,911 | 3.89% | 80.3s | budget=60s, floor=60 |
| ≤ 5.0% | Reactive: arrival rate | 11,826 | 3.40% | 99.0s | util=0.9, window=10m |
| ≤ 5.0% | Predictive: LightGBM + reactive | 13,312 | 0.26% | 33.2s | q=0.5, util=1 |
| ≤ 5.0% | Predictive: LightGBM alone | 12,207 | 4.91% | 8959.9s | q=0.9, util=1 |
| ≤ 5.0% | Seasonal naive + reactive | 13,709 | 0.09% | 22.8s | q=0.5, util=1 |
| ≤ 5.0% | Seasonal naive adj. + reactive | 13,515 | 0.13% | 25.2s | q=0.5, util=1 |
| | | | | | |
| ≤ 1.0% | Static | 27,360 | 0.21% | 7.7s | n=95 |
| ≤ 1.0% | Reactive: backlog/instance | 18,389 | 0.53% | 51.9s | budget=30s, floor=60 |
| ≤ 1.0% | Reactive: arrival rate | 13,246 | 0.82% | 53.0s | util=0.8, window=15m |
| ≤ 1.0% | Predictive: LightGBM + reactive | 13,312 | 0.26% | 33.2s | q=0.5, util=1 |
| ≤ 1.0% | Predictive: LightGBM alone | 13,113 | 0.27% | 36.1s | q=0.7, util=0.85 |
| ≤ 1.0% | Seasonal naive + reactive | 13,709 | 0.09% | 22.8s | q=0.5, util=1 |
| ≤ 1.0% | Seasonal naive adj. + reactive | 13,515 | 0.13% | 25.2s | q=0.5, util=1 |
| | | | | | |
| ≤ 0.1% | Static | 28,800 | 0.00% | 0.0s | n=100 |
| ≤ 0.1% | Reactive: backlog/instance | 23,221 | 0.03% | 21.7s | budget=30s, floor=80 |
| ≤ 0.1% | Reactive: arrival rate | 15,128 | 0.09% | 20.0s | util=0.7, window=15m |
| ≤ 0.1% | Predictive: LightGBM + reactive | 13,405 | 0.08% | 17.3s | q=0.8, util=1 |
| ≤ 0.1% | Predictive: LightGBM alone | 13,611 | 0.07% | 14.6s | q=0.8, util=0.85 |
| ≤ 0.1% | Seasonal naive + reactive | 13,709 | 0.09% | 22.8s | q=0.5, util=1 |
| ≤ 0.1% | Seasonal naive adj. + reactive | 13,807 | 0.03% | 14.8s | q=0.7, util=1 |
| | | | | | |
| ≤ 0.0% | Static | 28,800 | 0.00% | 0.0s | n=100 |
| ≤ 0.0% | Reactive: backlog/instance | 23,353 | 0.00% | 8.1s | budget=15s, floor=80 |
| ≤ 0.0% | Reactive: arrival rate | 21,120 | 0.00% | 0.0s | util=0.5, window=15m |
| ≤ 0.0% | Predictive: LightGBM + reactive | 14,482 | 0.00% | 3.6s | q=0.9, util=0.85 |
| ≤ 0.0% | Predictive: LightGBM alone | 14,343 | 0.00% | 6.3s | q=0.9, util=0.85 |
| ≤ 0.0% | Seasonal naive + reactive | 18,713 | 0.00% | 0.0s | q=0.95, util=0.85 |
| ≤ 0.0% | Seasonal naive adj. + reactive | 15,730 | 0.00% | 4.6s | q=0.8, util=0.85 |
| | | | | | |
