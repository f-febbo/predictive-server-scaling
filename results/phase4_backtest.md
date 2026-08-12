# Phase 4 — forecast backtest

Rolling origin, retrained every 7 days, 15-minute horizon, held-out period only.

Coverage is the calibration check: a q0.9 forecast should sit above the actual about 90% of the time.

| Model | Quantile | MAE | Pinball loss | Coverage | n |
|---|---|---|---|---|---|
| seasonal_naive | 0.5 | 14.48 | 7.238 | 49.8% | 17,265 |
| seasonal_naive | 0.7 | 16.18 | 6.696 | 67.9% | 17,265 |
| seasonal_naive | 0.8 | 18.82 | 5.648 | 77.1% | 17,265 |
| seasonal_naive | 0.9 | 24.27 | 3.839 | 87.1% | 17,265 |
| seasonal_naive | 0.95 | 31.25 | 2.452 | 93.1% | 17,265 |
| seasonal_naive | 0.99 | 52.52 | 0.733 | 98.4% | 17,265 |
| seasonal_naive_adjusted | 0.5 | 10.89 | 5.445 | 51.0% | 17,265 |
| seasonal_naive_adjusted | 0.7 | 12.78 | 4.898 | 70.6% | 17,265 |
| seasonal_naive_adjusted | 0.8 | 15.27 | 4.035 | 80.3% | 17,265 |
| seasonal_naive_adjusted | 0.9 | 20.33 | 2.637 | 90.1% | 17,265 |
| seasonal_naive_adjusted | 0.95 | 25.88 | 1.628 | 94.9% | 17,265 |
| seasonal_naive_adjusted | 0.99 | 40.91 | 0.474 | 99.0% | 17,265 |
| lightgbm | 0.5 | 7.70 | 3.848 | 48.1% | 17,265 |
| lightgbm | 0.7 | 8.33 | 3.482 | 66.9% | 17,265 |
| lightgbm | 0.8 | 9.61 | 2.907 | 76.6% | 17,265 |
| lightgbm | 0.9 | 12.34 | 1.963 | 86.6% | 17,265 |
| lightgbm | 0.95 | 15.09 | 1.212 | 92.3% | 17,265 |
| lightgbm | 0.99 | 21.59 | 0.396 | 97.3% | 17,265 |

## Best model by pinball loss at each quantile

| Quantile | Best model | Pinball loss |
|---|---|---|
| 0.5 | lightgbm | 3.848 |
| 0.7 | lightgbm | 3.482 |
| 0.8 | lightgbm | 2.907 |
| 0.9 | lightgbm | 1.963 |
| 0.95 | lightgbm | 1.212 |
| 0.99 | lightgbm | 0.396 |
