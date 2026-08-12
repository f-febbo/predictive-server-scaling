"""Tests for the forecast scoring functions.

Pinball loss is the objective the whole forecasting phase is built around, so
its asymmetry is pinned explicitly rather than assumed. Under-forecasting
causes SLO violations; over-forecasting costs a few cents of compute. A
symmetric loss like MSE or MAPE treats those as equally bad, which is the wrong
objective for this problem.
"""

import numpy as np
import pytest

from src.forecast.losses import coverage, mean_absolute_error, pinball_loss


# --- pinball loss -----------------------------------------------------------


def test_a_perfect_forecast_has_zero_loss():
    actual = np.array([10.0, 20.0, 30.0])

    assert pinball_loss(actual, actual, quantile=0.9) == pytest.approx(0.0)


def test_under_forecasting_is_penalised_by_the_quantile():
    # Actual 10, forecast 8: a shortfall of 2, weighted by q=0.9.
    loss = pinball_loss(np.array([10.0]), np.array([8.0]), quantile=0.9)

    assert loss == pytest.approx(1.8)


def test_over_forecasting_is_penalised_by_one_minus_the_quantile():
    # Actual 8, forecast 10: an excess of 2, weighted by 1-q = 0.1.
    loss = pinball_loss(np.array([8.0]), np.array([10.0]), quantile=0.9)

    assert loss == pytest.approx(0.2)


def test_a_high_quantile_punishes_shortfalls_nine_times_harder_than_excess():
    # This asymmetry is the entire point: it is what makes the model learn to
    # provision above the mean rather than at it.
    shortfall = pinball_loss(np.array([10.0]), np.array([8.0]), quantile=0.9)
    excess = pinball_loss(np.array([8.0]), np.array([10.0]), quantile=0.9)

    assert shortfall == pytest.approx(9.0 * excess)


def test_the_median_quantile_is_symmetric():
    shortfall = pinball_loss(np.array([10.0]), np.array([8.0]), quantile=0.5)
    excess = pinball_loss(np.array([8.0]), np.array([10.0]), quantile=0.5)

    assert shortfall == pytest.approx(excess)


def test_median_pinball_loss_is_half_the_absolute_error():
    actual = np.array([10.0, 20.0, 5.0])
    forecast = np.array([8.0, 25.0, 5.0])

    loss = pinball_loss(actual, forecast, quantile=0.5)

    assert loss == pytest.approx(0.5 * mean_absolute_error(actual, forecast))


def test_a_low_quantile_punishes_excess_harder_than_shortfall():
    shortfall = pinball_loss(np.array([10.0]), np.array([8.0]), quantile=0.1)
    excess = pinball_loss(np.array([8.0]), np.array([10.0]), quantile=0.1)

    assert excess == pytest.approx(9.0 * shortfall)


def test_loss_is_averaged_over_the_sample():
    actual = np.array([10.0, 10.0])
    forecast = np.array([8.0, 8.0])

    assert pinball_loss(actual, forecast, quantile=0.9) == pytest.approx(1.8)


def test_pinball_loss_rejects_a_quantile_outside_zero_to_one():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            pinball_loss(np.array([1.0]), np.array([1.0]), quantile=bad)


def test_pinball_loss_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        pinball_loss(np.array([1.0, 2.0]), np.array([1.0]), quantile=0.5)


def test_pinball_loss_rejects_an_empty_sample():
    with pytest.raises(ValueError):
        pinball_loss(np.array([]), np.array([]), quantile=0.5)


# --- coverage ---------------------------------------------------------------


def test_coverage_is_the_share_of_actuals_at_or_below_the_forecast():
    # The p90 forecast should sit above the actual about 90% of the time. This
    # is the calibration check: a model can have a good pinball loss and still
    # be systematically miscalibrated.
    actual = np.array([1.0, 2.0, 3.0, 4.0, 20.0])
    forecast = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

    assert coverage(actual, forecast) == pytest.approx(0.8)


def test_full_coverage_when_the_forecast_never_falls_short():
    actual = np.array([1.0, 2.0, 3.0])
    forecast = np.array([10.0, 10.0, 10.0])

    assert coverage(actual, forecast) == pytest.approx(1.0)


def test_zero_coverage_when_the_forecast_always_falls_short():
    actual = np.array([10.0, 20.0])
    forecast = np.array([1.0, 2.0])

    assert coverage(actual, forecast) == pytest.approx(0.0)


def test_an_exact_match_counts_as_covered():
    # Forecasting exactly the actual demand is not a shortfall.
    assert coverage(np.array([5.0]), np.array([5.0])) == pytest.approx(1.0)


# --- mean absolute error ----------------------------------------------------


def test_mean_absolute_error_is_the_average_absolute_gap():
    actual = np.array([10.0, 20.0])
    forecast = np.array([8.0, 25.0])

    assert mean_absolute_error(actual, forecast) == pytest.approx(3.5)


def test_mean_absolute_error_is_symmetric():
    over = mean_absolute_error(np.array([10.0]), np.array([12.0]))
    under = mean_absolute_error(np.array([10.0]), np.array([8.0]))

    assert over == pytest.approx(under)
