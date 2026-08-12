"""Feature construction for the arrival-rate forecast.

We forecast the arrival rate — the exogenous load — and nothing else. Queue
depth and CPU utilisation are tempting because they are right there in the
metrics, but both are contaminated by our own scaler's past actions: a model
trained on them learns to predict its own behaviour, and the feedback loop
quietly corrupts it in a way that is very hard to detect from the metrics
afterwards. Arrivals happen whatever we do with capacity.

Every feature row indexed at time `t` is built strictly from series values
before `t`. Lags start at 1, and rolling windows are computed on the shifted
series so that the minute containing `t` — which has not finished yet in a
real deployment — is never read. Calendar features come from the timestamp
itself, which is not lookahead: at time `t` we know what time it is.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

DAY_MINUTES = 24 * 60
WEEK_MINUTES = 7 * DAY_MINUTES

# Short lags capture the last hour of momentum; the daily and weekly lags carry
# the seasonal shape that dominates this trace.
SHORT_LAGS = tuple(range(1, 61))
SEASONAL_LAGS = (DAY_MINUTES, 2 * DAY_MINUTES, WEEK_MINUTES, 2 * WEEK_MINUTES)
ROLLING_WINDOWS = (5, 15, 60)


def build_features(series: pd.Series) -> pd.DataFrame:
    """Build the model input matrix, indexed by decision time.

    Args:
        series: Arrivals per minute, indexed by timestamp.

    Returns:
        One row per timestamp in `series`. Rows near the start carry missing
        values for lags that reach back before the data begins; those are left
        as NaN rather than filled, because inventing a value there would teach
        the model something untrue about early history. LightGBM handles NaN
        natively, and the backtest drops rows whose target is unknown.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must be indexed by a DatetimeIndex")

    features = pd.DataFrame(index=series.index)

    for lag in SHORT_LAGS + SEASONAL_LAGS:
        features[f"lag_{lag}"] = series.shift(lag)

    # Shift first, then roll: the window must end at t-1, not at t.
    past = series.shift(1)
    for window in ROLLING_WINDOWS:
        rolling = past.rolling(window)
        features[f"roll_mean_{window}"] = rolling.mean()
        features[f"roll_std_{window}"] = rolling.std()
        features[f"roll_max_{window}"] = rolling.max()

    # The level at this time last week, smoothed, so the model has a seasonal
    # baseline that is not hostage to one noisy minute.
    features["seasonal_level"] = series.shift(WEEK_MINUTES).rolling(15, center=True).mean()

    timestamps = series.index
    features["minute_of_day"] = timestamps.hour * 60 + timestamps.minute
    features["hour"] = timestamps.hour
    features["day_of_week"] = timestamps.dayofweek
    features["is_weekend"] = (timestamps.dayofweek >= 5).astype(int)
    features["is_holiday"] = _holiday_flags(timestamps)

    return features


def build_target(series: pd.Series, horizon_minutes: int) -> pd.Series:
    """The value to be predicted from each decision time.

    The target for decision time `t` is the arrival count at `t + horizon`.
    The final `horizon` rows have no observable outcome and are left as NaN so
    they cannot be trained on.
    """
    if horizon_minutes <= 0:
        raise ValueError(f"horizon_minutes must be positive, got {horizon_minutes}")

    return series.shift(-horizon_minutes)


def _holiday_flags(timestamps: pd.DatetimeIndex) -> pd.Series:
    """1 on US federal holidays, else 0.

    Holiday traffic follows the weekend shape rather than the weekday shape,
    and no combination of lags can tell the model that in advance.
    """
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(
        start=timestamps.min().normalize(), end=timestamps.max().normalize()
    )
    return pd.Series(
        timestamps.normalize().isin(holidays).astype(int), index=timestamps
    )
