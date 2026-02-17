"""Time-based feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_week_phase_hour_sunday(index: pd.DatetimeIndex) -> np.ndarray:
    # Monday=0..Sunday=6 -> Sunday=0..Saturday=6
    sunday_based_dow = (index.dayofweek.to_numpy() + 1) % 7
    hour = index.hour.to_numpy()
    return sunday_based_dow * 24 + hour


def create_time_features_1h(
    index: pd.DatetimeIndex,
    holiday_dates: list[str],
    include_weekends_as_holiday: bool,
) -> pd.DataFrame:
    df_time = pd.DataFrame(index=index)

    holiday_set = set(pd.to_datetime(holiday_dates).date) if holiday_dates else set()
    is_custom = pd.Series(index.date).isin(holiday_set).to_numpy()
    if include_weekends_as_holiday:
        is_weekend = index.dayofweek.to_numpy() >= 5
        df_time["is_holiday"] = (is_custom | is_weekend).astype(int)
    else:
        df_time["is_holiday"] = is_custom.astype(int)

    hour = index.hour.to_numpy()
    df_time["day_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df_time["day_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    week_hour = build_week_phase_hour_sunday(index).astype(float)
    df_time["week_sin"] = np.sin(2.0 * np.pi * week_hour / 168.0)
    df_time["week_cos"] = np.cos(2.0 * np.pi * week_hour / 168.0)
    return df_time
