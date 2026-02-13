"""SMARTCARE Tod preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_unit_averages(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["slot", "Auto Id"])["Tod"].mean().unstack("Auto Id")


def identify_valid_slots(
    cnt: pd.DataFrame,
    n_units: int,
    min_required_per_unit: int,
) -> pd.Series:
    if cnt.empty:
        return pd.Series(dtype=bool, index=cnt.index)

    units_ok = pd.Series(True, index=cnt.index) if n_units <= 0 else (cnt.notna().sum(axis=1) == n_units)
    sample_ok = (
        pd.Series(True, index=cnt.index)
        if min_required_per_unit <= 0
        else (cnt >= min_required_per_unit).all(axis=1)
    )
    return units_ok & sample_ok


def preprocess_smartcare_tod_1h(
    df_smart: pd.DataFrame,
    master_index: pd.DatetimeIndex,
    n_units: int,
    sample_interval_sec: int,
    per_unit_min_ratio: float,
    freq: str = "1h",
) -> pd.DataFrame:
    required_cols = ["Auto Id", "Tod"]
    missing = [c for c in required_cols if c not in df_smart.columns]
    if missing:
        raise ValueError(f"SMARTCARE required columns are missing: {missing}")
    if sample_interval_sec <= 0:
        raise ValueError("sample_interval_sec must be > 0")

    df = df_smart.copy()
    df["Tod"] = pd.to_numeric(df["Tod"], errors="coerce")
    df["slot"] = df.index.ceil(freq)

    expected_per_unit = int(round(3600.0 / sample_interval_sec))
    min_required = 0 if per_unit_min_ratio <= 0 else int(np.floor(expected_per_unit * per_unit_min_ratio))

    cnt = df.groupby(["slot", "Auto Id"])["Tod"].count().unstack("Auto Id")
    valid_slots = identify_valid_slots(cnt, n_units=n_units, min_required_per_unit=min_required)
    unit_mean = calculate_unit_averages(df)

    total_mean = unit_mean.mean(axis=1)
    valid_mask = valid_slots.reindex(total_mean.index).fillna(False)
    tod_1h = total_mean.where(valid_mask, np.nan)
    return pd.DataFrame({"Tod_1h": tod_1h}).reindex(master_index)
