"""Weather preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def clean_weather_columns(dfw: pd.DataFrame) -> pd.DataFrame:
    if "Time" not in dfw.columns:
        raise ValueError("'Time' column is missing in weather CSV.")
    for col in ["Temperature", "Humidity"]:
        if col not in dfw.columns:
            raise ValueError(f"'{col}' column is missing in weather CSV.")

    out = dfw.copy()
    out["Temperature"] = pd.to_numeric(out["Temperature"], errors="coerce")
    out["Humidity"] = pd.to_numeric(out["Humidity"], errors="coerce")
    out["datetime"] = pd.to_datetime(out["Time"], errors="coerce")
    out = out.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    return out[["Temperature", "Humidity"]]


def preprocess_and_interpolate_weather_1h(
    weather_csv: Path,
    master_index: pd.DatetimeIndex,
    interpolate_linear: bool = True,
    interpolate_limit: int | None = None,
) -> pd.DataFrame:
    raw = pd.read_csv(weather_csv, low_memory=False)
    clean = clean_weather_columns(raw)

    out = clean.resample("1h").mean().reindex(master_index)
    if interpolate_linear:
        out = out.interpolate(method="linear", limit=interpolate_limit, limit_direction="both")
    return out


# Backward-compatible alias
def preprocess_weather_1h(
    weather_csv: Path,
    master_index: pd.DatetimeIndex,
    interpolate_linear: bool = True,
    interpolate_limit: int | None = None,
) -> pd.DataFrame:
    return preprocess_and_interpolate_weather_1h(
        weather_csv=weather_csv,
        master_index=master_index,
        interpolate_linear=interpolate_linear,
        interpolate_limit=interpolate_limit,
    )
