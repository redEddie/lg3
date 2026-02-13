"""Top-level preprocessing orchestrator."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pandas as pd

from .ereport import preprocess_ereport_power_1h
from .fourier import fit_predict_fourier_midnight
from .io_utils import load_all_csv_time_from_filename
from .smartcare import preprocess_smartcare_tod_1h
from .time_features import create_time_features_1h
from .weather import preprocess_and_interpolate_weather_1h

if TYPE_CHECKING:
    from gru_fourier.src.config import SiteConfig


def _step_start(label: str) -> float:
    print(f"[STEP] {label} ...")
    return time.perf_counter()


def _step_end(label: str, t0: float, rows: int | None = None) -> None:
    elapsed = time.perf_counter() - t0
    if rows is None:
        print(f"[STEP] {label} done in {elapsed:.2f}s")
    else:
        print(f"[STEP] {label} done in {elapsed:.2f}s (rows={rows})")


def run_preprocess(site: "SiteConfig") -> pd.DataFrame:
    start = pd.Timestamp(site.start)
    end = pd.Timestamp(site.end)
    master_index = pd.date_range(start=start, end=end, freq="1h")

    required_history = max(1, int(site.fourier_fit_hours))
    raw_start = start - pd.Timedelta(hours=required_history)
    print(f"[INFO] master range: {start} ~ {end} ({len(master_index)} rows)")
    print(f"[INFO] raw_start for loading: {raw_start}")

    t0 = _step_start("time features")
    df_time = create_time_features_1h(
        master_index,
        holiday_dates=site.holiday_dates,
        include_weekends_as_holiday=site.include_weekends_as_holiday,
    )
    _step_end("time features", t0, rows=len(df_time))

    t0 = _step_start("smartcare load")
    df_smart_raw = load_all_csv_time_from_filename(
        site.smartcare_dir,
        site.smartcare_pattern,
        date_start=raw_start,
        date_end=end,
        usecols=["Time", "Auto Id", "Tod"],
    )
    _step_end("smartcare load", t0, rows=len(df_smart_raw))

    t0 = _step_start("smartcare aggregate")
    df_smart_raw = df_smart_raw.loc[(df_smart_raw.index >= raw_start) & (df_smart_raw.index <= end)]
    smart_1h = preprocess_smartcare_tod_1h(
        df_smart_raw,
        master_index=master_index,
        n_units=site.n_units,
        sample_interval_sec=site.smart_sample_sec,
        per_unit_min_ratio=site.smart_per_unit_min_ratio,
    )
    _step_end("smartcare aggregate", t0, rows=len(smart_1h))
    del df_smart_raw

    t0 = _step_start("ereport load")
    df_ereport_raw = load_all_csv_time_from_filename(
        site.ereport_dir,
        site.ereport_pattern,
        date_start=raw_start,
        date_end=end,
        usecols=["Time", "Power"],
    )
    _step_end("ereport load", t0, rows=len(df_ereport_raw))

    t0 = _step_start("ereport aggregate")
    power_1h = preprocess_ereport_power_1h(
        df_ereport_raw,
        master_index=master_index,
        coverage_threshold=site.power_coverage_threshold,
        forced_zero_ranges=site.forced_off_ranges,
    )
    _step_end("ereport aggregate", t0, rows=len(power_1h))
    del df_ereport_raw

    t0 = _step_start("weather aggregate")
    weather_1h = preprocess_and_interpolate_weather_1h(
        site.weather_csv,
        master_index=master_index,
        interpolate_linear=site.weather_interpolate_linear,
        interpolate_limit=site.weather_interpolate_limit,
    )
    _step_end("weather aggregate", t0, rows=len(weather_1h))

    t0 = _step_start("fourier feature")
    fourier_midnight = fit_predict_fourier_midnight(
        power_1h["Power_1h"],
        index=master_index,
        fit_hours=site.fourier_fit_hours,
        pred_hours=site.fourier_pred_hours,
        k_harmonics=site.fourier_k,
        period=site.fourier_s,
        l2=site.fourier_ridge_l2,
        carry_forward_on_missing=site.fourier_carry_forward_on_missing,
    )
    _step_end("fourier feature", t0, rows=len(fourier_midnight))

    t0 = _step_start("merge all features")
    df_all = (
        df_time.join(smart_1h, how="left")
        .join(power_1h, how="left")
        .join(weather_1h, how="left")
    )
    df_all["fourier_base_midnight"] = fourier_midnight
    _step_end("merge all features", t0, rows=len(df_all))
    return df_all
