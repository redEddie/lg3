"""Preprocessing, sliding-window evaluation, and metrics for baseline forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .methods import (
    fourier_trend_baseline,
    last_output_hold_baseline,
    naive_persistence_baseline,
    weekly_hour_mean_baseline,
)


@dataclass
class EvaluationResult:
    """Container for evaluated metrics and sample predictions."""

    overall_mse: Dict[str, float]
    hour_of_day_mse: Dict[str, List[float]]
    n_windows: int
    n_points: int
    per_window_mse: Dict[str, List[float]]
    sample_predictions: pd.DataFrame


def preprocess_hourly_series(csv_path: str, time_col: str, target_col: str) -> pd.Series:
    """Load CSV and produce a clean hourly target series.

    Notes:
        Datetimes are treated as already-local naive times (no timezone conversion).
    """
    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col])

    series = (
        df[[time_col, target_col]]
        .sort_values(time_col)
        .set_index(time_col)[target_col]
        .astype(float)
    )

    # Safe to always resample. If already hourly, this remains hourly means.
    hourly = series.resample("1H").mean()
    hourly = hourly.interpolate(method="time")
    hourly = hourly.dropna()

    if hourly.empty:
        raise ValueError("Series is empty after preprocessing.")

    return hourly


def _compute_hourly_mse(
    all_true: np.ndarray,
    all_pred: np.ndarray,
    all_hours: np.ndarray,
) -> List[float]:
    """Compute MSE for each hour-of-day (0..23)."""
    out: List[float] = []
    for hour in range(24):
        mask = all_hours == hour
        if np.any(mask):
            mse = float(np.mean((all_true[mask] - all_pred[mask]) ** 2))
        else:
            mse = float("nan")
        out.append(mse)
    return out


def evaluate_baselines(
    series: pd.Series,
    history_hours: int = 168,
    horizon_hours: int = 24,
    stride_hours: int = 24,
    fourier_k: int = 2,
    sample_window_count: int = 2,
) -> EvaluationResult:
    """Run sliding-window evaluation for all baselines."""
    values = series.to_numpy(dtype=float)
    index = series.index

    start_t = history_hours
    end_t = len(series) - horizon_hours

    windows: List[Tuple[int, int]] = []
    for t in range(start_t, end_t + 1, stride_hours):
        hist_start = t - history_hours
        future_end = t + horizon_hours
        if hist_start < 0 or future_end > len(series):
            continue
        windows.append((t, future_end))

    if not windows:
        raise ValueError("No valid windows. Check data length and stride settings.")

    baseline_names = [
        "last_output_hold",
        "naive_persistence",
        "weekly_hour_mean",
        "fourier_trend",
    ]

    all_true_by_model: Dict[str, List[np.ndarray]] = {k: [] for k in baseline_names}
    all_pred_by_model: Dict[str, List[np.ndarray]] = {k: [] for k in baseline_names}
    all_hours_by_model: Dict[str, List[np.ndarray]] = {k: [] for k in baseline_names}
    per_window_mse: Dict[str, List[float]] = {k: [] for k in baseline_names}

    sample_window_indices = set(range(min(sample_window_count, len(windows))))
    sample_window_indices |= set(range(max(0, len(windows) - sample_window_count), len(windows)))
    sample_rows: List[pd.DataFrame] = []

    for window_idx, (t, future_end) in enumerate(windows):
        history_vals = values[t - history_hours : t]
        history_index = index[t - history_hours : t]
        history_series = pd.Series(history_vals, index=history_index)

        future_true = values[t:future_end]
        future_index = index[t:future_end]
        future_hours = future_index.hour.to_numpy()

        preds = {
            "last_output_hold": last_output_hold_baseline(history_vals, horizon=horizon_hours),
            "naive_persistence": naive_persistence_baseline(history_vals, horizon=horizon_hours, period=24),
            "weekly_hour_mean": weekly_hour_mean_baseline(history_series, forecast_index=future_index),
            "fourier_trend": fourier_trend_baseline(history_vals, horizon=horizon_hours, fourier_k=fourier_k),
        }

        for name, y_hat in preds.items():
            all_true_by_model[name].append(future_true)
            all_pred_by_model[name].append(y_hat)
            all_hours_by_model[name].append(future_hours)
            mse = float(np.mean((future_true - y_hat) ** 2))
            per_window_mse[name].append(mse)

        if window_idx in sample_window_indices:
            sample_df = pd.DataFrame(
                {
                    "window_idx": window_idx,
                    "timestamp": future_index,
                    "y_true": future_true,
                    "y_hat_last_output_hold": preds["last_output_hold"],
                    "y_hat_naive_persistence": preds["naive_persistence"],
                    "y_hat_weekly_hour_mean": preds["weekly_hour_mean"],
                    "y_hat_fourier_trend": preds["fourier_trend"],
                }
            )
            sample_rows.append(sample_df)

    overall_mse: Dict[str, float] = {}
    hour_of_day_mse: Dict[str, List[float]] = {}

    for name in baseline_names:
        all_true = np.concatenate(all_true_by_model[name])
        all_pred = np.concatenate(all_pred_by_model[name])
        all_hours = np.concatenate(all_hours_by_model[name])

        overall_mse[name] = float(np.mean((all_true - all_pred) ** 2))
        hour_of_day_mse[name] = _compute_hourly_mse(all_true, all_pred, all_hours)

    samples = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()

    n_windows = len(windows)
    n_points = n_windows * horizon_hours

    return EvaluationResult(
        overall_mse=overall_mse,
        hour_of_day_mse=hour_of_day_mse,
        n_windows=n_windows,
        n_points=n_points,
        per_window_mse=per_window_mse,
        sample_predictions=samples,
    )
