"""Forecasting baseline methods for 24-step ahead prediction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def last_output_hold_baseline(history: np.ndarray, horizon: int = 24) -> np.ndarray:
    """Predict a constant sequence using the most recent observed value.

    Args:
        history: Historical values in chronological order.
        horizon: Forecast horizon length.

    Returns:
        A 1D array of length ``horizon``.
    """
    last_value = float(history[-1])
    return np.full(horizon, last_value, dtype=float)


def naive_persistence_baseline(history: np.ndarray, horizon: int = 24, period: int = 24) -> np.ndarray:
    """Seasonal-naive forecast using values from one period earlier.

    For the default ``period=24``, tomorrow at each hour is predicted with
    yesterday's value at the same hour.
    """
    if len(history) < period:
        raise ValueError(f"history length must be >= period ({period}).")
    return history[-period:].astype(float, copy=True)


def weekly_hour_mean_baseline(
    history_series: pd.Series,
    forecast_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Predict hour-of-day means computed over the history window.

    Args:
        history_series: Historical series indexed by naive local datetimes.
        forecast_index: Datetime index for the forecast horizon.

    Returns:
        Forecast values aligned to ``forecast_index``.
    """
    hour_means = history_series.groupby(history_series.index.hour).mean()
    default_mean = float(history_series.mean())
    preds = [float(hour_means.get(hour, default_mean)) for hour in forecast_index.hour]
    return np.asarray(preds, dtype=float)


def _build_fourier_design_matrix(timesteps: Sequence[int], k: int) -> np.ndarray:
    """Build an OLS design matrix with linear trend and daily Fourier terms."""
    t = np.asarray(timesteps, dtype=float)
    cols = [np.ones_like(t), t]
    for harmonic in range(1, k + 1):
        omega = 2.0 * np.pi * harmonic * t / 24.0
        cols.append(np.sin(omega))
        cols.append(np.cos(omega))
    return np.column_stack(cols)


def fourier_trend_baseline(history: np.ndarray, horizon: int = 24, fourier_k: int = 2) -> np.ndarray:
    """Fit trend + daily Fourier OLS on history and extrapolate for next steps.

    Args:
        history: Historical values in chronological order.
        horizon: Forecast horizon length.
        fourier_k: Number of Fourier harmonics (recommended 1 or 2).

    Returns:
        A 1D forecast array of length ``horizon``.
    """
    if fourier_k < 1:
        raise ValueError("fourier_k must be >= 1.")

    n_hist = len(history)
    t_hist = np.arange(n_hist)
    x_hist = _build_fourier_design_matrix(t_hist, k=fourier_k)
    y_hist = history.astype(float)

    coef, *_ = np.linalg.lstsq(x_hist, y_hist, rcond=None)

    t_future = np.arange(n_hist, n_hist + horizon)
    x_future = _build_fourier_design_matrix(t_future, k=fourier_k)
    y_pred = x_future @ coef
    return y_pred.astype(float)
