"""Fourier seasonality utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_fourier_basis(t: np.ndarray, k_harmonics: int, period: float) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    basis = np.empty((len(t), 2 * k_harmonics), dtype=float)
    for k in range(1, k_harmonics + 1):
        omega = 2.0 * np.pi * k * t / period
        basis[:, k - 1] = np.sin(omega)
        basis[:, k_harmonics + (k - 1)] = np.cos(omega)
    return basis


def add_bias_column(x: np.ndarray) -> np.ndarray:
    ones = np.ones((x.shape[0], 1), dtype=float)
    return np.concatenate([ones, x], axis=1)


def make_fourier_design(n_rows: int, k_harmonics: int, period: float, start_offset: int = 0) -> np.ndarray:
    t = np.arange(start_offset, start_offset + n_rows, dtype=float)
    return add_bias_column(build_fourier_basis(t, k_harmonics, period))


class FourierSeasonalityModel:
    def __init__(self, k_harmonics: int, period: float, l2: float) -> None:
        self.k_harmonics = int(k_harmonics)
        self.period = float(period)
        self.l2 = float(l2)
        self.theta: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> np.ndarray:
        y_fit = np.asarray(y, dtype=float).reshape(-1)
        x = make_fourier_design(len(y_fit), self.k_harmonics, self.period, start_offset=0)
        d = x.shape[1]
        self.theta = np.linalg.solve(x.T @ x + self.l2 * np.eye(d), x.T @ y_fit)
        return self.theta

    def predict(self, n_rows: int, start_offset: int = 0) -> np.ndarray:
        if self.theta is None:
            raise RuntimeError("Model is not fitted.")
        x = make_fourier_design(n_rows, self.k_harmonics, self.period, start_offset=start_offset)
        return x @ self.theta


def extract_fit_window(power_1h: pd.Series, t0: pd.Timestamp, fit_hours: int) -> np.ndarray | None:
    fit_start = t0 - pd.Timedelta(hours=fit_hours)
    y_fit = power_1h.loc[(power_1h.index >= fit_start) & (power_1h.index < t0)].to_numpy(dtype=float)
    if len(y_fit) != fit_hours:
        return None
    if np.any(~np.isfinite(y_fit)):
        return None
    return y_fit


def predict_next_window(model: FourierSeasonalityModel, fit_hours: int, pred_hours: int) -> np.ndarray:
    return model.predict(pred_hours, start_offset=fit_hours)


def fit_predict_fourier_midnight(
    power_1h: pd.Series,
    index: pd.DatetimeIndex,
    fit_hours: int,
    pred_hours: int,
    k_harmonics: int,
    period: float,
    l2: float,
    carry_forward_on_missing: bool = True,
) -> pd.Series:
    out = pd.Series(index=index, dtype=float)
    power = power_1h.reindex(index)

    midnight_locs = index.indexer_at_time("00:00")
    midnights = index[midnight_locs]

    prev_theta: np.ndarray | None = None
    for t0 in midnights:
        y_fit = extract_fit_window(power, t0=t0, fit_hours=fit_hours)
        model = FourierSeasonalityModel(k_harmonics=k_harmonics, period=period, l2=l2)

        if y_fit is not None:
            prev_theta = model.fit(y_fit)
        elif carry_forward_on_missing and prev_theta is not None:
            model.theta = prev_theta
        else:
            continue

        y_hat = predict_next_window(model, fit_hours=fit_hours, pred_hours=pred_hours)
        pred_idx = pd.date_range(start=t0, periods=pred_hours, freq="1h").intersection(index)
        out.loc[pred_idx] = y_hat[: len(pred_idx)]
    return out


# Backward-compatible alias
def build_midnight_fourier_feature(
    power_1h: pd.Series,
    index: pd.DatetimeIndex,
    fit_hours: int,
    pred_hours: int,
    k_harmonics: int,
    period: float,
    l2: float,
    carry_forward_on_missing: bool = True,
) -> pd.Series:
    return fit_predict_fourier_midnight(
        power_1h=power_1h,
        index=index,
        fit_hours=fit_hours,
        pred_hours=pred_hours,
        k_harmonics=k_harmonics,
        period=period,
        l2=l2,
        carry_forward_on_missing=carry_forward_on_missing,
    )
