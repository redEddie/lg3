"""Baseline forecasting package."""

from .methods import (
    fourier_trend_baseline,
    last_output_hold_baseline,
    naive_persistence_baseline,
    weekly_hour_mean_baseline,
)
from .evaluation import EvaluationResult, evaluate_baselines, preprocess_hourly_series

__all__ = [
    "last_output_hold_baseline",
    "naive_persistence_baseline",
    "weekly_hour_mean_baseline",
    "fourier_trend_baseline",
    "preprocess_hourly_series",
    "evaluate_baselines",
    "EvaluationResult",
]
