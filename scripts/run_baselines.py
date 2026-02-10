"""CLI entrypoint for baseline forecasting evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.evaluation import evaluate_baselines, preprocess_hourly_series


def _top_worst_hours(hourly_mse: List[float], top_k: int = 3) -> List[Tuple[int, float]]:
    pairs = [(hour, mse) for hour, mse in enumerate(hourly_mse) if mse == mse]  # drop NaN
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run 24h forecasting baseline evaluation.")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--time-col", required=True, help="Datetime column name")
    parser.add_argument("--target-col", required=True, help="Target column name")
    parser.add_argument("--fourier-k", type=int, choices=[1, 2], default=2, help="Fourier harmonics")
    parser.add_argument("--stride-hours", type=int, default=24, help="Sliding window stride in hours")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output artifacts")
    return parser.parse_args()


def main() -> None:
    """Run preprocessing, evaluate baselines, and write outputs."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series = preprocess_hourly_series(
        csv_path=args.csv,
        time_col=args.time_col,
        target_col=args.target_col,
    )

    result = evaluate_baselines(
        series=series,
        history_hours=168,
        horizon_hours=24,
        stride_hours=args.stride_hours,
        fourier_k=args.fourier_k,
    )

    summary: Dict[str, object] = {
        "overall_mse": result.overall_mse,
        "hour_of_day_mse": result.hour_of_day_mse,
        "n_windows": result.n_windows,
        "n_points": result.n_points,
        "per_window_mse": result.per_window_mse,
    }

    with (output_dir / "mse_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    result.sample_predictions.to_csv(output_dir / "prediction_samples.csv", index=False)

    print("\n=== Baseline Overall MSE ===")
    for name, mse in sorted(result.overall_mse.items(), key=lambda x: x[1]):
        print(f"{name:20s} {mse:.6f}")

    print(f"\nEvaluated windows: {result.n_windows}")
    print(f"Evaluated points : {result.n_points}")

    print("\n=== Hour-of-day MSE preview (top-3 worst hours) ===")
    for name, hourly in result.hour_of_day_mse.items():
        worst = _top_worst_hours(hourly, top_k=3)
        worst_txt = ", ".join([f"h{h}: {m:.6f}" for h, m in worst])
        print(f"{name:20s} {worst_txt}")


if __name__ == "__main__":
    main()
