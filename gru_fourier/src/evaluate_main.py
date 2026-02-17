"""CLI entrypoint for checkpoint evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_EVALUATE_CONFIG_PATH, load_evaluate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latest rolling run checkpoints.")
    parser.add_argument("--config", type=Path, default=DEFAULT_EVALUATE_CONFIG_PATH, help="Path to evaluate TOML config.")
    parser.add_argument("--csv-path", type=Path, default=None, help="Override input feature CSV path.")
    parser.add_argument("--latest-json", type=Path, default=None, help="Override latest run manifest path.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override evaluation output directory.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "cnn1d"],
        default=None,
        help="Override model type namespace for latest/out-dir defaults.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_evaluate_config(args.config)
    if args.csv_path is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "csv_path": args.csv_path})
    if args.latest_json is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "latest_json": args.latest_json})
    if args.out_dir is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "out_dir": args.out_dir})
    if args.model_type is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "model_type": args.model_type})

    try:
        from gru_fourier.evaluation.runner import run_evaluation
    except ModuleNotFoundError as e:
        if e.name == "torch":
            raise RuntimeError(
                "PyTorch is required for evaluation. Install torch first, then run evaluate_main again."
            ) from e
        raise

    summary_csv = run_evaluation(cfg)
    print(f"[INFO] evaluation complete, summary: {summary_csv}")


if __name__ == "__main__":
    main()
