"""CLI entrypoint for rolling training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_TRAIN_CONFIG_PATH, load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling LSTM24 training.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH, help="Path to train TOML config.")
    parser.add_argument("--csv-path", type=Path, default=None, help="Override input feature CSV path.")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Override runs output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    if args.csv_path is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "csv_path": args.csv_path})
    if args.runs_dir is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "runs_dir": args.runs_dir})
    if args.epochs is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "epochs": int(args.epochs)})

    try:
        from gru_fourier.training.runner import run_training
    except ModuleNotFoundError as e:
        if e.name == "torch":
            raise RuntimeError(
                "PyTorch is required for training. Install torch first, then run train_main again."
            ) from e
        raise

    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    latest_json = run_training(cfg)
    print(f"[INFO] training complete, latest manifest: {latest_json}")


if __name__ == "__main__":
    main()
