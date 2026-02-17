"""CLI entrypoint for preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_PREPROCESS_CONFIG_PATH, load_preprocess_config
from .preprocess import run_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preprocessing for GRU Fourier project.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PREPROCESS_CONFIG_PATH,
        help="Path to preprocess TOML config.",
    )
    parser.add_argument("--site", type=str, default=None, help="Site key under [sites] in config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_preprocess_config(args.config)

    site_key = args.site or cfg.default_site
    if site_key not in cfg.sites:
        raise ValueError(f"Unknown site '{site_key}'. Available: {sorted(cfg.sites)}")

    site = cfg.sites[site_key]
    output_dir = args.output_dir or cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] site={site_key}")
    print(f"[INFO] smartcare_dir={site.smartcare_dir}")
    print(f"[INFO] ereport_dir={site.ereport_dir}")
    print(f"[INFO] weather_csv={site.weather_csv}")

    df = run_preprocess(site)
    out_path = output_dir / site.out_filename
    df.to_csv(out_path, index_label="datetime")

    print(f"[INFO] rows={len(df)}, cols={len(df.columns)}")
    print(f"[INFO] saved={out_path}")


if __name__ == "__main__":
    main()
