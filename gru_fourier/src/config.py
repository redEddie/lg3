"""Configuration loaders for preprocess/train/evaluate pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPROCESS_CONFIG_PATH = PROJECT_ROOT / "gru_fourier" / "config" / "preprocess.toml"
DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "gru_fourier" / "config" / "train.toml"
DEFAULT_EVALUATE_CONFIG_PATH = PROJECT_ROOT / "gru_fourier" / "config" / "evaluate.toml"


@dataclass(frozen=True)
class SiteConfig:
    name: str
    smartcare_dir: Path
    smartcare_pattern: str
    ereport_dir: Path
    ereport_pattern: str
    weather_csv: Path
    start: str
    end: str
    out_filename: str
    holiday_dates: list[str]
    include_weekends_as_holiday: bool
    power_coverage_threshold: float
    smart_per_unit_min_ratio: float
    n_units: int
    smart_sample_sec: int
    fourier_s: float
    fourier_k: int
    fourier_fit_hours: int
    fourier_pred_hours: int
    fourier_ridge_l2: float
    fourier_carry_forward_on_missing: bool
    weather_interpolate_linear: bool
    weather_interpolate_limit: int | None
    forced_off_ranges: list[tuple[str, str]]


@dataclass(frozen=True)
class PreprocessConfig:
    default_site: str
    output_dir: Path
    sites: dict[str, SiteConfig]


@dataclass(frozen=True)
class TrainConfig:
    csv_path: Path
    dt_col: str
    target_col: str
    exog_cols_base: list[str]
    fourier_col: str
    weather_cols: list[str]
    use_tod: bool
    tod_col: str
    lookback: int
    horizon: int
    buffer_hours: int
    train_hours: int
    valid_hours: int
    step_hours: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    seed: int
    lambda_diff: float
    lambda_cos: float
    lambda_share: float
    cos_eps: float
    share_eps: float
    day_sum_mask_eps: float
    top_alpha: float
    runs_dir: Path
    device: str
    model_type: str
    hidden_size: int
    cnn_channels: int
    cnn_kernel_size: int
    log_every: int


@dataclass(frozen=True)
class EvaluateConfig:
    csv_path: Path
    dt_col: str
    target_col: str
    runs_dir: Path
    latest_json: Path | None
    out_dir: Path | None
    default_out_dirname: str
    device: str
    model_type: str
    hidden_size: int
    cnn_channels: int
    cnn_kernel_size: int
    top_alpha: float
    cos_eps: float
    share_eps: float
    day_sum_mask_eps: float


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load_toml(path: Path):
    with path.open("rb") as f:
        return tomllib.load(f)


def load_preprocess_config(config_path: Path | None = None) -> PreprocessConfig:
    conf_path = config_path or DEFAULT_PREPROCESS_CONFIG_PATH
    raw = _load_toml(conf_path)

    global_cfg = raw.get("global", {})
    default_site = str(global_cfg["default_site"])
    output_dir = _resolve_path(str(global_cfg.get("output_dir", "processed_data")))

    sites_raw = raw.get("sites", {})
    if not sites_raw:
        raise ValueError("No site configuration found under [sites].")

    sites: dict[str, SiteConfig] = {}
    for name, site_raw in sites_raw.items():
        ranges = [tuple(x) for x in site_raw.get("forced_off_ranges", [])]
        sites[name] = SiteConfig(
            name=name,
            smartcare_dir=_resolve_path(str(site_raw["smartcare_dir"])),
            smartcare_pattern=str(site_raw.get("smartcare_pattern", "LOG_SMARTCARE_*.csv")),
            ereport_dir=_resolve_path(str(site_raw["ereport_dir"])),
            ereport_pattern=str(site_raw.get("ereport_pattern", "DBG_EREPORT_*.csv")),
            weather_csv=_resolve_path(str(site_raw["weather_csv"])),
            start=str(site_raw["start"]),
            end=str(site_raw["end"]),
            out_filename=str(site_raw["out_filename"]),
            holiday_dates=list(site_raw.get("holiday_dates", [])),
            include_weekends_as_holiday=bool(site_raw.get("include_weekends_as_holiday", True)),
            power_coverage_threshold=float(site_raw.get("power_coverage_threshold", 0.85)),
            smart_per_unit_min_ratio=float(site_raw.get("smart_per_unit_min_ratio", 0.70)),
            n_units=int(site_raw.get("n_units", 7)),
            smart_sample_sec=int(site_raw.get("smart_sample_sec", 5)),
            fourier_s=float(site_raw.get("fourier_s", 168.0)),
            fourier_k=int(site_raw.get("fourier_k", 10)),
            fourier_fit_hours=int(site_raw.get("fourier_fit_hours", 14 * 24)),
            fourier_pred_hours=int(site_raw.get("fourier_pred_hours", 24)),
            fourier_ridge_l2=float(site_raw.get("fourier_ridge_l2", 1e-2)),
            fourier_carry_forward_on_missing=bool(site_raw.get("fourier_carry_forward_on_missing", True)),
            weather_interpolate_linear=bool(site_raw.get("weather_interpolate_linear", True)),
            weather_interpolate_limit=(
                None
                if site_raw.get("weather_interpolate_limit", None) is None
                else int(site_raw.get("weather_interpolate_limit"))
            ),
            forced_off_ranges=ranges,
        )

    if default_site not in sites:
        raise ValueError(f"default_site '{default_site}' does not exist in [sites].")

    return PreprocessConfig(default_site=default_site, output_dir=output_dir, sites=sites)


def load_train_config(config_path: Path | None = None) -> TrainConfig:
    conf_path = config_path or DEFAULT_TRAIN_CONFIG_PATH
    raw = _load_toml(conf_path)

    data = raw.get("data", {})
    feature = raw.get("feature", {})
    window = raw.get("window", {})
    train = raw.get("train", {})
    loss = raw.get("loss", {})
    metric = raw.get("metric", {})
    out = raw.get("output", {})
    model = raw.get("model", {})

    return TrainConfig(
        csv_path=_resolve_path(str(data.get("csv_path", "processed_data/preprocessed_1h_master_with_weather_delta_20250701_20250930_ohsungsa_f2.csv"))),
        dt_col=str(data.get("dt_col", "datetime")),
        target_col=str(data.get("target_col", "Power_1h")),
        exog_cols_base=list(feature.get("exog_cols_base", ["is_holiday", "day_sin", "day_cos"])),
        fourier_col=str(feature.get("fourier_col", "fourier_base_midnight")),
        weather_cols=list(feature.get("weather_cols", ["Temperature", "Humidity"])),
        use_tod=bool(feature.get("use_tod", False)),
        tod_col=str(feature.get("tod_col", "Tod_1h")),
        lookback=int(window.get("lookback", 168)),
        horizon=int(window.get("horizon", 24)),
        buffer_hours=int(window.get("buffer_hours", 14 * 24)),
        train_hours=int(window.get("train_hours", 28 * 24)),
        valid_hours=int(window.get("valid_hours", 7 * 24)),
        step_hours=int(window.get("step_hours", 7 * 24)),
        batch_size=int(train.get("batch_size", 64)),
        epochs=int(train.get("epochs", 400)),
        lr=float(train.get("lr", 1e-3)),
        weight_decay=float(train.get("weight_decay", 1e-4)),
        grad_clip=float(train.get("grad_clip", 5.0)),
        seed=int(train.get("seed", 42)),
        lambda_diff=float(loss.get("lambda_diff", 1.0)),
        lambda_cos=float(loss.get("lambda_cos", 1.0)),
        lambda_share=float(loss.get("lambda_share", 0.0)),
        cos_eps=float(metric.get("cos_eps", 1e-8)),
        share_eps=float(metric.get("share_eps", 1e-6)),
        day_sum_mask_eps=float(metric.get("day_sum_mask_eps", 1e-3)),
        top_alpha=float(metric.get("top_alpha", 0.20)),
        runs_dir=_resolve_path(str(out.get("runs_dir", "runs_lstm24_roll"))),
        device=str(train.get("device", "auto")),
        model_type=str(model.get("model_type", "lstm")),
        hidden_size=int(model.get("hidden_size", 32)),
        cnn_channels=int(model.get("cnn_channels", 32)),
        cnn_kernel_size=int(model.get("cnn_kernel_size", 5)),
        log_every=int(train.get("log_every", 10)),
    )


def load_evaluate_config(config_path: Path | None = None) -> EvaluateConfig:
    conf_path = config_path or DEFAULT_EVALUATE_CONFIG_PATH
    raw = _load_toml(conf_path)

    data = raw.get("data", {})
    runtime = raw.get("runtime", {})
    metric = raw.get("metric", {})
    output = raw.get("output", {})
    model = raw.get("model", {})

    latest_json_raw = data.get("latest_json", None)
    out_dir_raw = output.get("out_dir", None)

    return EvaluateConfig(
        csv_path=_resolve_path(str(data.get("csv_path", "processed_data/preprocessed_1h_master_with_weather_delta_20250701_20250930_ohsungsa_f2.csv"))),
        dt_col=str(data.get("dt_col", "datetime")),
        target_col=str(data.get("target_col", "Power_1h")),
        runs_dir=_resolve_path(str(data.get("runs_dir", "runs_lstm24_roll"))),
        latest_json=None if latest_json_raw is None else _resolve_path(str(latest_json_raw)),
        out_dir=None if out_dir_raw is None else _resolve_path(str(out_dir_raw)),
        default_out_dirname=str(output.get("default_out_dirname", "eval_last_midnight_on_own_val_inline_latest")),
        device=str(runtime.get("device", "auto")),
        model_type=str(model.get("model_type", "lstm")),
        hidden_size=int(model.get("hidden_size", 32)),
        cnn_channels=int(model.get("cnn_channels", 32)),
        cnn_kernel_size=int(model.get("cnn_kernel_size", 5)),
        top_alpha=float(metric.get("top_alpha", 0.20)),
        cos_eps=float(metric.get("cos_eps", 1e-8)),
        share_eps=float(metric.get("share_eps", 1e-6)),
        day_sum_mask_eps=float(metric.get("day_sum_mask_eps", 1e-3)),
    )
