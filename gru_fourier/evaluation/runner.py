"""Evaluation runner reproduced from notebook logic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gru_fourier.src.config import EvaluateConfig
from gru_fourier.training.common import (
    MetricParams,
    cos_centered_np,
    create_forecast_model,
    diff_rmse_np,
    share_overlap_percent_np,
    share_tv_np,
    topk_iou_np,
)


def _resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_cfg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' but CUDA is not available")
    return device_cfg


def _strict_index_loc(idx_dt: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    try:
        loc = idx_dt.get_loc(ts)
        if isinstance(loc, slice):
            return int(loc.start)
        if isinstance(loc, (np.ndarray, list)):
            return int(loc[0])
        return int(loc)
    except KeyError:
        loc = int(idx_dt.get_indexer([ts], method="nearest")[0])
        print(f"[WARN] timestamp not exactly in index: {ts} -> nearest {idx_dt[loc]}")
        return loc


def build_midnight_val_t_list(
    idx_dt: pd.DatetimeIndex,
    y: np.ndarray,
    xe: np.ndarray,
    lookback: int,
    horizon: int,
    val_start_ts: pd.Timestamp,
    val_end_ts: pd.Timestamp,
) -> np.ndarray:
    va_s = _strict_index_loc(idx_dt, pd.Timestamp(val_start_ts))
    va_e_incl = _strict_index_loc(idx_dt, pd.Timestamp(val_end_ts))
    va_e = va_e_incl + 1

    n_rows = len(idx_dt)
    is_finite_y = np.isfinite(y)
    is_finite_xe = np.isfinite(xe).all(axis=1)

    t_list: list[int] = []
    lo = max(va_s, lookback)
    hi = min(va_e, n_rows - horizon)
    for t in range(lo, hi):
        if idx_dt[t].hour != 0:
            continue
        if not is_finite_y[t - lookback : t].all():
            continue
        if not is_finite_y[t : t + horizon].all():
            continue
        if not is_finite_xe[t : t + horizon].all():
            continue
        t_list.append(t)
    return np.array(t_list, dtype=int)


@torch.no_grad()
def infer_one(
    model, y: np.ndarray, xe: np.ndarray, t: int, lookback: int, horizon: int, device: str
):
    x_power = torch.from_numpy(y[t - lookback : t][:, None]).float().unsqueeze(0).to(device)
    x_exog = torch.from_numpy(xe[t : t + horizon, :]).float().unsqueeze(0).to(device)
    y_pred = model(x_power, x_exog).squeeze(0).detach().cpu().numpy().astype(np.float32)
    y_true = y[t : t + horizon].astype(np.float32)
    return y_true, y_pred


def run_evaluation(cfg: EvaluateConfig) -> Path:
    device = _resolve_device(cfg.device)

    df = pd.read_csv(cfg.csv_path, parse_dates=[cfg.dt_col]).set_index(cfg.dt_col).sort_index()

    model_key = cfg.model_type.strip().lower()
    model_runs_dir = cfg.runs_dir / model_key

    if cfg.latest_json is not None:
        latest_json = cfg.latest_json
    else:
        latest_json = model_runs_dir / "LATEST_RUN.json"
        # Backward compatibility: old layout without model subdir
        if not latest_json.exists():
            legacy_latest = cfg.runs_dir / "LATEST_RUN.json"
            if legacy_latest.exists():
                latest_json = legacy_latest
    if not latest_json.exists():
        raise RuntimeError(f"LATEST_RUN.json not found: {latest_json}")

    with latest_json.open("r", encoding="utf-8") as f:
        latest = json.load(f)

    ckpt_items = latest.get("ckpts", [])
    last_ckpts = [c for c in ckpt_items if c.get("type") == "last"]
    last_ckpts = sorted(last_ckpts, key=lambda x: int(x.get("wi", 0)))
    if not last_ckpts:
        raise RuntimeError("No 'last' checkpoints found in latest run manifest.")

    out_dir = cfg.out_dir if cfg.out_dir is not None else (model_runs_dir / cfg.default_out_dirname)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_params = MetricParams(
        top_alpha=cfg.top_alpha,
        cos_eps=cfg.cos_eps,
        share_eps=cfg.share_eps,
        day_sum_mask_eps=cfg.day_sum_mask_eps,
    )

    summary_rows: list[dict] = []
    for item in last_ckpts:
        ckpt_path = Path(item["path"])
        wi = int(item.get("wi", -1))
        if not ckpt_path.exists():
            print(f"[WARN] ckpt missing -> skip: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        run_name = ckpt.get("run_name", ckpt_path.stem)
        window = ckpt.get("window", {})
        ckfg = ckpt.get("config", {})

        lookback = int(ckfg.get("LOOKBACK", 168))
        horizon = int(ckfg.get("HORIZON", 24))
        exog_cols = ckfg.get("EXOG_COLS")
        if exog_cols is None:
            raise ValueError(f"[{run_name}] missing EXOG_COLS in checkpoint config")

        val_start = pd.to_datetime(window.get("val_start"))
        val_end = pd.to_datetime(window.get("val_end"))

        df_feat = df[[cfg.target_col] + list(exog_cols)].copy()
        y_arr = df_feat[cfg.target_col].values.astype(np.float32)
        xe_arr = df_feat[list(exog_cols)].values.astype(np.float32)
        idx_dt = df_feat.index

        t_list = build_midnight_val_t_list(
            idx_dt,
            y_arr,
            xe_arr,
            lookback=lookback,
            horizon=horizon,
            val_start_ts=val_start,
            val_end_ts=val_end,
        )

        if len(t_list) == 0:
            summary_rows.append(
                {
                    "wi": wi,
                    "run_name": run_name,
                    "ckpt_path": str(ckpt_path),
                    "val_start": str(val_start),
                    "val_end": str(val_end),
                    "midnight_samples": 0,
                    "rmse": np.nan,
                    "diff_rmse": np.nan,
                    "cos_centered": np.nan,
                    "top_iou": np.nan,
                    "share_overlap_pct": np.nan,
                    "share_tv": np.nan,
                    "note": "no valid midnight samples",
                }
            )
            continue

        model_type = str(ckfg.get("MODEL_TYPE", cfg.model_type))
        hidden_size = int(ckfg.get("HIDDEN_SIZE", cfg.hidden_size))
        cnn_channels = int(ckfg.get("CNN_CHANNELS", cfg.cnn_channels))
        cnn_kernel_size = int(ckfg.get("CNN_KERNEL_SIZE", cfg.cnn_kernel_size))

        model = create_forecast_model(
            model_type=model_type,
            exog_dim=len(exog_cols),
            horizon=horizon,
            hidden_size=hidden_size,
            cnn_channels=cnn_channels,
            cnn_kernel_size=cnn_kernel_size,
        ).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        y_true_mat, y_pred_mat = [], []
        for t in t_list:
            yt, yp = infer_one(model, y_arr, xe_arr, t, lookback, horizon, device)
            y_true_mat.append(yt)
            y_pred_mat.append(yp)
        y_true_mat = np.stack(y_true_mat, axis=0)
        y_pred_mat = np.stack(y_pred_mat, axis=0)

        rmse = float(np.sqrt(np.mean((y_pred_mat - y_true_mat) ** 2)))
        drmse = diff_rmse_np(y_pred_mat, y_true_mat)
        cosc = cos_centered_np(y_pred_mat, y_true_mat, eps=metric_params.cos_eps)
        tiou = topk_iou_np(y_pred_mat, y_true_mat, alpha=metric_params.top_alpha)
        shov = share_overlap_percent_np(y_pred_mat, y_true_mat, eps=metric_params.share_eps)
        shtv = share_tv_np(
            y_pred_mat,
            y_true_mat,
            eps=metric_params.share_eps,
            day_sum_mask_eps=metric_params.day_sum_mask_eps,
        )

        summary_rows.append(
            {
                "wi": wi,
                "run_name": run_name,
                "ckpt_path": str(ckpt_path),
                "val_start": str(val_start),
                "val_end": str(val_end),
                "midnight_samples": int(len(t_list)),
                "rmse": rmse,
                "diff_rmse": drmse,
                "cos_centered": cosc,
                "top_iou": tiou,
                "share_overlap_pct": shov,
                "share_tv": shtv,
                "note": "",
            }
        )

        print(
            f"[w={wi:04d}] {run_name} | samples={len(t_list)} | "
            f"RMSE={rmse:.4f} dRMSE={drmse:.4f} cosC={cosc:.4f} IoU={tiou:.4f} shareOv={shov:.2f}%"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["wi"]) if summary_rows else pd.DataFrame()
    out_csv = out_dir / "summary_midnight_last_latest.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"[INFO] saved summary -> {out_csv}")
    return out_csv
