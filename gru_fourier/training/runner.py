"""Rolling-window trainer reproduced from notebook logic."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gru_fourier.src.config import TrainConfig

from .common import (
    LSTM24,
    MetricParams,
    Power24Dataset,
    build_total_loss_fn,
    eval_all_metrics,
    make_t_list_in_range,
    seed_all,
)


def _fmt_dt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d")


def _build_windows(n_rows: int, cfg: TrainConfig) -> list[tuple[int, int, int, int]]:
    train_start = cfg.buffer_hours
    train_end = train_start + cfg.train_hours
    val_start = train_end
    val_end = val_start + cfg.valid_hours

    windows: list[tuple[int, int, int, int]] = []
    while val_end <= n_rows:
        windows.append((train_start, train_end, val_start, val_end))
        train_start += cfg.step_hours
        train_end += cfg.step_hours
        val_start += cfg.step_hours
        val_end += cfg.step_hours
    return windows


def _resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_cfg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' but CUDA is not available")
    return device_cfg


def run_training(cfg: TrainConfig) -> Path:
    device = _resolve_device(cfg.device)
    seed_all(cfg.seed)

    df = pd.read_csv(cfg.csv_path, parse_dates=[cfg.dt_col]).set_index(cfg.dt_col).sort_index()

    req_min = [cfg.target_col] + cfg.exog_cols_base + [cfg.fourier_col] + cfg.weather_cols
    if cfg.use_tod:
        req_min += [cfg.tod_col]
    missing = [c for c in req_min if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    exog_cols = [cfg.fourier_col] + cfg.weather_cols + cfg.exog_cols_base
    if cfg.use_tod:
        exog_cols = [cfg.fourier_col] + cfg.weather_cols + [cfg.tod_col] + cfg.exog_cols_base

    df_feat = df[[cfg.target_col] + exog_cols].copy()
    y = df_feat[cfg.target_col].values.astype(np.float32)
    xe = df_feat[exog_cols].values.astype(np.float32)
    idx = df_feat.index
    n_rows = len(df_feat)

    is_finite_y = np.isfinite(y)
    is_finite_xe = np.isfinite(xe).all(axis=1)

    windows = _build_windows(n_rows, cfg)
    if not windows:
        raise RuntimeError("No rolling windows available. Check buffer/train/valid settings.")

    print(f"[INFO] rows={n_rows}, windows={len(windows)}")
    print(
        f"[INFO] first window train={idx[windows[0][0]]}~{idx[windows[0][1]-1]} | "
        f"val={idx[windows[0][2]]}~{idx[windows[0][3]-1]}"
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = cfg.runs_dir / f"exp_{run_id}"
    ckpt_flat_dir = exp_dir / "ckpts_flat"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_flat_dir.mkdir(parents=True, exist_ok=True)

    latest_json = cfg.runs_dir / "LATEST_RUN.json"
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exp_dir": str(exp_dir.resolve()),
        "ckpts": [],
    }

    metric_params = MetricParams(
        top_alpha=cfg.top_alpha,
        cos_eps=cfg.cos_eps,
        share_eps=cfg.share_eps,
        day_sum_mask_eps=cfg.day_sum_mask_eps,
    )
    total_loss = build_total_loss_fn(
        lambda_diff=cfg.lambda_diff,
        lambda_cos=cfg.lambda_cos,
        lambda_share=cfg.lambda_share,
        metric_params=metric_params,
    )

    for wi, (tr_s, tr_e, va_s, va_e) in enumerate(windows, start=1):
        t_list_train = make_t_list_in_range(
            is_finite_y,
            is_finite_xe,
            lookback=cfg.lookback,
            horizon=cfg.horizon,
            n_rows=n_rows,
            t_start_inclusive=tr_s,
            t_end_exclusive=tr_e,
        )
        t_list_val = make_t_list_in_range(
            is_finite_y,
            is_finite_xe,
            lookback=cfg.lookback,
            horizon=cfg.horizon,
            n_rows=n_rows,
            t_start_inclusive=va_s,
            t_end_exclusive=va_e,
        )

        if len(t_list_train) == 0 or len(t_list_val) == 0:
            print(f"[WARN] window {wi:04d}: skip (train_t={len(t_list_train)}, val_t={len(t_list_val)})")
            continue

        tr_start_dt = idx[tr_s]
        tr_end_dt = idx[tr_e - 1]
        va_start_dt = idx[va_s]
        va_end_dt = idx[va_e - 1]

        run_name = (
            f"run_{wi:04d}_"
            f"TR{_fmt_dt(tr_start_dt)}-{_fmt_dt(tr_end_dt)}_"
            f"VA{_fmt_dt(va_start_dt)}-{_fmt_dt(va_end_dt)}"
        )
        run_dir = exp_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n========== [{wi:04d}/{len(windows):04d}] {run_name} ==========")
        print(f"  train_t={len(t_list_train)} | val_t={len(t_list_val)}")

        ds_train = Power24Dataset(y, xe, t_list_train, cfg.lookback, cfg.horizon)
        ds_val = Power24Dataset(y, xe, t_list_val, cfg.lookback, cfg.horizon)
        dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        seed_all(cfg.seed)
        model = LSTM24(exog_dim=len(exog_cols), horizon=cfg.horizon, hidden=cfg.hidden_size).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_val_rmse = float("inf")
        best_state = None
        best_epoch = None

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total_loss_sum, n_batches = 0.0, 0
            for x_power, x_exog, y_true in dl_train:
                x_power = x_power.to(device)
                x_exog = x_exog.to(device)
                y_true = y_true.to(device)

                opt.zero_grad()
                y_pred = model(x_power, x_exog)
                loss = total_loss(y_pred, y_true)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

                total_loss_sum += float(loss.item())
                n_batches += 1

            train_rmse, train_drmse, train_cos, train_iou, train_shov, train_shtv = eval_all_metrics(
                model, dl_train, device, metric_params
            )
            val_rmse, val_drmse, val_cos, val_iou, val_shov, val_shtv = eval_all_metrics(
                model, dl_val, device, metric_params
            )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            avg_train_loss = total_loss_sum / max(n_batches, 1)
            if (epoch == 1) or (epoch % cfg.log_every == 0) or (epoch == cfg.epochs):
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_loss={avg_train_loss:.6f} | "
                    f"train_RMSE={train_rmse:.6f}, dRMSE={train_drmse:.6f}, cosC={train_cos:.4f}, "
                    f"IoU={train_iou:.4f}, shareOv={train_shov:.2f}%, shareTV={train_shtv:.4f} | "
                    f"val_RMSE={val_rmse:.6f}, dRMSE={val_drmse:.6f}, cosC={val_cos:.4f}, "
                    f"IoU={val_iou:.4f}, shareOv={val_shov:.2f}%, shareTV={val_shtv:.4f} | "
                    f"best_val_RMSE={best_val_rmse:.6f} (epoch={best_epoch})"
                )

        tr_s_ymd = _fmt_dt(tr_start_dt)
        tr_e_ymd = _fmt_dt(tr_end_dt)
        va_s_ymd = _fmt_dt(va_start_dt)
        va_e_ymd = _fmt_dt(va_end_dt)

        flat_last_name = f"last_w{wi:04d}_TR{tr_s_ymd}-{tr_e_ymd}_VA{va_s_ymd}-{va_e_ymd}.pt"
        flat_best_name = f"best_w{wi:04d}_TR{tr_s_ymd}-{tr_e_ymd}_VA{va_s_ymd}-{va_e_ymd}.pt"

        last_path = run_dir / "last.pt"
        best_path = run_dir / "best.pt"
        flat_last_path = ckpt_flat_dir / flat_last_name
        flat_best_path = ckpt_flat_dir / flat_best_name

        window_meta = {
            "wi": wi,
            "train_start": str(tr_start_dt),
            "train_end": str(tr_end_dt),
            "val_start": str(va_start_dt),
            "val_end": str(va_end_dt),
            "train_hours": cfg.train_hours,
            "val_hours": cfg.valid_hours,
            "step_hours": cfg.step_hours,
            "buffer_hours": cfg.buffer_hours,
        }
        config_meta = {
            "LOOKBACK": cfg.lookback,
            "HORIZON": cfg.horizon,
            "TARGET_COL": cfg.target_col,
            "EXOG_COLS": exog_cols,
            "USE_TOD": cfg.use_tod,
            "LOSS": f"MSE + {cfg.lambda_diff}*diffMSE + {cfg.lambda_cos}*(1-cosCentered) + {cfg.lambda_share}*shareTV",
            "TOP_ALPHA": cfg.top_alpha,
            "SHARE_EPS": cfg.share_eps,
            "DAY_SUM_MASK_EPS": cfg.day_sum_mask_eps,
            "N_TRAIN_T": int(len(t_list_train)),
            "N_VAL_T": int(len(t_list_val)),
            "SEED": cfg.seed,
            "LR": cfg.lr,
            "WEIGHT_DECAY": cfg.weight_decay,
            "EPOCHS": cfg.epochs,
            "BATCH_SIZE": cfg.batch_size,
            "GRAD_CLIP": cfg.grad_clip,
            "MODEL": f"LSTM24(hidden={cfg.hidden_size})+exogMLP+Softplus",
        }

        last_payload = {
            "epoch": cfg.epochs,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "best_val_rmse": best_val_rmse,
            "best_epoch": best_epoch,
            "run_name": run_name,
            "window": window_meta,
            "config": config_meta,
        }
        torch.save(last_payload, last_path)
        torch.save(last_payload, flat_last_path)
        manifest["ckpts"].append({"type": "last", "wi": int(wi), "path": str(flat_last_path.resolve())})

        if best_state is not None:
            best_payload = {
                "epoch": best_epoch,
                "model_state": best_state,
                "best_val_rmse": best_val_rmse,
                "run_name": run_name,
                "window": window_meta,
                "config": config_meta,
            }
            torch.save(best_payload, best_path)
            torch.save(best_payload, flat_best_path)
            manifest["ckpts"].append({"type": "best", "wi": int(wi), "path": str(flat_best_path.resolve())})

    with latest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] wrote latest run pointer -> {latest_json}")
    print(f"[INFO] this run ckpts count = {len(manifest['ckpts'])}")
    return latest_json
