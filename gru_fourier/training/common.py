"""Shared model/data/metric utilities for training and evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MetricParams:
    top_alpha: float = 0.20
    cos_eps: float = 1e-8
    share_eps: float = 1e-6
    day_sum_mask_eps: float = 1e-3


class Power24Dataset(Dataset):
    def __init__(
        self, y: np.ndarray, xe: np.ndarray, t_list: np.ndarray, lookback: int, horizon: int
    ):
        self.y = y
        self.xe = xe
        self.t_list = t_list
        self.lookback = int(lookback)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return len(self.t_list)

    def __getitem__(self, idx_: int):
        t = int(self.t_list[idx_])
        x_power = self.y[t - self.lookback : t][:, None]
        x_exog = self.xe[t : t + self.horizon, :]
        y_out = self.y[t : t + self.horizon]
        return torch.from_numpy(x_power), torch.from_numpy(x_exog), torch.from_numpy(y_out)


class LSTM24(nn.Module):
    def __init__(self, exog_dim: int, horizon: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.exog_mlp = nn.Sequential(
            nn.Linear(horizon * exog_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + 64, 128),
            nn.ReLU(),
            nn.Linear(128, horizon),
        )
        self.out_act = nn.Softplus(beta=1.0, threshold=20.0)

    def forward(self, x_power: torch.Tensor, x_exog_future: torch.Tensor) -> torch.Tensor:
        _, (h_n, c_n) = self.lstm(x_power)
        h = h_n[-1]
        c = c_n[-1]
        hc = torch.cat([h, c], dim=1)

        bsz = x_exog_future.size(0)
        ex = x_exog_future.reshape(bsz, -1)
        ex = self.exog_mlp(ex)

        z = torch.cat([hc, ex], dim=1)
        return self.out_act(self.head(z))


class CNN1D24(nn.Module):
    def __init__(self, exog_dim: int, horizon: int, channels: int = 32, kernel_size: int = 5):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.exog_mlp = nn.Sequential(
            nn.Linear(horizon * exog_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels + 64, 128),
            nn.ReLU(),
            nn.Linear(128, horizon),
        )
        self.out_act = nn.Softplus(beta=1.0, threshold=20.0)

    def forward(self, x_power: torch.Tensor, x_exog_future: torch.Tensor) -> torch.Tensor:
        # x_power: (B, L, 1) -> (B, 1, L)
        x = x_power.transpose(1, 2)
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)

        bsz = x_exog_future.size(0)
        ex = x_exog_future.reshape(bsz, -1)
        ex = self.exog_mlp(ex)

        z = torch.cat([h, ex], dim=1)
        return self.out_act(self.head(z))


def create_forecast_model(
    model_type: str,
    exog_dim: int,
    horizon: int,
    hidden_size: int = 32,
    cnn_channels: int = 32,
    cnn_kernel_size: int = 5,
) -> nn.Module:
    m = model_type.strip().lower()
    if m == "lstm":
        return LSTM24(exog_dim=exog_dim, horizon=horizon, hidden=hidden_size)
    if m in {"cnn1d", "1d-cnn", "cnn"}:
        return CNN1D24(
            exog_dim=exog_dim,
            horizon=horizon,
            channels=cnn_channels,
            kernel_size=cnn_kernel_size,
        )
    raise ValueError(f"Unsupported model_type: {model_type}. Use 'lstm' or 'cnn1d'.")


def model_signature(
    model_type: str,
    hidden_size: int,
    cnn_channels: int,
    cnn_kernel_size: int,
) -> str:
    m = model_type.strip().lower()
    if m == "lstm":
        return f"LSTM24(hidden={hidden_size})+exogMLP+Softplus"
    return f"CNN1D24(channels={cnn_channels},kernel={cnn_kernel_size})+exogMLP+Softplus"


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_t_list_in_range(
    is_finite_y: np.ndarray,
    is_finite_xe: np.ndarray,
    lookback: int,
    horizon: int,
    n_rows: int,
    t_start_inclusive: int,
    t_end_exclusive: int,
) -> np.ndarray:
    t_list: list[int] = []
    lo = max(t_start_inclusive, lookback)
    hi = min(t_end_exclusive, n_rows - horizon)
    for t in range(lo, hi):
        if not is_finite_y[t - lookback : t].all():
            continue
        if not is_finite_y[t : t + horizon].all():
            continue
        if not is_finite_xe[t : t + horizon].all():
            continue
        t_list.append(t)
    return np.array(t_list, dtype=int)


def diff_rmse_np(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    dy_p = y_pred[:, 1:] - y_pred[:, :-1]
    dy_t = y_true[:, 1:] - y_true[:, :-1]
    return float(np.sqrt(np.mean((dy_p - dy_t) ** 2)))


def cos_centered_np(y_pred: np.ndarray, y_true: np.ndarray, eps: float) -> float:
    yp = y_pred - y_pred.mean(axis=1, keepdims=True)
    yt = y_true - y_true.mean(axis=1, keepdims=True)
    num = np.sum(yp * yt, axis=1)
    den = np.linalg.norm(yp, axis=1) * np.linalg.norm(yt, axis=1) + eps
    return float(np.mean(num / den))


def topk_iou_np(y_pred: np.ndarray, y_true: np.ndarray, alpha: float, eps: float = 1e-12) -> float:
    m, h = y_true.shape
    k = int(math.ceil(alpha * h))
    ious: list[float] = []
    for i in range(m):
        idx_t = np.argpartition(-y_true[i], k - 1)[:k]
        idx_p = np.argpartition(-y_pred[i], k - 1)[:k]
        set_t, set_p = set(idx_t.tolist()), set(idx_p.tolist())
        inter = len(set_t & set_p)
        union = len(set_t | set_p)
        ious.append(inter / (union + eps))
    return float(np.mean(ious))


def share_overlap_percent_np(y_pred: np.ndarray, y_true: np.ndarray, eps: float) -> float:
    y_true_pos = np.clip(y_true, 0.0, None)
    y_pred_pos = np.clip(y_pred, 0.0, None)
    p = (y_true_pos + eps) / np.sum(y_true_pos + eps, axis=1, keepdims=True)
    q = (y_pred_pos + eps) / np.sum(y_pred_pos + eps, axis=1, keepdims=True)
    overlap = np.sum(np.minimum(p, q), axis=1)
    return float(100.0 * np.mean(overlap))


def share_tv_np(
    y_pred: np.ndarray, y_true: np.ndarray, eps: float, day_sum_mask_eps: float
) -> float:
    y_true_pos = np.clip(y_true, 0.0, None)
    y_pred_pos = np.clip(y_pred, 0.0, None)
    sum_t = np.sum(y_true_pos + eps, axis=1, keepdims=True)
    p = (y_true_pos + eps) / sum_t
    q = (y_pred_pos + eps) / np.sum(y_pred_pos + eps, axis=1, keepdims=True)
    tv = 0.5 * np.sum(np.abs(p - q), axis=1)
    w = (sum_t.squeeze(1) >= day_sum_mask_eps).astype(np.float32)
    denom = max(float(np.sum(w)), 1.0)
    return float(np.sum(tv * w) / denom)


def build_total_loss_fn(
    lambda_diff: float,
    lambda_cos: float,
    lambda_share: float,
    metric_params: MetricParams,
):
    mse_loss = nn.MSELoss()

    def diff_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        dy_true = y_true[:, 1:] - y_true[:, :-1]
        return (dy_pred - dy_true).pow(2).mean()

    def cos_centered(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        yp = y_pred - y_pred.mean(dim=1, keepdim=True)
        yt = y_true - y_true.mean(dim=1, keepdim=True)
        num = (yp * yt).sum(dim=1)
        den = yp.norm(p=2, dim=1) * yt.norm(p=2, dim=1) + metric_params.cos_eps
        return (num / den).mean()

    def share_tv_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true_pos = torch.clamp(y_true, min=0.0)
        y_pred_pos = torch.clamp(y_pred, min=0.0)

        sum_t = (y_true_pos + metric_params.share_eps).sum(dim=1, keepdim=True)
        sum_p = (y_pred_pos + metric_params.share_eps).sum(dim=1, keepdim=True)
        p = (y_true_pos + metric_params.share_eps) / sum_t
        q = (y_pred_pos + metric_params.share_eps) / sum_p

        tv = 0.5 * torch.sum(torch.abs(p - q), dim=1)
        w = (sum_t.squeeze(1) >= metric_params.day_sum_mask_eps).float()
        denom = w.sum().clamp(min=1.0)
        return (tv * w).sum() / denom

    def total_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        base = mse_loss(y_pred, y_true)
        dloss = diff_mse(y_pred, y_true)
        clos = 1.0 - cos_centered(y_pred, y_true)
        sh = share_tv_loss(y_pred, y_true)
        return base + lambda_diff * dloss + lambda_cos * clos + lambda_share * sh

    return total_loss


@torch.no_grad()
def eval_all_metrics(model: nn.Module, dataloader, device: str, metric_params: MetricParams):
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []

    for x_power, x_exog, y_true in dataloader:
        x_power = x_power.to(device)
        x_exog = x_exog.to(device)
        y_true = y_true.to(device)
        y_pred = model(x_power, x_exog)
        preds.append(y_pred.detach().cpu().numpy().astype(np.float32))
        trues.append(y_true.detach().cpu().numpy().astype(np.float32))

    if not preds:
        return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan

    y_pred_mat = np.concatenate(preds, axis=0)
    y_true_mat = np.concatenate(trues, axis=0)

    rmse = float(np.sqrt(np.mean((y_pred_mat - y_true_mat) ** 2)))
    drmse = diff_rmse_np(y_pred_mat, y_true_mat)
    cosc = cos_centered_np(y_pred_mat, y_true_mat, eps=metric_params.cos_eps)
    iou = topk_iou_np(y_pred_mat, y_true_mat, alpha=metric_params.top_alpha)
    shov = share_overlap_percent_np(y_pred_mat, y_true_mat, eps=metric_params.share_eps)
    shtv = share_tv_np(
        y_pred_mat,
        y_true_mat,
        eps=metric_params.share_eps,
        day_sum_mask_eps=metric_params.day_sum_mask_eps,
    )
    return rmse, drmse, cosc, iou, shov, shtv
