"""EREPORT Power preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_sampling_step(index: pd.DatetimeIndex, fallback_sec: float = 60.0) -> float:
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        return fallback_sec
    median_step = float(diffs.median())
    if not np.isfinite(median_step) or median_step <= 0:
        return fallback_sec
    return median_step


def calculate_power_adjustment(
    p_obs: float,
    n_obs: int,
    expected_rows: int,
    past_nonzero: bool,
    future_nonzero: bool,
) -> float:
    missing = expected_rows - n_obs
    if missing <= 0:
        return p_obs
    if past_nonzero and future_nonzero:
        return p_obs * (expected_rows / max(n_obs, 1))
    if past_nonzero ^ future_nonzero:
        avg_obs = p_obs / max(n_obs, 1)
        return p_obs + 0.5 * avg_obs * missing
    return p_obs


def _infer_gap_activity(obs_series: pd.Series, probe: list[pd.Timestamp]) -> tuple[bool, bool]:
    if obs_series.empty:
        return False, False
    obs_ts = obs_series.index.view("int64")
    obs_v = obs_series.to_numpy(dtype=float)

    past_nonzero = False
    future_nonzero = False
    for tmiss in probe:
        t_ns = pd.Timestamp(tmiss).value
        pos = int(np.searchsorted(obs_ts, t_ns, side="left"))
        if pos - 1 >= 0 and obs_v[pos - 1] != 0.0:
            past_nonzero = True
        if pos < len(obs_v) and obs_v[pos] != 0.0:
            future_nonzero = True
        if past_nonzero and future_nonzero:
            break
    return past_nonzero, future_nonzero


def apply_forced_zero_ranges(
    df: pd.DataFrame,
    forced_zero_ranges: list[tuple[str, str]] | None,
    col: str = "Power_1h",
) -> pd.DataFrame:
    if not forced_zero_ranges:
        return df
    out = df.copy()
    for a, b in forced_zero_ranges:
        a_ts = pd.Timestamp(a)
        b_ts = pd.Timestamp(b)
        mask = (out.index >= a_ts) & (out.index <= b_ts)
        out.loc[mask, col] = 0.0
    return out


def preprocess_ereport_power_1h(
    df_ereport: pd.DataFrame,
    master_index: pd.DatetimeIndex,
    coverage_threshold: float,
    forced_zero_ranges: list[tuple[str, str]] | None,
    freq: str = "1h",
) -> pd.DataFrame:
    if "Power" not in df_ereport.columns:
        raise ValueError(f"'Power' column is missing in EREPORT data: {df_ereport.columns.tolist()}")

    df = df_ereport.copy()
    df["Power"] = pd.to_numeric(df["Power"], errors="coerce")
    df["slot"] = df.index.ceil(freq)

    median_step = estimate_sampling_step(df.index)
    expected_rows = max(int(round(3600.0 / median_step)), 1)
    min_required_rows = 0 if coverage_threshold <= 0 else int(expected_rows * coverage_threshold)

    grp_sum = df.groupby("slot")["Power"].sum(min_count=1)
    grp_n = df.groupby("slot")["Power"].count()
    slot_power = {slot: grp["Power"].dropna().sort_index() for slot, grp in df.groupby("slot")}

    out = pd.DataFrame(index=grp_sum.index)
    out["Power_1h"] = np.nan

    for slot, p_obs in grp_sum.items():
        if not np.isfinite(p_obs):
            continue
        n = int(grp_n.get(slot, 0))
        if n < min_required_rows:
            continue

        obs_series = slot_power.get(slot)
        if obs_series is None or obs_series.empty:
            continue

        missing = expected_rows - n
        if missing <= 0:
            out.at[slot, "Power_1h"] = float(p_obs)
            continue

        w_end = pd.Timestamp(slot)
        step = pd.Timedelta(seconds=median_step)
        grid = pd.date_range(start=w_end - pd.Timedelta(freq) + step, end=w_end, freq=step)
        obs_set = set(obs_series.index)
        missing_times = [t for t in grid if t not in obs_set]
        probe = missing_times[:: max(1, len(missing_times) // 10)] if len(missing_times) > 10 else missing_times
        past_nonzero, future_nonzero = _infer_gap_activity(obs_series, probe)

        p_adj = calculate_power_adjustment(
            p_obs=float(p_obs),
            n_obs=n,
            expected_rows=expected_rows,
            past_nonzero=past_nonzero,
            future_nonzero=future_nonzero,
        )
        out.at[slot, "Power_1h"] = p_adj

    out = out.reindex(master_index)
    out = apply_forced_zero_ranges(out, forced_zero_ranges, col="Power_1h")
    return out[["Power_1h"]]
