"""I/O helpers for daily log files."""

from __future__ import annotations

import io
from pathlib import Path
import re

import pandas as pd


def extract_date_from_filename(filename: str) -> pd.Timestamp:
    match = re.search(r"(\d{8})", filename)
    if not match:
        raise ValueError(f"Could not find YYYYMMDD in file name: {filename}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def make_datetime_from_time_and_filename(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    if "Time" not in df.columns:
        raise ValueError(f"'Time' column is missing in {file_path}")
    file_date = extract_date_from_filename(file_path.name)
    out = df.copy()
    out["datetime"] = pd.to_datetime(
        file_date.strftime("%Y-%m-%d") + " " + out["Time"].astype(str),
        errors="coerce",
    )
    return out.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")


def read_csv_remove_nulls(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    with path.open("rb") as fh:
        raw = fh.read()
    if b"\x00" in raw:
        raw = raw.replace(b"\x00", b"")
    return pd.read_csv(io.BytesIO(raw), on_bad_lines="skip", usecols=usecols, low_memory=False)


def filter_files_by_date(
    files: list[Path],
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
) -> list[Path]:
    if date_start is None or date_end is None:
        return files
    lo = pd.Timestamp(date_start).date()
    hi = pd.Timestamp(date_end).date()
    selected: list[Path] = []
    for f in files:
        try:
            d = extract_date_from_filename(f.name).date()
        except ValueError:
            continue
        if lo <= d <= hi:
            selected.append(f)
    return selected


def load_all_csv_time_from_filename(
    folder: Path,
    pattern: str,
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    files = sorted(folder.glob(pattern))
    files = filter_files_by_date(files, date_start=date_start, date_end=date_end)
    if not files:
        raise FileNotFoundError(f"No files found: {folder} with pattern {pattern}")

    frames: list[pd.DataFrame] = []
    for f in files:
        raw = read_csv_remove_nulls(f, usecols=usecols)
        one = make_datetime_from_time_and_filename(raw, f)
        one["src_file"] = f.name
        frames.append(one)
    return pd.concat(frames, axis=0).sort_index()
