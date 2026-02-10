# LG3 Baseline Forecast Suite

This repository now includes a reproducible baseline suite for **next-day (24h) forecasting** using the **past 7 days (168h)** of history.

## What is implemented

### Baselines (4)
1. **Last-output hold**
   - Forecast all next 24 points as the last observed value.
2. **Naive persistence (daily seasonal naive)**
   - Forecast tomorrow using yesterday's same-hour values.
3. **Weekly hour-of-day mean**
   - Compute mean by hour-of-day over the last 7 days and forecast by matching forecast hours.
4. **Trend + daily Fourier regression**
   - OLS with intercept, linear trend, and daily seasonality terms
   - Uses `numpy.linalg.lstsq`
   - `K` harmonics configurable (`--fourier-k 1|2`, default `2`)

### Evaluation outputs
- **Overall MSE** across all forecast points and windows
- **Hour-of-day MSE** for each hour `0..23`
- Count metadata:
  - `n_windows`
  - `n_points`
- Optional diagnostics:
  - per-window MSE per baseline

## Data assumptions
- Input CSV has:
  - time column (`--time-col`)
  - target column (`--target-col`)
- Timestamps are already in **Asia/Seoul local time** and stored as **naive datetimes**.
  - No timezone localization/conversion is applied.
  - Hour-of-day uses parsed naive datetime directly.

## Preprocessing
The pipeline does the following:
1. Read CSV and parse datetime from `--time-col`
2. Sort ascending by time and set as index
3. Resample target to hourly (`1H`) with mean
4. Interpolate missing values with `method="time"`
5. Drop remaining edge NaNs

## Sliding-window setup
For each anchor `t`:
- History: `y[t-168 : t)`
- Future truth: `y[t : t+24)`
- Default stride: `24h` (`--stride-hours` configurable)
- Windows missing full history or horizon are skipped

## Project structure

```text
baselines/
  __init__.py
  methods.py
  evaluation.py
scripts/
  run_baselines.py
outputs/
README.md
```

## Usage

```bash
python scripts/run_baselines.py \
  --csv <PATH_TO_CSV> \
  --time-col <TIME_COL> \
  --target-col <TARGET_COL> \
  --fourier-k 2 \
  --stride-hours 24 \
  --output-dir outputs
```

### CLI arguments
- `--csv` (required): input CSV path
- `--time-col` (required): datetime column name
- `--target-col` (required): target column name
- `--fourier-k` (optional): `1` or `2` (default: `2`)
- `--stride-hours` (optional): sliding stride in hours (default: `24`)
- `--output-dir` (optional): output folder (default: `outputs`)

## Output files
- `outputs/mse_summary.json`
  - `overall_mse` per baseline
  - `hour_of_day_mse` (24-length array) per baseline
  - `n_windows`, `n_points`
  - `per_window_mse` per baseline
- `outputs/prediction_samples.csv`
  - sampled windows (first 2 + last 2)
  - columns include: `timestamp`, `y_true`, and all `y_hat_*`

## Example

```bash
python scripts/run_baselines.py \
  --csv processed_data/snu/lg3_train.csv \
  --time-col Timestamp \
  --target-col Power
```

The script prints:
- overall MSE table
- top-3 worst hour-of-day MSE entries per baseline
