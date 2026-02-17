## 개발 환경 정리 (Conda + Ruff)

아래 명령으로 `<python_env>` 환경을 기준으로 lint/fix/format을 실행할 수 있습니다.

```bash
# 방법 1) 환경을 활성화한 뒤 실행
conda activate <python_env>
ruff check .
ruff check . --fix
ruff format .

# 방법 2) 한 줄 실행 (환경 활성화 없이)
conda run -n <python_env> ruff check .
conda run -n <python_env> ruff check . --fix
conda run -n <python_env> ruff format .
```

`pyproject.toml`에 Ruff 설정과 기본 의존성이 정리되어 있습니다.

Ruff의 핵심 기능은 아래 3가지입니다.

- lint: 잠재 버그/나쁜 스타일/불필요 코드 검사 (`ruff check`)
- fix: 자동 수정 가능한 항목 바로 고침 (`ruff check --fix`)
- format: 코드 포맷 통일 (`ruff format`)

## 🚀 실행 방법 (Usage)

각 단계별 실행 옵션은 --help 플래그를 통해 확인할 수 있습니다.

```bash
python3 -m gru_fourier.src.preprocess_main --help
python3 -m gru_fourier.src.train_main --help
python3 -m gru_fourier.src.evaluate_main --help
```

### 1. 데이터 전처리 (Preprocessing)

```bash
python3 -m gru_fourier.src.preprocess_main --config gru_fourier/config/preprocess.toml
```

### 2. 모델 학습 (Training)

* **LSTM (기본)**
```bash
python3 -m gru_fourier.src.train_main --config gru_fourier/config/train.toml
```

* **1D-CNN**
```bash
python3 -m gru_fourier.src.train_main --config gru_fourier/config/train.toml --model-type cnn1d
```

### 3. 성능 평가 (Evaluation)

* **LSTM**
```bash
python3 -m gru_fourier.src.evaluate_main --config gru_fourier/config/evaluate.toml
```

* **1D-CNN**
```bash
python3 -m gru_fourier.src.evaluate_main --config gru_fourier/config/evaluate.toml --model-type cnn1d
```

### 4. 결과 시각화 (Visualization)

* **LSTM**
```bash
python3 -m gru_fourier.src.evaluate_plot_main --config gru_fourier/config/evaluate.toml --save-plots
```

* **1D-CNN**
```bash
python3 -m gru_fourier.src.evaluate_plot_main --config gru_fourier/config/evaluate.toml --save-plots --model-type cnn1d
```

# 결과
## 1D-CNN
last=400epoch, window=0007

| METRIC  | LSTM@0007 | 1D-CNN@0007 |
|---------|-----------|-------------|
| RMSE    | 46.2322   | 31.6983     |
| dRMSE   | 40.1051   | 38.0844     |
| cosC    | 0.5708    | 0.6038      |
| IoU     | 0.1803    | 0.2993      |
| shareOv | 60.72%    | 62.65%      |

## [성능평가법]
| METRIC  | VALUE   | DESCRIPTION |
|---------|---------|-------------|
| RMSE    | 43.2866 |  평균 오차 크기: 실제값과 예측값 차이의 제곱평균제곱근입니다. 값이 낮을수록 전체적인 수치 예측이 정확함을 의미합니다.  |
| dRMSE   | 50.4444 | 변화량 오차 (diff-RMSE): 시계열의 **'기울기(변화량)'**에 대한 RMSE입니다. 현재 RMSE보다 dRMSE가 높다는 것은, 수치 자체보다 **값이 오르내리는 시점이나 속도(변동성)**를 맞추는 데 더 큰 어려움을 겪고 있음을 시사합니다.  |
| cosC    | 0.6465  |  방향 유사도 (Cosine Similarity): 두 벡터 사이의 각도를 측정합니다. 1에 가까울수록 패턴의 '모양'이 일치함을 뜻합니다. 0.64는 전체적인 흐름은 따라가고 있으나 세밀한 굴곡에서 차이가 있음을 나타냅니다. |
| IoU     | 0.2098  |  이벤트 일치도 (Intersection over Union): 주로 특정 임계치(Peak 등)를 넘는 구간이 얼마나 겹치는지 측정합니다. 0.20은 현재 모델의 가장 취약한 부분으로, 피크 타임 예측이 빗나가고 있을 확률이 높습니다. |
| shareOv | 66.07%  |  영역 중첩도 (Shared Overlap): 전체 면적 중 실제와 예측이 겹치는 비율입니다. 값이 높을수록 모델이 데이터의 전체적인 규모(Scale)를 잘 파악하고 있다는 뜻입니다.  |

## 생성되는 파일
### 1. preprocess
```
.
└── processed_data/
    └── preprocessed_1h_master_with_weather_delta_20250701_20250930_ohsungsa_f2.csv
        (preprocess 결과 feature 마스터 CSV)
```
### 2. train
```
.
└── runs_lstm24_roll/
    ├─── LATEST_RUN.json
    │    (가장 최근 train 실행의 메타/체크포인트 포인터)
    │
    └─── exp_YYYYMMDD_HHMMSS/
         ├─ ckpts_flat/
         │  ├─ last_w0001_TR..._VA....pt
         │  ├─ best_w0001_TR..._VA....pt
         │  └─ ...
         │  (윈도우별 체크포인트를 한 폴더에 평탄화 저장)
         │
         ├─ run_0001_TR..._VA.../
         │  ├─ last.pt
         │  └─ best.pt
         ├─ run_0002_TR..._VA.../
         │  ├─ last.pt
         │  └─ best.pt
         └─ ...
          (각 rolling window 실험 폴더)\
```
### 3. eval
```
.
└─ eval_last_midnight_on_own_val_inline_latest/
    └── summary_midnight_last_latest.csv
        (eval / eval_plot 공통: 윈도우별 지표 요약)
```
### 4. eval(plot)
```
.
└─ eval_last_midnight_on_own_val_inline_latest/
    ├── summary_midnight_last_latest.csv
    │   (eval / eval_plot 공통: 윈도우별 지표 요약)
    │
    └── plots/                      [eval_plot에서 --save-plots일 때만 생성]
        ├─ run_0001_TR..._VA.../
        │  ├─ w0001_day_000_YYYYMMDD.png
        │  ├─ w0001_day_001_YYYYMMDD.png
        │  └─ ... (최대 7개/모델 기본)
        └─ ...
```
