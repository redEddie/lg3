## [실행 방법]
사용가능한 옵션 보는 법. 

`python -m gru_fourier.src.preprocess_main --help`
`python -m gru_fourier.src.train_main --help`
`python -m gru_fourier.src.evaluate_main --help`

1. 전처리
`python3 -m gru_fourier.src.preprocess_main --config gru_fourier/config/preprocess.toml`
2. 학습
`python3 -m gru_fourier.src.train_main --config gru_fourier/config/train.toml`
3. 평가
`python3 -m gru_fourier.src.evaluate_main --config gru_fourier/config/evaluate.toml`
4. 평가(시각화)
`python3 -m gru_fourier.src.evaluate_plot_main --config gru_fourier/config/evaluate.toml --save-plots`


# 결과
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