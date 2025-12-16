# MapleStory Item Price Prediction

메이플스토리 아이템 시세 예측을 위한 End-to-End 머신러닝 파이프라인 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 메이플스토리 아이템 거래 데이터(2.6M+ 거래 기록)를 사용하여 아이템 가격을 예측하는 LightGBM 모델을 구축합니다.

### 최신 모델 성능

| Metric | Value |
|--------|-------|
| **Test R²** | 0.7885 |
| **Test MAE** | 427M Meso |
| **Test MAPE** | 354% |
| **Log-scale R²** | 0.9691 |

### 주요 기능

- **데이터 추출**: MySQL 또는 JSONL 파일에서 거래 데이터 로드
- **고급 특징 추출**: 134개 특징 (잠재능력 % 파싱, 아이템 메타데이터 등)
- **로그 변환**: 가격 범위가 넓은 데이터에 적합한 log1p 변환 적용
- **LightGBM 모델**: 하이퍼파라미터 튜닝된 Gradient Boosting 모델
- **모델 저장/로드**: 훈련된 모델 및 전처리 파이프라인 저장

## 프로젝트 구조

```
maple-meso/
├── README.md                      # 이 파일
├── FINAL_REPORT.md               # 최종 보고서
├── requirements.txt              # Python 의존성
├── pyproject.toml               # 프로젝트 설정
├── main.py                      # 기본 실행 스크립트
│
├── src/                         # 소스 코드
│   ├── preprocess.py            # 데이터 전처리 (134개 특징 추출)
│   ├── train.py                 # 모델 훈련 모듈
│   ├── train_improved.py        # 개선된 훈련 파이프라인
│   ├── predict.py               # 가격 예측 인터페이스
│   ├── ensemble.py              # 앙상블 모델
│   ├── tune_hyperparameters.py  # 하이퍼파라미터 튜닝
│   ├── config.py                # 데이터베이스 설정
│   ├── db_connection.py         # DB 연결 유틸리티
│   └── pipeline.py              # E2E 파이프라인
│
├── data/
│   ├── raw/                     # 원본 데이터
│   │   └── raw_data.jsonl       # JSONL 형식 거래 데이터
│   └── processed/               # 전처리된 데이터
│       └── preprocessed_data.parquet
│
├── models/
│   ├── improved/                # 최신 개선된 모델 ⭐
│   │   ├── model.joblib         # 훈련된 LightGBM 모델
│   │   ├── scaler.joblib        # 특징 스케일러
│   │   ├── label_encoders.joblib # 레이블 인코더
│   │   ├── feature_importance.json
│   │   └── metrics.json
│   ├── archive/                 # 이전 버전 모델
│   └── hyperparameter_tuning_results.json
│
├── docs/                        # 문서
│   ├── PROGRESS_REPORT.md       # 진행 보고서
│   ├── MODEL_COMPARISON.md      # 모델 비교
│   ├── IMPROVEMENT_GUIDE.md     # 개선 가이드
│   └── colab/                   # Google Colab 관련
│       ├── maple_meso_colab.ipynb
│       └── COLAB_SETUP.md
│
└── logs/                        # 훈련 로그
    ├── training_improved.log
    └── ...
```

## 설치 및 설정

### 필요 조건

- Python >= 3.10
- pip 또는 uv 패키지 매니저

### 설치

```bash
# pip 사용
pip install -r requirements.txt

# 또는 uv 사용
uv sync
```

## 사용 방법

### 1. 개선된 모델 훈련 (권장)

```bash
# JSONL 파일에서 데이터 로드하여 훈련 (DB 연결 불필요)
python -m src.train_improved --save-dir models

# 샘플로 빠른 테스트
python -m src.train_improved --sample 50000 --save-dir models

# 로그 변환 없이 훈련
python -m src.train_improved --no-log-transform --save-dir models
```

### 2. 가격 예측

```bash
# JSON 파일로 예측
python src/predict.py --json-file <파일경로>

# JSON 문자열로 예측
python src/predict.py --json-string '{"name": "...", "item_id": 1004423, ...}'

# 인터랙티브 모드
python src/predict.py --interactive
```

### 3. Python에서 사용

```python
from src.predict import predict_price

payload_json = {
    "name": "앱솔랩스 메이지크라운",
    "item_id": 1004423,
    "star_force": 22,
    "potential_grade": 4,
    "additional_grade": 4,
    "payload_json": {
        "detail_json": "...",
        "summary_json": "...",
    }
}

result = predict_price(payload_json)
print(f"예측 가격: {result['predicted_price_formatted']} 메소")
```

## 데이터 구조

### 입력 데이터 (raw_data.jsonl)

주요 필드:
- `trade_sn`: 거래 일련번호
- `name`: 아이템 이름
- `item_id`: 아이템 ID
- `price`: 가격 (타겟 변수)
- `star_force`: 스타포스
- `potential_grade`: 잠재능력 등급 (0-4)
- `additional_grade`: 추가옵션 등급 (0-4)
- `payload_json`: 중첩된 JSON (detail_json, summary_json 포함)
- `created_at`: 거래 시간

### 추출되는 특징 (134개)

1. **기본 특징**: `item_id`, `name`, `category`, `star_force`, `potential_grade`
2. **스탯 특징**: `detail_PAD_sum`, `detail_MAD_max`, `total_stat_sum` 등
3. **잠재능력 파싱**: `pot_str_percent`, `pot_boss_dmg`, `pot_ied` 등
4. **아이템 메타데이터**: `level_requirement`, `job_type`, `star_force_max`
5. **시간 특징**: `year`, `month`, `day_of_week`, `hour`, `day_of_year`
6. **복합 특징**: `enhancement_score`, `value_score`, `potential_value_score`

## 모델 정보

### 현재 최적 모델: LightGBM + Log Transform

- **알고리즘**: LightGBM (Gradient Boosting)
- **타겟 변환**: log1p (가격의 로그 변환)
- **특징 수**: 134개
- **훈련 샘플**: 1,840,936개
- **테스트 샘플**: 525,982개

### Top 10 중요 특징

1. `name` - 아이템 이름
2. `day_of_year` - 연중 일자 (시장 트렌드)
3. `item_id` - 아이템 ID
4. `value_score` - 복합 가치 점수
5. `job_type` - 직업군
6. `detail_percent_StatR_sum` - 스탯 % 합계
7. `day` - 일자
8. `detail_cuttable` - 가위 사용 가능 횟수
9. `level_requirement` - 착용 레벨
10. `max_stat` - 최대 스탯 수치

## 평가 지표

| 지표 | 설명 |
|------|------|
| **R²** | 결정계수 (0.7885 = 79% 분산 설명) |
| **RMSE** | 평균 제곱근 오차 |
| **MAE** | 평균 절대 오차 (427M Meso) |
| **MAPE** | 평균 절대 백분율 오차 (354%) |
| **Log R²** | 로그 스케일 R² (0.969) |

## 문서

- [FINAL_REPORT.md](FINAL_REPORT.md) - 최종 보고서 (문제 정의, 방법론, 결과 분석)
- [docs/PROGRESS_REPORT.md](docs/PROGRESS_REPORT.md) - 프로젝트 진행 보고서
- [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) - 모델 비교 분석

## 라이선스

이 프로젝트는 개인 프로젝트입니다.
