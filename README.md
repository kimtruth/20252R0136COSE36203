# MapleStory Item Price Prediction

메이플스토리 아이템 시세 예측을 위한 End-to-End 머신러닝 파이프라인 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 MySQL 데이터베이스에 저장된 메이플스토리 아이템 거래 데이터를 사용하여 아이템 가격을 예측하는 회귀 모델을 구축합니다.

### 주요 기능

- **데이터 추출 (Extract)**: MySQL 데이터베이스에서 거래 데이터 추출
- **데이터 변환 (Transform)**: 중첩된 JSON 데이터를 평평한 특징 벡터로 변환
- **데이터 적재 (Load)**: 전처리된 데이터를 parquet 형식으로 저장
- **모델 훈련**: Random Forest 및 Gradient Boosting 모델 훈련
- **모델 평가**: 다양한 지표로 모델 성능 평가
- **모델 저장**: 훈련된 모델 및 전처리 파이프라인 저장

## 프로젝트 구조

```
maple-meso/
├── src/
│   ├── config.py              # 데이터베이스 설정
│   ├── db_connection.py        # 데이터베이스 연결 유틸리티
│   ├── explore_data.py        # 데이터 탐색 스크립트
│   ├── preprocess.py          # 데이터 전처리 모듈
│   ├── train.py               # 모델 훈련 모듈
│   └── pipeline.py            # E2E 파이프라인
├── data/
│   └── processed/             # 전처리된 데이터 저장
├── models/                    # 훈련된 모델 저장
├── main.py                    # 메인 실행 스크립트
├── pyproject.toml             # 프로젝트 의존성
└── README.md
```

## 설치 및 설정

### 필요 조건

- Python >= 3.11
- uv 패키지 매니저

### 설치

```bash
# 의존성 설치
uv sync

# 환경 활성화 (선택사항)
source .venv/bin/activate  # macOS/Linux
```

## 사용 방법

### 1. 데이터베이스 연결 확인

데이터베이스 설정은 `src/config.py`에 있습니다.

### 2. 데이터 탐색

```bash
uv run python src/explore_data.py
```

### 3. 전체 파이프라인 실행

```bash
# 기본 실행 (10000개 샘플로 테스트)
uv run python main.py

# 전체 데이터셋으로 실행
uv run python main.py --limit None

# 커스텀 설정으로 실행
uv run python src/pipeline.py --limit 50000 --model random_forest
```

### 4. 개별 단계 실행

#### 데이터 전처리만 실행

```bash
uv run python src/preprocess.py
```

#### 모델 훈련만 실행 (전처리된 데이터 필요)

```bash
uv run python src/train.py
```

### 5. 가격 예측 (테스트)

훈련된 모델을 사용하여 새로운 아이템의 가격을 예측할 수 있습니다.

```bash
# JSON 파일로 예측
uv run python src/predict.py --json-file <파일경로>

# JSON 문자열로 예측
uv run python src/predict.py --json-string '{"name": "...", "item_id": 1004423, ...}'

# 인터랙티브 모드
uv run python src/predict.py --interactive

# 예제 실행 (기본 예제 사용)
uv run python src/predict.py
```

**입력 형식**: payload_json 형태의 딕셔너리 또는 전체 데이터베이스 행 구조

**예제**:
```python
from src.predict import predict_price

payload_json = {
    "name": "앱솔랩스 메이지크라운",
    "item_id": 1004423,
    "star_force": 22,
    "potential_grade": 4,
    "additional_grade": 4,
    "payload_json": {
        "detail_json": "...",  # JSON 문자열 또는 딕셔너리
        "summary_json": "...",
        # ... 기타 필드
    }
}

result = predict_price(payload_json)
print(f"예측 가격: {result['predicted_price_formatted']} 메소")
```

## 데이터 구조

### 입력 데이터 (auction_history 테이블)

주요 컬럼:
- `trade_sn`: 거래 일련번호
- `name`: 아이템 이름
- `item_id`: 아이템 ID
- `price`: 가격 (타겟 변수)
- `star_force`: 스타포스
- `potential_grade`: 잠재능력 등급
- `additional_grade`: 추가옵션 등급
- `payload_json`: 중첩된 JSON 데이터 (스탯, 옵션 등)
- `created_at`: 생성 시간
- `trade_date`: 거래 날짜

### 전처리 후 특징

전처리 과정에서 다음 특징들이 생성됩니다:

1. **기본 특징**: `item_id`, `name`, `star_force`, `potential_grade`, `additional_grade`, `count`
2. **JSON에서 추출된 특징**:
   - 기본 스탯: `base_STR`, `base_DEX`, `base_INT`, `base_LUK`
   - 스크롤 스탯: `scroll_STR_sum`, `scroll_STR_max`, 등
   - 능력치: `MHP_sum`, `MAD_sum`, `PAD_sum`, 등
   - 잠재능력/추가옵션 특징: `potential_options_count`, `additional_has_skill_cooldown`, 등
3. **시간 특징**: `year`, `month`, `day_of_week`, `hour`, 등

## 모델

현재 지원하는 모델:
- **Random Forest Regressor**: 기본 모델
- **Gradient Boosting Regressor**: 대안 모델

모델은 `models/` 디렉토리에 저장됩니다:
- `price_prediction_model.joblib`: 훈련된 모델
- `scaler.joblib`: 특징 스케일러
- `label_encoders.joblib`: 범주형 변수 인코더
- `feature_importance.json`: 특징 중요도
- `metrics.json`: 모델 성능 지표

## 평가 지표

모델은 다음 지표로 평가됩니다:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-squared)
- **MAPE** (Mean Absolute Percentage Error)

## 개발 가이드

### 코드 스타일

- Python 3.11+ 사용
- 타입 힌트 사용
- 함수 및 클래스에 docstring 작성

### 테스트

```bash
# 작은 샘플로 테스트
uv run python main.py --limit 1000
```

## 라이선스

이 프로젝트는 개인 프로젝트입니다.

## 참고사항

- 대용량 데이터셋의 경우 메모리 사용량을 고려하여 배치 처리나 샘플링을 사용하세요
- 데이터베이스 쿼리는 인덱스를 활용하여 최적화할 수 있습니다
- 모델 성능 향상을 위해 하이퍼파라미터 튜닝을 고려하세요

