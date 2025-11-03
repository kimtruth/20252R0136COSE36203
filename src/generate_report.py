"""Generate training report from model results"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from pathlib import Path


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_number(num, decimals=2):
    """Format large numbers"""
    if num >= 1e12:
        return f"{num/1e12:.{decimals}f}조"
    elif num >= 1e8:
        return f"{num/1e8:.{decimals}f}억"
    elif num >= 1e4:
        return f"{num/1e4:.{decimals}f}만"
    else:
        return f"{num:,.{decimals}f}"


def generate_report(models_dir='models', output_path='TRAINING_REPORT.md'):
    """Generate training report markdown"""
    
    # Load metrics
    metrics_path = os.path.join(models_dir, 'metrics.json')
    feature_importance_path = os.path.join(models_dir, 'feature_importance.json')
    
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        return
    
    metrics = load_json(metrics_path)
    feature_importance = load_json(feature_importance_path) if os.path.exists(feature_importance_path) else {}
    
    # Generate report
    report = f"""# 메이플스토리 아이템 시세 예측 모델 훈련 리포트

**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 실행 요약

이 리포트는 MySQL 데이터베이스의 모든 거래 데이터를 사용하여 훈련된 아이템 시세 예측 모델의 성능을 정리합니다.

### 모델 정보
- **모델 타입**: Random Forest Regressor
- **데이터셋**: 전체 데이터베이스 데이터 (제한 없음)
- **데이터 분할**: Train 70%, Validation 10%, Test 20%
- **분할 방식**: 시간 순서 기반 분할 (최근 데이터를 테스트 세트로 사용)

---

## 📈 모델 성능 지표

### Train Set (훈련 데이터)

| 지표 | 값 | 설명 |
|------|-----|------|
| **RMSE** | {metrics.get('train_rmse', 0):,.2f} | {format_number(metrics.get('train_rmse', 0))} |
| **MAE** | {metrics.get('train_mae', 0):,.2f} | {format_number(metrics.get('train_mae', 0))} |
| **R² Score** | {metrics.get('train_r2', 0):.4f} | {metrics.get('train_r2', 0)*100:.2f}% 설명력 |
| **MAPE** | {metrics.get('train_mape', 0):.2f}% | 평균 절대 백분율 오차 |

### Validation Set (검증 데이터)

| 지표 | 값 | 설명 |
|------|-----|------|
| **RMSE** | {metrics.get('val_rmse', 0):,.2f} | {format_number(metrics.get('val_rmse', 0))} |
| **MAE** | {metrics.get('val_mae', 0):,.2f} | {format_number(metrics.get('val_mae', 0))} |
| **R² Score** | {metrics.get('val_r2', 0):.4f} | {metrics.get('val_r2', 0)*100:.2f}% 설명력 |
| **MAPE** | {metrics.get('val_mape', 0):.2f}% | 평균 절대 백분율 오차 |

### Test Set (테스트 데이터)

| 지표 | 값 | 설명 |
|------|-----|------|
| **RMSE** | {metrics.get('test_rmse', 0):,.2f} | {format_number(metrics.get('test_rmse', 0))} |
| **MAE** | {metrics.get('test_mae', 0):,.2f} | {format_number(metrics.get('test_mae', 0))} |
| **R² Score** | {metrics.get('test_r2', 0):.4f} | {metrics.get('test_r2', 0)*100:.2f}% 설명력 |
| **MAPE** | {metrics.get('test_mape', 0):.2f}% | 평균 절대 백분율 오차 |

---

## 🔍 성능 분석

### 1. 모델 일반화 성능

- **Train vs Test R² 차이**: {abs(metrics.get('train_r2', 0) - metrics.get('test_r2', 0)):.4f}
  - 차이가 작을수록 과적합이 적습니다
  - 차이가 크면 과적합 가능성이 있습니다

- **Train vs Test RMSE 비율**: {metrics.get('test_rmse', 1) / max(metrics.get('train_rmse', 1), 1):.2f}x
  - 테스트 세트의 RMSE가 훈련 세트보다 {metrics.get('test_rmse', 1) / max(metrics.get('train_rmse', 1), 1):.2f}배 높습니다

### 2. 예측 정확도

- **평균 절대 오차 (MAE)**: {format_number(metrics.get('test_mae', 0))}
  - 평균적으로 예측 가격과 실제 가격의 차이가 {format_number(metrics.get('test_mae', 0))} 정도입니다

- **평균 절대 백분율 오차 (MAPE)**: {metrics.get('test_mape', 0):.2f}%
  - 예측 오차가 평균적으로 {metrics.get('test_mape', 0):.2f}% 수준입니다

### 3. 모델 설명력

- **R² Score (Test)**: {metrics.get('test_r2', 0):.4f} ({metrics.get('test_r2', 0)*100:.2f}%)
  - 모델이 가격 변동의 약 {metrics.get('test_r2', 0)*100:.2f}%를 설명합니다
  - {'매우 우수' if metrics.get('test_r2', 0) > 0.9 else '우수' if metrics.get('test_r2', 0) > 0.8 else '양호' if metrics.get('test_r2', 0) > 0.7 else '보통' if metrics.get('test_r2', 0) > 0.5 else '개선 필요'}한 수준입니다

---

## 🎯 주요 특징 중요도 (Top 20)

모델이 가격 예측에 사용한 주요 특징들의 중요도 순위입니다:

"""
    
    # Add feature importance table
    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        report += "| 순위 | 특징명 | 중요도 | 비율 |\n"
        report += "|------|--------|--------|------|\n"
        
        total_importance = sum(feature_importance.values())
        for i, (feature, importance) in enumerate(sorted_features[:20], 1):
            percentage = (importance / total_importance * 100) if total_importance > 0 else 0
            report += f"| {i} | `{feature}` | {importance:.6f} | {percentage:.2f}% |\n"
    else:
        report += "특징 중요도 정보를 불러올 수 없습니다.\n"
    
    report += f"""

---

## 💡 주요 특징 분석

"""
    
    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:10]
        
        report += "### 가장 중요한 특징 Top 10\n\n"
        for i, (feature, importance) in enumerate(top_features, 1):
            report += f"{i}. **{feature}**\n"
            report += f"   - 중요도: {importance:.6f}\n"
            report += f"   - 특징 설명: {get_feature_description(feature)}\n\n"
    
    report += f"""---

## 📝 결론 및 권장사항

### 모델 성능 요약

1. **전반적인 성능**: {'매우 우수' if metrics.get('test_r2', 0) > 0.9 else '우수' if metrics.get('test_r2', 0) > 0.8 else '양호' if metrics.get('test_r2', 0) > 0.7 else '보통'}
   - R² Score가 {metrics.get('test_r2', 0):.4f}로 가격 변동의 상당 부분을 잘 설명합니다

2. **예측 정확도**: {'높음' if metrics.get('test_mape', 100) < 10 else '보통' if metrics.get('test_mape', 100) < 20 else '낮음'}
   - MAPE가 {metrics.get('test_mape', 0):.2f}%로 {'매우' if metrics.get('test_mape', 0) < 5 else '적절한' if metrics.get('test_mape', 0) < 10 else '개선이 필요한'} 수준입니다

3. **과적합 여부**: {'과적합이 적음' if abs(metrics.get('train_r2', 0) - metrics.get('test_r2', 0)) < 0.1 else '과적합 가능성 있음'}
   - Train R²와 Test R²의 차이가 {abs(metrics.get('train_r2', 0) - metrics.get('test_r2', 0)):.4f}입니다

### 개선 권장사항

"""
    
    # Add recommendations based on metrics
    recommendations = []
    
    if abs(metrics.get('train_r2', 0) - metrics.get('test_r2', 0)) > 0.15:
        recommendations.append("- **과적합 완화**: Train/Test 성능 차이가 큽니다. 정규화 강화, 모델 복잡도 감소, 또는 더 많은 데이터 수집 고려")
    
    if metrics.get('test_r2', 0) < 0.7:
        recommendations.append("- **모델 성능 개선**: 더 많은 특징 엔지니어링, 다른 모델 시도 (Gradient Boosting, XGBoost 등), 하이퍼파라미터 튜닝 고려")
    
    if metrics.get('test_mape', 100) > 20:
        recommendations.append("- **예측 정확도 개선**: 가격 범위가 넓어서 오차가 큽니다. 로그 변환, 가격 범위별 모델 분리, 이상치 처리 고려")
    
    if not recommendations:
        recommendations.append("- 모델 성능이 우수합니다. 현재 설정을 유지하거나 추가 데이터로 재훈련을 고려하세요")
        recommendations.append("- 다양한 모델 타입을 비교하여 최적 모델 선택 고려")
    
    for rec in recommendations:
        report += f"{rec}\n"
    
    report += f"""

---

## 📁 저장된 파일

모델 및 전처리 파이프라인은 다음 위치에 저장되었습니다:

- `models/price_prediction_model.joblib`: 훈련된 모델
- `models/scaler.joblib`: 특징 스케일러
- `models/label_encoders.joblib`: 범주형 변수 인코더
- `models/feature_importance.json`: 특징 중요도 (JSON)
- `models/metrics.json`: 성능 지표 (JSON)

---

**보고서 생성 완료**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Training report saved to: {output_path}")
    return output_path


def get_feature_description(feature_name):
    """Get human-readable description of feature"""
    descriptions = {
        'price_per_unit': '단위당 가격 (타겟과 유사하여 제외됨)',
        'detail_PAD_max': '물리 공격력 최대값',
        'detail_MAD_sum': '마법 공격력 합계',
        'detail_scroll_STR_sum': 'STR 스크롤 합계',
        'detail_MAD_max': '마법 공격력 최대값',
        'detail_PAD_sum': '물리 공격력 합계',
        'name': '아이템 이름',
        'payload_item_id': '아이템 ID (payload에서)',
        'item_id': '아이템 ID',
        'star_force': '스타포스',
        'potential_grade': '잠재능력 등급',
        'additional_grade': '추가옵션 등급',
        'detail_scroll_count': '스크롤 사용 횟수',
        'potential_options_count': '잠재능력 옵션 개수',
        'additional_options_count': '추가옵션 개수',
        'year': '년도',
        'month': '월',
        'day_of_week': '요일',
        'hour': '시간',
    }
    
    # Try to match partial names
    for key, desc in descriptions.items():
        if key in feature_name.lower():
            return desc
    
    # Default descriptions based on prefixes
    if feature_name.startswith('detail_base_'):
        stat = feature_name.replace('detail_base_', '')
        return f'기본 {stat} 스탯'
    elif feature_name.startswith('detail_scroll_'):
        stat = feature_name.replace('detail_scroll_', '').replace('_sum', '').replace('_max', '')
        return f'{stat} 스크롤 증가량'
    elif feature_name.startswith('detail_'):
        stat = feature_name.replace('detail_', '').replace('_sum', '').replace('_max', '')
        return f'{stat} 관련 능력치'
    elif feature_name.startswith('potential_'):
        return '잠재능력 관련 특징'
    elif feature_name.startswith('additional_'):
        return '추가옵션 관련 특징'
    elif 'year' in feature_name or 'month' in feature_name or 'day' in feature_name or 'hour' in feature_name:
        return '시간 관련 특징'
    else:
        return '기타 특징'


if __name__ == "__main__":
    generate_report()

