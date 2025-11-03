"""Compare different model performances"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path


def compare_models(models_dir='models', output_path='MODEL_COMPARISON.md'):
    """Compare performance of different models"""
    
    models_dir = Path(models_dir)
    comparison = {}
    
    # Load metrics for each model
    for metrics_file in models_dir.glob('metrics*.json'):
        model_name = metrics_file.stem.replace('metrics_', '') or 'random_forest'
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        comparison[model_name] = metrics
    
    if not comparison:
        print("No model metrics found for comparison")
        return
    
    # Generate comparison report
    report = f"""# 모델 성능 비교 리포트

**생성 일시**: {os.popen('date').read().strip()}

---

## 📊 모델 비교 개요

이 리포트는 다양한 모델의 성능을 비교합니다.

"""
    
    # Comparison table
    report += "## 📈 테스트 세트 성능 비교\n\n"
    report += "| 모델 | R² Score | RMSE | MAE | MAPE |\n"
    report += "|------|----------|------|-----|------|\n"
    
    for model_name, metrics in sorted(comparison.items()):
        r2 = metrics.get('test_r2', 0)
        rmse = metrics.get('test_rmse', 0)
        mae = metrics.get('test_mae', 0)
        mape = metrics.get('test_mape', 0)
        
        report += f"| {model_name} | {r2:.4f} | {rmse:,.0f} | {mae:,.0f} | {mape:.2f}% |\n"
    
    # Find best model
    best_model = max(comparison.items(), key=lambda x: x[1].get('test_r2', 0))
    best_r2 = best_model[1].get('test_r2', 0)
    
    report += f"\n**최고 성능 모델**: {best_model[0]} (R²: {best_r2:.4f})\n\n"
    
    # Detailed comparison
    report += "## 🔍 상세 비교\n\n"
    
    for model_name, metrics in sorted(comparison.items()):
        report += f"### {model_name}\n\n"
        report += "#### Train Set\n"
        report += f"- R²: {metrics.get('train_r2', 0):.4f}\n"
        report += f"- RMSE: {metrics.get('train_rmse', 0):,.0f}\n"
        report += f"- MAE: {metrics.get('train_mae', 0):,.0f}\n\n"
        
        report += "#### Validation Set\n"
        report += f"- R²: {metrics.get('val_r2', 0):.4f}\n"
        report += f"- RMSE: {metrics.get('val_rmse', 0):,.0f}\n"
        report += f"- MAE: {metrics.get('val_mae', 0):,.0f}\n\n"
        
        report += "#### Test Set\n"
        report += f"- R²: {metrics.get('test_r2', 0):.4f}\n"
        report += f"- RMSE: {metrics.get('test_rmse', 0):,.0f}\n"
        report += f"- MAE: {metrics.get('test_mae', 0):,.0f}\n\n"
        
        # Overfitting analysis
        train_r2 = metrics.get('train_r2', 0)
        test_r2 = metrics.get('test_r2', 0)
        overfit_diff = train_r2 - test_r2
        report += f"**과적합 분석**: Train R² - Test R² = {overfit_diff:.4f}\n"
        if overfit_diff < 0.1:
            report += "- ✅ 과적합이 적습니다\n\n"
        elif overfit_diff < 0.2:
            report += "- ⚠️ 약간의 과적합 가능성이 있습니다\n\n"
        else:
            report += "- ❌ 과적합이 심합니다\n\n"
    
    # Improvement analysis
    if 'random_forest' in comparison and len(comparison) > 1:
        report += "## 📈 개선 분석\n\n"
        rf_metrics = comparison['random_forest']
        rf_r2 = rf_metrics.get('test_r2', 0)
        
        for model_name, metrics in sorted(comparison.items()):
            if model_name == 'random_forest':
                continue
            
            model_r2 = metrics.get('test_r2', 0)
            improvement = model_r2 - rf_r2
            improvement_pct = (improvement / rf_r2 * 100) if rf_r2 > 0 else 0
            
            report += f"### {model_name} vs Random Forest\n"
            report += f"- R² 개선: {improvement:+.4f} ({improvement_pct:+.2f}%)\n"
            report += f"- RMSE 개선: {rf_metrics.get('test_rmse', 0) - metrics.get('test_rmse', 0):,.0f}\n"
            report += f"- MAE 개선: {rf_metrics.get('test_mae', 0) - metrics.get('test_mae', 0):,.0f}\n\n"
    
    report += "---\n\n"
    report += "**비교 리포트 생성 완료**\n"
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Model comparison report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    compare_models()

