"""
Improved Model Training Script
==============================
This script trains the improved model with:
1. Enhanced feature extraction (category, level_req, job, parsed potentials)
2. Log transformation of target variable
3. Ensemble model with optimized hyperparameters
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime

from src.preprocess import load_and_preprocess_data, load_from_jsonl, preprocess_data
from src.train import (
    train_price_prediction_model, 
    prepare_features, 
    encode_categorical_features, 
    scale_features,
    evaluate_model,
    train_model
)
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def train_improved_model(
    data_path: str = 'data/raw/raw_data.jsonl',
    save_dir: str = 'models',
    sample_size: int = None,
    use_log_transform: bool = True,
    run_hyperparameter_tuning: bool = False
):
    """
    Train improved model with enhanced features and log transformation.
    
    Args:
        data_path: Path to raw JSONL data file
        save_dir: Directory to save models and results
        sample_size: Number of rows to sample (None for all data)
        use_log_transform: Whether to apply log transform to target
        run_hyperparameter_tuning: Whether to run hyperparameter tuning
    """
    print("=" * 80)
    print("IMPROVED MODEL TRAINING PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # =========================================================================
    # STEP 1: Load and preprocess data with enhanced features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    print(f"\nLoading data from: {data_path}")
    
    # Load raw data from JSONL
    df_raw = load_from_jsonl(data_path, limit=sample_size)
    print(f"Loaded {len(df_raw):,} rows")
    
    # Preprocess with enhanced feature extraction
    print("\nPreprocessing with enhanced feature extraction...")
    df = preprocess_data(df_raw)
    print(f"Preprocessed data shape: {df.shape}")
    print(f"Number of features: {df.shape[1]}")
    
    # Save preprocessed data
    processed_path = 'data/processed/preprocessed_improved.parquet'
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_parquet(processed_path, index=False)
    print(f"Saved preprocessed data to: {processed_path}")
    
    # Print feature summary
    print(f"\nFeature columns ({len(df.columns)}):")
    feature_types = df.dtypes.value_counts()
    for dtype, count in feature_types.items():
        print(f"  {dtype}: {count} columns")
    
    # =========================================================================
    # STEP 2: Prepare features and target
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: PREPARING FEATURES AND TARGET")
    print("=" * 80)
    
    X, y = prepare_features(df, target_col='price')
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nTarget (price) statistics:")
    print(f"  Min: {y.min():,.0f}")
    print(f"  Max: {y.max():,.0f}")
    print(f"  Mean: {y.mean():,.0f}")
    print(f"  Median: {y.median():,.0f}")
    print(f"  Std: {y.std():,.0f}")
    
    # Apply log transform if requested
    if use_log_transform:
        print("\nApplying log1p transform to target...")
        y_original = y.copy()
        y = np.log1p(y)
        print(f"Log-transformed target statistics:")
        print(f"  Min: {y.min():.2f}")
        print(f"  Max: {y.max():.2f}")
        print(f"  Mean: {y.mean():.2f}")
        print(f"  Median: {y.median():.2f}")
    
    # =========================================================================
    # STEP 3: Split data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: SPLITTING DATA")
    print("=" * 80)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42  # 0.125 of 0.8 = 0.1 of total
    )
    
    print(f"Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # =========================================================================
    # STEP 4: Encode and scale features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: ENCODING AND SCALING FEATURES")
    print("=" * 80)
    
    print("Encoding categorical features...")
    X_train_enc, X_test_enc, X_val_enc, label_encoders = encode_categorical_features(
        X_train.copy(), X_test.copy(), X_val.copy()
    )
    
    print("Scaling numerical features...")
    X_train_scaled, X_test_scaled, X_val_scaled, scaler = scale_features(
        X_train_enc, X_test_enc, X_val_enc
    )
    
    print(f"Final feature count: {X_train_scaled.shape[1]}")
    
    # =========================================================================
    # STEP 5: Train LightGBM model
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING LIGHTGBM MODEL")
    print("=" * 80)
    
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is required. Install with: pip install lightgbm")
    
    # Load tuned hyperparameters if available
    tuning_path = os.path.join(save_dir, 'hyperparameter_tuning_results.json')
    if os.path.exists(tuning_path):
        print(f"Loading tuned hyperparameters from: {tuning_path}")
        with open(tuning_path, 'r') as f:
            tuning_data = json.load(f)
            best_params = tuning_data.get('best_params', {})
    else:
        print("Using default hyperparameters")
        best_params = {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.1,
            'num_leaves': 50,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }
    
    print(f"Hyperparameters: {best_params}")
    
    # Create and train model
    model_params = {
        **best_params,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'objective': 'rmse'  # Use RMSE for log-scale target
    }
    
    print("\nTraining LightGBM model...")
    model = lgb.LGBMRegressor(**model_params)
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    print(f"Best iteration: {model.best_iteration_}")
    
    # =========================================================================
    # STEP 6: Evaluate model
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: EVALUATING MODEL")
    print("=" * 80)
    
    train_metrics = evaluate_model(model, X_train_scaled, y_train, 'train', use_log_transform)
    val_metrics = evaluate_model(model, X_val_scaled, y_val, 'val', use_log_transform)
    test_metrics = evaluate_model(model, X_test_scaled, y_test, 'test', use_log_transform)
    
    print("\n" + "-" * 60)
    print("PERFORMANCE METRICS (on original price scale)")
    print("-" * 60)
    
    print(f"\nTrain Set:")
    print(f"  R²: {train_metrics['train_r2']:.4f}")
    print(f"  RMSE: {train_metrics['train_rmse']:,.0f}")
    print(f"  MAE: {train_metrics['train_mae']:,.0f}")
    print(f"  MAPE: {train_metrics['train_mape']:.2f}%")
    if 'train_log_r2' in train_metrics:
        print(f"  Log-scale R²: {train_metrics['train_log_r2']:.4f}")
    
    print(f"\nValidation Set:")
    print(f"  R²: {val_metrics['val_r2']:.4f}")
    print(f"  RMSE: {val_metrics['val_rmse']:,.0f}")
    print(f"  MAE: {val_metrics['val_mae']:,.0f}")
    print(f"  MAPE: {val_metrics['val_mape']:.2f}%")
    if 'val_log_r2' in val_metrics:
        print(f"  Log-scale R²: {val_metrics['val_log_r2']:.4f}")
    
    print(f"\nTest Set:")
    print(f"  R²: {test_metrics['test_r2']:.4f}")
    print(f"  RMSE: {test_metrics['test_rmse']:,.0f}")
    print(f"  MAE: {test_metrics['test_mae']:,.0f}")
    print(f"  MAPE: {test_metrics['test_mape']:.2f}%")
    if 'test_log_r2' in test_metrics:
        print(f"  Log-scale R²: {test_metrics['test_log_r2']:.4f}")
    
    # =========================================================================
    # STEP 7: Save model and artifacts
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: SAVING MODEL AND ARTIFACTS")
    print("=" * 80)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'improved_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(save_dir, 'scaler_improved.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save label encoders
    encoders_path = os.path.join(save_dir, 'label_encoders_improved.joblib')
    joblib.dump(label_encoders, encoders_path)
    print(f"Label encoders saved to: {encoders_path}")
    
    # Save feature importance
    feature_importance = dict(zip(
        X_train_scaled.columns,
        model.feature_importances_.tolist()
    ))
    feature_importance_sorted = dict(sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    importance_path = os.path.join(save_dir, 'feature_importance_improved.json')
    with open(importance_path, 'w') as f:
        json.dump(feature_importance_sorted, f, indent=2)
    print(f"Feature importance saved to: {importance_path}")
    
    # Save metrics
    all_metrics = {
        **train_metrics,
        **val_metrics,
        **test_metrics,
        'model_type': 'lightgbm_improved',
        'use_log_transform': use_log_transform,
        'n_features': X_train_scaled.shape[1],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'hyperparameters': best_params,
        'training_date': datetime.now().isoformat()
    }
    
    metrics_path = os.path.join(save_dir, 'metrics_improved.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # =========================================================================
    # STEP 8: Print top features
    # =========================================================================
    print("\n" + "=" * 80)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 80)
    
    for i, (feature, importance) in enumerate(list(feature_importance_sorted.items())[:15], 1):
        print(f"{i:2d}. {feature:45s} {importance:.4f}")
    
    # =========================================================================
    # STEP 9: Compare with baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)
    
    baseline_path = os.path.join(save_dir, 'metrics_ensemble.json')
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        print("\n                    Baseline (Ensemble)    Improved Model    Change")
        print("-" * 70)
        
        baseline_r2 = baseline.get('test_r2', 0)
        improved_r2 = test_metrics['test_r2']
        print(f"Test R²:            {baseline_r2:.4f}               {improved_r2:.4f}           {improved_r2 - baseline_r2:+.4f}")
        
        baseline_rmse = baseline.get('test_rmse', 0)
        improved_rmse = test_metrics['test_rmse']
        rmse_pct = (baseline_rmse - improved_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0
        print(f"Test RMSE:          {baseline_rmse:,.0f}         {improved_rmse:,.0f}      {rmse_pct:+.1f}%")
        
        baseline_mae = baseline.get('test_mae', 0)
        improved_mae = test_metrics['test_mae']
        mae_pct = (baseline_mae - improved_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
        print(f"Test MAE:           {baseline_mae:,.0f}         {improved_mae:,.0f}      {mae_pct:+.1f}%")
        
        baseline_mape = baseline.get('test_mape', 0)
        improved_mape = test_metrics['test_mape']
        print(f"Test MAPE:          {baseline_mape:,.1f}%            {improved_mape:.1f}%         {improved_mape - baseline_mape:+.1f}%")
    
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_importance': feature_importance_sorted,
        'metrics': all_metrics
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved model')
    parser.add_argument('--data', type=str, default='data/raw/raw_data.jsonl',
                        help='Path to raw data file')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of samples to use (None for all)')
    parser.add_argument('--no-log-transform', action='store_true',
                        help='Disable log transformation')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save models')
    
    args = parser.parse_args()
    
    results = train_improved_model(
        data_path=args.data,
        save_dir=args.save_dir,
        sample_size=args.sample,
        use_log_transform=not args.no_log_transform
    )

