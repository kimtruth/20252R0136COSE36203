"""Improved model training with hyperparameter tuning and MAE optimization"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from src.preprocess import load_and_preprocess_data
from src.train import train_price_prediction_model
from src.tune_hyperparameters import tune_lightgbm_hyperparameters
from src.ensemble import train_ensemble_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.train import prepare_features, encode_categorical_features, scale_features


def run_improved_training(preprocessed_data_path: str = None,
                         run_tuning: bool = True,
                         tune_iterations: int = 30,
                         train_mae_model: bool = True,
                         train_ensemble: bool = True):
    """
    Run improved training pipeline with:
    1. Hyperparameter tuning
    2. MAE-optimized model
    3. Ensemble model
    
    Args:
        preprocessed_data_path: Path to preprocessed data (if None, will load from DB)
        run_tuning: Whether to run hyperparameter tuning
        tune_iterations: Number of iterations for RandomizedSearchCV
        train_mae_model: Whether to train MAE-optimized model
        train_ensemble: Whether to train ensemble model
    """
    print("=" * 80)
    print("IMPROVED MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # Load data
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    
    if preprocessed_data_path and os.path.exists(preprocessed_data_path):
        print(f"Loading preprocessed data from: {preprocessed_data_path}")
        df = pd.read_parquet(preprocessed_data_path)
        print(f"Loaded {len(df):,} rows")
    else:
        default_path = 'data/processed/preprocessed_data.parquet'
        if os.path.exists(default_path):
            print(f"Loading preprocessed data from: {default_path}")
            df = pd.read_parquet(default_path)
            print(f"Loaded {len(df):,} rows")
        else:
            print("No preprocessed data found. Loading from database...")
            df = load_and_preprocess_data(limit=None, save_path=default_path)
            print(f"Loaded and preprocessed {len(df):,} rows")
    
    print(f"Data shape: {df.shape}")
    print(f"Target (price) statistics:")
    print(df['price'].describe())
    
    # Prepare features
    print("\n" + "=" * 80)
    print("STEP 2: PREPARING FEATURES")
    print("=" * 80)
    
    X, y = prepare_features(df, target_col='price')
    print(f"Features shape: {X.shape}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1/(1-0.2), random_state=42
    )
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Encode and scale
    print("\nEncoding and scaling features...")
    X_train_encoded, X_val_encoded, X_test_encoded, label_encoders = encode_categorical_features(
        X_train, X_val, X_test
    )
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_encoded, X_val_encoded, X_test_encoded
    )
    
    results = {}
    
    # Step 3: Hyperparameter tuning
    if run_tuning:
        print("\n" + "=" * 80)
        print("STEP 3: HYPERPARAMETER TUNING")
        print("=" * 80)
        
        tuning_results = tune_lightgbm_hyperparameters(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            n_iter=tune_iterations,
            cv=3,
            scoring='r2'
        )
        
        # Save tuning results
        os.makedirs('models', exist_ok=True)
        with open('models/hyperparameter_tuning_results.json', 'w') as f:
            json.dump({
                'best_params': tuning_results['best_params'],
                'best_score': tuning_results['best_score'],
                'validation_metrics': tuning_results['validation_metrics']
            }, f, indent=2)
        
        print(f"\n✓ Tuning results saved to: models/hyperparameter_tuning_results.json")
        results['tuning'] = tuning_results
        best_params = tuning_results['best_params']
    else:
        # Load existing tuning results
        tuning_path = 'models/hyperparameter_tuning_results.json'
        if os.path.exists(tuning_path):
            print(f"\nLoading existing tuning results from: {tuning_path}")
            with open(tuning_path, 'r') as f:
                tuning_data = json.load(f)
                best_params = tuning_data.get('best_params', {})
                print(f"Using parameters: {list(best_params.keys())}")
        else:
            print("\nNo tuning results found. Using default parameters.")
            best_params = {}
    
    # Step 4: Train RMSE-optimized model with tuned hyperparameters
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING RMSE-OPTIMIZED MODEL (with tuned hyperparameters)")
    print("=" * 80)
    
    rmse_results = train_price_prediction_model(
        df,
        target_col='price',
        model_type='lightgbm',
        test_size=0.2,
        val_size=0.1,
        save_dir='models',
        hyperparameters=best_params,
        objective='rmse'
    )
    
    results['rmse_optimized'] = rmse_results
    
    # Step 5: Train MAE-optimized model
    if train_mae_model:
        print("\n" + "=" * 80)
        print("STEP 5: TRAINING MAE-OPTIMIZED MODEL")
        print("=" * 80)
        
        mae_results = train_price_prediction_model(
            df,
            target_col='price',
            model_type='lightgbm',
            test_size=0.2,
            val_size=0.1,
            save_dir='models',
            hyperparameters=best_params,
            objective='mae'
        )
        
        # Save with MAE suffix
        import joblib
        model_path = 'models/price_prediction_model_lightgbm_mae.joblib'
        joblib.dump(mae_results['model'], model_path)
        print(f"✓ MAE-optimized model saved to: {model_path}")
        
        results['mae_optimized'] = mae_results
    
    # Step 6: Train ensemble model
    if train_ensemble:
        print("\n" + "=" * 80)
        print("STEP 6: TRAINING ENSEMBLE MODEL")
        print("=" * 80)
        
        # Use tuned hyperparameters for LightGBM in ensemble
        lgb_params = best_params.copy() if best_params else {}
        lgb_params.setdefault('n_estimators', 100)
        lgb_params.setdefault('max_depth', -1)
        lgb_params.setdefault('learning_rate', 0.1)
        lgb_params.setdefault('num_leaves', 31)
        lgb_params.setdefault('feature_fraction', 0.9)
        lgb_params.setdefault('bagging_fraction', 0.8)
        lgb_params.setdefault('min_child_samples', 20)
        lgb_params.setdefault('random_state', 42)
        lgb_params.setdefault('n_jobs', -1)
        lgb_params.setdefault('verbose', -1)
        
        ensemble_results = train_ensemble_model(
            df,
            target_col='price',
            test_size=0.2,
            val_size=0.1,
            save_dir='models',
            rf_weight=0.3,
            lgb_weight=0.7,
            rf_params={'n_estimators': 100, 'max_depth': None, 'random_state': 42, 'n_jobs': -1},
            lgb_params=lgb_params
        )
        
        results['ensemble'] = ensemble_results
    
    # Step 7: Summary
    print("\n" + "=" * 80)
    print("STEP 7: PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print("\nRMSE-Optimized Model (with tuned hyperparameters):")
    print(f"  Test R²: {results['rmse_optimized']['metrics']['test_r2']:.4f}")
    print(f"  Test RMSE: {results['rmse_optimized']['metrics']['test_rmse']:,.2f}")
    print(f"  Test MAE: {results['rmse_optimized']['metrics']['test_mae']:,.2f}")
    
    if train_mae_model:
        print("\nMAE-Optimized Model:")
        print(f"  Test R²: {results['mae_optimized']['metrics']['test_r2']:.4f}")
        print(f"  Test RMSE: {results['mae_optimized']['metrics']['test_rmse']:,.2f}")
        print(f"  Test MAE: {results['mae_optimized']['metrics']['test_mae']:,.2f}")
    
    if train_ensemble:
        print("\nEnsemble Model:")
        print(f"  Test R²: {results['ensemble']['metrics']['test_r2']:.4f}")
        print(f"  Test RMSE: {results['ensemble']['metrics']['test_rmse']:,.2f}")
        print(f"  Test MAE: {results['ensemble']['metrics']['test_mae']:,.2f}")
    
    # Compare with baseline
    print("\n" + "=" * 80)
    print("IMPROVEMENT vs BASELINE")
    print("=" * 80)
    
    # Load baseline metrics
    baseline_path = 'models/metrics_lightgbm.json'
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        baseline_r2 = baseline.get('test_r2', 0.8009)
        baseline_rmse = baseline.get('test_rmse', 3928335387)
        baseline_mae = baseline.get('test_mae', 655342117)
        
        print(f"\nBaseline (LightGBM default):")
        print(f"  R²: {baseline_r2:.4f}")
        print(f"  RMSE: {baseline_rmse:,.2f}")
        print(f"  MAE: {baseline_mae:,.2f}")
        
        print(f"\nRMSE-Optimized Model:")
        r2_improvement = results['rmse_optimized']['metrics']['test_r2'] - baseline_r2
        rmse_improvement = (baseline_rmse - results['rmse_optimized']['metrics']['test_rmse']) / baseline_rmse * 100
        mae_change = (results['rmse_optimized']['metrics']['test_mae'] - baseline_mae) / baseline_mae * 100
        
        print(f"  R²: {results['rmse_optimized']['metrics']['test_r2']:.4f} ({r2_improvement:+.4f})")
        print(f"  RMSE: {results['rmse_optimized']['metrics']['test_rmse']:,.2f} ({rmse_improvement:+.2f}%)")
        print(f"  MAE: {results['rmse_optimized']['metrics']['test_mae']:,.2f} ({mae_change:+.2f}%)")
        
        if train_mae_model:
            print(f"\nMAE-Optimized Model:")
            r2_improvement = results['mae_optimized']['metrics']['test_r2'] - baseline_r2
            rmse_improvement = (baseline_rmse - results['mae_optimized']['metrics']['test_rmse']) / baseline_rmse * 100
            mae_improvement = (baseline_mae - results['mae_optimized']['metrics']['test_mae']) / baseline_mae * 100
            
            print(f"  R²: {results['mae_optimized']['metrics']['test_r2']:.4f} ({r2_improvement:+.4f})")
            print(f"  RMSE: {results['mae_optimized']['metrics']['test_rmse']:,.2f} ({rmse_improvement:+.2f}%)")
            print(f"  MAE: {results['mae_optimized']['metrics']['test_mae']:,.2f} ({mae_improvement:+.2f}%)")
        
        if train_ensemble:
            print(f"\nEnsemble Model:")
            r2_improvement = results['ensemble']['metrics']['test_r2'] - baseline_r2
            rmse_improvement = (baseline_rmse - results['ensemble']['metrics']['test_rmse']) / baseline_rmse * 100
            mae_improvement = (baseline_mae - results['ensemble']['metrics']['test_mae']) / baseline_mae * 100
            
            print(f"  R²: {results['ensemble']['metrics']['test_r2']:.4f} ({r2_improvement:+.4f})")
            print(f"  RMSE: {results['ensemble']['metrics']['test_rmse']:,.2f} ({rmse_improvement:+.2f}%)")
            print(f"  MAE: {results['ensemble']['metrics']['test_mae']:,.2f} ({mae_improvement:+.2f}%)")
    
    print("\n" + "=" * 80)
    print("IMPROVED TRAINING COMPLETE!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved model training with hyperparameter tuning')
    parser.add_argument('--data', type=str, default=None, help='Path to preprocessed data')
    parser.add_argument('--no-tune', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--tune-iterations', type=int, default=30, help='Number of tuning iterations')
    parser.add_argument('--no-mae', action='store_true', help='Skip MAE-optimized model')
    parser.add_argument('--no-ensemble', action='store_true', help='Skip ensemble model')
    
    args = parser.parse_args()
    
    results = run_improved_training(
        preprocessed_data_path=args.data,
        run_tuning=not args.no_tune,
        tune_iterations=args.tune_iterations,
        train_mae_model=not args.no_mae,
        train_ensemble=not args.no_ensemble
    )


