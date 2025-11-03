"""Hyperparameter tuning for LightGBM model"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from typing import Dict, Any
import json

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    raise ImportError("LightGBM is required for hyperparameter tuning. Install with: pip install lightgbm")


def tune_lightgbm_hyperparameters(X_train: pd.DataFrame,
                                  y_train: pd.Series,
                                  X_val: pd.DataFrame,
                                  y_val: pd.Series,
                                  n_iter: int = 50,
                                  cv: int = 3,
                                  scoring: str = 'r2') -> Dict[str, Any]:
    """
    Tune LightGBM hyperparameters using RandomizedSearchCV
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_iter: Number of iterations for randomized search
        cv: Number of cross-validation folds
        scoring: Scoring metric ('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error')
    
    Returns:
        Dictionary with best parameters and results
    """
    print("=" * 80)
    print("LIGHTGBM HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [-1, 5, 7, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'num_leaves': [31, 50, 100, 200],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.7, 0.8, 0.9],
        'bagging_freq': [3, 5, 7],
        'min_child_samples': [10, 20, 30],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5, 1.0],
    }
    
    # Create base model
    base_model = lgb.LGBMRegressor(
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Use RandomizedSearchCV for faster tuning
    print(f"\nRunning RandomizedSearchCV with {n_iter} iterations...")
    print(f"Parameter grid size: {len(param_grid)} parameters")
    
    # Custom scorer for R²
    if scoring == 'r2':
        scorer = make_scorer(r2_score)
    elif scoring == 'neg_mean_squared_error':
        scorer = 'neg_mean_squared_error'
    elif scoring == 'neg_mean_absolute_error':
        scorer = 'neg_mean_absolute_error'
    else:
        scorer = scoring
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Combine train and val for CV
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    
    # Fit
    print("\nFitting RandomizedSearchCV...")
    search.fit(X_combined, y_combined)
    
    # Get best model
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    
    print("\n" + "=" * 80)
    print("BEST PARAMETERS FOUND")
    print("=" * 80)
    for param, value in sorted(best_params.items()):
        print(f"  {param}: {value}")
    print(f"\nBest CV Score ({scoring}): {best_score:.6f}")
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION SET PERFORMANCE")
    print("=" * 80)
    y_val_pred = best_model.predict(X_val)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"RMSE: {val_rmse:,.2f}")
    print(f"MAE: {val_mae:,.2f}")
    print(f"R²: {val_r2:.6f}")
    
    return {
        'best_params': best_params,
        'best_score': float(best_score),
        'best_model': best_model,
        'validation_metrics': {
            'rmse': float(val_rmse),
            'mae': float(val_mae),
            'r2': float(val_r2)
        },
        'cv_results': {
            'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': search.cv_results_['std_test_score'].tolist(),
            'params': search.cv_results_['params']
        }
    }


def tune_with_validation_set(X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_val: pd.DataFrame,
                             y_val: pd.Series,
                             param_grid: Dict = None) -> Dict[str, Any]:
    """
    Manual hyperparameter tuning with validation set
    
    Faster than CV but uses validation set directly
    """
    print("=" * 80)
    print("LIGHTGBM HYPERPARAMETER TUNING (Validation Set)")
    print("=" * 80)
    
    if param_grid is None:
        # Focused parameter grid for faster tuning
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [5, 7, 10, -1],
            'learning_rate': [0.05, 0.1, 0.15],
            'num_leaves': [31, 50, 100],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.7, 0.8, 0.9],
            'min_child_samples': [10, 20, 30],
        }
    
    best_score = -np.inf
    best_params = None
    best_model = None
    results = []
    
    # Generate all combinations
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)
    
    print(f"\nTesting {total_combinations} parameter combinations...")
    print("This may take a while. Progress will be shown below.\n")
    
    count = 0
    for params in itertools.product(*values):
        count += 1
        param_dict = dict(zip(keys, params))
        
        model = lgb.LGBMRegressor(
            **param_dict,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        y_val_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_val_pred)
        
        results.append({
            'params': param_dict,
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'mae': mean_absolute_error(y_val, y_val_pred)
        })
        
        if r2 > best_score:
            best_score = r2
            best_params = param_dict.copy()
            best_model = model
        
        if count % 10 == 0:
            print(f"Progress: {count}/{total_combinations} ({count/total_combinations*100:.1f}%) - Best R²: {best_score:.6f}")
    
    print(f"\nCompleted {count} combinations")
    print("\n" + "=" * 80)
    print("BEST PARAMETERS FOUND")
    print("=" * 80)
    for param, value in sorted(best_params.items()):
        print(f"  {param}: {value}")
    print(f"\nBest Validation R²: {best_score:.6f}")
    
    return {
        'best_params': best_params,
        'best_score': float(best_score),
        'best_model': best_model,
        'all_results': results
    }


if __name__ == "__main__":
    from src.train import prepare_features, encode_categorical_features, scale_features
    
    print("Loading preprocessed data...")
    df = pd.read_parquet('data/processed/preprocessed_data.parquet')
    
    print(f"Data shape: {df.shape}")
    
    # Prepare features
    print("\nPreparing features...")
    X, y = prepare_features(df, target_col='price')
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1/(1-0.2), random_state=42)
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Encode and scale
    print("\nEncoding categorical features...")
    X_train_encoded, X_val_encoded, X_test_encoded, label_encoders = encode_categorical_features(
        X_train, X_val, X_test
    )
    
    print("\nScaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_encoded, X_val_encoded, X_test_encoded
    )
    
    # Run tuning (use RandomizedSearchCV for faster results)
    print("\n" + "=" * 80)
    print("Starting hyperparameter tuning...")
    print("=" * 80)
    
    results = tune_lightgbm_hyperparameters(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        n_iter=30,  # Reduced for faster execution
        cv=3,
        scoring='r2'
    )
    
    # Save results
    os.makedirs('models', exist_ok=True)
    
    with open('models/hyperparameter_tuning_results.json', 'w') as f:
        json.dump({
            'best_params': results['best_params'],
            'best_score': results['best_score'],
            'validation_metrics': results['validation_metrics']
        }, f, indent=2)
    
    print(f"\nResults saved to: models/hyperparameter_tuning_results.json")
    
    # Test on test set
    print("\n" + "=" * 80)
    print("TEST SET PERFORMANCE")
    print("=" * 80)
    y_test_pred = results['best_model'].predict(X_test_scaled)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"RMSE: {test_rmse:,.2f}")
    print(f"MAE: {test_mae:,.2f}")
    print(f"R²: {test_r2:.6f}")
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("=" * 80)

