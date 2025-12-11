"""Ensemble models for MapleStory item price prediction"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")


class EnsembleModel:
    """Ensemble model combining Random Forest and LightGBM"""
    
    def __init__(self, 
                 rf_weight: float = 0.3,
                 lgb_weight: float = 0.7,
                 rf_params: Dict = None,
                 lgb_params: Dict = None):
        """
        Initialize ensemble model
        
        Args:
            rf_weight: Weight for Random Forest predictions
            lgb_weight: Weight for LightGBM predictions (should sum to 1.0 with rf_weight)
            rf_params: Parameters for Random Forest
            lgb_params: Parameters for LightGBM
        """
        self.rf_weight = rf_weight
        self.lgb_weight = lgb_weight
        
        # Normalize weights
        total_weight = rf_weight + lgb_weight
        if total_weight > 0:
            self.rf_weight = rf_weight / total_weight
            self.lgb_weight = lgb_weight / total_weight
        
        # Initialize models
        self.rf_model = None
        self.lgb_model = None
        
        # Default parameters
        if rf_params is None:
            rf_params = {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42,
                'n_jobs': -1
            }
        self.rf_params = rf_params
        
        if lgb_params is None:
            lgb_params = {
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        self.lgb_params = lgb_params
    
    def fit(self, 
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None):
        """Train both models"""
        print("=" * 80)
        print("TRAINING ENSEMBLE MODEL")
        print("=" * 80)
        
        # Train Random Forest
        print("\n1. Training Random Forest model...")
        self.rf_model = RandomForestRegressor(**self.rf_params)
        self.rf_model.fit(X_train, y_train)
        
        # Train LightGBM
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is required for ensemble. Install with: pip install lightgbm")
        
        print("\n2. Training LightGBM model...")
        self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)
        
        if X_val is not None and y_val is not None:
            self.lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
        else:
            self.lgb_model.fit(X_train, y_train)
        
        print("\nEnsemble model training complete!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using weighted average"""
        if self.rf_model is None or self.lgb_model is None:
            raise ValueError("Models not trained. Call fit() first.")
        
        rf_pred = self.rf_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        
        # Weighted average
        ensemble_pred = self.rf_weight * rf_pred + self.lgb_weight * lgb_pred
        
        return ensemble_pred
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance"""
        if self.rf_model is None or self.lgb_model is None:
            return {}
        
        rf_importance = self.rf_model.feature_importances_
        lgb_importance = self.lgb_model.feature_importances_
        
        # Weighted average of importances
        combined_importance = (
            self.rf_weight * rf_importance + 
            self.lgb_weight * lgb_importance
        )
        
        feature_names = self.rf_model.feature_names_in_ if hasattr(self.rf_model, 'feature_names_in_') else \
                       [f'feature_{i}' for i in range(len(combined_importance))]
        
        importance_dict = {}
        for name, importance in zip(feature_names, combined_importance):
            importance_dict[name] = float(importance)
        
        return importance_dict
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble model"""
        y_pred = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / (y + 1))) * 100
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }


def train_ensemble_model(df: pd.DataFrame,
                         target_col: str = 'price',
                         test_size: float = 0.2,
                         val_size: float = 0.1,
                         save_dir: str = 'models',
                         rf_weight: float = 0.3,
                         lgb_weight: float = 0.7,
                         rf_params: Dict = None,
                         lgb_params: Dict = None) -> Dict:
    """Train ensemble model"""
    from src.train import prepare_features, encode_categorical_features, scale_features
    from sklearn.model_selection import train_test_split
    
    print("=" * 80)
    print("ENSEMBLE MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # Prepare features
    print("\n1. Preparing features...")
    X, y = prepare_features(df, target_col)
    print(f"   Features shape: {X.shape}")
    
    # Split data
    print("\n2. Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42
    )
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Encode categorical features
    print("\n3. Encoding categorical features...")
    X_train_encoded, X_test_encoded, X_val_encoded, label_encoders = encode_categorical_features(
        X_train, X_test, X_val
    )
    
    # Scale features
    print("\n4. Scaling features...")
    X_train_scaled, X_test_scaled, X_val_scaled, scaler = scale_features(
        X_train_encoded, X_test_encoded, X_val_encoded
    )
    
    # Train ensemble
    print("\n5. Training ensemble model...")
    ensemble = EnsembleModel(
        rf_weight=rf_weight,
        lgb_weight=lgb_weight,
        rf_params=rf_params,
        lgb_params=lgb_params
    )
    ensemble.fit(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Evaluate
    print("\n6. Evaluating ensemble model...")
    train_metrics = ensemble.evaluate(X_train_scaled, y_train)
    val_metrics = ensemble.evaluate(X_val_scaled, y_val)
    test_metrics = ensemble.evaluate(X_test_scaled, y_test)
    
    # Print metrics
    print("\n" + "=" * 80)
    print("ENSEMBLE MODEL PERFORMANCE")
    print("=" * 80)
    print(f"\nTrain Set:")
    print(f"  RMSE: {train_metrics['rmse']:,.2f}")
    print(f"  MAE: {train_metrics['mae']:,.2f}")
    print(f"  R²: {train_metrics['r2']:.4f}")
    
    print(f"\nValidation Set:")
    print(f"  RMSE: {val_metrics['rmse']:,.2f}")
    print(f"  MAE: {val_metrics['mae']:,.2f}")
    print(f"  R²: {val_metrics['r2']:.4f}")
    
    print(f"\nTest Set:")
    print(f"  RMSE: {test_metrics['rmse']:,.2f}")
    print(f"  MAE: {test_metrics['mae']:,.2f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    # Save model
    print("\n7. Saving ensemble model...")
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'ensemble_model.joblib')
    scaler_path = os.path.join(save_dir, 'scaler_ensemble.joblib')
    encoders_path = os.path.join(save_dir, 'label_encoders_ensemble.joblib')
    feature_importance_path = os.path.join(save_dir, 'feature_importance_ensemble.json')
    metrics_path = os.path.join(save_dir, 'metrics_ensemble.json')
    
    joblib.dump(ensemble, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoders, encoders_path)
    
    feature_importance = ensemble.get_feature_importance()
    import json
    with open(feature_importance_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    all_metrics = {
        **{f'train_{k}': v for k, v in train_metrics.items()},
        **{f'val_{k}': v for k, v in val_metrics.items()},
        **{f'test_{k}': v for k, v in test_metrics.items()},
        'model_type': 'ensemble',
        'rf_weight': rf_weight,
        'lgb_weight': lgb_weight
    }
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"   Ensemble model saved to: {model_path}")
    print(f"   Metrics saved to: {metrics_path}")
    
    # Top features
    if feature_importance:
        print("\n" + "=" * 80)
        print("TOP 10 MOST IMPORTANT FEATURES (ENSEMBLE)")
        print("=" * 80)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:40s} {importance:.4f}")
    
    return {
        'ensemble': ensemble,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_importance': feature_importance,
        'metrics': all_metrics
    }


if __name__ == "__main__":
    print("Loading preprocessed data...")
    df = pd.read_parquet('data/processed/preprocessed_data.parquet')
    
    print(f"Data shape: {df.shape}")
    
    # Train ensemble
    results = train_ensemble_model(
        df,
        target_col='price',
        test_size=0.2,
        val_size=0.1,
        save_dir='models',
        rf_weight=0.3,
        lgb_weight=0.7
    )
    
    print("\n" + "=" * 80)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("=" * 80)


