"""Machine learning model training for MapleStory item price prediction"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, List
import json

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")


def prepare_features(df: pd.DataFrame, target_col: str = 'price') -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for model training"""
    # Separate target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Remove non-feature columns
    columns_to_drop = [
        'trade_sn', 'payload_json', 'created_at', 'trade_date', 
        'register_date', 'end_date', 'price_per_unit',  # price_per_unit is too similar to target
        'payload_trade_sn', 'payload_item_id'  # IDs should not be used as features
    ]
    for col in columns_to_drop:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    return X, y


def encode_categorical_features(X_train: pd.DataFrame, 
                               X_test: pd.DataFrame = None,
                               X_val: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Encode categorical features"""
    label_encoders = {}
    
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        label_encoders[col] = le
        
        if X_test is not None:
            # Handle unseen categories
            test_values = X_test[col].astype(str)
            unseen_mask = ~test_values.isin(le.classes_)
            if unseen_mask.any():
                # Replace unseen with most common class
                test_values[unseen_mask] = le.classes_[0]
            X_test[col] = le.transform(test_values)
        
        if X_val is not None:
            val_values = X_val[col].astype(str)
            unseen_mask = ~val_values.isin(le.classes_)
            if unseen_mask.any():
                val_values[unseen_mask] = le.classes_[0]
            X_val[col] = le.transform(val_values)
    
    return X_train, X_test, X_val, label_encoders


def scale_features(X_train: pd.DataFrame, 
                   X_test: pd.DataFrame = None,
                   X_val: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale numerical features"""
    scaler = StandardScaler()
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = X_test.copy()
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = X_val.copy()
        X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    
    return X_train_scaled, X_test_scaled, X_val_scaled, scaler


def train_model(X_train: pd.DataFrame, 
                y_train: pd.Series,
                model_type: str = 'random_forest',
                X_val: pd.DataFrame = None,
                y_val: pd.Series = None,
                **kwargs) -> Tuple[object, Dict]:
    """Train a regression model"""
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            random_state=42,
            n_jobs=-1
        )
        print(f"Training {model_type} model...")
        model.fit(X_train, y_train)
        
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 3),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=42
        )
        print(f"Training {model_type} model...")
        model.fit(X_train, y_train)
        
    elif model_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is not installed. Install with: pip install lightgbm")
        
        print(f"Training {model_type} model...")
        
        # Check if hyperparameters are provided (from tuning)
        # If not, use defaults or load from saved tuning results
        objective = kwargs.get('objective', 'rmse')  # 'rmse', 'mae', 'quantile'
        alpha = kwargs.get('alpha', 0.5)  # For quantile regression
        
        # Build model with hyperparameters
        model_params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', -1),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'num_leaves': kwargs.get('num_leaves', 31),
            'feature_fraction': kwargs.get('feature_fraction', 0.9),
            'bagging_fraction': kwargs.get('bagging_fraction', 0.8),
            'bagging_freq': kwargs.get('bagging_freq', 5),
            'min_child_samples': kwargs.get('min_child_samples', 20),
            'reg_alpha': kwargs.get('reg_alpha', 0.0),
            'reg_lambda': kwargs.get('reg_lambda', 0.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Set objective
        if objective == 'mae':
            model_params['objective'] = 'mae'
        elif objective == 'quantile':
            model_params['objective'] = 'quantile'
            model_params['alpha'] = alpha
        else:  # default to rmse
            model_params['objective'] = 'rmse'
        
        model = lgb.LGBMRegressor(**model_params)
        
        # Use validation set for early stopping if available
        early_stopping_rounds = kwargs.get('early_stopping_rounds', 10)
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            )
        else:
            model.fit(X_train, y_train)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: random_forest, gradient_boosting, lightgbm")
    
    # Get feature importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        for idx, importance in enumerate(model.feature_importances_):
            feature_importance[X_train.columns[idx]] = float(importance)
    
    return model, feature_importance


def evaluate_model(model: object, 
                   X: pd.DataFrame, 
                   y: pd.Series,
                   set_name: str = 'test',
                   use_log_transform: bool = False) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X: Features
        y: Target (in original scale or log scale depending on use_log_transform)
        set_name: Name of the dataset (train, val, test)
        use_log_transform: If True, y is in log scale and predictions need inverse transform
    """
    y_pred = model.predict(X)
    
    # If log transform was used, we need to inverse transform for proper metrics
    if use_log_transform:
        # Inverse transform: expm1(log1p(x)) = x
        y_original = np.expm1(y)  # Convert log scale back to original
        y_pred_original = np.expm1(y_pred)  # Convert predictions back to original
        
        # Clip negative predictions (can happen due to model extrapolation)
        y_pred_original = np.clip(y_pred_original, 0, None)
        
        # Calculate metrics on original scale
        mse = mean_squared_error(y_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_original, y_pred_original)
        r2 = r2_score(y_original, y_pred_original)
        
        # MAPE on original scale (much more meaningful now)
        mape = np.mean(np.abs((y_original - y_pred_original) / (y_original + 1))) * 100
        
        # Also calculate log-scale R² for comparison
        log_r2 = r2_score(y, y_pred)
        
        metrics = {
            f'{set_name}_mse': float(mse),
            f'{set_name}_rmse': float(rmse),
            f'{set_name}_mae': float(mae),
            f'{set_name}_r2': float(r2),
            f'{set_name}_mape': float(mape),
            f'{set_name}_log_r2': float(log_r2)
        }
    else:
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y - y_pred) / (y + 1))) * 100
        
        metrics = {
            f'{set_name}_mse': float(mse),
            f'{set_name}_rmse': float(rmse),
            f'{set_name}_mae': float(mae),
            f'{set_name}_r2': float(r2),
            f'{set_name}_mape': float(mape)
        }
    
    return metrics


def split_data_by_time(df: pd.DataFrame, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       time_col: str = 'created_at') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data by time order (chronological split)
    Train: oldest data
    Validation: middle period
    Test: most recent data
    """
    # Sort by time
    if time_col in df.columns:
        df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
        X_sorted = X.iloc[df_sorted.index].reset_index(drop=True)
        y_sorted = y.iloc[df_sorted.index].reset_index(drop=True)
        
        # Calculate split indices
        total_size = len(df_sorted)
        test_start_idx = int(total_size * (1 - test_size))
        val_start_idx = int(total_size * (1 - test_size - val_size))
        
        # Split chronologically
        X_train = X_sorted.iloc[:val_start_idx]
        X_val = X_sorted.iloc[val_start_idx:test_start_idx]
        X_test = X_sorted.iloc[test_start_idx:]
        
        y_train = y_sorted.iloc[:val_start_idx]
        y_val = y_sorted.iloc[val_start_idx:test_start_idx]
        y_test = y_sorted.iloc[test_start_idx:]
        
        print(f"   Time-based split:")
        if time_col in df.columns:
            print(f"   Train period: {df_sorted[time_col].iloc[0]} ~ {df_sorted[time_col].iloc[val_start_idx-1]}")
            print(f"   Validation period: {df_sorted[time_col].iloc[val_start_idx]} ~ {df_sorted[time_col].iloc[test_start_idx-1]}")
            print(f"   Test period: {df_sorted[time_col].iloc[test_start_idx]} ~ {df_sorted[time_col].iloc[-1]}")
        
    else:
        # Fallback to random split if time column not available
        print(f"   Warning: Time column '{time_col}' not found. Using random split.")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_price_prediction_model(df: pd.DataFrame,
                                target_col: str = 'price',
                                model_type: str = 'random_forest',
                                test_size: float = 0.2,
                                val_size: float = 0.1,
                                save_dir: str = 'models',
                                use_time_split: bool = False,
                                hyperparameters: Dict = None,
                                objective: str = 'rmse',
                                use_log_transform: bool = False) -> Dict:
    """
    Complete training pipeline.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        model_type: Model type ('random_forest', 'lightgbm', etc.)
        test_size: Test set size ratio
        val_size: Validation set size ratio
        save_dir: Directory to save models
        use_time_split: Whether to use time-based split
        hyperparameters: Model hyperparameters
        objective: Loss objective ('rmse', 'mae')
        use_log_transform: If True, apply log1p transform to target variable
    """
    
    print("=" * 80)
    print("TRAINING PRICE PREDICTION MODEL")
    if use_log_transform:
        print("(Using LOG TRANSFORM on target variable)")
    print("=" * 80)
    
    # Prepare features
    print("\n1. Preparing features...")
    X, y = prepare_features(df, target_col)
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # Apply log transformation if requested
    if use_log_transform:
        print("\n   Applying log1p transform to target variable...")
        print(f"   Original price range: {y.min():,.0f} ~ {y.max():,.0f}")
        y = np.log1p(y)  # log(1 + x) to handle zeros
        print(f"   Log-transformed range: {y.min():.2f} ~ {y.max():.2f}")
    
    # Split data using random sampling
    # This ensures time patterns are learned across all time periods
    # Time features (year, month, day_of_week, etc.) are preserved in features
    print("\n2. Splitting data (random sampling)...")
    print("   Note: Using random split to ensure time patterns are learned across all periods")
    print("   Time features (year, month, day_of_week, etc.) are preserved in the feature set")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42
    )
    
    print(f"   Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"   Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"   Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    
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
    
    # Train model
    print("\n5. Training model...")
    
    # Load hyperparameters if provided or from saved tuning results
    if hyperparameters is None and model_type == 'lightgbm':
        tuning_results_path = os.path.join(save_dir, 'hyperparameter_tuning_results.json')
        if os.path.exists(tuning_results_path):
            print(f"   Loading hyperparameters from {tuning_results_path}")
            with open(tuning_results_path, 'r') as f:
                tuning_data = json.load(f)
                hyperparameters = tuning_data.get('best_params', {})
                print(f"   Using tuned hyperparameters: {list(hyperparameters.keys())}")
    
    # Prepare kwargs for model training
    model_kwargs = {}
    if hyperparameters:
        model_kwargs.update(hyperparameters)
    if model_type == 'lightgbm':
        model_kwargs['objective'] = objective
    
    # Pass validation set for LightGBM early stopping
    if model_type == 'lightgbm':
        model, feature_importance = train_model(
            X_train_scaled, y_train, 
            model_type=model_type,
            X_val=X_val_scaled,
            y_val=y_val,
            **model_kwargs
        )
    else:
        model, feature_importance = train_model(
            X_train_scaled, y_train, model_type=model_type, **model_kwargs
        )
    
    # Evaluate
    print("\n6. Evaluating model...")
    train_metrics = evaluate_model(model, X_train_scaled, y_train, 'train', use_log_transform)
    val_metrics = evaluate_model(model, X_val_scaled, y_val, 'val', use_log_transform)
    test_metrics = evaluate_model(model, X_test_scaled, y_test, 'test', use_log_transform)
    
    # Print metrics
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE METRICS")
    if use_log_transform:
        print("(Metrics computed on ORIGINAL scale after inverse transform)")
    print("=" * 80)
    print(f"\nTrain Set:")
    print(f"  RMSE: {train_metrics['train_rmse']:,.2f}")
    print(f"  MAE: {train_metrics['train_mae']:,.2f}")
    print(f"  R²: {train_metrics['train_r2']:.4f}")
    print(f"  MAPE: {train_metrics['train_mape']:.2f}%")
    if use_log_transform and 'train_log_r2' in train_metrics:
        print(f"  Log-scale R²: {train_metrics['train_log_r2']:.4f}")
    
    print(f"\nValidation Set:")
    print(f"  RMSE: {val_metrics['val_rmse']:,.2f}")
    print(f"  MAE: {val_metrics['val_mae']:,.2f}")
    print(f"  R²: {val_metrics['val_r2']:.4f}")
    print(f"  MAPE: {val_metrics['val_mape']:.2f}%")
    if use_log_transform and 'val_log_r2' in val_metrics:
        print(f"  Log-scale R²: {val_metrics['val_log_r2']:.4f}")
    
    print(f"\nTest Set:")
    print(f"  RMSE: {test_metrics['test_rmse']:,.2f}")
    print(f"  MAE: {test_metrics['test_mae']:,.2f}")
    print(f"  R²: {test_metrics['test_r2']:.4f}")
    print(f"  MAPE: {test_metrics['test_mape']:.2f}%")
    if use_log_transform and 'test_log_r2' in test_metrics:
        print(f"  Log-scale R²: {test_metrics['test_log_r2']:.4f}")
    
    # Save model and artifacts
    print("\n7. Saving model and artifacts...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save with model type suffix to keep separate models
    model_suffix = f'_{model_type}' if model_type != 'random_forest' else ''
    model_path = os.path.join(save_dir, f'price_prediction_model{model_suffix}.joblib')
    scaler_path = os.path.join(save_dir, f'scaler{model_suffix}.joblib')
    encoders_path = os.path.join(save_dir, f'label_encoders{model_suffix}.joblib')
    feature_importance_path = os.path.join(save_dir, f'feature_importance{model_suffix}.json')
    metrics_path = os.path.join(save_dir, f'metrics{model_suffix}.json')
    
    # Also save scaler and encoders without suffix for compatibility (use latest)
    scaler_path_compat = os.path.join(save_dir, 'scaler.joblib')
    encoders_path_compat = os.path.join(save_dir, 'label_encoders.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoders, encoders_path)
    
    # Save compatibility versions (for predict.py to use latest)
    joblib.dump(scaler, scaler_path_compat)
    joblib.dump(label_encoders, encoders_path_compat)
    
    with open(feature_importance_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    all_metrics = {**train_metrics, **val_metrics, **test_metrics}
    all_metrics['model_type'] = model_type  # Add model type to metrics
    all_metrics['use_log_transform'] = use_log_transform  # Track if log transform was used
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"   Model saved to: {model_path}")
    print(f"   Scaler saved to: {scaler_path}")
    print(f"   Label encoders saved to: {encoders_path}")
    print(f"   Feature importance saved to: {feature_importance_path}")
    print(f"   Metrics saved to: {metrics_path}")
    
    # Top features
    if feature_importance:
        print("\n" + "=" * 80)
        print("TOP 10 MOST IMPORTANT FEATURES")
        print("=" * 80)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:40s} {importance:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_importance': feature_importance,
        'metrics': all_metrics,
        'use_log_transform': use_log_transform
    }


if __name__ == "__main__":
    # Load preprocessed data
    import sys
    from src.preprocess import load_and_preprocess_data
    
    print("Loading preprocessed data...")
    df = pd.read_parquet('data/processed/sample_preprocessed.parquet')
    
    print(f"Data shape: {df.shape}")
    print(f"Target distribution:\n{df['price'].describe()}")
    
    # Train model
    results = train_price_prediction_model(
        df,
        target_col='price',
        model_type='random_forest',
        test_size=0.2,
        val_size=0.1,
        save_dir='models'
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

