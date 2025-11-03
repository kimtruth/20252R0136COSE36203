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
        'register_date', 'end_date', 'price_per_unit'  # price_per_unit is too similar to target
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
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 3),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        for idx, importance in enumerate(model.feature_importances_):
            feature_importance[X_train.columns[idx]] = float(importance)
    
    return model, feature_importance


def evaluate_model(model: object, 
                   X: pd.DataFrame, 
                   y: pd.Series,
                   set_name: str = 'test') -> Dict[str, float]:
    """Evaluate model performance"""
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Calculate percentage errors
    mape = np.mean(np.abs((y - y_pred) / (y + 1))) * 100  # Add 1 to avoid division by zero
    
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
                                use_time_split: bool = True) -> Dict:
    """Complete training pipeline"""
    
    print("=" * 80)
    print("TRAINING PRICE PREDICTION MODEL")
    print("=" * 80)
    
    # Prepare features
    print("\n1. Preparing features...")
    X, y = prepare_features(df, target_col)
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # Split data
    print("\n2. Splitting data...")
    if use_time_split:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_time(
            df, X, y, test_size=test_size, val_size=val_size
        )
    else:
        print("   Using random split...")
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
    model, feature_importance = train_model(
        X_train_scaled, y_train, model_type=model_type
    )
    
    # Evaluate
    print("\n6. Evaluating model...")
    train_metrics = evaluate_model(model, X_train_scaled, y_train, 'train')
    val_metrics = evaluate_model(model, X_val_scaled, y_val, 'val')
    test_metrics = evaluate_model(model, X_test_scaled, y_test, 'test')
    
    # Print metrics
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 80)
    print(f"\nTrain Set:")
    print(f"  RMSE: {train_metrics['train_rmse']:,.2f}")
    print(f"  MAE: {train_metrics['train_mae']:,.2f}")
    print(f"  R²: {train_metrics['train_r2']:.4f}")
    print(f"  MAPE: {train_metrics['train_mape']:.2f}%")
    
    print(f"\nValidation Set:")
    print(f"  RMSE: {val_metrics['val_rmse']:,.2f}")
    print(f"  MAE: {val_metrics['val_mae']:,.2f}")
    print(f"  R²: {val_metrics['val_r2']:.4f}")
    print(f"  MAPE: {val_metrics['val_mape']:.2f}%")
    
    print(f"\nTest Set:")
    print(f"  RMSE: {test_metrics['test_rmse']:,.2f}")
    print(f"  MAE: {test_metrics['test_mae']:,.2f}")
    print(f"  R²: {test_metrics['test_r2']:.4f}")
    print(f"  MAPE: {test_metrics['test_mape']:.2f}%")
    
    # Save model and artifacts
    print("\n7. Saving model and artifacts...")
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'price_prediction_model.joblib')
    scaler_path = os.path.join(save_dir, 'scaler.joblib')
    encoders_path = os.path.join(save_dir, 'label_encoders.joblib')
    feature_importance_path = os.path.join(save_dir, 'feature_importance.json')
    metrics_path = os.path.join(save_dir, 'metrics.json')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoders, encoders_path)
    
    with open(feature_importance_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    all_metrics = {**train_metrics, **val_metrics, **test_metrics}
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
        'metrics': all_metrics
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

