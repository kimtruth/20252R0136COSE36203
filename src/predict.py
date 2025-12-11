"""Predict item price from payload_json"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from src.preprocess import (
    parse_json_field,
    extract_stat_features,
    extract_options_features,
    extract_time_features
)

# Import EnsembleModel for unpickling saved ensemble models
try:
    from src.ensemble import EnsembleModel
except ImportError:
    pass


def flatten_payload_json_for_prediction(payload_json: Dict) -> Dict:
    """Flatten payload_json for prediction (same as preprocessing)"""
    flat_data = {}
    
    # Extract basic fields from payload_json
    for key in ['star_force', 'potential_grade', 'additional_grade', 
                'scroll_count', 'item_id', 'trade_sn']:
        if key in payload_json:
            flat_data[f'payload_{key}'] = payload_json[key]
    
    # Parse detail_json
    detail_json = parse_json_field(payload_json.get('detail_json'))
    if detail_json:
        # Extract stats
        stats = detail_json.get('stats', {})
        stat_features = extract_stat_features(stats)
        flat_data.update({f'detail_{k}': v for k, v in stat_features.items()})
        
        # Extract potential and additional options
        potential_options = detail_json.get('potential_options', [])
        additional_options = detail_json.get('additional_options', [])
        
        pot_features = extract_options_features(potential_options)
        add_features = extract_options_features(additional_options)
        
        flat_data.update({f'potential_{k}': v for k, v in pot_features.items()})
        flat_data.update({f'additional_{k}': v for k, v in add_features.items()})
    
    # Parse summary_json
    summary_json = parse_json_field(payload_json.get('summary_json'))
    if summary_json:
        star_force_info = summary_json.get('star_force', {})
        if isinstance(star_force_info, dict):
            flat_data['summary_star_force_current'] = star_force_info.get('current', 0)
            flat_data['summary_star_force_max'] = star_force_info.get('max', 0)
    
    return flat_data


def prepare_features_for_prediction(payload_json: Dict, 
                                   label_encoders: Dict,
                                   created_at: Optional[str] = None,
                                   trade_date: Optional[str] = None,
                                   register_date: Optional[str] = None) -> pd.DataFrame:
    """Prepare features from payload_json for prediction"""
    
    # Start with a structure similar to original database row
    row_data = {}
    
    # Add basic fields (same as original columns)
    if 'name' in payload_json:
        row_data['name'] = payload_json['name']
    if 'item_id' in payload_json:
        row_data['item_id'] = payload_json['item_id']
    if 'count' in payload_json:
        row_data['count'] = payload_json['count']
    
    # Add grade fields from payload_json (these exist in original table)
    if 'potential_grade' in payload_json:
        row_data['potential_grade'] = payload_json['potential_grade']
    if 'additional_grade' in payload_json:
        row_data['additional_grade'] = payload_json['additional_grade']
    
    # Add time features
    if created_at:
        row_data['created_at'] = created_at
    elif 'created_at' in payload_json:
        row_data['created_at'] = payload_json['created_at']
    else:
        # Use current time if not provided
        from datetime import datetime
        row_data['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if trade_date:
        row_data['trade_date'] = trade_date
    elif 'trade_date' in payload_json:
        row_data['trade_date'] = payload_json['trade_date']
    
    if register_date:
        row_data['register_date'] = register_date
    elif 'register_date' in payload_json:
        row_data['register_date'] = payload_json['register_date']
    
    # Flatten payload_json for detailed features
    flat_data = flatten_payload_json_for_prediction(payload_json)
    
    # Merge flat_data into row_data
    row_data.update(flat_data)
    
    # Create DataFrame
    df = pd.DataFrame([row_data])
    
    # Extract time features (this adds year, month, day, etc.)
    df = extract_time_features(df)
    
    # Handle categorical variables (same as training)
    if 'item_id' in df.columns:
        df['item_id'] = df['item_id'].astype(str)
    if 'name' in df.columns:
        df['name'] = df['name'].astype(str)
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('unknown')
    
    # Encode categorical features
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            value = df[col].iloc[0]
            # Handle unseen categories
            if value not in le.classes_:
                # Use first class (most common) as fallback
                df.loc[0, col] = le.classes_[0]
            else:
                df.loc[0, col] = le.transform([value])[0]
    
    # Remove columns that shouldn't be features (same as training)
    columns_to_drop = [
        'trade_sn', 'payload_json', 'created_at', 'trade_date', 
        'register_date', 'end_date', 'price_per_unit'
    ]
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df


def predict_price(payload_json: Dict,
                  model_path: str = 'models/price_prediction_model_lightgbm.joblib',
                  scaler_path: str = 'models/scaler_lightgbm.joblib',
                  encoders_path: str = 'models/label_encoders_lightgbm.joblib',
                  created_at: Optional[str] = None,
                  trade_date: Optional[str] = None,
                  register_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Predict item price from payload_json
    
    Args:
        payload_json: Dictionary containing item information (can be full DB row or just payload_json)
        model_path: Path to trained model
        scaler_path: Path to scaler
        encoders_path: Path to label encoders
        created_at: Optional datetime string for created_at
        trade_date: Optional date string for trade_date
        register_date: Optional date string for register_date
    
    Returns:
        Dictionary with prediction results
    """
    # Load model and preprocessors
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)
    
    # Use preprocess module to ensure same preprocessing as training
    from src.preprocess import preprocess_data
    
    # If payload_json is just the payload_json field, convert to full row format
    if 'payload_json' not in payload_json and ('name' in payload_json or 'item_id' in payload_json):
        # Assume it's already a full row or payload_json dict
        row_data = payload_json.copy()
    else:
        # Create row structure from payload_json
        row_data = {}
        if isinstance(payload_json, dict):
            # If it has payload_json field, use it
            if 'payload_json' in payload_json:
                row_data.update(payload_json)
                # Extract payload_json if it's nested
                if isinstance(payload_json['payload_json'], dict):
                    row_data['payload_json'] = payload_json['payload_json']
            else:
                # Assume it's payload_json itself
                row_data = payload_json.copy()
                row_data['payload_json'] = payload_json
    
    # Add time fields if provided
    if created_at:
        row_data['created_at'] = created_at
    elif 'created_at' not in row_data:
        from datetime import datetime
        row_data['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if trade_date:
        row_data['trade_date'] = trade_date
    elif 'trade_date' not in row_data and 'payload_json' in row_data:
        if isinstance(row_data['payload_json'], dict) and 'trade_date' in row_data['payload_json']:
            row_data['trade_date'] = row_data['payload_json']['trade_date']
    
    if register_date:
        row_data['register_date'] = register_date
    elif 'register_date' not in row_data and 'payload_json' in row_data:
        if isinstance(row_data['payload_json'], dict) and 'register_date' in row_data['payload_json']:
            row_data['register_date'] = row_data['payload_json']['register_date']
    
    # Ensure payload_json is parsed if it's a string (before creating DataFrame)
    if 'payload_json' in row_data:
        payload_val = row_data['payload_json']
        if isinstance(payload_val, str):
            try:
                import json
                row_data['payload_json'] = json.loads(payload_val)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, keep as is or set to empty dict
                if not isinstance(payload_val, dict):
                    row_data['payload_json'] = {}
    
    # Create DataFrame with single row
    df = pd.DataFrame([row_data])
    
    # Preprocess using same pipeline as training
    df_processed = preprocess_data(df, target_col=None)  # No target for prediction
    
    # Add dummy price column if missing (needed for prepare_features)
    if 'price' not in df_processed.columns:
        df_processed['price'] = 0
    
    # Prepare features (same as training)
    from src.train import prepare_features
    X, _ = prepare_features(df_processed, target_col='price')
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories
            if X[col].iloc[0] not in le.classes_:
                X.loc[0, col] = le.classes_[0]
            else:
                X.loc[0, col] = le.transform([X[col].iloc[0]])[0]
            # Convert to numeric after encoding
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Ensure all expected features are present
    # Handle ensemble model (has rf_model and lgb_model)
    expected_features = None
    if hasattr(model, 'rf_model') and hasattr(model, 'lgb_model'):
        # Ensemble model: get features from one of the internal models
        if hasattr(model.rf_model, 'feature_names_in_'):
            expected_features = list(model.rf_model.feature_names_in_)
        elif hasattr(model.lgb_model, 'feature_names_in_'):
            expected_features = list(model.lgb_model.feature_names_in_)
    elif hasattr(model, 'feature_names_in_'):
        # Regular model
        expected_features = list(model.feature_names_in_)
    
    if expected_features:
        # Remove features that weren't seen during training
        unseen_features = [f for f in X.columns if f not in expected_features]
        if unseen_features:
            # Drop unseen features (these are new engineered features not in training)
            X = X.drop(columns=unseen_features)
        
        # Add missing features with default values
        missing_features = [f for f in expected_features if f not in X.columns]
        if missing_features:
            for feat in missing_features:
                # Try to determine if it should be numeric or categorical
                if any(x in feat.lower() for x in ['id', 'name']):
                    X[feat] = 0  # Will be encoded if categorical
                else:
                    X[feat] = 0
        
        # Reorder columns to match expected order
        X = X[expected_features]
    
    # Scale features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.transform(X[numeric_cols])
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    
    # Get feature importance if available
    feature_importance = {}
    # Check if it's an ensemble model (has get_feature_importance method)
    if hasattr(model, 'get_feature_importance'):
        feature_importance = model.get_feature_importance()
    elif hasattr(model, 'feature_importances_'):
        for idx, importance in enumerate(model.feature_importances_):
            feature_importance[X.columns[idx]] = float(importance)
    
    return {
        'predicted_price': float(prediction),
        'predicted_price_formatted': f"{prediction:,.0f}",
        'item_name': row_data.get('name', payload_json.get('name', 'Unknown')),
        'item_id': row_data.get('item_id', payload_json.get('item_id', 'Unknown')),
        'feature_importance_top5': dict(sorted(feature_importance.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True)[:5]) if feature_importance else {}
    }


def predict_from_json_file(json_file_path: str) -> Dict[str, Any]:
    """Predict price from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        payload_json = json.load(f)
    
    return predict_price(payload_json)


def predict_from_json_string(json_string: str) -> Dict[str, Any]:
    """Predict price from JSON string"""
    payload_json = json.loads(json_string)
    return predict_price(payload_json)


def format_price(price: float) -> str:
    """Format price in Korean style"""
    if price >= 1e12:
        return f"{price/1e12:.2f}조"
    elif price >= 1e8:
        return f"{price/1e8:.2f}억"
    elif price >= 1e4:
        return f"{price/1e4:.2f}만"
    else:
        return f"{price:,.0f}"


def predict_from_database(n_samples: int = 5,
                          model_path: str = 'models/price_prediction_model_lightgbm.joblib',
                          scaler_path: str = 'models/scaler_lightgbm.joblib',
                          encoders_path: str = 'models/label_encoders_lightgbm.joblib') -> None:
    """
    Predict prices for random samples from database and compare with actual prices
    
    Args:
        n_samples: Number of random samples to fetch from database
        model_path: Path to model file
        scaler_path: Path to scaler file
        encoders_path: Path to label encoders file
    """
    from src.db_connection import get_db_connection
    from src.config import TABLE_NAME
    
    print("=" * 80)
    print(f"데이터베이스에서 랜덤 {n_samples}개 샘플 가져오기")
    print("=" * 80)
    
    # Connect to database
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cursor:
            # Get random samples
            query = f"""
                SELECT * FROM {TABLE_NAME}
                LIMIT {n_samples}
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                print("데이터베이스에서 샘플을 찾을 수 없습니다.")
                return
            
            print(f"\n✓ {len(rows)}개 샘플 로드 완료\n")
            
            # Load model and preprocessors
            print("모델 로딩 중...")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            label_encoders = joblib.load(encoders_path)
            print("✓ 모델 로드 완료\n")
            
            results = []
            
            # Predict for each sample
            for idx, row in enumerate(rows, 1):
                try:
                    # Convert row to dict if needed
                    if isinstance(row, dict):
                        row_data = row
                    else:
                        row_data = dict(row)
                    
                    # Predict
                    result = predict_price(
                        row_data,
                        model_path=model_path,
                        scaler_path=scaler_path,
                        encoders_path=encoders_path
                    )
                    
                    actual_price = row_data.get('price', 0)
                    predicted_price = result['predicted_price']
                    error = abs(predicted_price - actual_price)
                    error_pct = (error / actual_price * 100) if actual_price > 0 else 0
                    
                    results.append({
                        'index': idx,
                        'item_name': result['item_name'],
                        'item_id': result['item_id'],
                        'actual_price': actual_price,
                        'predicted_price': predicted_price,
                        'error': error,
                        'error_pct': error_pct
                    })
                    
                except Exception as e:
                    print(f"⚠ 샘플 {idx} 예측 실패: {e}")
                    continue
            
            # Print results
            print("=" * 80)
            print("예측 결과 요약")
            print("=" * 80)
            
            total_error = 0
            total_error_pct = 0
            valid_samples = 0
            
            for res in results:
                print(f"\n[샘플 {res['index']}]")
                print(f"  아이템명: {res['item_name']}")
                print(f"  아이템 ID: {res['item_id']}")
                print(f"  실제 가격: {res['actual_price']:,.0f} 메소 ({format_price(res['actual_price'])})")
                print(f"  예측 가격: {res['predicted_price']:,.0f} 메소 ({format_price(res['predicted_price'])})")
                print(f"  오차: {res['error']:,.0f} 메소 ({format_price(res['error'])}, {res['error_pct']:.2f}%)")
                
                total_error += res['error']
                total_error_pct += res['error_pct']
                valid_samples += 1
            
            if valid_samples > 0:
                avg_error = total_error / valid_samples
                avg_error_pct = total_error_pct / valid_samples
                
                print("\n" + "=" * 80)
                print("전체 통계")
                print("=" * 80)
                print(f"평균 절대 오차: {avg_error:,.0f} 메소 ({format_price(avg_error)})")
                print(f"평균 오차율: {avg_error_pct:.2f}%")
                print(f"성공한 샘플: {valid_samples}/{len(rows)}")
            
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict item price from payload_json or database')
    parser.add_argument(
        '--json-file',
        type=str,
        help='Path to JSON file containing payload_json'
    )
    parser.add_argument(
        '--json-string',
        type=str,
        help='JSON string containing payload_json'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode: enter JSON interactively'
    )
    parser.add_argument(
        '--db-samples',
        type=int,
        default=5,
        help='Number of random samples to fetch from database (default: 5)'
    )
    parser.add_argument(
        '--use-db',
        action='store_true',
        help='Use random samples from database (default mode)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/price_prediction_model_lightgbm.joblib',
        help='Path to model file (default: LightGBM model)'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='models/scaler_lightgbm.joblib',
        help='Path to scaler file (default: LightGBM scaler)'
    )
    parser.add_argument(
        '--encoders',
        type=str,
        default='models/label_encoders_lightgbm.joblib',
        help='Path to label encoders file (default: LightGBM encoders)'
    )
    
    args = parser.parse_args()
    
    # Default: use database samples
    if args.use_db or (not args.json_file and not args.json_string and not args.interactive):
        predict_from_database(
            n_samples=args.db_samples,
            model_path=args.model,
            scaler_path=args.scaler,
            encoders_path=args.encoders
        )
    else:
        # Get payload_json from other sources
        if args.json_file:
            with open(args.json_file, 'r', encoding='utf-8') as f:
                payload_json = json.load(f)
        elif args.json_string:
            payload_json = json.loads(args.json_string)
        elif args.interactive:
            print("Enter payload_json (paste JSON and press Enter, then Ctrl+D or Ctrl+Z to finish):")
            payload_json = json.load(sys.stdin)
        else:
            print("No input provided.")
            print("Usage:")
            print("  python src/predict.py --use-db --db-samples 10  # Use database (default)")
            print("  python src/predict.py --json-file <file>       # Use JSON file")
            print("  python src/predict.py --json-string '<json>'  # Use JSON string")
            print("  python src/predict.py --interactive            # Interactive mode")
            sys.exit(1)
        
        # Predict
        try:
            result = predict_price(
                payload_json,
                model_path=args.model,
                scaler_path=args.scaler,
                encoders_path=args.encoders
            )
            
            print("\n" + "=" * 80)
            print("가격 예측 결과")
            print("=" * 80)
            print(f"\n아이템명: {result['item_name']}")
            print(f"아이템 ID: {result['item_id']}")
            print(f"\n예측 가격: {result['predicted_price_formatted']} 메소 ({format_price(result['predicted_price'])} 메소)")
            
            if 'price' in payload_json:
                actual_price = payload_json['price']
                error = abs(result['predicted_price'] - actual_price)
                error_pct = (error / actual_price * 100) if actual_price > 0 else 0
                print(f"\n실제 가격: {actual_price:,.0f} 메소 ({format_price(actual_price)} 메소)")
                print(f"오차: {error:,.0f} 메소 ({format_price(error)} 메소, {error_pct:.2f}%)")
            
            print("\n" + "=" * 80)
            print("주요 특징 중요도 (Top 5)")
            print("=" * 80)
            for feature, importance in result['feature_importance_top5'].items():
                print(f"  {feature}: {importance:.4f}")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()

