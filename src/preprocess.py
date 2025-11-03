"""Data preprocessing module for MapleStory auction data"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
from src.db_connection import get_db_connection
from src.config import TABLE_NAME


def parse_json_field(json_str: Optional[str]) -> Optional[Dict]:
    """Parse JSON string field"""
    if json_str is None:
        return None
    if isinstance(json_str, dict):
        return json_str
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_stat_features(stats: Optional[Dict]) -> Dict[str, float]:
    """Extract stat features from stats dictionary"""
    features = {}
    
    if not stats:
        return features
    
    # Base stats
    base = stats.get('base', {})
    features['base_STR'] = base.get('STR', 0)
    features['base_DEX'] = base.get('DEX', 0)
    features['base_INT'] = base.get('INT', 0)
    features['base_LUK'] = base.get('LUK', 0)
    
    # Scroll stats (sum of arrays)
    scroll = stats.get('scroll', {})
    for stat_name in ['STR', 'DEX', 'INT', 'LUK']:
        scroll_values = scroll.get(stat_name, [])
        if isinstance(scroll_values, list):
            features[f'scroll_{stat_name}_sum'] = sum(scroll_values)
            features[f'scroll_{stat_name}_max'] = max(scroll_values) if scroll_values else 0
        else:
            features[f'scroll_{stat_name}_sum'] = 0
            features[f'scroll_{stat_name}_max'] = 0
    
    # MHP, MMP, PAD, MAD (sum of arrays)
    for stat_name in ['MHP', 'MMP', 'PAD', 'MAD', 'PDD']:
        stat_values = stats.get(stat_name, [])
        if isinstance(stat_values, list):
            features[f'{stat_name}_sum'] = sum(stat_values)
            features[f'{stat_name}_max'] = max(stat_values) if stat_values else 0
        else:
            features[f'{stat_name}_sum'] = 0
            features[f'{stat_name}_max'] = 0
    
    # Percent stats
    percent = stats.get('percent', {})
    for stat_name in ['MHP', 'MMP', 'BDR', 'IMDR', 'Damage', 'StatR']:
        stat_values = percent.get(stat_name, [])
        if isinstance(stat_values, list):
            features[f'percent_{stat_name}_sum'] = sum(stat_values)
        else:
            features[f'percent_{stat_name}_sum'] = stat_values if isinstance(stat_values, (int, float)) else 0
    
    # Requirements
    requirements = stats.get('requirements', {})
    features['scroll_count'] = requirements.get('scroll_count', 0)
    features['cuttable'] = requirements.get('cuttable', 0)
    
    return features


def extract_options_features(options: List[str]) -> Dict[str, Any]:
    """Extract features from options list"""
    features = {}
    
    if not options or not isinstance(options, list):
        return {
            'options_count': 0,
            'has_skill_cooldown': 0,
            'has_stat_percent': 0,
            'has_damage': 0
        }
    
    features['options_count'] = len(options)
    
    # Check for specific option types
    option_text = ' '.join(options).lower()
    features['has_skill_cooldown'] = 1 if '재사용' in option_text or 'cooldown' in option_text else 0
    features['has_stat_percent'] = 1 if '%' in option_text else 0
    features['has_damage'] = 1 if '데미지' in option_text or 'damage' in option_text else 0
    
    return features


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features"""
    df = df.copy()
    
    # Parse datetime columns
    if 'created_at' in df.columns:
        try:
            # Check if already datetime
            if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            
            df['year'] = df['created_at'].dt.year
            df['month'] = df['created_at'].dt.month
            df['day'] = df['created_at'].dt.day
            df['day_of_week'] = df['created_at'].dt.dayofweek
            df['hour'] = df['created_at'].dt.hour
            df['day_of_year'] = df['created_at'].dt.dayofyear
        except Exception as e:
            print(f"Warning: Could not parse created_at: {e}")
    
    if 'trade_date' in df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
            
            df['trade_year'] = df['trade_date'].dt.year
            df['trade_month'] = df['trade_date'].dt.month
            df['trade_day_of_week'] = df['trade_date'].dt.dayofweek
        except Exception as e:
            print(f"Warning: Could not parse trade_date: {e}")
    
    if 'register_date' in df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df['register_date']):
                df['register_date'] = pd.to_datetime(df['register_date'], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not parse register_date: {e}")
    
    return df


def flatten_payload_json(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten payload_json column"""
    df = df.copy()
    
    # Parse payload_json
    payload_data = []
    for idx, row in df.iterrows():
        payload = row.get('payload_json')
        if payload is None:
            payload_data.append({})
            continue
        
        # Parse if string
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except:
                payload = {}
        
        flat_data = {}
        
        # Extract basic fields from payload_json
        for key in ['star_force', 'potential_grade', 'additional_grade', 
                    'scroll_count', 'item_id', 'trade_sn']:
            if key in payload:
                flat_data[f'payload_{key}'] = payload[key]
        
        # Parse detail_json
        detail_json = parse_json_field(payload.get('detail_json'))
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
        summary_json = parse_json_field(payload.get('summary_json'))
        if summary_json:
            # Extract star_force info
            star_force_info = summary_json.get('star_force', {})
            if isinstance(star_force_info, dict):
                flat_data['summary_star_force_current'] = star_force_info.get('current', 0)
                flat_data['summary_star_force_max'] = star_force_info.get('max', 0)
        
        payload_data.append(flat_data)
    
    # Create DataFrame from flattened payload data
    payload_df = pd.DataFrame(payload_data)
    
    # Merge with original dataframe
    df = pd.concat([df, payload_df], axis=1)
    
    return df


def preprocess_data(df: pd.DataFrame, target_col: str = 'price') -> pd.DataFrame:
    """Main preprocessing function"""
    df = df.copy()
    
    # Remove rows with missing target
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
    
    # Flatten payload_json
    print("Flattening payload_json...")
    df = flatten_payload_json(df)
    
    # Extract time features
    print("Extracting time features...")
    df = extract_time_features(df)
    
    # Handle categorical variables
    print("Handling categorical variables...")
    
    # Convert item_id to string for encoding
    if 'item_id' in df.columns:
        df['item_id'] = df['item_id'].astype(str)
    
    # Handle name column (keep as categorical, will be encoded later)
    if 'name' in df.columns:
        df['name'] = df['name'].astype(str)
    
    # Fill missing values
    print("Filling missing values...")
    
    # Fill numeric columns with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Fill categorical columns with 'unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('unknown')
    
    return df


def load_and_preprocess_data(limit: Optional[int] = None, 
                            save_path: Optional[str] = None) -> pd.DataFrame:
    """Load data from database and preprocess"""
    print("Loading data from database...")
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cursor:
            query = f"SELECT * FROM {TABLE_NAME}"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows)
            print(f"Loaded {len(df)} rows")
        
        # Preprocess
        df_processed = preprocess_data(df)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df_processed.to_parquet(save_path, index=False)
            print(f"Saved preprocessed data to {save_path}")
        
        return df_processed
        
    finally:
        conn.close()


if __name__ == "__main__":
    # Test with small sample
    print("Testing preprocessing with 1000 rows...")
    df = load_and_preprocess_data(limit=1000, save_path='data/processed/sample_preprocessed.parquet')
    print(f"\nPreprocessed data shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:\n{df.head()}")

