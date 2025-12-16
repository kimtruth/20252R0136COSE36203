"""Data preprocessing module for MapleStory auction data"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional

# Lazy import for database connection (only needed when loading from DB)
def _get_db_connection():
    from src.db_connection import get_db_connection
    return get_db_connection()

def _get_table_name():
    from src.config import TABLE_NAME
    return TABLE_NAME


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


def parse_potential_option_value(option_str: str) -> Dict[str, Any]:
    """
    Parse a potential option string like "STR +6%" or "공격력 +10" to extract numeric values.
    
    Returns dict with stat_type, value, is_percent
    """
    import re
    
    result = {'stat_type': 'unknown', 'value': 0, 'is_percent': False}
    
    if not option_str or not isinstance(option_str, str):
        return result
    
    # Check if percentage
    result['is_percent'] = '%' in option_str
    
    # Extract numeric value
    numbers = re.findall(r'[-+]?\d+', option_str)
    if numbers:
        result['value'] = int(numbers[0])
    
    # Identify stat type
    stat_keywords = {
        'STR': 'STR',
        'DEX': 'DEX', 
        'INT': 'INT',
        'LUK': 'LUK',
        '공격력': 'ATK',
        '마력': 'MATK',
        '보스': 'BOSS_DMG',
        '데미지': 'DMG',
        '크리티컬': 'CRIT',
        '올스탯': 'ALL_STAT',
        '방어율 무시': 'IED',
        '방어력': 'DEF',
        '최대 HP': 'HP',
        '최대 MP': 'MP',
        '이동속도': 'SPEED',
        '점프력': 'JUMP',
        '재사용': 'COOLDOWN'
    }
    
    for keyword, stat_type in stat_keywords.items():
        if keyword in option_str:
            result['stat_type'] = stat_type
            break
    
    return result


def extract_detailed_options_features(options: List[str], prefix: str = '') -> Dict[str, Any]:
    """
    Extract detailed features from potential/additional options.
    Parses numeric values from options like "STR +6%", "공격력 +10"
    """
    features = {}
    
    if not options or not isinstance(options, list):
        return {
            f'{prefix}total_percent_value': 0,
            f'{prefix}total_flat_value': 0,
            f'{prefix}str_percent': 0,
            f'{prefix}dex_percent': 0,
            f'{prefix}int_percent': 0,
            f'{prefix}luk_percent': 0,
            f'{prefix}all_stat_percent': 0,
            f'{prefix}atk_flat': 0,
            f'{prefix}matk_flat': 0,
            f'{prefix}boss_dmg': 0,
            f'{prefix}ied': 0,
            f'{prefix}crit_dmg': 0,
        }
    
    # Initialize accumulators
    total_percent = 0
    total_flat = 0
    stat_percents = {'STR': 0, 'DEX': 0, 'INT': 0, 'LUK': 0, 'ALL_STAT': 0}
    flat_values = {'ATK': 0, 'MATK': 0}
    special_stats = {'BOSS_DMG': 0, 'IED': 0, 'DMG': 0, 'CRIT': 0}
    
    for option in options:
        parsed = parse_potential_option_value(option)
        
        if parsed['is_percent']:
            total_percent += parsed['value']
            if parsed['stat_type'] in stat_percents:
                stat_percents[parsed['stat_type']] += parsed['value']
            if parsed['stat_type'] in special_stats:
                special_stats[parsed['stat_type']] += parsed['value']
        else:
            total_flat += parsed['value']
            if parsed['stat_type'] in flat_values:
                flat_values[parsed['stat_type']] += parsed['value']
    
    features[f'{prefix}total_percent_value'] = total_percent
    features[f'{prefix}total_flat_value'] = total_flat
    features[f'{prefix}str_percent'] = stat_percents['STR']
    features[f'{prefix}dex_percent'] = stat_percents['DEX']
    features[f'{prefix}int_percent'] = stat_percents['INT']
    features[f'{prefix}luk_percent'] = stat_percents['LUK']
    features[f'{prefix}all_stat_percent'] = stat_percents['ALL_STAT']
    features[f'{prefix}atk_flat'] = flat_values['ATK']
    features[f'{prefix}matk_flat'] = flat_values['MATK']
    features[f'{prefix}boss_dmg'] = special_stats['BOSS_DMG']
    features[f'{prefix}ied'] = special_stats['IED']
    features[f'{prefix}crit_dmg'] = special_stats['CRIT']
    
    return features


def extract_item_metadata(detail_json: Dict) -> Dict[str, Any]:
    """
    Extract item metadata from detail_json including category, level requirement, job type, etc.
    """
    features = {}
    
    if not detail_json:
        return {
            'category': 'unknown',
            'level_requirement': 0,
            'job_type': 0,
            'star_force_max': 0,
            'item_grade': 0,
            'masterpiece': 0,
            'item_quality': 0
        }
    
    # Extract category (신발, 상의, 장갑, etc.)
    features['category'] = detail_json.get('category', 'unknown')
    
    # Extract level requirements
    level_req = detail_json.get('level_requirements', {})
    if isinstance(level_req, dict):
        features['level_requirement'] = level_req.get('level_is_cash_item', 0)
        features['job_type'] = level_req.get('job', 0)
        features['is_superior'] = level_req.get('superior', 0)
        features['seed_ring_level'] = level_req.get('seed_ring_level', 0)
    else:
        features['level_requirement'] = 0
        features['job_type'] = 0
        features['is_superior'] = 0
        features['seed_ring_level'] = 0
    
    # Extract star force info
    features['star_force_max'] = detail_json.get('star_force_max', 0)
    features['star_force_current'] = detail_json.get('star_force', 0)
    
    # Extract grade/quality
    features['item_grade'] = detail_json.get('grade', 0)
    features['masterpiece'] = detail_json.get('masterpiece', 0)
    features['item_quality'] = detail_json.get('item_quality', 0)
    
    # Extract potential and additional grades from detail_json
    features['detail_potential_grade'] = detail_json.get('potential_grade', 0)
    features['detail_additional_grade'] = detail_json.get('additional_grade', 0)
    
    # Is cash item
    features['is_cash_item'] = 1 if detail_json.get('cash_item', False) else 0
    
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
    """Flatten payload_json column with enhanced feature extraction"""
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
            
            # Extract potential and additional options (basic features)
            potential_options = detail_json.get('potential_options', [])
            additional_options = detail_json.get('additional_options', [])
            
            pot_features = extract_options_features(potential_options)
            add_features = extract_options_features(additional_options)
            
            flat_data.update({f'potential_{k}': v for k, v in pot_features.items()})
            flat_data.update({f'additional_{k}': v for k, v in add_features.items()})
            
            # NEW: Extract detailed potential option values (numeric parsing)
            detailed_pot_features = extract_detailed_options_features(potential_options, 'pot_')
            detailed_add_features = extract_detailed_options_features(additional_options, 'add_')
            flat_data.update(detailed_pot_features)
            flat_data.update(detailed_add_features)
            
            # NEW: Extract item metadata (category, level_req, job, grade, etc.)
            metadata_features = extract_item_metadata(detail_json)
            flat_data.update(metadata_features)
        
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


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to improve model performance"""
    df = df.copy()
    
    # 1. Total attack power (PAD + MAD)
    if 'detail_PAD_sum' in df.columns and 'detail_MAD_sum' in df.columns:
        df['total_attack'] = df['detail_PAD_sum'] + df['detail_MAD_sum']
    
    # 2. Total attack power (max values)
    if 'detail_PAD_max' in df.columns and 'detail_MAD_max' in df.columns:
        df['total_attack_max'] = df['detail_PAD_max'] + df['detail_MAD_max']
    
    # 3. Total stat points (STR + DEX + INT + LUK)
    stat_cols = ['detail_scroll_STR_sum', 'detail_scroll_DEX_sum', 
                 'detail_scroll_INT_sum', 'detail_scroll_LUK_sum']
    if all(col in df.columns for col in stat_cols):
        df['total_stat_sum'] = df[stat_cols].sum(axis=1)
    
    # 4. Stat efficiency (total stats per scroll count)
    if 'total_stat_sum' in df.columns and 'detail_scroll_count' in df.columns:
        df['stat_efficiency'] = df['total_stat_sum'] / (df['detail_scroll_count'] + 1)
        df['stat_efficiency'] = df['stat_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 5. Attack efficiency (total attack per scroll count)
    if 'total_attack' in df.columns and 'detail_scroll_count' in df.columns:
        df['attack_efficiency'] = df['total_attack'] / (df['detail_scroll_count'] + 1)
        df['attack_efficiency'] = df['attack_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 6. Potential + Additional grade combination
    if 'potential_grade' in df.columns and 'additional_grade' in df.columns:
        df['total_grade'] = df['potential_grade'] + df['additional_grade']
    
    # 7. Star force efficiency (star force per potential grade)
    if 'payload_star_force' in df.columns and 'potential_grade' in df.columns:
        df['star_force_efficiency'] = df['payload_star_force'] / (df['potential_grade'] + 1)
        df['star_force_efficiency'] = df['star_force_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 8. Total option count (potential + additional)
    pot_count_col = 'potential_options_count' if 'potential_options_count' in df.columns else None
    add_count_col = 'additional_options_count' if 'additional_options_count' in df.columns else None
    if pot_count_col and add_count_col:
        df['total_options_count'] = df[pot_count_col] + df[add_count_col]
    
    # 9. Defense ratio (PDD / total attack)
    if 'detail_PDD_sum' in df.columns and 'total_attack' in df.columns:
        df['defense_ratio'] = df['detail_PDD_sum'] / (df['total_attack'] + 1)
        df['defense_ratio'] = df['defense_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 10. HP efficiency (MHP / total attack)
    if 'detail_MHP_sum' in df.columns and 'total_attack' in df.columns:
        df['hp_efficiency'] = df['detail_MHP_sum'] / (df['total_attack'] + 1)
        df['hp_efficiency'] = df['hp_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 11. Maximum stat (max of STR, DEX, INT, LUK)
    if all(col in df.columns for col in ['detail_scroll_STR_max', 'detail_scroll_DEX_max', 
                                          'detail_scroll_INT_max', 'detail_scroll_LUK_max']):
        df['max_stat'] = df[['detail_scroll_STR_max', 'detail_scroll_DEX_max', 
                             'detail_scroll_INT_max', 'detail_scroll_LUK_max']].max(axis=1)
    
    # 12. Total percent stat (sum of all percent stats)
    percent_cols = [col for col in df.columns if col.startswith('detail_percent_')]
    if percent_cols:
        df['total_percent_stat'] = df[percent_cols].sum(axis=1)
    
    # 13. Attack to stat ratio (total attack / total stat)
    if 'total_attack' in df.columns and 'total_stat_sum' in df.columns:
        df['attack_stat_ratio'] = df['total_attack'] / (df['total_stat_sum'] + 1)
        df['attack_stat_ratio'] = df['attack_stat_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 14. Star force weighted by attack (star_force * total_attack)
    if 'payload_star_force' in df.columns and 'total_attack' in df.columns:
        df['star_force_attack_score'] = df['payload_star_force'] * df['total_attack']
    
    # 15. Grade weighted attack (total_grade * total_attack)
    if 'total_grade' in df.columns and 'total_attack' in df.columns:
        df['grade_attack_score'] = df['total_grade'] * df['total_attack']
    
    # 16. Option quality score (has_skill_cooldown + has_stat_percent + has_damage)
    pot_opt_cols = ['potential_has_skill_cooldown', 'potential_has_stat_percent', 'potential_has_damage']
    add_opt_cols = ['additional_has_skill_cooldown', 'additional_has_stat_percent', 'additional_has_damage']
    
    if all(col in df.columns for col in pot_opt_cols):
        df['potential_quality_score'] = df[pot_opt_cols].sum(axis=1)
    if all(col in df.columns for col in add_opt_cols):
        df['additional_quality_score'] = df[add_opt_cols].sum(axis=1)
    if 'potential_quality_score' in df.columns and 'additional_quality_score' in df.columns:
        df['total_quality_score'] = df['potential_quality_score'] + df['additional_quality_score']
    
    # 17. Percent damage ratio (if available)
    if 'detail_percent_Damage_sum' in df.columns and 'total_attack' in df.columns:
        df['damage_percent_ratio'] = df['detail_percent_Damage_sum'] / (df['total_attack'] + 1)
        df['damage_percent_ratio'] = df['damage_percent_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 18. Base stat total (base STR + DEX + INT + LUK)
    base_stat_cols = ['detail_base_STR', 'detail_base_DEX', 'detail_base_INT', 'detail_base_LUK']
    if all(col in df.columns for col in base_stat_cols):
        df['base_stat_total'] = df[base_stat_cols].sum(axis=1)
    
    # 19. Scroll enhancement level (scroll_count weighted by star_force)
    if 'detail_scroll_count' in df.columns and 'payload_star_force' in df.columns:
        df['enhancement_level'] = df['detail_scroll_count'] * df['payload_star_force']
    
    # 20. Price tier estimation (based on multiple factors)
    # This is a synthetic feature combining multiple important factors
    if all(col in df.columns for col in ['total_attack_max', 'total_grade', 'payload_star_force']):
        # Normalize and combine (simplified scoring)
        df['value_score'] = (
            (df['total_attack_max'] / (df['total_attack_max'].max() + 1)) * 0.4 +
            (df['total_grade'] / (df['total_grade'].max() + 1)) * 0.3 +
            (df['payload_star_force'] / (df['payload_star_force'].max() + 1)) * 0.3
        )
    
    # ============ NEW ENGINEERED FEATURES ============
    
    # 21. Total potential stat percent (sum of all stat % from potentials)
    pot_percent_cols = ['pot_str_percent', 'pot_dex_percent', 'pot_int_percent', 'pot_luk_percent']
    if all(col in df.columns for col in pot_percent_cols):
        df['pot_total_main_stat_percent'] = df[pot_percent_cols].sum(axis=1)
    
    # 22. Total additional stat percent
    add_percent_cols = ['add_str_percent', 'add_dex_percent', 'add_int_percent', 'add_luk_percent']
    if all(col in df.columns for col in add_percent_cols):
        df['add_total_main_stat_percent'] = df[add_percent_cols].sum(axis=1)
    
    # 23. Combined main stat percent (potential + additional)
    if 'pot_total_main_stat_percent' in df.columns and 'add_total_main_stat_percent' in df.columns:
        df['combined_main_stat_percent'] = df['pot_total_main_stat_percent'] + df['add_total_main_stat_percent']
    
    # 24. Total attack from potentials (flat ATK + MATK)
    if 'pot_atk_flat' in df.columns and 'pot_matk_flat' in df.columns:
        df['pot_total_attack_flat'] = df['pot_atk_flat'] + df['pot_matk_flat']
    if 'add_atk_flat' in df.columns and 'add_matk_flat' in df.columns:
        df['add_total_attack_flat'] = df['add_atk_flat'] + df['add_matk_flat']
    
    # 25. Combined boss damage (potential + additional)
    if 'pot_boss_dmg' in df.columns and 'add_boss_dmg' in df.columns:
        df['combined_boss_dmg'] = df['pot_boss_dmg'] + df['add_boss_dmg']
    
    # 26. Combined IED (potential + additional)
    if 'pot_ied' in df.columns and 'add_ied' in df.columns:
        df['combined_ied'] = df['pot_ied'] + df['add_ied']
    
    # 27. Star force ratio (current / max)
    if 'star_force_current' in df.columns and 'star_force_max' in df.columns:
        df['star_force_ratio'] = df['star_force_current'] / (df['star_force_max'] + 1)
        df['star_force_ratio'] = df['star_force_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 28. Level tier (binned level requirement)
    if 'level_requirement' in df.columns:
        df['level_tier'] = pd.cut(df['level_requirement'], 
                                   bins=[0, 100, 140, 160, 200, 250, 300], 
                                   labels=[0, 1, 2, 3, 4, 5],
                                   include_lowest=True)
        df['level_tier'] = df['level_tier'].astype(float).fillna(0)
    
    # 29. Enhancement score (combination of star force, scroll, and grade)
    if all(col in df.columns for col in ['payload_star_force', 'detail_scroll_count', 'total_grade']):
        df['enhancement_score'] = (
            df['payload_star_force'] * 2 + 
            df['detail_scroll_count'] * 3 + 
            df['total_grade'] * 5
        )
    
    # 30. Potential value score (weighted sum of potential stats)
    if all(col in df.columns for col in ['pot_total_percent_value', 'pot_boss_dmg', 'pot_ied', 'pot_total_attack_flat']):
        df['potential_value_score'] = (
            df['pot_total_percent_value'] * 1.0 +
            df['pot_boss_dmg'] * 0.8 +
            df['pot_ied'] * 0.6 +
            df['pot_total_attack_flat'] * 0.3
        )
    
    # 31. Star force squared (non-linear relationship with price)
    if 'payload_star_force' in df.columns:
        df['star_force_squared'] = df['payload_star_force'] ** 2
    
    # 32. Grade interaction with star force
    if 'total_grade' in df.columns and 'payload_star_force' in df.columns:
        df['grade_star_interaction'] = df['total_grade'] * df['payload_star_force']
    
    # 33. High-tier indicator (star force >= 17)
    if 'payload_star_force' in df.columns:
        df['is_high_star'] = (df['payload_star_force'] >= 17).astype(int)
    
    # 34. Legendary potential indicator (grade == 3)
    if 'detail_potential_grade' in df.columns:
        df['is_legendary_potential'] = (df['detail_potential_grade'] == 3).astype(int)
    
    # 35. Unique or higher potential indicator
    if 'detail_potential_grade' in df.columns:
        df['is_unique_or_higher'] = (df['detail_potential_grade'] >= 2).astype(int)
    
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
    
    # Add engineered features
    print("Adding engineered features...")
    df = add_engineered_features(df)
    
    # Handle categorical variables
    print("Handling categorical variables...")
    
    # Convert item_id to string for encoding
    if 'item_id' in df.columns:
        df['item_id'] = df['item_id'].astype(str)
    
    # Handle name column (keep as categorical, will be encoded later)
    if 'name' in df.columns:
        df['name'] = df['name'].astype(str)
    
    # Handle category column (keep as categorical, will be encoded later)
    if 'category' in df.columns:
        df['category'] = df['category'].astype(str)
    
    # Fill missing values
    print("Filling missing values...")
    
    # Fill numeric columns with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Fill categorical columns with 'unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('unknown')
    
    return df


def load_from_jsonl(file_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Load data from JSONL file (useful for Colab/Google Drive)"""
    print(f"Loading data from JSONL file: {file_path}")
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {i+1}: {e}")
                continue
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} rows from JSONL file")
    return df


def load_and_preprocess_data(limit: Optional[int] = None, 
                            save_path: Optional[str] = None,
                            jsonl_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from database or JSONL file and preprocess
    
    Args:
        limit: Limit number of rows to load
        save_path: Path to save preprocessed data
        jsonl_path: Path to JSONL file (if provided, loads from file instead of database)
    """
    # Load from JSONL file if provided (for Colab compatibility)
    if jsonl_path:
        print("Loading from JSONL file...")
        df = load_from_jsonl(jsonl_path, limit=limit)
    else:
        # Load from database
        print("Loading data from database...")
        try:
            conn = _get_db_connection()
            table_name = _get_table_name()
            try:
                with conn.cursor() as cursor:
                    query = f"SELECT * FROM {table_name}"
                    if limit:
                        query += f" LIMIT {limit}"
                    
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(rows)
                    print(f"Loaded {len(df)} rows")
            finally:
                conn.close()
        except Exception as e:
            print(f"Database connection failed: {e}")
            print("Hint: For Colab, use jsonl_path parameter to load from file")
            raise
    
    # Preprocess
    df_processed = preprocess_data(df)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_processed.to_parquet(save_path, index=False)
        print(f"Saved preprocessed data to {save_path}")
    
    return df_processed


if __name__ == "__main__":
    # Test with small sample
    print("Testing preprocessing with 1000 rows...")
    df = load_and_preprocess_data(limit=1000, save_path='data/processed/sample_preprocessed.parquet')
    print(f"\nPreprocessed data shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:\n{df.head()}")

