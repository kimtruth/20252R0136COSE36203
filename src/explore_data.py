"""Explore and analyze the database structure and data"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.db_connection import get_db_connection
from src.config import TABLE_NAME
import json


def explore_table_structure():
    """Explore the structure of the auction_history table"""
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cursor:
            # Get table structure
            cursor.execute(f"DESCRIBE {TABLE_NAME}")
            columns = cursor.fetchall()
            
            print("=" * 80)
            print("TABLE STRUCTURE:")
            print("=" * 80)
            for col in columns:
                print(f"{col['Field']:30} {col['Type']:20} {col['Null']:5} {col['Key']:5}")
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
            samples = cursor.fetchall()
            
            print("\n" + "=" * 80)
            print("SAMPLE DATA (first 5 rows):")
            print("=" * 80)
            for i, row in enumerate(samples, 1):
                print(f"\n--- Row {i} ---")
                for key, value in row.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"{key}: {value[:100]}... (truncated)")
                    else:
                        print(f"{key}: {value}")
            
            # Skip total row count - too slow
            print(f"\n" + "=" * 80)
            print("DATA EXPLORATION (sample only)")
            print("=" * 80)
            
            # Deep dive into payload_json structure
            print("\n" + "=" * 80)
            print("DETAILED JSON COLUMN ANALYSIS:")
            print("=" * 80)
            cursor.execute(f"SELECT payload_json FROM {TABLE_NAME} LIMIT 5")
            json_samples = cursor.fetchall()
            
            all_keys = set()
            for i, row in enumerate(json_samples, 1):
                json_data = row['payload_json']
                
                # Handle both dict and string JSON
                if isinstance(json_data, str):
                    try:
                        json_data = json.loads(json_data)
                    except:
                        pass
                
                if isinstance(json_data, dict):
                    print(f"\n--- JSON Sample {i} ---")
                    print(f"Top-level keys: {list(json_data.keys())}")
                    all_keys.update(json_data.keys())
                    
                    # Check for nested structures
                    for key, value in json_data.items():
                        if isinstance(value, dict):
                            print(f"  {key}: dict with keys {list(value.keys())[:10]}")
                        elif isinstance(value, list):
                            print(f"  {key}: list with {len(value)} items")
                            if value and isinstance(value[0], dict):
                                print(f"    First item keys: {list(value[0].keys())[:10]}")
                                print(f"    First item: {value[0]}")
                        elif isinstance(value, str) and len(value) > 50:
                            print(f"  {key}: {value[:50]}...")
                        else:
                            print(f"  {key}: {value}")
            
            print(f"\nAll unique top-level keys found: {sorted(all_keys)}")
            
            # Get one complete JSON to see full structure
            cursor.execute(f"SELECT payload_json FROM {TABLE_NAME} LIMIT 1")
            sample = cursor.fetchone()
            if sample:
                json_data = sample['payload_json']
                if isinstance(json_data, str):
                    json_data = json.loads(json_data)
                
                print("\n" + "=" * 80)
                print("COMPLETE JSON STRUCTURE (first sample, formatted):")
                print("=" * 80)
                print(json.dumps(json_data, indent=2, ensure_ascii=False)[:3000])
            
            # Sample date and price info from already fetched samples
            print("\n" + "=" * 80)
            print("SAMPLE STATS (from first 5 rows):")
            print("=" * 80)
            prices = [row['price'] for row in samples if row.get('price')]
            dates = [row['created_at'] for row in samples if row.get('created_at')]
            if prices:
                print(f"Sample prices: min={min(prices):,}, max={max(prices):,}")
            if dates:
                print(f"Sample dates: {dates[0]} to {dates[-1]}")
            
    finally:
        conn.close()


if __name__ == "__main__":
    explore_table_structure()

