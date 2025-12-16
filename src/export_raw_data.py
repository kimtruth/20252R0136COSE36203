"""Export raw data from database to JSON file(s)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime, date
from src.db_connection import get_db_connection
from src.config import TABLE_NAME


def export_to_json(output_path: str = None, 
                   limit: int = None,
                   chunk_size: int = 10000,
                   format: str = 'jsonl'):
    """
    Export raw data from database to JSON file(s)
    
    Args:
        output_path: Output file path (default: data/raw/raw_data_YYYYMMDD_HHMMSS.jsonl)
        limit: Limit number of rows (None for all data)
        chunk_size: Number of rows to process at a time (for memory efficiency)
        format: Output format - 'jsonl' (JSON Lines, one object per line) or 'json' (single JSON array)
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if limit:
            output_path = f'data/raw/raw_data_{limit}_{timestamp}.jsonl'
        else:
            output_path = f'data/raw/raw_data_{timestamp}.jsonl'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 80)
    print("EXPORTING RAW DATA FROM DATABASE")
    print("=" * 80)
    print(f"\nOutput file: {output_path}")
    print(f"Format: {format.upper()}")
    if limit:
        print(f"Limit: {limit:,} rows")
    else:
        print("Limit: None (all data)")
    print(f"Chunk size: {chunk_size:,} rows")
    print("\nThis may take a while...")
    print("Progress will be shown below.\n")
    
    conn = get_db_connection()
    total_rows = 0
    
    try:
        with conn.cursor() as cursor:
            # Get total count if no limit
            if limit is None:
                cursor.execute(f"SELECT COUNT(*) as count FROM {TABLE_NAME}")
                total_count = cursor.fetchone()['count']
                print(f"Total rows in database: {total_count:,}\n")
            else:
                total_count = limit
            
            # Build query
            query = f"SELECT * FROM {TABLE_NAME}"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            
            # Export based on format
            if format.lower() == 'jsonl':
                # JSON Lines format (one JSON object per line)
                with open(output_path, 'w', encoding='utf-8') as f:
                    processed = 0
                    while True:
                        rows = cursor.fetchmany(chunk_size)
                        if not rows:
                            break
                        
                        for row in rows:
                            # Convert datetime and other non-serializable types
                            json_row = _convert_to_json_serializable(row)
                            f.write(json.dumps(json_row, ensure_ascii=False) + '\n')
                            processed += 1
                        
                        total_rows = processed
                        if limit and processed >= limit:
                            break
                        
                        # Progress update
                        if processed % (chunk_size * 10) == 0:
                            print(f"Exported {processed:,} / {total_count:,} rows...")
                    
                    print(f"Exported {processed:,} / {total_count:,} rows...")
            
            elif format.lower() == 'json':
                # Single JSON array format
                all_rows = []
                processed = 0
                
                while True:
                    rows = cursor.fetchmany(chunk_size)
                    if not rows:
                        break
                    
                    for row in rows:
                        json_row = _convert_to_json_serializable(row)
                        all_rows.append(json_row)
                        processed += 1
                    
                    total_rows = processed
                    if limit and processed >= limit:
                        break
                    
                    # Progress update
                    if processed % (chunk_size * 10) == 0:
                        print(f"Loaded {processed:,} / {total_count:,} rows into memory...")
                
                print(f"Writing {processed:,} rows to JSON file...")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_rows, f, ensure_ascii=False, indent=2)
            
            else:
                raise ValueError(f"Unknown format: {format}. Use 'jsonl' or 'json'")
        
        print("\n" + "=" * 80)
        print("EXPORT COMPLETE!")
        print("=" * 80)
        print(f"\nRaw data saved to: {output_path}")
        print(f"Total rows exported: {total_rows:,}")
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        if format.lower() == 'jsonl':
            print(f"\nFormat: JSON Lines (one JSON object per line)")
            print(f"You can read it line by line or use: json.loads() for each line")
        else:
            print(f"\nFormat: Single JSON array")
            print(f"You can read it with: json.load()")
        
    finally:
        conn.close()


def _convert_to_json_serializable(obj):
    """Convert database row to JSON-serializable format"""
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            result[key] = _convert_to_json_serializable(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, bytes):
        # Handle binary data (e.g., BLOB columns)
        return obj.decode('utf-8', errors='ignore')
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export raw data from database to JSON')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: data/raw/raw_data_TIMESTAMP.jsonl)'
    )
    parser.add_argument(
        '--limit',
        type=lambda x: None if x.lower() == 'none' else int(x),
        default=None,
        help='Limit number of rows (None for all data)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Number of rows to process at a time (default: 10000)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['jsonl', 'json'],
        default='jsonl',
        help='Output format: jsonl (JSON Lines, recommended for large data) or json (single JSON array)'
    )
    
    args = parser.parse_args()
    
    export_to_json(
        output_path=args.output,
        limit=args.limit,
        chunk_size=args.chunk_size,
        format=args.format
    )

