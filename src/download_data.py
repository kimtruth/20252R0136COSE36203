"""Download and preprocess all data from database for faster training"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_and_preprocess_data
import argparse


def download_and_preprocess_data(limit=None, output_path=None):
    """
    Download and preprocess all data from database
    
    Args:
        limit: Limit number of rows (None for all data)
        output_path: Path to save preprocessed data (default: data/processed/preprocessed_data.parquet)
    """
    if output_path is None:
        if limit:
            output_path = f'data/processed/preprocessed_data_{limit}.parquet'
        else:
            output_path = 'data/processed/preprocessed_data.parquet'
    
    print("=" * 80)
    print("DOWNLOADING AND PREPROCESSING DATA")
    print("=" * 80)
    print(f"\nOutput file: {output_path}")
    if limit:
        print(f"Limit: {limit:,} rows")
    else:
        print("Limit: None (all data)")
    
    print("\nThis may take a while...")
    print("Progress will be shown below.\n")
    
    df = load_and_preprocess_data(limit=limit, save_path=output_path)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nPreprocessed data saved to: {output_path}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nFile size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"\nNow you can train models faster using:")
    print(f"  uv run python src/pipeline.py --preprocessed-data {output_path} --model lightgbm")
    print(f"\nOr the pipeline will automatically use this file if it exists.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and preprocess data from database')
    parser.add_argument(
        '--limit',
        type=lambda x: None if x.lower() == 'none' else int(x),
        default=None,
        help='Limit number of rows (None for all data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: data/processed/preprocessed_data.parquet)'
    )
    
    args = parser.parse_args()
    
    download_and_preprocess_data(limit=args.limit, output_path=args.output)

