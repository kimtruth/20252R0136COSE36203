"""End-to-end pipeline for MapleStory item price prediction"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.preprocess import load_and_preprocess_data
from src.train import train_price_prediction_model
import pandas as pd


def run_pipeline(
    data_limit: int = None,
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    val_size: float = 0.1,
    preprocessed_data_path: str = None,
    save_processed: bool = True
):
    """
    Run complete E2E pipeline: Extract -> Transform -> Load -> Train -> Evaluate -> Save
    
    Args:
        data_limit: Limit number of rows to load from database (None for all)
        model_type: Type of model ('random_forest' or 'gradient_boosting')
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        preprocessed_data_path: Path to preprocessed data file (if None, will load from DB)
        save_processed: Whether to save preprocessed data
    """
    print("=" * 80)
    print("MAPLESTORY ITEM PRICE PREDICTION - E2E PIPELINE")
    print("=" * 80)
    
    # Step 1: Extract and Transform
    print("\n" + "=" * 80)
    print("STEP 1: EXTRACT & TRANSFORM")
    print("=" * 80)
    
    if preprocessed_data_path and os.path.exists(preprocessed_data_path):
        print(f"Loading preprocessed data from {preprocessed_data_path}...")
        df = pd.read_parquet(preprocessed_data_path)
        print(f"Loaded {len(df)} rows")
    else:
        processed_path = None
        if save_processed:
            processed_path = 'data/processed/preprocessed_data.parquet'
            if data_limit:
                processed_path = f'data/processed/preprocessed_data_{data_limit}.parquet'
        
        df = load_and_preprocess_data(limit=data_limit, save_path=processed_path)
    
    print(f"\nPreprocessed data shape: {df.shape}")
    print(f"Target (price) statistics:")
    print(df['price'].describe())
    
    # Step 2: Train
    print("\n" + "=" * 80)
    print("STEP 2: TRAIN MODEL")
    print("=" * 80)
    
    results = train_price_prediction_model(
        df,
        target_col='price',
        model_type=model_type,
        test_size=test_size,
        val_size=val_size,
        save_dir='models',
        use_time_split=False  # Use random sampling to learn time patterns across all periods
    )
    
    # Step 3: Summary
    print("\n" + "=" * 80)
    print("STEP 3: PIPELINE SUMMARY")
    print("=" * 80)
    print(f"\nModel Type: {model_type}")
    print(f"\nFinal Test Set Performance:")
    print(f"  RMSE: {results['metrics']['test_rmse']:,.2f}")
    print(f"  MAE: {results['metrics']['test_mae']:,.2f}")
    print(f"  R²: {results['metrics']['test_r2']:.4f}")
    print(f"  MAPE: {results['metrics']['test_mape']:.2f}%")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nModel and artifacts saved in 'models/' directory")
    print("Preprocessed data saved in 'data/processed/' directory")
    
    # Generate report
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING REPORT")
    print("=" * 80)
    try:
        from src.generate_report import generate_report
        report_filename = f'TRAINING_REPORT_{model_type}.md' if model_type != 'random_forest' else 'TRAINING_REPORT.md'
        report_path = generate_report(models_dir='models', output_path=report_filename)
        print(f"\nTraining report generated: {report_path}")
        
        # Generate comparison if multiple models exist
        from src.compare_models import compare_models
        compare_models(models_dir='models', output_path='MODEL_COMPARISON.md')
        print(f"Model comparison report generated: MODEL_COMPARISON.md")
    except Exception as e:
        print(f"Warning: Could not generate report: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MapleStory Item Price Prediction Pipeline')
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of rows to load from database (for testing)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'gradient_boosting', 'lightgbm'],
        help='Model type to train'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Proportion of data for validation set'
    )
    parser.add_argument(
        '--preprocessed-data',
        type=str,
        default=None,
        help='Path to preprocessed data file (skips data loading if provided)'
    )
    parser.add_argument(
        '--no-save-processed',
        action='store_true',
        help='Do not save preprocessed data'
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        data_limit=args.limit,
        model_type=args.model,
        test_size=args.test_size,
        val_size=args.val_size,
        preprocessed_data_path=args.preprocessed_data,
        save_processed=not args.no_save_processed
    )


if __name__ == "__main__":
    main()

