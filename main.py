"""Main entry point for MapleStory item price prediction pipeline"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline


def main():
    """
    Main function to run the E2E pipeline.
    
    You can customize the pipeline by modifying the parameters:
    - data_limit: Limit number of rows (None for all data, use small number for testing)
    - model_type: 'random_forest' or 'gradient_boosting'
    """
    # Run pipeline with ALL data from database
    run_pipeline(
        data_limit=None,  # Use all data from database
        model_type='random_forest',
        test_size=0.2,
        val_size=0.1,
        save_processed=True
    )


if __name__ == "__main__":
    main()
