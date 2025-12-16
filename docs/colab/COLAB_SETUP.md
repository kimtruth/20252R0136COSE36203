# Google Colab Setup Guide

This guide will help you run the MapleStory item price prediction project in Google Colab.

## Quick Start

1. **Upload the project to Colab:**
   - Option A: Upload the entire project folder to Google Drive and mount it
   - Option B: Use the provided `maple_meso_colab.ipynb` notebook which handles everything

2. **Open the notebook:**
   - Upload `maple_meso_colab.ipynb` to Google Colab
   - Or create a new notebook and copy the cells from the provided notebook

3. **Run the cells in order**

## Setup Methods

### Method 1: Using the Colab Notebook (Recommended)

The `maple_meso_colab.ipynb` notebook includes:
- Automatic dependency installation
- File upload support (for raw data JSONL files)
- Google Drive integration (optional)
- All preprocessing and training steps

### Method 2: Manual Setup

1. **Install dependencies:**
```python
!pip install -r requirements.txt
```

2. **Upload your data:**
   - Upload raw data files (`.jsonl`) or preprocessed data (`.parquet`)
   - Or mount Google Drive and point to your data files

3. **Run the pipeline:**
```python
from src.pipeline import run_pipeline

# Using uploaded preprocessed data
run_pipeline(
    data_limit=None,
    model_type='lightgbm',  # or 'random_forest', 'gradient_boosting'
    preprocessed_data_path='/content/preprocessed_data.parquet',
    save_processed=True
)
```

## Data Options

### Option 1: Upload Raw Data Files
- Upload `.jsonl` files from `data/raw/` directory
- The notebook will process them automatically

### Option 2: Use Preprocessed Data
- Upload `.parquet` files from `data/processed/` directory
- Faster startup, skips preprocessing step

### Option 3: Connect to Database (Advanced)
- Set environment variables for database connection
- Requires database credentials and network access

## Important Notes

- **File Paths**: Colab uses `/content/` as the default directory
- **Memory**: Large datasets may require Colab Pro for sufficient RAM
- **GPU**: Not required for this project, but Colab Pro provides more resources
- **Session Timeout**: Colab sessions timeout after inactivity - save your work frequently

## Troubleshooting

### Import Errors
- Make sure all project files are uploaded to Colab
- Check that `src/` directory structure is preserved

### Memory Issues
- Use `data_limit` parameter to process smaller batches
- Process data in chunks if needed

### File Not Found
- Check file paths - Colab uses `/content/` not relative paths
- Use absolute paths or `os.chdir()` to set working directory

