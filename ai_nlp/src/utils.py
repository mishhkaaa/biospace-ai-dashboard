"""
NASA Bioscience AI Pipeline Utilities
Shared utility functions for data processing, validation, and pipeline management
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
from rich.console import Console
from rich.logging import RichHandler

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types and pandas DataFrames"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return super(NumpyEncoder, self).default(obj)
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

# Initialize rich console for beautiful output
console = Console()

class PipelineLogger:
    """Custom logger for the NASA Bioscience AI Pipeline"""
    
    def __init__(self, name: str = "nasa_pipeline", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
        try:
            console.print(f"ℹ️  {message}", style="blue")
        except UnicodeEncodeError:
            console.print(f"INFO: {message}", style="blue")
    
    def success(self, message: str):
        self.logger.info(message)
        try:
            console.print(f"✅ {message}", style="green")
        except UnicodeEncodeError:
            console.print(f"SUCCESS: {message}", style="green")
    
    def warning(self, message: str):
        self.logger.warning(message)
        try:
            console.print(f"⚠️  {message}", style="yellow")
        except UnicodeEncodeError:
            console.print(f"WARNING: {message}", style="yellow")
    
    def error(self, message: str):
        self.logger.error(message)
        try:
            console.print(f"❌ {message}", style="red")
        except UnicodeEncodeError:
            console.print(f"ERROR: {message}", style="red")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                      name: str = "DataFrame") -> bool:
    """
    Validate that a DataFrame has required columns and basic data quality
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for logging purposes
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger = PipelineLogger()
    
    # Check if DataFrame is empty
    if df.empty:
        logger.error(f"{name} is empty")
        return False
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"{name} missing required columns: {missing_columns}")
        return False
    
    # Check for excessive null values
    for col in required_columns:
        null_percentage = df[col].isnull().sum() / len(df) * 100
        if null_percentage > 50:
            logger.warning(f"{name} column '{col}' has {null_percentage:.1f}% null values")
    
    logger.success(f"{name} validation passed ({len(df)} rows, {len(df.columns)} columns)")
    return True

def load_and_validate_data(filepath: str, required_columns: List[str]) -> pd.DataFrame:
    """
    Load CSV data with validation and error handling
    
    Args:
        filepath: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        pd.DataFrame: Loaded and validated DataFrame
    """
    logger = PipelineLogger()
    
    try:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        if validate_dataframe(df, required_columns, Path(filepath).name):
            return df
        else:
            raise ValueError(f"Data validation failed for {filepath}")
            
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        raise

def clean_text_data(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Clean and preprocess text data for NLP tasks
    
    Args:
        df: Input DataFrame
        text_columns: List of text column names to clean
        
    Returns:
        pd.DataFrame: DataFrame with cleaned text
    """
    logger = PipelineLogger()
    df_clean = df.copy()
    
    for col in text_columns:
        if col in df_clean.columns:
            logger.info(f"Cleaning text column: {col}")
            
            # Fill NaN values
            df_clean[col] = df_clean[col].fillna("")
            
            # Basic text cleaning
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].str.strip()
            
            # Remove excessive whitespace
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
            
            # Remove very short texts (likely errors)
            min_length = 10 if col == 'abstract' else 5
            short_text_mask = df_clean[col].str.len() < min_length
            if short_text_mask.sum() > 0:
                logger.warning(f"Found {short_text_mask.sum()} very short texts in {col}")
                df_clean.loc[short_text_mask, col] = ""
    
    return df_clean

def extract_publication_year(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """
    Extract publication year from various date formats
    
    Args:
        df: Input DataFrame
        date_columns: List of potential date column names
        
    Returns:
        pd.DataFrame: DataFrame with standardized 'year' column
    """
    logger = PipelineLogger()
    df_with_year = df.copy()
    
    # If year column already exists and is valid, use it
    if 'year' in df_with_year.columns:
        try:
            df_with_year['year'] = pd.to_numeric(df_with_year['year'], errors='coerce')
            valid_years = df_with_year['year'].notna()
            if valid_years.sum() > len(df_with_year) * 0.8:  # If 80% have valid years
                logger.info("Using existing 'year' column")
                return df_with_year
        except:
            pass
    
    # Try to extract year from other date columns
    for col in date_columns:
        if col in df_with_year.columns:
            logger.info(f"Attempting to extract year from {col}")
            try:
                # Try direct numeric conversion first
                df_with_year['year'] = pd.to_numeric(df_with_year[col], errors='coerce')
                
                # If that doesn't work, try date parsing
                if df_with_year['year'].isna().all():
                    dates = pd.to_datetime(df_with_year[col], errors='coerce')
                    df_with_year['year'] = dates.dt.year
                
                # Validate year range (reasonable for scientific publications)
                current_year = datetime.now().year
                valid_year_mask = (df_with_year['year'] >= 1950) & (df_with_year['year'] <= current_year)
                
                if valid_year_mask.sum() > 0:
                    logger.success(f"Extracted {valid_year_mask.sum()} valid years from {col}")
                    break
                    
            except Exception as e:
                logger.warning(f"Could not extract year from {col}: {str(e)}")
                continue
    
    # Fill missing years with median or placeholder
    if 'year' not in df_with_year.columns or df_with_year['year'].isna().all():
        logger.warning("No valid years found, using placeholder")
        df_with_year['year'] = 'unknown'
    else:
        # Fill missing with median year
        median_year = df_with_year['year'].median()
        df_with_year['year'] = df_with_year['year'].fillna(median_year)
    
    return df_with_year

def save_results_with_metadata(data: Any, filepath: str, metadata: Optional[Dict] = None):
    """
    Save results with metadata for pipeline tracking
    
    Args:
        data: Data to save (DataFrame, dict, list)
        filepath: Output file path
        metadata: Optional metadata to include
    """
    logger = PipelineLogger()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare metadata
    meta = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_version': '2.0',
        'file_path': filepath,
        **(metadata or {})
    }
    
    # Save based on file extension
    file_ext = Path(filepath).suffix.lower()
    
    try:
        if file_ext == '.csv' and isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
            meta['rows'] = len(data)
            meta['columns'] = list(data.columns)
            
        elif file_ext == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                
        elif file_ext == '.jsonl':
            with open(filepath, 'w', encoding='utf-8') as f:
                if isinstance(data, pd.DataFrame):
                    for _, row in data.iterrows():
                        json.dump(row.to_dict(), f, ensure_ascii=False, cls=NumpyEncoder)
                        f.write('\n')
                else:
                    for item in data:
                        json.dump(item, f, ensure_ascii=False, cls=NumpyEncoder)
                        f.write('\n')
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Save metadata
        meta_filepath = filepath.replace(file_ext, '_metadata.json')
        with open(meta_filepath, 'w') as f:
            json.dump(meta, f, indent=2, cls=NumpyEncoder)
        
        logger.success(f"Saved results to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving {filepath}: {str(e)}")
        raise

def generate_pipeline_report(results: Dict[str, Any]) -> str:
    """
    Generate a comprehensive pipeline execution report
    
    Args:
        results: Dictionary containing pipeline results and metrics
        
    Returns:
        str: Formatted report text
    """
    # Create rich table for results summary
    table = Table(title="🚀 NASA Bioscience AI Pipeline Results", show_header=True)
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Output", style="blue")
    table.add_column("Metrics", style="yellow")
    
    for component, data in results.items():
        status = "✅ Complete" if data.get('success', False) else "❌ Failed"
        output = data.get('output_file', 'N/A')
        metrics = data.get('metrics', 'N/A')
        table.add_row(component, status, output, str(metrics))
    
    console.print(table)
    
    # Generate text report
    report_lines = [
        "=" * 60,
        "NASA BIOSCIENCE AI PIPELINE EXECUTION REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    for component, data in results.items():
        report_lines.extend([
            f"Component: {component}",
            f"Status: {'SUCCESS' if data.get('success', False) else 'FAILED'}",
            f"Output: {data.get('output_file', 'N/A')}",
            f"Metrics: {data.get('metrics', 'N/A')}",
            f"Duration: {data.get('duration', 'N/A')}",
            "-" * 40,
        ])
    
    return "\n".join(report_lines)

def calculate_text_statistics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate comprehensive text statistics for analysis
    
    Args:
        texts: List of text strings
        
    Returns:
        Dict with text statistics
    """
    if not texts:
        return {}
    
    # Filter out empty texts
    valid_texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 0]
    
    if not valid_texts:
        return {}
    
    lengths = [len(t) for t in valid_texts]
    word_counts = [len(t.split()) for t in valid_texts]
    
    return {
        'total_texts': len(texts),
        'valid_texts': len(valid_texts),
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'avg_words': np.mean(word_counts),
        'median_words': np.median(word_counts),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'empty_texts': len(texts) - len(valid_texts)
    }

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
