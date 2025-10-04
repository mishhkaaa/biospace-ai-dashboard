"""
NASA Bioscience AI Pipeline - Enhanced Summarization Module
Advanced text summarization using state-of-the-art transformer models
"""

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import os
import time
import torch
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import config
from utils import (
    PipelineLogger, 
    load_and_validate_data, 
    clean_text_data, 
    save_results_with_metadata,
    calculate_text_statistics
)

class AdvancedSummarizer:
    """
    Advanced summarization with multiple models and adaptive parameters
    """
    
    def __init__(self, model_name: str = None, use_gpu: bool = True):
        self.logger = PipelineLogger("Summarizer")
        self.model_name = model_name or config.models.summarization_model
        self.device = 0 if torch.cuda.is_available() and use_gpu else -1
        
        self.logger.info(f"Initializing summarizer with model: {self.model_name}")
        self.logger.info(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
        
        # Initialize model and tokenizer
        try:
            self.summarizer = pipeline(
                "summarization", 
                model=self.model_name,
                device=self.device,
                torch_dtype=torch.float16 if self.device == 0 else torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.logger.success("Summarization model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _calculate_adaptive_params(self, text: str) -> Tuple[int, int]:
        """
        Calculate adaptive summarization parameters based on text length
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (max_length, min_length)
        """
        text_length = len(text.split())
        
        if text_length < 50:
            return 30, 10
        elif text_length < 100:
            return 50, 20
        elif text_length < 200:
            return 80, 30
        elif text_length < 500:
            return 130, 50
        else:
            return 200, 80
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better summarization
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        if len(text) < 10:
            return ""
        
        # Truncate if too long for model
        max_tokens = self.tokenizer.model_max_length - 100  # Leave room for special tokens
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return text
    
    def summarize_single(self, text: str, custom_max_length: Optional[int] = None,
                        custom_min_length: Optional[int] = None) -> str:
        """
        Summarize a single text with error handling and adaptive parameters
        
        Args:
            text: Input text to summarize
            custom_max_length: Override max length
            custom_min_length: Override min length
            
        Returns:
            Summary text
        """
        try:
            # Preprocess text
            clean_text = self._preprocess_text(text)
            if not clean_text:
                return ""
            
            # Calculate adaptive parameters
            max_length, min_length = self._calculate_adaptive_params(clean_text)
            if custom_max_length:
                max_length = custom_max_length
            if custom_min_length:
                min_length = custom_min_length
            
            # Ensure min_length <= max_length and reasonable values
            min_length = min(min_length, max_length - 10)
            max_length = min(max_length, 1024)  # Model limit
            min_length = max(min_length, 5)     # Minimum meaningful length
            
            # Generate summary
            summary = self.summarizer(
                clean_text,
                max_length=int(max_length),
                min_length=int(min_length),
                do_sample=False,
                truncation=True
            )[0]['summary_text']
            
            return summary.strip()
            
        except Exception as e:
            self.logger.warning(f"Error summarizing text: {str(e)}")
            return ""
    
    def summarize_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Summarize multiple texts efficiently in batches
        
        Args:
            texts: List of texts to summarize
            batch_size: Number of texts to process at once
            
        Returns:
            List of summaries
        """
        summaries = []
        
        with tqdm(total=len(texts), desc="Summarizing texts") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_summaries = []
                
                for text in batch:
                    summary = self.summarize_single(text)
                    batch_summaries.append(summary)
                    pbar.update(1)
                
                summaries.extend(batch_summaries)
        
        return summaries

def create_enhanced_summaries(df: pd.DataFrame, text_column: str = 'abstract') -> pd.DataFrame:
    """
    Create enhanced summaries with quality metrics and fallbacks
    
    Args:
        df: Input DataFrame with text data
        text_column: Name of column containing text to summarize
        
    Returns:
        DataFrame with summary and quality metrics
    """
    logger = PipelineLogger("Enhanced Summarization")
    
    # Validate input
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    # Initialize summarizer
    summarizer = AdvancedSummarizer()
    
    # Clean and prepare texts
    texts = df[text_column].fillna("").astype(str).tolist()
    
    # Calculate text statistics
    text_stats = calculate_text_statistics(texts)
    logger.info(f"Text statistics: {text_stats}")
    
    # Generate summaries
    logger.info(f"Generating summaries for {len(texts)} texts")
    start_time = time.time()
    
    summaries = summarizer.summarize_batch(texts)
    
    duration = time.time() - start_time
    logger.success(f"Summarization completed in {duration:.2f} seconds")
    
    # Create output DataFrame
    df_output = df.copy()
    df_output['summary'] = summaries
    
    # Add quality metrics
    df_output['summary_length'] = df_output['summary'].str.len()
    df_output['summary_words'] = df_output['summary'].str.split().str.len()
    df_output['compression_ratio'] = df_output[text_column].str.len() / df_output['summary_length'].replace(0, 1)
    
    # Flag potential quality issues
    df_output['summary_quality'] = 'good'
    df_output.loc[df_output['summary'].str.len() < 10, 'summary_quality'] = 'too_short'
    df_output.loc[df_output['summary'] == "", 'summary_quality'] = 'failed'
    df_output.loc[df_output['compression_ratio'] < 1.5, 'summary_quality'] = 'low_compression'
    
    # Create fallback summaries for failed cases
    failed_mask = df_output['summary_quality'] == 'failed'
    if failed_mask.sum() > 0:
        logger.warning(f"Creating fallback summaries for {failed_mask.sum()} failed cases")
        
        # Use first N words as fallback
        fallback_summaries = df_output.loc[failed_mask, text_column].str.split().str[:30].str.join(' ')
        df_output.loc[failed_mask, 'summary'] = fallback_summaries
        df_output.loc[failed_mask, 'summary_quality'] = 'fallback'
    
    # Log quality distribution
    quality_counts = df_output['summary_quality'].value_counts()
    logger.info(f"Summary quality distribution: {quality_counts.to_dict()}")
    
    return df_output

def main():
    """Main execution function for summarization pipeline"""
    logger = PipelineLogger("Summarization Pipeline")
    
    try:
        # Load and validate data
        data_path = config.get_data_file_path("person_a_metadata.csv", "raw")
        df = load_and_validate_data(data_path, config.data.required_columns)
        
        # Clean text data
        df = clean_text_data(df, config.data.text_columns)
        
        logger.info(f"Loaded {len(df)} papers for summarization")
        
        # Create enhanced summaries
        df_with_summaries = create_enhanced_summaries(df, text_column='abstract')
        
        # Prepare output with essential columns
        essential_columns = ['paper_id', 'title', 'abstract', 'summary', 'year', 
                           'summary_quality', 'summary_length', 'compression_ratio']
        
        # Only include columns that exist
        output_columns = [col for col in essential_columns if col in df_with_summaries.columns]
        df_output = df_with_summaries[output_columns]
        
        # Save results
        output_path = config.get_output_file_path("paper_summaries.csv", "summaries")
        
        metadata = {
            'component': 'summarization',
            'model': config.models.summarization_model,
            'total_papers': len(df_output),
            'summary_quality_distribution': df_output['summary_quality'].value_counts().to_dict(),
            'avg_compression_ratio': df_output['compression_ratio'].mean(),
        }
        
        save_results_with_metadata(df_output, output_path, metadata)
        
        # Create summary statistics
        stats_summary = {
            'total_papers': len(df_output),
            'successful_summaries': (df_output['summary_quality'] == 'good').sum(),
            'failed_summaries': (df_output['summary_quality'] == 'failed').sum(),
            'fallback_summaries': (df_output['summary_quality'] == 'fallback').sum(),
            'average_compression_ratio': float(df_output['compression_ratio'].mean()),
            'average_summary_length': float(df_output['summary_length'].mean()),
        }
        
        logger.success(f"Summarization pipeline completed successfully!")
        logger.info(f"Summary statistics: {stats_summary}")
        
        return df_output, stats_summary
        
    except Exception as e:
        logger.error(f"Summarization pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
