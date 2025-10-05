"""
NASA Bioscience AI Pipeline - Enhanced Keywords & Cluster Summarization Module
Advanced keyword extraction and cluster-level summarization with multiple techniques
"""

import pandas as pd
import numpy as np
from keybert import KeyBERT
from transformers import pipeline
import os
import time
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

from config import config
from utils import (
    PipelineLogger, 
    load_and_validate_data, 
    save_results_with_metadata
)

class AdvancedKeywordExtractor:
    """
    Advanced keyword extraction using multiple techniques
    """
    
    def __init__(self, model_name: str = None):
        self.logger = PipelineLogger("KeywordExtractor")
        self.model_name = model_name or config.models.keyword_model
        
        # Initialize KeyBERT model
        try:
            self.keybert_model = KeyBERT(self.model_name)
            self.logger.success(f"KeyBERT model initialized: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize KeyBERT: {str(e)}")
            raise
        
        # Initialize summarization model
        try:
            self.summarizer = pipeline(
                "summarization", 
                model=config.models.summarization_model,
                device=0 if config.models.summarization_model != "facebook/bart-large-cnn" else -1  # Use CPU for large models
            )
            self.logger.success("Summarization model initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize summarizer: {str(e)}")
            raise
    
    def _preprocess_text_for_keywords(self, text: str) -> str:
        """
        Preprocess text for better keyword extraction
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Handle None, NaN, or non-string inputs safely
        if text is None or pd.isna(text):
            return ""
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_keybert_keywords(self, text: str, top_k: int = 10, 
                                diversity: float = 0.7) -> List[Tuple[str, float]]:
        """
        Extract keywords using KeyBERT with diversity control
        
        Args:
            text: Input text
            top_k: Number of keywords to extract
            diversity: Diversity parameter (0-1, higher = more diverse)
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            clean_text = self._preprocess_text_for_keywords(text)
            if len(clean_text.split()) < 5:  # Too short for meaningful keywords
                return []
            
            # Extract keywords with diversity
            keywords = self.keybert_model.extract_keywords(
                clean_text, 
                keyphrase_ngram_range=(1, 3),  # Allow 1-3 word phrases
                stop_words='english',
                top_n=top_k,  # Fixed: KeyBERT uses 'top_n' not 'top_k'
                use_mmr=True,  # Use Maximal Marginal Relevance for diversity
                diversity=diversity
            )
            
            return keywords
            
        except Exception as e:
            self.logger.warning(f"KeyBERT extraction failed: {str(e)}")
            return []
    
    def extract_frequency_keywords(self, text: str, top_k: int = 10, 
                                 min_word_length: int = 3) -> List[Tuple[str, int]]:
        """
        Extract keywords using frequency analysis with scientific term filtering
        
        Args:
            text: Input text
            top_k: Number of keywords to extract
            min_word_length: Minimum word length
            
        Returns:
            List of (keyword, frequency) tuples
        """
        try:
            clean_text = self._preprocess_text_for_keywords(text)
            
            # Scientific stopwords (extend standard stopwords)
            scientific_stopwords = {
                'study', 'research', 'analysis', 'method', 'result', 'conclusion',
                'finding', 'data', 'experiment', 'test', 'sample', 'group',
                'control', 'treatment', 'effect', 'significant', 'show', 'demonstrate',
                'observe', 'measure', 'evaluate', 'assess', 'examine', 'investigate',
                'report', 'describe', 'present', 'discuss', 'suggest', 'indicate',
                'paper', 'article', 'journal', 'publication', 'author', 'et', 'al'
            }
            
            # Extract words and filter
            words = re.findall(r'\b[a-zA-Z]+\b', clean_text.lower())
            
            # Filter words
            filtered_words = [
                word for word in words 
                if len(word) >= min_word_length 
                and word not in scientific_stopwords
                and not word.isdigit()
            ]
            
            # Count frequencies
            word_counts = Counter(filtered_words)
            
            return word_counts.most_common(top_k)
            
        except Exception as e:
            self.logger.warning(f"Frequency extraction failed: {str(e)}")
            return []
    
    def extract_scientific_terms(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract scientific terms using pattern matching
        
        Args:
            text: Input text
            top_k: Number of terms to extract
            
        Returns:
            List of scientific terms
        """
        try:
            clean_text = self._preprocess_text_for_keywords(text)
            
            # Patterns for scientific terms
            patterns = [
                r'\b[A-Z][a-z]+ [a-z]+\b',  # Species names (Genus species)
                r'\b[A-Z]{2,}\b',  # Acronyms
                r'\b\w+(?:tion|sion|ment|ence|ance|ity|osis|esis)\b',  # Scientific suffixes
                r'\b(?:bio|micro|nano|macro|multi|inter|intra|trans|pre|post|anti|pro)\w+\b',  # Scientific prefixes
                r'\b\w*(?:protein|enzyme|gene|DNA|RNA|cell|tissue|organ)\w*\b',  # Biological terms
                r'\b\w*(?:radiation|gravity|magnetic|electric|thermal|chemical)\w*\b',  # Physics/chemistry terms
            ]
            
            scientific_terms = set()
            for pattern in patterns:
                matches = re.findall(pattern, clean_text, re.IGNORECASE)
                scientific_terms.update([match.lower() for match in matches])
            
            # Filter out common words that matched patterns
            common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use'}
            scientific_terms = [term for term in scientific_terms if term not in common_words and len(term) > 3]
            
            return list(scientific_terms)[:top_k]
            
        except Exception as e:
            self.logger.warning(f"Scientific term extraction failed: {str(e)}")
            return []
    
    def extract_comprehensive_keywords(self, text: str, top_k: int = 15) -> Dict[str, Any]:
        """
        Extract keywords using multiple techniques and combine them
        
        Args:
            text: Input text
            top_k: Total number of keywords to extract
            
        Returns:
            Dictionary with different keyword extraction results
        """
        # Extract using different methods
        keybert_keywords = self.extract_keybert_keywords(text, top_k)
        frequency_keywords = self.extract_frequency_keywords(text, top_k)
        scientific_terms = self.extract_scientific_terms(text, top_k)
        
        # Combine and score keywords
        keyword_scores = {}
        
        # Add KeyBERT keywords (weighted higher)
        for keyword, score in keybert_keywords:
            keyword_scores[keyword] = keyword_scores.get(keyword, 0) + score * 2.0
        
        # Add frequency keywords (weighted medium)
        max_freq = max([freq for _, freq in frequency_keywords], default=1)
        for keyword, freq in frequency_keywords:
            normalized_score = freq / max_freq
            keyword_scores[keyword] = keyword_scores.get(keyword, 0) + normalized_score * 1.0
        
        # Add scientific terms (weighted lower but boost scientific relevance)
        for term in scientific_terms:
            keyword_scores[term] = keyword_scores.get(term, 0) + 0.5
        
        # Sort by combined score
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'combined_keywords': [kw for kw, _ in sorted_keywords[:top_k]],
            'keybert_keywords': [kw for kw, _ in keybert_keywords],
            'frequency_keywords': [kw for kw, _ in frequency_keywords],
            'scientific_terms': scientific_terms,
            'keyword_scores': dict(sorted_keywords[:top_k])
        }

class AdvancedClusterSummarizer:
    """
    Advanced cluster-level summarization with quality control
    """
    
    def __init__(self):
        self.logger = PipelineLogger("ClusterSummarizer")
        
        # Initialize summarization model
        try:
            self.summarizer = pipeline(
                "summarization", 
                model=config.models.summarization_model,
                device=-1  # Use CPU for stability
            )
            self.logger.success("Cluster summarization model initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize cluster summarizer: {str(e)}")
            raise
    
    def _prepare_cluster_text(self, texts: List[str]) -> str:
        """
        Prepare cluster text for summarization
        
        Args:
            texts: List of texts in the cluster
            
        Returns:
            Combined and cleaned text
        """
        # Filter out empty texts and handle non-string values safely
        valid_texts = []
        for text in texts:
            # Skip None, NaN, or non-string values
            if text is None or pd.isna(text):
                continue
            # Convert to string if not already a string
            if not isinstance(text, str):
                try:
                    text = str(text)
                except:
                    continue
            # Check if string is long enough after stripping
            text_stripped = text.strip()
            if len(text_stripped) > 10:
                valid_texts.append(text_stripped)
        
        if not valid_texts:
            return ""
        
        # Combine texts with separators
        combined_text = " [SEP] ".join(valid_texts)
        
        # Truncate if too long (leave room for model processing)
        max_length = 4000  # Conservative limit
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length] + "..."
        
        return combined_text
    
    def summarize_cluster(self, texts: List[str], cluster_id: int) -> Dict[str, Any]:
        """
        Create comprehensive cluster summary
        
        Args:
            texts: List of texts in the cluster
            cluster_id: Cluster identifier
            
        Returns:
            Dictionary with cluster summary information
        """
        try:
            # Prepare text
            combined_text = self._prepare_cluster_text(texts)
            
            if not combined_text:
                return {
                    'cluster_id': cluster_id,
                    'summary': f"Cluster {cluster_id}: Insufficient text for summarization",
                    'summary_quality': 'failed',
                    'text_count': len(texts),
                    'valid_text_count': 0
                }
            
            # Calculate adaptive parameters based on cluster size
            # Count valid texts safely (handle floats, NaN, None)
            valid_count = 0
            for t in texts:
                if t is None or pd.isna(t):
                    continue
                if not isinstance(t, str):
                    try:
                        t = str(t)
                    except:
                        continue
                if len(t.strip()) > 10:
                    valid_count += 1
            cluster_size = valid_count
            
            if cluster_size <= 2:
                max_length = config.models.summary_max_length // 2
                min_length = config.models.summary_min_length // 2
            elif cluster_size <= 5:
                max_length = int(config.models.summary_max_length * 0.8)
                min_length = int(config.models.summary_min_length * 0.8)
            else:
                max_length = config.models.cluster_summary_max_length
                min_length = config.models.cluster_summary_min_length
            
            # Ensure min_length <= max_length
            min_length = min(min_length, max_length - 10)
            
            # Generate summary
            summary_result = self.summarizer(
                combined_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )[0]['summary_text']
            
            # Quality assessment
            summary_quality = 'good'
            if len(summary_result) < min_length * 0.8:
                summary_quality = 'short'
            elif 'cluster' in summary_result.lower() and str(cluster_id) not in summary_result:
                # Add cluster context if missing
                summary_result = f"Cluster {cluster_id}: {summary_result}"
            
            return {
                'cluster_id': cluster_id,
                'summary': summary_result.strip() if isinstance(summary_result, str) else str(summary_result),
                'summary_quality': summary_quality,
                'summary_length': len(summary_result),
                'text_count': len(texts),
                'valid_text_count': cluster_size,
                'compression_ratio': len(combined_text) / len(summary_result) if summary_result else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Cluster {cluster_id} summarization failed: {str(e)}")
            
            # Fallback summary
            fallback_summary = f"Cluster {cluster_id}: Research cluster containing {len(texts)} studies"
            if texts and len(texts) > 0:
                # Use first few words from first valid text (handle non-strings safely)
                first_valid = ""
                for t in texts:
                    if t is None or pd.isna(t):
                        continue
                    if not isinstance(t, str):
                        try:
                            t = str(t)
                        except:
                            continue
                    t_stripped = t.strip()
                    if len(t_stripped) > 20:
                        first_valid = t_stripped
                        break
                if first_valid:
                    fallback_summary += f" focusing on {' '.join(first_valid.split()[:10])}..."
            
            # Calculate valid text count safely
            valid_text_count = 0
            for t in texts:
                if t is None or pd.isna(t):
                    continue
                if not isinstance(t, str):
                    try:
                        t = str(t)
                    except:
                        continue
                if len(t.strip()) > 10:
                    valid_text_count += 1
            
            return {
                'cluster_id': cluster_id,
                'summary': fallback_summary,
                'summary_quality': 'fallback',
                'text_count': len(texts),
                'valid_text_count': valid_text_count
            }

def create_enhanced_cluster_analysis(summaries_path: str, clusters_path: str) -> pd.DataFrame:
    """
    Create enhanced cluster analysis with keywords and summaries
    
    Args:
        summaries_path: Path to paper summaries
        clusters_path: Path to cluster assignments
        
    Returns:
        DataFrame with cluster analysis
    """
    logger = PipelineLogger("Enhanced Cluster Analysis")
    
    # Load data
    summaries_df = load_and_validate_data(summaries_path, ['paper_id', 'summary'])
    clusters_df = load_and_validate_data(clusters_path, ['paper_id', 'cluster_id'])
    
    # Merge data
    df = summaries_df.merge(clusters_df, on='paper_id', how='inner')
    
    # Check if journal information is available in summaries
    has_journal_info = 'journal' in summaries_df.columns
    if has_journal_info:
        logger.info("Journal information detected - will include in cluster analysis")
    
    logger.info(f"Loaded {len(df)} papers with summaries and cluster assignments")
    
    # Initialize extractors
    keyword_extractor = AdvancedKeywordExtractor()
    cluster_summarizer = AdvancedClusterSummarizer()
    
    # Process each cluster
    cluster_results = []
    
    unique_clusters = sorted(df['cluster_id'].unique())
    logger.info(f"Processing {len(unique_clusters)} clusters")
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise points if using DBSCAN
            continue
            
        logger.info(f"Processing cluster {cluster_id}")
        
        # Get cluster data
        cluster_data = df[df['cluster_id'] == cluster_id]
        cluster_texts = cluster_data['summary'].tolist()
        
        # Add journal context if available
        journal_context = ""
        if has_journal_info:
            cluster_journals = cluster_data['journal'].dropna()
            if len(cluster_journals) > 0:
                journal_counts = cluster_journals.value_counts()
                top_journals = journal_counts.head(3).index.tolist()
                if top_journals:
                    journal_context = f" (primarily published in: {', '.join(top_journals)})"
        
        # Extract keywords - safely combine text handling floats/NaN/None
        valid_cluster_texts = []
        for text in cluster_texts:
            if text is None or pd.isna(text):
                continue
            if not isinstance(text, str):
                try:
                    text = str(text)
                except:
                    continue
            text_stripped = text.strip()
            if text_stripped:  # Add any non-empty string
                valid_cluster_texts.append(text_stripped)
        
        combined_text = " ".join(valid_cluster_texts)
        keyword_results = keyword_extractor.extract_comprehensive_keywords(combined_text)
        
        # Create cluster summary
        summary_results = cluster_summarizer.summarize_cluster(cluster_texts, cluster_id)
        
        # Enhance summary with journal context if available
        enhanced_summary = summary_results['summary']
        if journal_context:
            # Add journal context to summary if it's meaningful
            if not journal_context.lower() in enhanced_summary.lower():
                enhanced_summary = f"{enhanced_summary}{journal_context}"
        
        # Combine results
        cluster_result = {
            'cluster_id': cluster_id,
            'paper_count': len(cluster_texts),
            'keywords': ", ".join([str(kw) for kw in keyword_results['combined_keywords']]),
            'keybert_keywords': ", ".join([str(kw) for kw in keyword_results['keybert_keywords']]),
            'scientific_terms': ", ".join([str(kw) for kw in keyword_results['scientific_terms']]),
            'cluster_summary': enhanced_summary,
            'summary_quality': summary_results['summary_quality'],
            'summary_length': len(enhanced_summary),
            'compression_ratio': summary_results.get('compression_ratio', 0),
            'keyword_scores': keyword_results['keyword_scores']
        }
        
        # Add journal information if available
        if has_journal_info:
            cluster_journals = cluster_data['journal'].dropna()
            if len(cluster_journals) > 0:
                journal_counts = cluster_journals.value_counts()
                cluster_result['primary_journals'] = ", ".join(journal_counts.head(3).index.tolist())
                cluster_result['journal_diversity'] = len(journal_counts)
            else:
                cluster_result['primary_journals'] = ""
                cluster_result['journal_diversity'] = 0
        
        cluster_results.append(cluster_result)
    
    # Create output DataFrame
    df_output = pd.DataFrame(cluster_results)
    
    # Add cluster quality metrics
    df_output['cluster_quality'] = 'good'
    df_output.loc[df_output['paper_count'] < config.clustering.min_cluster_size, 'cluster_quality'] = 'small'
    df_output.loc[df_output['summary_quality'].isin(['failed', 'fallback']), 'cluster_quality'] = 'low_quality'
    
    # Sort by cluster_id
    df_output = df_output.sort_values('cluster_id').reset_index(drop=True)
    
    logger.success(f"Cluster analysis completed for {len(df_output)} clusters")
    
    return df_output

def main():
    """Main execution function for keywords and cluster analysis pipeline"""
    logger = PipelineLogger("Keywords & Cluster Analysis Pipeline")
    
    try:
        # Load data
        summaries_path = config.get_output_file_path("paper_summaries.csv", "summaries")
        clusters_path = config.get_output_file_path("paper_clusters.csv", "clusters")
        
        # Create enhanced cluster analysis
        df_cluster_analysis = create_enhanced_cluster_analysis(summaries_path, clusters_path)
        
        # Save results
        output_path = config.get_output_file_path("cluster_keywords_summaries.csv", "clusters")
        
        metadata = {
            'component': 'keywords_and_summarization',
            'keyword_model': config.models.keyword_model,
            'summarization_model': config.models.summarization_model,
            'total_clusters': len(df_cluster_analysis),
            'small_clusters': (df_cluster_analysis['cluster_quality'] == 'small').sum(),
            'avg_keywords_per_cluster': df_cluster_analysis['keywords'].str.split(',').str.len().mean(),
            'avg_summary_length': df_cluster_analysis['summary_length'].mean()
        }
        
        save_results_with_metadata(df_cluster_analysis, output_path, metadata)
        
        # Create summary statistics
        stats = {
            'total_clusters': len(df_cluster_analysis),
            'avg_papers_per_cluster': float(df_cluster_analysis['paper_count'].mean()),
            'small_clusters': int((df_cluster_analysis['cluster_quality'] == 'small').sum()),
            'high_quality_summaries': int((df_cluster_analysis['summary_quality'] == 'good').sum()),
            'avg_keywords_per_cluster': float(df_cluster_analysis['keywords'].str.split(',').str.len().mean()),
        }
        
        logger.success("Keywords & cluster analysis pipeline completed successfully!")
        logger.info(f"Statistics: {stats}")
        
        return df_cluster_analysis, stats
        
    except Exception as e:
        logger.error(f"Keywords & cluster analysis pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
