"""
NASA Bioscience AI Pipeline - Enhanced Embeddings Module
Advanced semantic embeddings using multiple state-of-the-art models
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
import time
import torch
from typing import List, Dict, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from utils import PipelineLogger, validate_dataframe, save_results_with_metadata, NumpyEncoder
from config import ModelConfig, config
import warnings
warnings.filterwarnings('ignore')
from utils import (
    PipelineLogger, 
    load_and_validate_data, 
    save_results_with_metadata,
    calculate_text_statistics
)

class AdvancedEmbeddingGenerator:
    """
    Advanced embedding generation with multiple models and optimization
    """
    
    def __init__(self, model_name: str = None, use_scientific_model: bool = True, use_gpu: bool = True):
        self.logger = PipelineLogger("EmbeddingGenerator")
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Choose model based on scientific content
        if use_scientific_model and 'scibert' in config.models.scientific_embedding_model:
            self.model_name = config.models.scientific_embedding_model
            self.logger.info("Using scientific domain model for better accuracy")
        else:
            self.model_name = model_name or config.models.embedding_model
        
        self.logger.info(f"Initializing embedding model: {self.model_name}")
        self.logger.info(f"Using device: {'GPU' if self.use_gpu else 'CPU'}")
        
        # Initialize model
        try:
            device = 'cuda' if self.use_gpu else 'cpu'
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            self.logger.success(f"Model loaded successfully (dimension: {self.embedding_dim})")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts for better embeddings
        
        Args:
            texts: List of input texts
            
        Returns:
            List of cleaned texts
        """
        processed_texts = []
        
        for text in texts:
            if not text or not isinstance(text, str):
                processed_texts.append("")
                continue
            
            # Basic cleaning
            clean_text = text.strip()
            
            # Remove excessive whitespace
            clean_text = ' '.join(clean_text.split())
            
            # Truncate if too long (typical limit is 512 tokens)
            words = clean_text.split()
            if len(words) > 400:  # Conservative limit
                clean_text = ' '.join(words[:400]) + "..."
            
            processed_texts.append(clean_text)
        
        return processed_texts
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, 
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings
        """
        # Preprocess texts
        processed_texts = self._preprocess_texts(texts)
        
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(processed_texts):
            if text and len(text.strip()) > 0:
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            self.logger.warning("No valid texts found for embedding generation")
            return np.zeros((len(texts), self.embedding_dim))
        
        self.logger.info(f"Generating embeddings for {len(valid_texts)}/{len(texts)} valid texts")
        
        # Generate embeddings for valid texts
        try:
            embeddings_valid = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity calculations
            )
        except Exception as e:
            self.logger.error(f"Error during embedding generation: {str(e)}")
            raise
        
        # Create full embeddings array with zeros for invalid texts
        embeddings_full = np.zeros((len(texts), self.embedding_dim))
        embeddings_full[valid_indices] = embeddings_valid
        
        return embeddings_full
    
    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity matrix for embeddings
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Similarity matrix
        """
        self.logger.info("Calculating similarity matrix")
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def find_similar_papers(self, embeddings: np.ndarray, paper_ids: List[str], 
                           top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find most similar papers for each paper
        
        Args:
            embeddings: Array of embeddings
            paper_ids: List of paper IDs
            top_k: Number of similar papers to find
            
        Returns:
            Dictionary mapping paper_id to list of (similar_paper_id, similarity_score)
        """
        similarity_matrix = self.calculate_similarity_matrix(embeddings)
        similar_papers = {}
        
        for i, paper_id in enumerate(paper_ids):
            # Get similarities for this paper (excluding self)
            similarities = similarity_matrix[i]
            
            # Get top-k most similar (excluding self)
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            similar_papers[paper_id] = [
                (paper_ids[idx], float(similarities[idx]))
                for idx in similar_indices
                if similarities[idx] > 0.1  # Minimum similarity threshold
            ]
        
        return similar_papers

def create_enhanced_embeddings(df: pd.DataFrame, text_column: str = 'summary') -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Create enhanced embeddings with quality analysis and similarity calculations
    
    Args:
        df: Input DataFrame with text data
        text_column: Name of column containing text to embed
        
    Returns:
        Tuple of (DataFrame with embeddings, embeddings array, similarity info)
    """
    logger = PipelineLogger("Enhanced Embeddings")
    
    # Validate input
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    # Initialize embedding generator
    embedding_generator = AdvancedEmbeddingGenerator()
    
    # Prepare texts
    texts = df[text_column].fillna("").astype(str).tolist()
    
    # Calculate text statistics
    text_stats = calculate_text_statistics(texts)
    logger.info(f"Text statistics: {text_stats}")
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} texts")
    start_time = time.time()
    
    embeddings = embedding_generator.generate_embeddings(texts)
    
    duration = time.time() - start_time
    logger.success(f"Embedding generation completed in {duration:.2f} seconds")
    
    # Calculate quality metrics
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    zero_embeddings = (embedding_norms == 0).sum()
    
    logger.info(f"Generated embeddings: {embeddings.shape}")
    logger.info(f"Zero embeddings (failed): {zero_embeddings}")
    logger.info(f"Average embedding norm: {embedding_norms.mean():.4f}")
    
    # Find similar papers
    logger.info("Finding similar papers")
    similar_papers = embedding_generator.find_similar_papers(
        embeddings, df['paper_id'].tolist(), top_k=5
    )
    
    # Create output DataFrame
    df_output = df.copy()
    df_output['embedding'] = embeddings.tolist()
    df_output['embedding_norm'] = embedding_norms
    df_output['embedding_quality'] = 'good'
    df_output.loc[embedding_norms == 0, 'embedding_quality'] = 'failed'
    df_output.loc[embedding_norms < 0.5, 'embedding_quality'] = 'low_quality'
    
    # Add dimensionality reduction for visualization
    if embeddings.shape[0] > 2:
        logger.info("Computing PCA for visualization")
        try:
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings)
            df_output['pca_x'] = embeddings_2d[:, 0]
            df_output['pca_y'] = embeddings_2d[:, 1]
            explained_variance = pca.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance: {explained_variance:.3f}")
        except Exception as e:
            logger.warning(f"PCA computation failed: {str(e)}")
    
    # Prepare similarity information
    similarity_info = {
        'similar_papers': similar_papers,
        'embedding_stats': {
            'dimension': embeddings.shape[1],
            'total_embeddings': embeddings.shape[0],
            'failed_embeddings': zero_embeddings,
            'average_norm': float(embedding_norms.mean()),
            'std_norm': float(embedding_norms.std()),
        }
    }
    
    return df_output, embeddings, similarity_info

def save_embeddings_multiple_formats(df: pd.DataFrame, embeddings: np.ndarray, 
                                   similarity_info: Dict, base_path: str):
    """
    Save embeddings in multiple formats for different use cases
    
    Args:
        df: DataFrame with embedding data
        embeddings: NumPy array of embeddings
        similarity_info: Dictionary with similarity information
        base_path: Base path for output files
    """
    logger = PipelineLogger("Embedding Saver")
    
    # Save as JSONL (for pipeline compatibility)
    jsonl_path = base_path.replace('.jsonl', '.jsonl')
    embedding_data = []
    
    for _, row in df.iterrows():
        embedding_data.append({
            'paper_id': row['paper_id'],
            'embedding': row['embedding'],
            'embedding_quality': row['embedding_quality'],
            'embedding_norm': float(row['embedding_norm'])
        })
    
    save_results_with_metadata(
        embedding_data, 
        jsonl_path, 
        {
            'component': 'embeddings',
            'format': 'jsonl',
            'model': config.models.embedding_model,
            **similarity_info['embedding_stats']
        }
    )
    
    # Save embeddings as NumPy array (for fast loading)
    npy_path = jsonl_path.replace('.jsonl', '_embeddings.npy')
    np.save(npy_path, embeddings)
    logger.success(f"Saved embeddings array to {npy_path}")
    
    # Save similarity information
    similarity_path = jsonl_path.replace('.jsonl', '_similarities.json')
    with open(similarity_path, 'w') as f:
        json.dump(similarity_info, f, indent=2, cls=NumpyEncoder)
    logger.success(f"Saved similarity information to {similarity_path}")
    
    # Save metadata CSV for easy inspection
    metadata_path = jsonl_path.replace('.jsonl', '_metadata.csv')
    metadata_cols = ['paper_id', 'embedding_quality', 'embedding_norm']
    if 'pca_x' in df.columns:
        metadata_cols.extend(['pca_x', 'pca_y'])
    
    df[metadata_cols].to_csv(metadata_path, index=False)
    logger.success(f"Saved embedding metadata to {metadata_path}")

def main():
    """Main execution function for embeddings pipeline"""
    logger = PipelineLogger("Embeddings Pipeline")
    
    try:
        # Load summarized data
        summaries_path = config.get_output_file_path("paper_summaries.csv", "summaries")
        df = load_and_validate_data(summaries_path, ['paper_id', 'summary'])
        
        logger.info(f"Loaded {len(df)} papers with summaries")
        
        # Create enhanced embeddings
        df_with_embeddings, embeddings, similarity_info = create_enhanced_embeddings(df, 'summary')
        
        # Save embeddings in multiple formats
        output_path = config.get_output_file_path("paper_embeddings.jsonl", "embeddings")
        save_embeddings_multiple_formats(df_with_embeddings, embeddings, similarity_info, output_path)
        
        # Log completion statistics
        quality_counts = df_with_embeddings['embedding_quality'].value_counts()
        logger.success("Embeddings pipeline completed successfully!")
        logger.info(f"Embedding quality distribution: {quality_counts.to_dict()}")
        
        return df_with_embeddings, embeddings, similarity_info
        
    except Exception as e:
        logger.error(f"Embeddings pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
