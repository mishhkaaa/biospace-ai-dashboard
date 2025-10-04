"""
NASA Bioscience AI Pipeline Configuration
Central configuration for all pipeline components
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Configuration for AI models used in the pipeline"""
    summarization_model: str = "sshleifer/distilbart-cnn-12-6"  # More reliable for demo
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    scientific_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fallback to stable model
    keyword_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Generation parameters
    summary_max_length: int = 130
    summary_min_length: int = 30
    cluster_summary_max_length: int = 200
    cluster_summary_min_length: int = 60

@dataclass
class ClusterConfig:
    """Configuration for clustering analysis"""
    default_n_clusters: int = 8
    min_cluster_size: int = 3
    max_cluster_size: int = 100
    use_adaptive_clustering: bool = True
    clustering_algorithm: str = "kmeans"  # Options: kmeans, hdbscan, agglomerative

@dataclass
class InsightsConfig:
    """Configuration for insights generation"""
    knowledge_gap_threshold: int = 3
    trend_analysis_years: int = 20
    consensus_threshold: float = 0.7
    disagreement_threshold: float = 0.3
    min_papers_for_trend: int = 5

@dataclass
class DataConfig:
    """Configuration for data processing"""
    required_columns: List[str] = None
    text_columns: List[str] = None
    date_columns: List[str] = None
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ['paper_id', 'title', 'abstract']
        if self.text_columns is None:
            self.text_columns = ['abstract', 'title', 'finding']
        if self.date_columns is None:
            self.date_columns = ['year', 'publication_date', 'pub_date', 'date']

@dataclass
class PathConfig:
    """Configuration for file paths"""
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = "data"
    output_dir: str = "outputs"
    models_dir: str = "models"
    
    @property
    def raw_data_dir(self) -> str:
        return os.path.join(self.base_dir, self.data_dir, "raw")
    
    @property
    def processed_data_dir(self) -> str:
        return os.path.join(self.base_dir, self.data_dir, "processed")
    
    @property
    def summaries_dir(self) -> str:
        return os.path.join(self.base_dir, self.output_dir, "summaries")
    
    @property
    def embeddings_dir(self) -> str:
        return os.path.join(self.base_dir, self.output_dir, "embeddings")
    
    @property
    def clusters_dir(self) -> str:
        return os.path.join(self.base_dir, self.output_dir, "clusters")
    
    @property
    def insights_dir(self) -> str:
        return os.path.join(self.base_dir, self.output_dir, "insights")

class PipelineConfig:
    """Master configuration class for the entire pipeline"""
    
    def __init__(self):
        self.models = ModelConfig()
        self.clustering = ClusterConfig()
        self.insights = InsightsConfig()
        self.data = DataConfig()
        self.paths = PathConfig()
        
        # Ensure output directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary output directories"""
        dirs_to_create = [
            self.paths.processed_data_dir,
            self.paths.summaries_dir,
            self.paths.embeddings_dir,
            self.paths.clusters_dir,
            self.paths.insights_dir,
            os.path.join(self.paths.base_dir, self.paths.models_dir)
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_data_file_path(self, filename: str, data_type: str = "raw") -> str:
        """Get full path to a data file"""
        if data_type == "raw":
            return os.path.join(self.paths.raw_data_dir, filename)
        elif data_type == "processed":
            return os.path.join(self.paths.processed_data_dir, filename)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    def get_output_file_path(self, filename: str, output_type: str) -> str:
        """Get full path to an output file"""
        type_to_dir = {
            "summaries": self.paths.summaries_dir,
            "embeddings": self.paths.embeddings_dir,
            "clusters": self.paths.clusters_dir,
            "insights": self.paths.insights_dir
        }
        
        if output_type not in type_to_dir:
            raise ValueError(f"Unknown output_type: {output_type}")
        
        return os.path.join(type_to_dir[output_type], filename)

# Global configuration instance
config = PipelineConfig()