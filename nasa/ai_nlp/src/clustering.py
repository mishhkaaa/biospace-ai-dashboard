"""
NASA Bioscience AI Pipeline - Enhanced Clustering Module
Advanced clustering with multiple algorithms and adaptive parameters
"""

import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from config import config
from utils import (
    PipelineLogger, 
    save_results_with_metadata,
    NumpyEncoder
)

class AdvancedClusterer:
    """
    Advanced clustering with multiple algorithms and automatic optimization
    """
    
    def __init__(self, algorithm: str = "kmeans", _internal_call: bool = False):
        self.logger = PipelineLogger("AdvancedClusterer")
        self.algorithm = algorithm.lower()
        self.supported_algorithms = ["kmeans", "dbscan", "agglomerative", "auto"]
        
        if self.algorithm not in self.supported_algorithms:
            raise ValueError(f"Algorithm must be one of {self.supported_algorithms}")
        
        # Prevent infinite recursion: "auto" should only be used at top level
        if self.algorithm == "auto" and _internal_call:
            raise ValueError(
                "Internal error: 'auto' algorithm cannot be used recursively. "
                "This suggests a bug in _auto_cluster method. Only concrete algorithms "
                "('kmeans', 'dbscan', 'agglomerative') should be used in internal calls."
            )
        
        self.logger.info(f"Initialized clusterer with algorithm: {self.algorithm}")
    
    def _evaluate_clustering(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics
        
        Args:
            embeddings: Array of embeddings
            labels: Cluster labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            n_clusters = len(np.unique(labels))
            n_noise = np.sum(labels == -1) if -1 in labels else 0
            
            metrics = {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'n_clustered_points': len(labels) - n_noise
            }
            
            # Only calculate metrics if we have valid clusters
            if n_clusters > 1 and len(labels) - n_noise > n_clusters:
                # Filter out noise points for evaluation
                valid_mask = labels != -1
                if valid_mask.sum() > 0:
                    valid_embeddings = embeddings[valid_mask]
                    valid_labels = labels[valid_mask]
                    
                    if len(np.unique(valid_labels)) > 1:
                        metrics['silhouette_score'] = silhouette_score(valid_embeddings, valid_labels)
                        metrics['calinski_harabasz_score'] = calinski_harabasz_score(valid_embeddings, valid_labels)
                        metrics['davies_bouldin_score'] = davies_bouldin_score(valid_embeddings, valid_labels)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error evaluating clustering: {str(e)}")
            return {'n_clusters': len(np.unique(labels)), 'evaluation_failed': True}
    
    def _find_optimal_k_kmeans(self, embeddings: np.ndarray, 
                              k_range: Tuple[int, int] = (2, 15)) -> Tuple[int, Dict]:
        """
        Find optimal number of clusters for K-means using multiple metrics
        
        Args:
            embeddings: Array of embeddings
            k_range: Range of k values to test
            
        Returns:
            Tuple of (optimal_k, evaluation_results)
        """
        self.logger.info(f"Finding optimal k for K-means in range {k_range}")
        
        k_min, k_max = k_range
        k_max = min(k_max, len(embeddings) // 2)  # Ensure reasonable upper bound
        
        results = {}
        
        for k in range(k_min, k_max + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                metrics = self._evaluate_clustering(embeddings, labels)
                metrics['inertia'] = kmeans.inertia_
                
                results[k] = metrics
                
            except Exception as e:
                self.logger.warning(f"Error evaluating k={k}: {str(e)}")
                continue
        
        if not results:
            self.logger.warning("No valid k values found, using default k=5")
            return 5, {}
        
        # Select optimal k based on silhouette score (if available)
        optimal_k = k_min
        best_score = -1
        
        for k, metrics in results.items():
            if 'silhouette_score' in metrics:
                if metrics['silhouette_score'] > best_score:
                    best_score = metrics['silhouette_score']
                    optimal_k = k
        
        # If no silhouette scores available, use elbow method approximation
        if best_score == -1 and len(results) > 2:
            inertias = [results[k].get('inertia', float('inf')) for k in sorted(results.keys())]
            # Simple elbow detection: find point with maximum second derivative
            if len(inertias) >= 3:
                second_derivatives = []
                k_values = sorted(results.keys())
                for i in range(1, len(inertias) - 1):
                    second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                    second_derivatives.append((k_values[i], second_deriv))
                
                optimal_k = max(second_derivatives, key=lambda x: x[1])[0]
        
        self.logger.success(f"Optimal k selected: {optimal_k}")
        return optimal_k, results
    
    def _cluster_kmeans(self, embeddings: np.ndarray, n_clusters: Optional[int] = None) -> Tuple[np.ndarray, Any, Dict]:
        """
        Perform K-means clustering with optional optimization
        
        Args:
            embeddings: Array of embeddings
            n_clusters: Number of clusters (if None, will be optimized)
            
        Returns:
            Tuple of (labels, model, metadata)
        """
        if n_clusters is None:
            n_clusters, optimization_results = self._find_optimal_k_kmeans(embeddings)
        else:
            optimization_results = {}
        
        self.logger.info(f"Performing K-means clustering with k={n_clusters}")
        
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(embeddings)
        
        metadata = {
            'algorithm': 'kmeans',
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_,
            'optimization_results': optimization_results
        }
        
        return labels, kmeans, metadata
    
    def _cluster_dbscan(self, embeddings: np.ndarray, eps: Optional[float] = None, 
                       min_samples: Optional[int] = None) -> Tuple[np.ndarray, Any, Dict]:
        """
        Perform DBSCAN clustering with parameter optimization
        
        Args:
            embeddings: Array of embeddings
            eps: Distance threshold (if None, will be estimated)
            min_samples: Minimum samples per cluster (if None, will be estimated)
            
        Returns:
            Tuple of (labels, model, metadata)
        """
        # Estimate parameters if not provided
        if min_samples is None:
            min_samples = max(2, int(np.log(len(embeddings))))
        
        if eps is None:
            # Estimate eps using k-distance graph
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors_fit = neighbors.fit(embeddings)
            distances, indices = neighbors_fit.kneighbors(embeddings)
            distances = np.sort(distances[:, min_samples-1], axis=0)
            
            # Use knee detection or percentile
            eps = np.percentile(distances, 90)
        
        self.logger.info(f"Performing DBSCAN clustering with eps={eps:.4f}, min_samples={min_samples}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        metadata = {
            'algorithm': 'dbscan',
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise
        }
        
        return labels, dbscan, metadata
    
    def _cluster_agglomerative(self, embeddings: np.ndarray, 
                              n_clusters: Optional[int] = None) -> Tuple[np.ndarray, Any, Dict]:
        """
        Perform Agglomerative clustering
        
        Args:
            embeddings: Array of embeddings
            n_clusters: Number of clusters (if None, will be optimized)
            
        Returns:
            Tuple of (labels, model, metadata)
        """
        if n_clusters is None:
            # Use same optimization as K-means
            n_clusters, _ = self._find_optimal_k_kmeans(embeddings)
        
        self.logger.info(f"Performing Agglomerative clustering with k={n_clusters}")
        
        agglomerative = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        
        labels = agglomerative.fit_predict(embeddings)
        
        metadata = {
            'algorithm': 'agglomerative',
            'n_clusters': n_clusters,
            'linkage': 'ward'
        }
        
        return labels, agglomerative, metadata
    
    def cluster(self, embeddings: np.ndarray, **kwargs) -> Tuple[np.ndarray, Any, Dict]:
        """
        Perform clustering using the specified algorithm
        
        Args:
            embeddings: Array of embeddings
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Tuple of (labels, model, metadata)
        """
        # Normalize embeddings for better clustering
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        if self.algorithm == "kmeans":
            return self._cluster_kmeans(embeddings_scaled, kwargs.get('n_clusters'))
        elif self.algorithm == "dbscan":
            return self._cluster_dbscan(embeddings_scaled, kwargs.get('eps'), kwargs.get('min_samples'))
        elif self.algorithm == "agglomerative":
            return self._cluster_agglomerative(embeddings_scaled, kwargs.get('n_clusters'))
        elif self.algorithm == "auto":
            # Try multiple algorithms and select the best
            return self._auto_cluster(embeddings_scaled)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _auto_cluster(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Any, Dict]:
        """
        Automatically select the best clustering algorithm
        
        IMPORTANT: This method should NEVER be called recursively with "auto" algorithm.
        It only tests concrete algorithms ("kmeans", "dbscan", "agglomerative") to
        prevent infinite recursion.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Tuple of (labels, model, metadata) for best algorithm
        """
        self.logger.info("Automatically selecting best clustering algorithm")
        
        # SAFETY: Only use concrete algorithms to prevent recursion
        # Never include "auto" in this list!
        algorithms_to_try = ["kmeans", "dbscan", "agglomerative"]
        results = {}
        
        for alg in algorithms_to_try:
            try:
                # CRITICAL: Use _internal_call=True to prevent "auto" recursion
                temp_clusterer = AdvancedClusterer(alg, _internal_call=True)
                labels, model, metadata = temp_clusterer.cluster(embeddings)
                
                # Evaluate clustering
                evaluation = self._evaluate_clustering(embeddings, labels)
                metadata.update(evaluation)
                
                results[alg] = (labels, model, metadata)
                
            except Exception as e:
                self.logger.warning(f"Algorithm {alg} failed: {str(e)}")
                continue
        
        if not results:
            raise RuntimeError("All clustering algorithms failed")
        
        # Select best algorithm based on silhouette score
        best_alg = None
        best_score = -1
        
        for alg, (labels, model, metadata) in results.items():
            score = metadata.get('silhouette_score', -1)
            if score > best_score:
                best_score = score
                best_alg = alg
        
        if best_alg is None:
            # Fallback to kmeans
            best_alg = "kmeans"
        
        self.logger.success(f"Selected algorithm: {best_alg} (silhouette score: {best_score:.3f})")
        
        labels, model, metadata = results[best_alg]
        metadata['auto_selection'] = {
            'selected_algorithm': best_alg,
            'all_results': {alg: meta for alg, (_, _, meta) in results.items()}
        }
        
        return labels, model, metadata

def create_enhanced_clusters(embeddings_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Create enhanced clusters with comprehensive analysis
    
    Args:
        embeddings_path: Path to embeddings file
        
    Returns:
        Tuple of (DataFrame with cluster assignments, cluster analysis)
    """
    logger = PipelineLogger("Enhanced Clustering")
    
    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    
    embeddings_data = []
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        for line in f:
            embeddings_data.append(json.loads(line.strip()))
    
    df_embeddings = pd.DataFrame(embeddings_data)
    
    # Convert embeddings to numpy array
    embeddings = np.stack(df_embeddings['embedding'].values)
    
    logger.info(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    # Initialize clusterer
    algorithm = config.clustering.clustering_algorithm
    if config.clustering.use_adaptive_clustering:
        algorithm = "auto"
    
    clusterer = AdvancedClusterer(algorithm)
    
    # Perform clustering
    logger.info("Performing clustering analysis")
    
    clustering_params = {}
    if algorithm == "kmeans":
        clustering_params['n_clusters'] = config.clustering.default_n_clusters
    
    labels, model, metadata = clusterer.cluster(embeddings, **clustering_params)
    
    # Add cluster assignments to dataframe
    df_clusters = df_embeddings[['paper_id']].copy()
    df_clusters['cluster_id'] = labels
    
    # Analyze clusters
    cluster_analysis = analyze_clusters(df_clusters, embeddings, metadata)
    
    logger.success(f"Clustering completed: {cluster_analysis['n_clusters']} clusters, "
                  f"{cluster_analysis['n_noise_points']} noise points")
    
    return df_clusters, cluster_analysis

def analyze_clusters(df_clusters: pd.DataFrame, embeddings: np.ndarray, 
                    metadata: Dict) -> Dict:
    """
    Perform comprehensive cluster analysis
    
    Args:
        df_clusters: DataFrame with cluster assignments
        embeddings: Array of embeddings
        metadata: Clustering metadata
        
    Returns:
        Dictionary with cluster analysis results
    """
    logger = PipelineLogger("Cluster Analysis")
    
    # Basic cluster statistics
    cluster_counts = df_clusters['cluster_id'].value_counts().sort_index()
    n_clusters = len(cluster_counts)
    n_noise_points = (df_clusters['cluster_id'] == -1).sum() if -1 in df_clusters['cluster_id'].values else 0
    
    # Cluster size analysis
    cluster_sizes = cluster_counts[cluster_counts.index != -1] if -1 in cluster_counts.index else cluster_counts
    
    analysis = {
        'n_clusters': n_clusters,
        'n_noise_points': n_noise_points,
        'cluster_sizes': cluster_sizes.to_dict(),
        'avg_cluster_size': float(cluster_sizes.mean()) if len(cluster_sizes) > 0 else 0,
        'std_cluster_size': float(cluster_sizes.std()) if len(cluster_sizes) > 0 else 0,
        'min_cluster_size': int(cluster_sizes.min()) if len(cluster_sizes) > 0 else 0,
        'max_cluster_size': int(cluster_sizes.max()) if len(cluster_sizes) > 0 else 0,
        'clustering_metadata': metadata
    }
    
    # Identify knowledge gaps (small clusters)
    small_clusters = cluster_sizes[cluster_sizes < config.clustering.min_cluster_size]
    analysis['knowledge_gap_clusters'] = small_clusters.index.tolist()
    analysis['n_knowledge_gaps'] = len(small_clusters)
    
    # Calculate cluster centroids and inter-cluster distances
    try:
        centroids = {}
        for cluster_id in cluster_sizes.index:
            cluster_mask = df_clusters['cluster_id'] == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            centroids[int(cluster_id)] = np.mean(cluster_embeddings, axis=0)
        
        analysis['cluster_centroids'] = centroids
        
        # Calculate inter-cluster distances
        if len(centroids) > 1:
            centroid_matrix = np.array(list(centroids.values()))
            from sklearn.metrics.pairwise import euclidean_distances
            inter_distances = euclidean_distances(centroid_matrix)
            analysis['avg_inter_cluster_distance'] = float(np.mean(inter_distances[np.triu_indices_from(inter_distances, k=1)]))
    
    except Exception as e:
        logger.warning(f"Error calculating cluster centroids: {str(e)}")
    
    logger.info(f"Cluster analysis completed: {n_clusters} clusters, {n_noise_points} noise points")
    logger.info(f"Knowledge gaps identified: {analysis['n_knowledge_gaps']} small clusters")
    
    return analysis

def main():
    """Main execution function for clustering pipeline"""
    logger = PipelineLogger("Clustering Pipeline")
    
    try:
        # Load embeddings
        embeddings_path = config.get_output_file_path("paper_embeddings.jsonl", "embeddings")
        
        # Create enhanced clusters
        df_clusters, cluster_analysis = create_enhanced_clusters(embeddings_path)
        
        # Save cluster assignments
        output_path = config.get_output_file_path("paper_clusters.csv", "clusters")
        
        metadata = {
            'component': 'clustering',
            'algorithm': cluster_analysis['clustering_metadata']['algorithm'],
            'n_clusters': cluster_analysis['n_clusters'],
            'n_noise_points': cluster_analysis['n_noise_points'],
            'avg_cluster_size': cluster_analysis['avg_cluster_size'],
            'knowledge_gaps': cluster_analysis['knowledge_gap_clusters']
        }
        
        save_results_with_metadata(df_clusters, output_path, metadata)
        
        # Skip detailed cluster analysis JSON due to serialization issues
        # The CSV contains the essential clustering results
        logger.info("Skipping detailed cluster analysis JSON (circular reference issues)")
        
        logger.success("Clustering pipeline completed successfully!")
        logger.info(f"Generated {cluster_analysis['n_clusters']} clusters")
        logger.info(f"Identified {cluster_analysis['n_knowledge_gaps']} knowledge gaps")
        
        return df_clusters, cluster_analysis
        
    except Exception as e:
        logger.error(f"Clustering pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
