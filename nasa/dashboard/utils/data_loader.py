"""
Data loading utilities for the NASA Space Biology Knowledge Engine
"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import streamlit as st

class DataLoader:
    """Handles loading and processing of all dashboard data"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent / "ai_nlp"
        self.papers_df = None
        self.embeddings_data = None
        self.clusters_data = None
        self.keywords_data = None
        self.insights_data = None
        self.neo4j_data = None
        
    def _clean_data_types(self, df):
        """Clean data types for Arrow compatibility"""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        # Convert object columns that should be strings
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to string, handle mixed types
                try:
                    df[col] = df[col].astype(str)
                except:
                    # If conversion fails, keep as is
                    pass
        return df
        
    @st.cache_data
    def load_all_data(_self):
        """Load all required data files"""
        try:
            _self._load_papers_data()
            _self._load_embeddings_data()
            _self._load_clusters_data()
            _self._load_keywords_data()
            _self._load_insights_data()
            _self._validate_data()
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def _load_papers_data(self):
        """Load and process papers data"""
        try:
            # Load summaries data
            summaries_file = self.base_path / "outputs" / "summaries" / "paper_summaries.csv"
            if summaries_file.exists():
                self.papers_df = pd.read_csv(summaries_file)
                
                # Load metadata if available
                metadata_file = self.base_path / "outputs" / "summaries" / "paper_summaries_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        self.summaries_metadata = json.load(f)
            else:
                # Fallback to raw data
                raw_file = self.base_path / "data" / "raw" / "person_a_metadata.csv"
                if raw_file.exists():
                    self.papers_df = pd.read_csv(raw_file)
                    # Generate dummy summaries for demo
                    self.papers_df['summary'] = self.papers_df.get('abstract', 'Summary not available')
                else:
                    raise FileNotFoundError("No papers data found")
            
            # Ensure required columns exist
            required_columns = ['title', 'summary', 'publication_year']
            for col in required_columns:
                if col not in self.papers_df.columns:
                    if col == 'publication_year':
                        self.papers_df[col] = 2020  # Default year
                    else:
                        self.papers_df[col] = f"Unknown {col}"
            
            # Add derived columns if not present
            if 'keywords' not in self.papers_df.columns:
                self.papers_df['keywords'] = "space biology, NASA, research"
            
            # Clean data types for Arrow compatibility
            self.papers_df = self._clean_data_types(self.papers_df)
            
            if 'cluster' not in self.papers_df.columns:
                self.papers_df['cluster'] = np.random.randint(0, 5, len(self.papers_df))
            
            if 'organisms' not in self.papers_df.columns:
                organisms_list = ['Arabidopsis', 'E. coli', 'Yeast', 'Mice', 'Humans', 'Drosophila']
                self.papers_df['organisms'] = np.random.choice(organisms_list, len(self.papers_df))
            
            if 'mission_type' not in self.papers_df.columns:
                missions = ['ISS', 'SpaceX', 'Artemis', 'Mars', 'Ground-based']
                self.papers_df['mission_type'] = np.random.choice(missions, len(self.papers_df))
            
            # Clean data types to prevent Arrow serialization errors
            if self.papers_df is not None:
                self.papers_df = self._clean_data_types(self.papers_df)
            
        except Exception as e:
            st.error(f"Error loading papers data: {str(e)}")
            # Create dummy data as fallback
            self._create_dummy_papers_data()
            
        # Clean data types and ensure proper format
        if self.papers_df is not None:
            self.papers_df = self._clean_data_types(self.papers_df)
            
            if 'publication_year' in self.papers_df.columns:
                self.papers_df['publication_year'] = pd.to_numeric(
                    self.papers_df['publication_year'], errors='coerce'
                ).fillna(2020).astype(int)
            
            if 'cluster' in self.papers_df.columns:
                self.papers_df['cluster'] = pd.to_numeric(
                    self.papers_df['cluster'], errors='coerce'
                ).fillna(0).astype(int)
    
    def _load_embeddings_data(self):
        """Load embeddings and similarity data"""
        try:
            embeddings_file = self.base_path / "outputs" / "embeddings" / "paper_embeddings.jsonl"
            if embeddings_file.exists():
                self.embeddings_data = []
                with open(embeddings_file, 'r') as f:
                    for line in f:
                        self.embeddings_data.append(json.loads(line))
            
            # Load similarities if available
            similarities_file = self.base_path / "outputs" / "embeddings" / "paper_embeddings_similarities.json"
            if similarities_file.exists():
                with open(similarities_file, 'r') as f:
                    self.similarities_data = json.load(f)
                    
        except Exception as e:
            st.warning(f"Could not load embeddings data: {str(e)}")
            self.embeddings_data = []
    
    def _load_clusters_data(self):
        """Load clustering results"""
        try:
            clusters_file = self.base_path / "outputs" / "clusters" / "paper_clusters.csv"
            if clusters_file.exists():
                self.clusters_data = pd.read_csv(clusters_file)
                self.clusters_data = self._clean_data_types(self.clusters_data)
                
            # Load cluster metadata
            cluster_meta_file = self.base_path / "outputs" / "clusters" / "paper_clusters_metadata.json"
            if cluster_meta_file.exists():
                with open(cluster_meta_file, 'r') as f:
                    self.clusters_metadata = json.load(f)
                    
        except Exception as e:
            st.warning(f"Could not load clusters data: {str(e)}")
            self.clusters_data = None
    
    def _load_keywords_data(self):
        """Load keywords and cluster summaries"""
        try:
            keywords_file = self.base_path / "outputs" / "clusters" / "cluster_keywords_summaries.csv"
            if keywords_file.exists():
                self.keywords_data = pd.read_csv(keywords_file)
                self.keywords_data = self._clean_data_types(self.keywords_data)
                
        except Exception as e:
            st.warning(f"Could not load keywords data: {str(e)}")
            self.keywords_data = None
    
    def _load_insights_data(self):
        """Load insights and analysis results"""
        try:
            insights_files = {
                'knowledge_gaps': self.base_path / "outputs" / "insights" / "knowledge_gaps.json",
                'research_opportunities': self.base_path / "outputs" / "insights" / "research_opportunities.json",
                'comprehensive_insights': self.base_path / "outputs" / "insights" / "comprehensive_insights.json",
                'publication_trends': self.base_path / "outputs" / "insights" / "publication_trends.csv"
            }
            
            self.insights_data = {}
            for key, file_path in insights_files.items():
                if file_path.exists():
                    if file_path.suffix == '.json':
                        with open(file_path, 'r') as f:
                            self.insights_data[key] = json.load(f)
                    else:
                        self.insights_data[key] = pd.read_csv(file_path)
                        
        except Exception as e:
            st.warning(f"Could not load insights data: {str(e)}")
            self.insights_data = {}
    
    def _create_dummy_papers_data(self):
        """Create dummy data for demo purposes"""
        np.random.seed(42)
        
        titles = [
            "Effects of Microgravity on Plant Growth in Space",
            "Protein Crystallization in Low Earth Orbit",
            "Bone Density Changes in Long-Duration Spaceflight",
            "Microbial Behavior in Spacecraft Environments",
            "Radiation Effects on DNA Repair Mechanisms",
            "Cardiovascular Adaptations to Space Environment",
            "Neural Plasticity in Altered Gravity Conditions",
            "Immune System Response to Space Travel",
            "Cellular Regeneration in Weightlessness",
            "Metabolic Changes During Extended Space Missions"
        ]
        
        # Create dummy dataframe
        n_papers = max(50, len(titles) * 5)
        self.papers_df = pd.DataFrame({
            'title': np.random.choice(titles, n_papers),
            'summary': [f"This study investigates {topic} with important implications for space biology research." 
                       for topic in np.random.choice(titles, n_papers)],
            'publication_year': np.random.choice(range(2015, 2025), n_papers),
            'keywords': np.random.choice(['space biology', 'microgravity', 'radiation', 'plants', 'humans'], n_papers),
            'cluster': np.random.randint(0, 5, n_papers),
            'organisms': np.random.choice(['Arabidopsis', 'E. coli', 'Mice', 'Humans'], n_papers),
            'mission_type': np.random.choice(['ISS', 'SpaceX', 'Ground-based'], n_papers)
        })
        
        # Clean data types
        self.papers_df = self._clean_data_types(self.papers_df)
    
    def _validate_data(self):
        """Validate loaded data for consistency"""
        if self.papers_df is None or len(self.papers_df) == 0:
            raise ValueError("No papers data loaded")
        
        # Check for required columns
        required_cols = ['title', 'summary']
        missing_cols = [col for col in required_cols if col not in self.papers_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def get_unique_values(self, column):
        """Get unique values from a column for filter options"""
        if self.papers_df is not None and column in self.papers_df.columns:
            if column in ['keywords', 'organisms']:
                # Handle comma-separated values
                all_values = []
                for val in self.papers_df[column].dropna():
                    if isinstance(val, str):
                        all_values.extend([v.strip() for v in val.split(',')])
                return sorted(list(set(all_values)))
            else:
                return sorted(self.papers_df[column].dropna().unique().tolist())
        return []
    
    def get_statistics(self):
        """Get summary statistics for the overview tab"""
        if self.papers_df is None:
            return {}
        
        stats = {
            'total_papers': len(self.papers_df),
            'unique_organisms': len(self.get_unique_values('organisms')),
            'year_range': (self.papers_df['publication_year'].min(), 
                          self.papers_df['publication_year'].max()),
            'clusters_count': len(self.papers_df['cluster'].unique()) if 'cluster' in self.papers_df.columns else 0
        }
        
        return stats