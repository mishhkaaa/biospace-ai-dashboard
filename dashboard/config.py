"""
Configuration file for the NASA Space Biology Knowledge Engine Dashboard
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
AI_NLP_DIR = BASE_DIR.parent / "ai_nlp"
DATA_DIR = AI_NLP_DIR / "outputs"

# Data file paths
DATA_PATHS = {
    'summaries': DATA_DIR / "summaries" / "paper_summaries.csv",
    'summaries_metadata': DATA_DIR / "summaries" / "paper_summaries_metadata.json",
    'embeddings': DATA_DIR / "embeddings" / "paper_embeddings.jsonl",
    'embeddings_metadata': DATA_DIR / "embeddings" / "paper_embeddings_metadata.json",
    'similarities': DATA_DIR / "embeddings" / "paper_embeddings_similarities.json",
    'clusters': DATA_DIR / "clusters" / "paper_clusters.csv",
    'cluster_metadata': DATA_DIR / "clusters" / "paper_clusters_metadata.json",
    'keywords': DATA_DIR / "clusters" / "cluster_keywords_summaries.csv",
    'keywords_metadata': DATA_DIR / "clusters" / "cluster_keywords_summaries_metadata.json",
    'knowledge_gaps': DATA_DIR / "insights" / "knowledge_gaps.json",
    'research_opportunities': DATA_DIR / "insights" / "research_opportunities.json",
    'comprehensive_insights': DATA_DIR / "insights" / "comprehensive_insights.json",
    'publication_trends': DATA_DIR / "insights" / "publication_trends.csv",
    'raw_metadata': AI_NLP_DIR / "data" / "raw" / "person_a_metadata.csv"
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': "NASA Space Biology Knowledge Engine",
    'page_icon': "🚀",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Dashboard settings
DASHBOARD_CONFIG = {
    'max_papers_display': 1000,
    'papers_per_page_options': [10, 25, 50, 100],
    'default_papers_per_page': 10,
    'max_graph_nodes': 100,
    'default_graph_nodes': 50,
    'chatbot_max_results': 10,
    'chatbot_default_results': 3
}

# Visualization settings
VIZ_CONFIG = {
    'color_schemes': {
        'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'scientific': ['#440154', '#31688e', '#26828e', '#1f9e89', '#6ece58'],
        'space': ['#0d1b2a', '#1b263b', '#415a77', '#778da9', '#e0e1dd']
    },
    'default_height': 400,
    'graph_height': 600,
    'wordcloud_size': (800, 400)
}

# Neo4j configuration (if available)
NEO4J_CONFIG = {
    'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
    'password': os.getenv('NEO4J_PASSWORD', 'password'),
    'database': os.getenv('NEO4J_DATABASE', 'neo4j')
}

# Search and filter settings
SEARCH_CONFIG = {
    'min_search_length': 2,
    'max_search_results': 100,
    'search_fields': ['title', 'summary', 'keywords'],
    'filter_fields': ['keywords', 'cluster', 'publication_year', 'organisms', 'mission_type']
}

# Performance settings
PERFORMANCE_CONFIG = {
    'cache_ttl': 3600,  # 1 hour
    'max_cache_entries': 1000,
    'chunk_size': 1000,
    'lazy_loading': True
}