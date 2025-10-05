"""
Session state management for the Streamlit dashboard
"""

import streamlit as st

def initialize_session_state():
    """Initialize session state variables"""
    
    # Data loading state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Search and filter state
    if 'global_search' not in st.session_state:
        st.session_state.global_search = ""
    
    if 'selected_keywords' not in st.session_state:
        st.session_state.selected_keywords = []
    
    if 'selected_clusters' not in st.session_state:
        st.session_state.selected_clusters = []
    
    if 'selected_years' not in st.session_state:
        st.session_state.selected_years = []
    
    if 'selected_organisms' not in st.session_state:
        st.session_state.selected_organisms = []
    
    if 'selected_missions' not in st.session_state:
        st.session_state.selected_missions = []
    
    # Pagination state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    if 'papers_per_page' not in st.session_state:
        st.session_state.papers_per_page = 10
    
    # Chatbot state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    # Chatbot initialization state
    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False
    
    if 'qa_pipeline' not in st.session_state:
        st.session_state.qa_pipeline = None
    
    if 'query_router' not in st.session_state:
        st.session_state.query_router = None
    
    if 'embeddings_available' not in st.session_state:
        st.session_state.embeddings_available = False
    
    # Knowledge graph state
    if 'selected_node' not in st.session_state:
        st.session_state.selected_node = None
    
    if 'graph_layout' not in st.session_state:
        st.session_state.graph_layout = "spring"

def reset_filters():
    """Reset all filters to default state"""
    st.session_state.global_search = ""
    st.session_state.selected_keywords = []
    st.session_state.selected_clusters = []
    st.session_state.selected_years = []
    st.session_state.selected_organisms = []
    st.session_state.selected_missions = []
    st.session_state.current_page = 1

def get_filtered_data(data_loader):
    """Apply current filters to the data and return filtered results"""
    
    # Start with all papers
    filtered_papers = data_loader.papers_df.copy()
    
    # Apply global search filter
    if st.session_state.global_search:
        search_term = st.session_state.global_search.lower()
        mask = (
            filtered_papers['title'].str.lower().str.contains(search_term, na=False) |
            filtered_papers['summary'].str.lower().str.contains(search_term, na=False) |
            filtered_papers['keywords'].str.lower().str.contains(search_term, na=False)
        )
        filtered_papers = filtered_papers[mask]
    
    # Apply keyword filters
    if st.session_state.selected_keywords:
        keyword_mask = filtered_papers['keywords'].apply(
            lambda x: any(keyword.lower() in str(x).lower() 
                         for keyword in st.session_state.selected_keywords)
        )
        filtered_papers = filtered_papers[keyword_mask]
    
    # Apply cluster filters
    if st.session_state.selected_clusters:
        cluster_mask = filtered_papers['cluster'].isin(st.session_state.selected_clusters)
        filtered_papers = filtered_papers[cluster_mask]
    
    # Apply year filters
    if st.session_state.selected_years:
        year_mask = filtered_papers['publication_year'].isin(st.session_state.selected_years)
        filtered_papers = filtered_papers[year_mask]
    
    # Apply organism filters
    if st.session_state.selected_organisms:
        organism_mask = filtered_papers['organisms'].apply(
            lambda x: any(organism.lower() in str(x).lower() 
                         for organism in st.session_state.selected_organisms)
        )
        filtered_papers = filtered_papers[organism_mask]
    
    # Apply mission filters
    if st.session_state.selected_missions:
        mission_mask = filtered_papers['mission_type'].apply(
            lambda x: any(mission.lower() in str(x).lower() 
                         for mission in st.session_state.selected_missions)
        )
        filtered_papers = filtered_papers[mission_mask]
    
    return filtered_papers