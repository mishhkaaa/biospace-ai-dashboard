"""
Sidebar component for global search and filtering
"""

import streamlit as st
from utils.session_state import reset_filters

def render_sidebar(data_loader):
    """Render the sidebar with search and filter options"""
    
    st.sidebar.title("🔍 Search & Filters")
    
    # Global search
    search_term = st.sidebar.text_input(
        "Search papers", 
        value=st.session_state.global_search,
        placeholder="Enter keywords, titles, or concepts...",
        help="Search across titles, summaries, and keywords",
        key="global_search_input"
    )
    
    if search_term != st.session_state.global_search:
        st.session_state.global_search = search_term
        st.session_state.current_page = 1  # Reset pagination
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Filter options
    st.sidebar.subheader("📋 Filters")
    
    # Keywords filter
    available_keywords = data_loader.get_unique_values('keywords')
    if available_keywords:
        selected_keywords = st.sidebar.multiselect(
            "Keywords",
            options=available_keywords,
            default=st.session_state.selected_keywords,
            help="Filter by research keywords",
            key="keywords_filter"
        )
        if selected_keywords != st.session_state.selected_keywords:
            st.session_state.selected_keywords = selected_keywords
            st.session_state.current_page = 1
            st.rerun()
    
    # Clusters filter
    available_clusters = data_loader.get_unique_values('cluster')
    if available_clusters:
        selected_clusters = st.sidebar.multiselect(
            "Research Clusters",
            options=available_clusters,
            default=st.session_state.selected_clusters,
            help="Filter by thematic research clusters",
            key="clusters_filter"
        )
        if selected_clusters != st.session_state.selected_clusters:
            st.session_state.selected_clusters = selected_clusters
            st.session_state.current_page = 1
            st.rerun()
    
    # Publication year filter
    available_years = data_loader.get_unique_values('publication_year')
    if available_years:
        selected_years = st.sidebar.multiselect(
            "Publication Year",
            options=available_years,
            default=st.session_state.selected_years,
            help="Filter by publication year",
            key="years_filter"
        )
        if selected_years != st.session_state.selected_years:
            st.session_state.selected_years = selected_years
            st.session_state.current_page = 1
            st.rerun()
    
    # Organisms filter
    available_organisms = data_loader.get_unique_values('organisms')
    if available_organisms:
        selected_organisms = st.sidebar.multiselect(
            "Study Organisms",
            options=available_organisms,
            default=st.session_state.selected_organisms,
            help="Filter by organisms studied",
            key="organisms_filter"
        )
        if selected_organisms != st.session_state.selected_organisms:
            st.session_state.selected_organisms = selected_organisms
            st.session_state.current_page = 1
            st.rerun()
    
    # Mission type filter
    available_missions = data_loader.get_unique_values('mission_type')
    if available_missions:
        selected_missions = st.sidebar.multiselect(
            "Mission Type",
            options=available_missions,
            default=st.session_state.selected_missions,
            help="Filter by mission or experiment type",
            key="missions_filter"
        )
        if selected_missions != st.session_state.selected_missions:
            st.session_state.selected_missions = selected_missions
            st.session_state.current_page = 1
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Reset filters button
    if st.sidebar.button("🔄 Reset All Filters", key="reset_filters"):
        reset_filters()
        st.rerun()
    
    # Display active filters count
    active_filters = 0
    if st.session_state.global_search:
        active_filters += 1
    if st.session_state.selected_keywords:
        active_filters += 1
    if st.session_state.selected_clusters:
        active_filters += 1
    if st.session_state.selected_years:
        active_filters += 1
    if st.session_state.selected_organisms:
        active_filters += 1
    if st.session_state.selected_missions:
        active_filters += 1
    
    if active_filters > 0:
        st.sidebar.info(f"🎯 {active_filters} active filter{'s' if active_filters > 1 else ''}")
    
    # Data statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Summary")
    
    stats = data_loader.get_statistics()
    if stats:
        st.sidebar.metric("Total Papers", stats.get('total_papers', 0))
        st.sidebar.metric("Study Organisms", stats.get('unique_organisms', 0))
        st.sidebar.metric("Research Clusters", stats.get('clusters_count', 0))
        
        year_range = stats.get('year_range', (2020, 2024))
        st.sidebar.metric("Year Range", f"{year_range[0]} - {year_range[1]}")
    
    # Help section
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ Help & Tips"):
        st.markdown("""
        **Using the Dashboard:**
        - Use the search bar to find specific topics
        - Apply multiple filters to narrow results
        - Click on papers for detailed information
        - Use the chatbot for intelligent Q&A
        
        **Navigation:**
        - Overview: Key statistics and trends
        - Research Papers: Browse and search papers
        - Knowledge Graph: Explore connections
        - Chatbot: Ask questions about the research
        """)