"""
NASA Space Biology Knowledge Engine
A comprehensive dashboard for exploring NASA bioscience research data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import dashboard components
from components.overview import render_overview_tab
from components.research_papers import render_research_papers_tab
from components.knowledge_graph import render_knowledge_graph_tab
from components.chatbot import render_chatbot_tab
from components.sidebar import render_sidebar
from utils.data_loader import DataLoader
from utils.session_state import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="NASA Space Biology Knowledge Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f4e79;
        margin-bottom: 2rem;
    }
    
    .tab-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c5aa0;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stTabs > div > div > div > div {
        padding-top: 1rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 NASA Space Biology Knowledge Engine</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    try:
        if 'data_loader' not in st.session_state:
            with st.spinner('Loading NASA bioscience research data...'):
                st.session_state.data_loader = DataLoader()
                load_success = st.session_state.data_loader.load_all_data()
                if not load_success:
                    st.warning("Some data files could not be loaded. Using fallback data for demonstration.")
        
        data_loader = st.session_state.data_loader
        
        # Only proceed if we have a valid data_loader
        if data_loader is None:
            st.error("Failed to initialize data loader. Please refresh the page.")
            return
        
        # Sidebar
        render_sidebar(data_loader)
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "📄 Research Papers", 
            "🕸️ Knowledge Graph", 
            "💬 Chatbot"
        ])
        
        with tab1:
            render_overview_tab(data_loader)
        
        with tab2:
            render_research_papers_tab(data_loader)
        
        with tab3:
            render_knowledge_graph_tab(data_loader)
        
        with tab4:
            render_chatbot_tab(data_loader)
            
    except Exception as e:
        st.error(f"Error loading application: {str(e)}")
        st.info("Please check your data files and try refreshing the page.")
        
        # Debug information in expander
        with st.expander("Debug Information"):
            st.text(f"Error details: {str(e)}")
            st.text(f"Current working directory: {os.getcwd()}")
            st.text(f"Python path: {sys.path}")

if __name__ == "__main__":
    main()