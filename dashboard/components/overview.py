"""
Overview tab component showing key statistics and visualizations
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils.session_state import get_filtered_data

def render_overview_tab(data_loader):
    """Render the overview tab with statistics and visualizations"""
    
    st.markdown('<h2 class="tab-header">📊 Research Overview</h2>', unsafe_allow_html=True)
    
    # Get filtered data
    filtered_papers = get_filtered_data(data_loader)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📄 Total Papers",
            value=len(filtered_papers),
            delta=f"{len(filtered_papers) - len(data_loader.papers_df)} from filters" if len(filtered_papers) != len(data_loader.papers_df) else None
        )
    
    with col2:
        unique_organisms = len(filtered_papers['organisms'].unique()) if 'organisms' in filtered_papers.columns else 0
        st.metric(
            label="🧬 Study Organisms",
            value=unique_organisms
        )
    
    with col3:
        unique_clusters = len(filtered_papers['cluster'].unique()) if 'cluster' in filtered_papers.columns else 0
        st.metric(
            label="🎯 Research Clusters",
            value=unique_clusters
        )
    
    with col4:
        year_span = filtered_papers['publication_year'].max() - filtered_papers['publication_year'].min() if len(filtered_papers) > 0 else 0
        st.metric(
            label="📅 Year Span",
            value=f"{year_span} years"
        )
    
    st.markdown("---")
    
    # Visualizations row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Publication Timeline")
        if len(filtered_papers) > 0:
            timeline_data = filtered_papers.groupby('publication_year').size().reset_index(name='count')
            fig_timeline = px.line(
                timeline_data, 
                x='publication_year', 
                y='count',
                title="Papers Published Over Time",
                markers=True,
                color_discrete_sequence=['#1f77b4']
            )
            fig_timeline.update_layout(
                xaxis_title="Publication Year",
                yaxis_title="Number of Papers",
                showlegend=False,
                height=400
            )
            config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']}
            st.plotly_chart(fig_timeline, use_container_width=True, config=config)
        else:
            st.info("No data available for the current filters.")
    
    with col2:
        st.subheader("🧬 Organism Distribution")
        if len(filtered_papers) > 0 and 'organisms' in filtered_papers.columns:
            organism_counts = filtered_papers['organisms'].value_counts().head(10)
            fig_organisms = px.bar(
                x=organism_counts.values,
                y=organism_counts.index,
                orientation='h',
                title="Top Study Organisms",
                color=organism_counts.values,
                color_continuous_scale='viridis'
            )
            fig_organisms.update_layout(
                xaxis_title="Number of Studies",
                yaxis_title="Organism",
                showlegend=False,
                height=400
            )
            config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']}
            st.plotly_chart(fig_organisms, use_container_width=True, config=config)
        else:
            st.info("No organism data available.")
    
    # Visualizations row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Cluster Distribution")
        if len(filtered_papers) > 0 and 'cluster' in filtered_papers.columns:
            cluster_counts = filtered_papers['cluster'].value_counts().sort_index()
            fig_clusters = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {i}" for i in cluster_counts.index],
                title="Research Cluster Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_clusters.update_traces(textposition='inside', textinfo='percent+label')
            fig_clusters.update_layout(height=400)
            config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']}
            st.plotly_chart(fig_clusters, use_container_width=True, config=config)
        else:
            st.info("No cluster data available.")
    
    with col2:
        st.subheader("🚀 Mission Type Analysis")
        if len(filtered_papers) > 0 and 'mission_type' in filtered_papers.columns:
            mission_counts = filtered_papers['mission_type'].value_counts()
            fig_missions = px.pie(
                values=mission_counts.values,
                names=mission_counts.index,
                title="Mission Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4  # This makes it a donut chart
            )
            fig_missions.update_traces(textposition='inside', textinfo='percent+label')
            fig_missions.update_layout(height=400)
            config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']}
            st.plotly_chart(fig_missions, use_container_width=True, config=config)
        else:
            st.info("No mission type data available.")
    
    # Word cloud section
    st.markdown("---")
    st.subheader("☁️ Research Keywords Cloud")
    
    if len(filtered_papers) > 0:
        # Generate word cloud from keywords and summaries
        text_data = ""
        if 'keywords' in filtered_papers.columns:
            text_data += " ".join(filtered_papers['keywords'].dropna().astype(str))
        if 'summary' in filtered_papers.columns:
            text_data += " ".join(filtered_papers['summary'].dropna().astype(str))
        
        if text_data:
            try:
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='viridis',
                    max_words=100,
                    relative_scaling=0.5
                ).generate(text_data)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not generate word cloud: {str(e)}")
                st.info("Word cloud requires sufficient text data.")
    else:
        st.info("No data available for word cloud generation.")
    
    # Research insights section
    st.markdown("---")
    st.subheader("🔍 Research Insights")
    
    if hasattr(data_loader, 'insights_data') and data_loader.insights_data:
        
        # Knowledge gaps
        if 'knowledge_gaps' in data_loader.insights_data:
            with st.expander("📊 Knowledge Gaps Analysis"):
                gaps_data = data_loader.insights_data['knowledge_gaps']
                if isinstance(gaps_data, dict) and 'gaps' in gaps_data:
                    for i, gap in enumerate(gaps_data['gaps'][:5]):  # Show top 5
                        st.write(f"**Gap {i+1}:** {gap}")
                else:
                    st.write("Knowledge gaps analysis available in data files.")
        
        # Research opportunities
        if 'research_opportunities' in data_loader.insights_data:
            with st.expander("🎯 Research Opportunities"):
                opportunities_data = data_loader.insights_data['research_opportunities']
                if isinstance(opportunities_data, dict) and 'opportunities' in opportunities_data:
                    for i, opp in enumerate(opportunities_data['opportunities'][:5]):  # Show top 5
                        st.write(f"**Opportunity {i+1}:** {opp}")
                else:
                    st.write("Research opportunities analysis available in data files.")
        
        # Publication trends
        if 'publication_trends' in data_loader.insights_data:
            with st.expander("📈 Detailed Publication Trends"):
                trends_data = data_loader.insights_data['publication_trends']
                if isinstance(trends_data, pd.DataFrame):
                    # Clean data types for Arrow compatibility
                    trends_data_clean = trends_data.copy()
                    for col in trends_data_clean.columns:
                        if trends_data_clean[col].dtype == 'object':
                            trends_data_clean[col] = trends_data_clean[col].astype(str)
                    st.dataframe(trends_data_clean, use_container_width=True)
                else:
                    st.write("Publication trends analysis available in data files.")
    
    else:
        st.info("📋 Advanced insights will be displayed here when analysis data is available.")
    
    # Quick statistics table
    st.markdown("---")
    st.subheader("📋 Quick Statistics")
    
    if len(filtered_papers) > 0:
        # Safely get statistics with proper type handling
        try:
            unique_keywords = len(set(" ".join(filtered_papers['keywords'].dropna()).split()))
        except:
            unique_keywords = 0
            
        try:
            avg_papers_per_year = len(filtered_papers) // max(1, filtered_papers['publication_year'].nunique())
        except:
            avg_papers_per_year = 0
            
        try:
            most_common_organism = str(filtered_papers['organisms'].mode()[0]) if len(filtered_papers['organisms'].mode()) > 0 else "N/A"
        except:
            most_common_organism = "N/A"
            
        try:
            most_active_year = str(filtered_papers['publication_year'].mode()[0]) if len(filtered_papers['publication_year'].mode()) > 0 else "N/A"
        except:
            most_active_year = "N/A"
        
        stats_data = {
            "Metric": [
                "Total Papers",
                "Unique Keywords", 
                "Average Papers per Year",
                "Most Common Organism",
                "Most Active Year"
            ],
            "Value": [
                str(len(filtered_papers)),
                str(unique_keywords),
                str(avg_papers_per_year),
                most_common_organism,
                most_active_year
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        # Ensure all columns are strings for Arrow compatibility
        stats_df['Metric'] = stats_df['Metric'].astype(str)
        stats_df['Value'] = stats_df['Value'].astype(str)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)