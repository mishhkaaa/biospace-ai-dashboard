"""
Knowledge Graph tab component for interactive graph visualization
"""

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import tempfile
import os
from utils.session_state import get_filtered_data

def render_knowledge_graph_tab(data_loader):
    """Render the knowledge graph tab with interactive visualization"""
    
    st.markdown('<h2 class="tab-header">🕸️ Knowledge Graph</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore the interconnected relationships between research papers, organisms, missions, 
    and key findings in NASA's space biology research.
    """)
    
    # Graph configuration options
    st.markdown("### 🔧 Graph Configuration")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        layout_type = st.selectbox(
            "Layout Algorithm",
            options=["spring", "circular", "kamada_kawai", "spectral"],
            index=0,
            help="Choose the graph layout algorithm",
            key="knowledge_graph_layout_algorithm"
        )
    
    with col2:
        node_size_metric = st.selectbox(
            "Node Size Based On",
            options=["degree", "cluster_size", "uniform"],
            index=0,
            help="What determines node size",
            key="knowledge_graph_node_size_metric"
        )
    
    with col3:
        show_labels = st.checkbox(
            "Show Node Labels",
            value=True,
            help="Display labels on graph nodes",
            key="knowledge_graph_show_labels"
        )
    
    with col4:
        max_nodes = st.slider(
            "Max Nodes",
            min_value=10,
            max_value=100,
            value=50,
            help="Maximum number of nodes to display",
            key="knowledge_graph_max_nodes"
        )
    
    # Get filtered data
    filtered_papers = get_filtered_data(data_loader)
    
    if len(filtered_papers) == 0:
        st.warning("No papers match the current filters. Adjust your search criteria to see the knowledge graph.")
        return
    
    # Create graph selection tabs
    graph_tab1, graph_tab2, graph_tab3 = st.tabs([
        "📊 Paper Relationships", 
        "🧬 Organism Network", 
        "🚀 Mission Connections"
    ])
    
    with graph_tab1:
        st.markdown("#### Paper Similarity Network")
        create_paper_network(filtered_papers, layout_type, node_size_metric, show_labels, max_nodes)
    
    with graph_tab2:
        st.markdown("#### Organism Study Network")
        create_organism_network(filtered_papers, layout_type, show_labels, max_nodes)
    
    with graph_tab3:
        st.markdown("#### Mission and Research Network")
        create_mission_network(filtered_papers, layout_type, show_labels, max_nodes)
    
    # Graph statistics
    st.markdown("---")
    st.markdown("### 📈 Graph Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", len(filtered_papers))
    
    with col2:
        unique_organisms = filtered_papers['organisms'].nunique() if 'organisms' in filtered_papers.columns else 0
        st.metric("Unique Organisms", unique_organisms)
    
    with col3:
        unique_missions = filtered_papers['mission_type'].nunique() if 'mission_type' in filtered_papers.columns else 0
        st.metric("Mission Types", unique_missions)
    
    with col4:
        unique_clusters = filtered_papers['cluster'].nunique() if 'cluster' in filtered_papers.columns else 0
        st.metric("Research Clusters", unique_clusters)

def create_paper_network(papers_df, layout_type, node_size_metric, show_labels, max_nodes):
    """Create a network of papers based on similarities and clusters"""
    
    if len(papers_df) == 0:
        st.info("No papers available for network visualization.")
        return
    
    # Limit papers for performance
    display_papers = papers_df.head(max_nodes)
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes (papers)
    for idx, paper in display_papers.iterrows():
        title = paper.get('title', f'Paper {idx}')
        # Truncate long titles
        if len(title) > 50:
            title = title[:47] + "..."
        
        G.add_node(
            idx,
            title=title,
            cluster=paper.get('cluster', 0),
            year=paper.get('publication_year', 2020),
            organisms=paper.get('organisms', 'Unknown'),
            summary=paper.get('summary', 'No summary available')[:200] + "..." if len(str(paper.get('summary', ''))) > 200 else paper.get('summary', 'No summary available')
        )
    
    # Add edges based on shared attributes
    papers_list = list(display_papers.iterrows())
    for i, (idx1, paper1) in enumerate(papers_list):
        for j, (idx2, paper2) in enumerate(papers_list[i+1:], i+1):
            # Connect papers in same cluster
            if paper1.get('cluster') == paper2.get('cluster'):
                G.add_edge(idx1, idx2, weight=1.0, relationship="same_cluster")
            
            # Connect papers with same organisms
            elif paper1.get('organisms') == paper2.get('organisms'):
                G.add_edge(idx1, idx2, weight=0.7, relationship="same_organism")
            
            # Connect papers from same year
            elif paper1.get('publication_year') == paper2.get('publication_year'):
                G.add_edge(idx1, idx2, weight=0.3, relationship="same_year")
    
    # Create visualization
    if len(G.nodes()) > 0:
        create_plotly_network(G, layout_type, node_size_metric, show_labels, "paper")
    else:
        st.info("No connections found between papers with current filters.")

def create_organism_network(papers_df, layout_type, show_labels, max_nodes):
    """Create a network showing organism relationships"""
    
    if 'organisms' not in papers_df.columns:
        st.info("No organism data available for network visualization.")
        return
    
    # Create bipartite graph: papers and organisms
    G = nx.Graph()
    
    # Add organism nodes
    organisms = papers_df['organisms'].dropna().unique()[:max_nodes//2]
    for organism in organisms:
        G.add_node(f"org_{organism}", type="organism", label=organism, size=20)
    
    # Add paper nodes and connections
    for idx, paper in papers_df.head(max_nodes//2).iterrows():
        title = paper.get('title', f'Paper {idx}')
        if len(title) > 30:
            title = title[:27] + "..."
        
        G.add_node(
            f"paper_{idx}", 
            type="paper", 
            label=title,
            size=10,
            cluster=paper.get('cluster', 0)
        )
        
        # Connect to organism
        if pd.notna(paper.get('organisms')):
            organism = paper['organisms']
            if f"org_{organism}" in G.nodes():
                G.add_edge(f"paper_{idx}", f"org_{organism}")
    
    create_plotly_network(G, layout_type, "uniform", show_labels, "organism")

def create_mission_network(papers_df, layout_type, show_labels, max_nodes):
    """Create a network showing mission and research connections"""
    
    if 'mission_type' not in papers_df.columns:
        st.info("No mission data available for network visualization.")
        return
    
    # Create network with missions, clusters, and papers
    G = nx.Graph()
    
    # Add mission nodes
    missions = papers_df['mission_type'].dropna().unique()
    for mission in missions:
        G.add_node(f"mission_{mission}", type="mission", label=mission, size=25)
    
    # Add cluster nodes
    clusters = papers_df['cluster'].dropna().unique()
    for cluster in clusters:
        G.add_node(f"cluster_{cluster}", type="cluster", label=f"Cluster {cluster}", size=15)
    
    # Add paper nodes and connections
    for idx, paper in papers_df.head(max_nodes//3).iterrows():
        title = paper.get('title', f'Paper {idx}')
        if len(title) > 25:
            title = title[:22] + "..."
        
        G.add_node(f"paper_{idx}", type="paper", label=title, size=8)
        
        # Connect to mission
        if pd.notna(paper.get('mission_type')):
            mission = paper['mission_type']
            G.add_edge(f"paper_{idx}", f"mission_{mission}")
        
        # Connect to cluster
        if pd.notna(paper.get('cluster')):
            cluster = paper['cluster']
            G.add_edge(f"paper_{idx}", f"cluster_{cluster}")
    
    create_plotly_network(G, layout_type, "uniform", show_labels, "mission")

def create_plotly_network(G, layout_type, node_size_metric, show_labels, graph_type):
    """Create Plotly network visualization"""
    
    if len(G.nodes()) == 0:
        st.info("No nodes to display in the network.")
        return
    
    # Calculate layout
    if layout_type == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:  # spectral
        pos = nx.spectral_layout(G)
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node]
        label = node_data.get('label', str(node))
        node_text.append(label if show_labels else "")
        
        # Color based on type
        if graph_type == "paper":
            cluster = node_data.get('cluster', 0)
            node_color.append(cluster)
        elif graph_type == "organism":
            node_type = node_data.get('type', 'unknown')
            color_map = {'organism': 1, 'paper': 0}
            node_color.append(color_map.get(node_type, 0))
        else:  # mission
            node_type = node_data.get('type', 'unknown')
            color_map = {'mission': 2, 'cluster': 1, 'paper': 0}
            node_color.append(color_map.get(node_type, 0))
        
        # Size based on metric
        if node_size_metric == "degree":
            size = max(10, G.degree(node) * 5)
        elif node_size_metric == "cluster_size":
            size = node_data.get('size', 15)
        else:  # uniform
            size = 15
        
        node_size.append(size)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(125,125,125,0.5)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_labels else 'markers',
        text=node_text,
        textposition='middle center',
        textfont=dict(size=8),
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='viridis',
            line=dict(width=1, color='white'),
            showscale=True,
            colorbar=dict(title="Node Type")
        ),
        hovertemplate='<b>%{text}</b><extra></extra>',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text=f"Network visualization ({len(G.nodes())} nodes, {len(G.edges())} edges)",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    config = {
        'displayModeBar': True, 
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'scrollZoom': True
    }
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Display network statistics
    with st.expander("📊 Network Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nodes", len(G.nodes()))
            st.metric("Edges", len(G.edges()))
        
        with col2:
            if len(G.nodes()) > 0:
                density = nx.density(G)
                st.metric("Network Density", f"{density:.3f}")
                
                if nx.is_connected(G):
                    diameter = nx.diameter(G)
                    st.metric("Diameter", diameter)
                else:
                    components = nx.number_connected_components(G)
                    st.metric("Components", components)
        
        with col3:
            if len(G.nodes()) > 0:
                avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                st.metric("Avg Degree", f"{avg_degree:.2f}")
                
                clustering = nx.average_clustering(G)
                st.metric("Clustering Coeff", f"{clustering:.3f}")
    
    # Node selection for details
    if st.button("🔍 Analyze Selected Node", key=f"knowledge_graph_tab_analyze_node_button_{hash(str(G.nodes()))}"):
        st.info("Click on a node in the graph above to see detailed information. This feature would be enhanced with interactive callbacks in a full implementation.")