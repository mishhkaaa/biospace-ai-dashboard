"""
Research Papers tab component for browsing and searching papers
"""

import streamlit as st
import pandas as pd
import math
from utils.session_state import get_filtered_data

def render_research_papers_tab(data_loader):
    """Render the research papers tab with search and pagination"""
    
    st.markdown('<h2 class="tab-header">📄 Research Papers</h2>', unsafe_allow_html=True)
    
    # Get filtered data
    filtered_papers = get_filtered_data(data_loader)
    
    # Display filter summary
    if len(filtered_papers) != len(data_loader.papers_df):
        st.info(f"Showing {len(filtered_papers)} of {len(data_loader.papers_df)} papers based on current filters.")
    else:
        st.success(f"Showing all {len(filtered_papers)} papers.")
    
    if len(filtered_papers) == 0:
        st.warning("No papers match the current filters. Try adjusting your search criteria.")
        return
    
    # Sorting options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            options=['title', 'publication_year', 'cluster'],
            index=1,  # Default to publication_year
            help="Choose how to sort the papers",
            key="sort_by_select"
        )
    
    with col2:
        sort_order = st.selectbox(
            "Order:",
            options=['Descending', 'Ascending'],
            index=0,
            help="Choose sort order",
            key="sort_order_select"
        )
    
    with col3:
        papers_per_page = st.selectbox(
            "Papers per page:",
            options=[10, 25, 50, 100],
            index=0,
            help="Number of papers to show per page",
            key="papers_per_page_select"
        )
    
    # Sort the data
    ascending = sort_order == 'Ascending'
    filtered_papers = filtered_papers.sort_values(by=sort_by, ascending=ascending)
    
    # Pagination
    total_papers = len(filtered_papers)
    total_pages = math.ceil(total_papers / papers_per_page)
    
    # Pagination controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", disabled=st.session_state.current_page <= 1, key="first_top"):
            st.session_state.current_page = 1
            st.rerun()
    
    with col2:
        if st.button("⏪ Previous", disabled=st.session_state.current_page <= 1, key="prev_top"):
            st.session_state.current_page -= 1
            st.rerun()
    
    with col3:
        page_input = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=max(1, total_pages),
            value=st.session_state.current_page,
            step=1,
            help=f"Navigate to specific page (Total: {total_pages} pages)",
            key="page_input_top"
        )
        if page_input != st.session_state.current_page:
            st.session_state.current_page = page_input
            st.rerun()
    
    with col4:
        if st.button("Next ⏩", disabled=st.session_state.current_page >= total_pages, key="next_top"):
            st.session_state.current_page += 1
            st.rerun()
    
    with col5:
        if st.button("Last ⏭️", disabled=st.session_state.current_page >= total_pages, key="last_top"):
            st.session_state.current_page = total_pages
            st.rerun()
    
    # Calculate start and end indices for current page
    start_idx = (st.session_state.current_page - 1) * papers_per_page
    end_idx = min(start_idx + papers_per_page, total_papers)
    
    # Display papers for current page
    current_page_papers = filtered_papers.iloc[start_idx:end_idx]
    
    st.markdown(f"**Showing papers {start_idx + 1}-{end_idx} of {total_papers}**")
    st.markdown("---")
    
    # Display papers
    for idx, (paper_idx, paper) in enumerate(current_page_papers.iterrows()):
        with st.container():
            # Paper header
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"### {paper.get('title', 'Untitled Paper')}")
                
                # Paper metadata
                metadata_items = []
                if 'publication_year' in paper and pd.notna(paper['publication_year']):
                    metadata_items.append(f"📅 {int(paper['publication_year'])}")
                if 'cluster' in paper and pd.notna(paper['cluster']):
                    metadata_items.append(f"🎯 Cluster {paper['cluster']}")
                if 'organisms' in paper and pd.notna(paper['organisms']):
                    metadata_items.append(f"🧬 {paper['organisms']}")
                if 'mission_type' in paper and pd.notna(paper['mission_type']):
                    metadata_items.append(f"🚀 {paper['mission_type']}")
                
                if metadata_items:
                    st.markdown(" | ".join(metadata_items))
            
            with col2:
                # Expand button
                expand_key = f"expand_{paper_idx}_{idx}"
                if st.button("📖 Details", key=expand_key):
                    if expand_key not in st.session_state:
                        st.session_state[expand_key] = False
                    st.session_state[expand_key] = not st.session_state.get(expand_key, False)
                    st.rerun()
            
            # Paper summary
            if 'summary' in paper and pd.notna(paper['summary']):
                summary_text = str(paper['summary'])
                if len(summary_text) > 300:
                    st.markdown(f"**Summary:** {summary_text[:300]}...")
                else:
                    st.markdown(f"**Summary:** {summary_text}")
            
            # Keywords
            if 'keywords' in paper and pd.notna(paper['keywords']):
                keywords = str(paper['keywords']).split(',')
                keyword_tags = [f"`{kw.strip()}`" for kw in keywords[:5]]  # Show first 5 keywords
                st.markdown(f"**Keywords:** {' '.join(keyword_tags)}")
            
            # Expanded details
            if st.session_state.get(f"expand_{paper_idx}_{idx}", False):
                with st.expander("📋 Full Details", expanded=True):
                    
                    # Full summary
                    if 'summary' in paper and pd.notna(paper['summary']):
                        st.markdown("**Full Summary:**")
                        st.write(paper['summary'])
                    
                    # Additional metadata
                    st.markdown("**Metadata:**")
                    metadata_df = pd.DataFrame({
                        'Field': ['Paper ID', 'Publication Year', 'Cluster', 'Organisms', 'Mission Type', 'Keywords'],
                        'Value': [
                            str(paper_idx),
                            str(paper.get('publication_year', 'N/A')),
                            str(paper.get('cluster', 'N/A')),
                            str(paper.get('organisms', 'N/A')),
                            str(paper.get('mission_type', 'N/A')),
                            str(paper.get('keywords', 'N/A'))
                        ]
                    })
                    # Ensure all columns are strings for Arrow compatibility
                    metadata_df['Field'] = metadata_df['Field'].astype(str)
                    metadata_df['Value'] = metadata_df['Value'].astype(str)
                    st.dataframe(metadata_df, hide_index=True, use_container_width=True)
                    
                    # Citation information
                    st.markdown("**Citation:**")
                    citation = f"NASA Space Biology Research Paper #{paper_idx} ({paper.get('publication_year', 'Year Unknown')}). {paper.get('title', 'Untitled')}."
                    st.code(citation, language="text")
                    
                    # Related papers (mock)
                    if hasattr(data_loader, 'similarities_data') and data_loader.similarities_data:
                        st.markdown("**Related Papers:**")
                        st.info("Similar papers based on content analysis would be shown here.")
                    
                    # Download/export options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📄 Export as Text", key=f"export_text_{paper_idx}"):
                            export_text = f"""
Title: {paper.get('title', 'Untitled')}
Year: {paper.get('publication_year', 'Unknown')}
Summary: {paper.get('summary', 'No summary available')}
Keywords: {paper.get('keywords', 'No keywords')}
Cluster: {paper.get('cluster', 'Unknown')}
Organisms: {paper.get('organisms', 'Unknown')}
Mission: {paper.get('mission_type', 'Unknown')}
                            """
                            st.download_button(
                                label="Download",
                                data=export_text,
                                file_name=f"paper_{paper_idx}.txt",
                                mime="text/plain",
                                key=f"download_text_{paper_idx}"
                            )
                    
                    with col2:
                        if st.button("📊 View in Graph", key=f"view_graph_{paper_idx}"):
                            st.info("This would navigate to the Knowledge Graph tab with this paper highlighted.")
            
            st.markdown("---")
    
    # Bottom pagination
    st.markdown("### Navigation")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⏮️ First Page", disabled=st.session_state.current_page <= 1, key="bottom_first"):
            st.session_state.current_page = 1
            st.rerun()
    
    with col2:
        st.markdown(f"**Page {st.session_state.current_page} of {total_pages}**")
        progress = st.session_state.current_page / total_pages if total_pages > 0 else 0
        st.progress(progress)
    
    with col3:
        if st.button("Last Page ⏭️", disabled=st.session_state.current_page >= total_pages, key="bottom_last"):
            st.session_state.current_page = total_pages
            st.rerun()
    
    # Summary statistics for current view
    st.markdown("---")
    st.markdown("### Current View Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Papers on Page", len(current_page_papers))
    
    with col2:
        unique_years = current_page_papers['publication_year'].nunique()
        st.metric("Unique Years", unique_years)
    
    with col3:
        unique_clusters = current_page_papers['cluster'].nunique() if 'cluster' in current_page_papers.columns else 0
        st.metric("Unique Clusters", unique_clusters)
    
    with col4:
        unique_organisms = current_page_papers['organisms'].nunique() if 'organisms' in current_page_papers.columns else 0
        st.metric("Unique Organisms", unique_organisms)
    
    # Export current view
    st.markdown("---")
    if st.button("📥 Export Current View", key="export_current_view"):
        csv_data = current_page_papers.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"nasa_papers_page_{st.session_state.current_page}.csv",
            mime="text/csv",
            key="download_csv"
        )