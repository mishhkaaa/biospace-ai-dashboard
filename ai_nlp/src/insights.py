"""
NASA Bioscience AI Pipeline - Enhanced Insights Module
Advanced insights generation with comprehensive analysis and visualization-ready outputs
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

from config import config
from utils import (
    PipelineLogger, 
    load_and_validate_data, 
    extract_publication_year,
    save_results_with_metadata
)

class AdvancedInsightsGenerator:
    """
    Advanced insights generation with multiple analysis techniques
    """
    
    def __init__(self):
        self.logger = PipelineLogger("InsightsGenerator")
        self.current_year = datetime.now().year
    
    def identify_knowledge_gaps(self, df: pd.DataFrame, 
                              threshold: Optional[int] = None) -> Dict[str, Any]:
        """
        Identify knowledge gaps with detailed analysis
        
        Args:
            df: DataFrame with cluster assignments
            threshold: Minimum papers per cluster (uses config if None)
            
        Returns:
            Dictionary with knowledge gap analysis
        """
        if threshold is None:
            threshold = config.insights.knowledge_gap_threshold
        
        self.logger.info(f"Identifying knowledge gaps with threshold: {threshold}")
        
        # Calculate cluster sizes
        cluster_counts = df['cluster_id'].value_counts().sort_index()
        
        # Identify gaps
        gap_clusters = cluster_counts[cluster_counts < threshold]
        well_represented = cluster_counts[cluster_counts >= threshold]
        
        # Additional analysis
        total_papers = len(df)
        papers_in_gaps = gap_clusters.sum()
        
        gap_analysis = {
            'knowledge_gap_clusters': gap_clusters.index.tolist(),
            'gap_cluster_sizes': gap_clusters.to_dict(),
            'total_gap_clusters': len(gap_clusters),
            'total_papers_in_gaps': int(papers_in_gaps),
            'percentage_papers_in_gaps': float(papers_in_gaps / total_papers * 100),
            'well_represented_clusters': well_represented.index.tolist(),
            'avg_papers_per_gap_cluster': float(gap_clusters.mean()) if len(gap_clusters) > 0 else 0,
            'threshold_used': threshold,
            'recommendations': []
        }
        
        # Generate recommendations
        if len(gap_clusters) > 0:
            gap_analysis['recommendations'].append(
                f"Consider increasing research focus on {len(gap_clusters)} underexplored areas"
            )
            
            if papers_in_gaps / total_papers > 0.3:
                gap_analysis['recommendations'].append(
                    "High fragmentation detected - consider research consolidation strategies"
                )
        
        self.logger.info(f"Found {len(gap_clusters)} knowledge gaps affecting {papers_in_gaps} papers")
        
        return gap_analysis
    
    def analyze_publication_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive publication trends analysis
        
        Args:
            df: DataFrame with year and cluster information
            
        Returns:
            Dictionary with trends analysis
        """
        self.logger.info("Analyzing publication trends")
        
        # Ensure we have year data
        df = extract_publication_year(df, config.data.date_columns)
        
        # Basic trends calculation
        trends = df.groupby(['cluster_id', 'year']).size().reset_index(name='paper_count')
        
        # Time-based analysis
        current_year = self.current_year
        recent_years = [current_year - i for i in range(5)]  # Last 5 years
        
        # Calculate trends per cluster
        cluster_trends = {}
        yearly_totals = df.groupby('year').size()
        
        for cluster_id in df['cluster_id'].unique():
            cluster_data = df[df['cluster_id'] == cluster_id]
            cluster_yearly = cluster_data.groupby('year').size()
            
            # Calculate growth metrics
            recent_papers = cluster_data[cluster_data['year'].isin(recent_years)]
            older_papers = cluster_data[~cluster_data['year'].isin(recent_years)]
            
            trend_direction = "stable"
            if len(recent_papers) > 0 and len(older_papers) > 0:
                recent_rate = len(recent_papers) / len(recent_years)
                older_rate = len(older_papers) / max(1, len(cluster_data['year'].unique()) - len(recent_years))
                
                if recent_rate > older_rate * 1.2:
                    trend_direction = "growing"
                elif recent_rate < older_rate * 0.8:
                    trend_direction = "declining"
            
            cluster_trends[int(cluster_id)] = {
                'total_papers': len(cluster_data),
                'recent_papers': len(recent_papers),
                'trend_direction': trend_direction,
                'peak_year': cluster_yearly.idxmax() if len(cluster_yearly) > 0 else None,
                'peak_papers': int(cluster_yearly.max()) if len(cluster_yearly) > 0 else 0,
                'years_active': len(cluster_yearly),
                'avg_papers_per_year': float(cluster_yearly.mean()) if len(cluster_yearly) > 0 else 0
            }
        
        # Overall field trends
        field_trends = {
            'total_publications': len(df),
            'years_covered': len(yearly_totals),
            'peak_year': yearly_totals.idxmax() if len(yearly_totals) > 0 else None,
            'peak_publications': int(yearly_totals.max()) if len(yearly_totals) > 0 else 0,
            'recent_growth': self._calculate_recent_growth(yearly_totals, recent_years),
            'publication_timeline': yearly_totals.to_dict()
        }
        
        # Generate insights
        insights = []
        
        # Growing clusters
        growing_clusters = [cid for cid, data in cluster_trends.items() 
                          if data['trend_direction'] == 'growing']
        if growing_clusters:
            insights.append(f"Growing research areas: Clusters {growing_clusters}")
        
        # Declining clusters
        declining_clusters = [cid for cid, data in cluster_trends.items() 
                            if data['trend_direction'] == 'declining']
        if declining_clusters:
            insights.append(f"Declining research areas: Clusters {declining_clusters}")
        
        # Peak activity
        if field_trends['peak_year']:
            insights.append(f"Peak research activity in {field_trends['peak_year']}")
        
        return {
            'trends_data': trends,
            'cluster_trends': cluster_trends,
            'field_trends': field_trends,
            'insights': insights,
            'recent_years': recent_years
        }
    
    def _calculate_recent_growth(self, yearly_totals: pd.Series, recent_years: List[int]) -> float:
        """Calculate growth rate for recent years"""
        try:
            recent_data = yearly_totals[yearly_totals.index.isin(recent_years)]
            if len(recent_data) >= 2:
                first_year_count = recent_data.iloc[0]
                last_year_count = recent_data.iloc[-1]
                years_span = len(recent_data) - 1
                
                if years_span > 0 and first_year_count > 0:
                    growth_rate = ((last_year_count / first_year_count) ** (1/years_span) - 1) * 100
                    return float(growth_rate)
            
            return 0.0
        except:
            return 0.0
    
    def detect_consensus_disagreement(self, df: pd.DataFrame, 
                                    cluster_keywords_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced consensus/disagreement detection using multiple indicators
        
        Args:
            df: DataFrame with cluster and content information
            cluster_keywords_path: Path to cluster keywords file for additional analysis
            
        Returns:
            Dictionary with consensus/disagreement analysis
        """
        self.logger.info("Detecting consensus vs disagreement patterns")
        
        consensus_results = []
        
        # Load cluster keywords if available
        cluster_keywords = {}
        if cluster_keywords_path:
            try:
                keywords_df = pd.read_csv(cluster_keywords_path)
                cluster_keywords = dict(zip(keywords_df['cluster_id'], keywords_df['keywords']))
            except Exception as e:
                self.logger.warning(f"Could not load cluster keywords: {str(e)}")
        
        # Analyze each cluster
        for cluster_id, group in df.groupby('cluster_id'):
            cluster_analysis = self._analyze_cluster_consensus(group, cluster_id, cluster_keywords.get(cluster_id, ""))
            consensus_results.append(cluster_analysis)
        
        consensus_df = pd.DataFrame(consensus_results)
        
        # Overall statistics
        consensus_stats = {
            'total_clusters': len(consensus_df),
            'consensus_clusters': (consensus_df['consensus_status'] == 'consensus').sum(),
            'disagreement_clusters': (consensus_df['consensus_status'] == 'disagreement').sum(),
            'mixed_clusters': (consensus_df['consensus_status'] == 'mixed').sum(),
            'consensus_percentage': float((consensus_df['consensus_status'] == 'consensus').mean() * 100)
        }
        
        # Generate insights
        insights = []
        
        if consensus_stats['consensus_percentage'] > 70:
            insights.append("High consensus across research areas - field shows alignment")
        elif consensus_stats['consensus_percentage'] < 30:
            insights.append("High disagreement across research areas - field shows active debate")
        else:
            insights.append("Balanced mix of consensus and disagreement areas")
        
        # Identify specific areas of concern
        high_disagreement = consensus_df[
            (consensus_df['consensus_status'] == 'disagreement') & 
            (consensus_df['paper_count'] >= config.insights.min_papers_for_trend)
        ]
        
        if len(high_disagreement) > 0:
            insights.append(f"Major disagreement areas: Clusters {high_disagreement['cluster_id'].tolist()}")
        
        return {
            'consensus_data': consensus_df,
            'consensus_statistics': consensus_stats,
            'insights': insights
        }
    
    def _analyze_cluster_consensus(self, group: pd.DataFrame, cluster_id: int, 
                                 cluster_keywords: str) -> Dict[str, Any]:
        """
        Analyze consensus within a single cluster
        
        Args:
            group: DataFrame for single cluster
            cluster_id: Cluster identifier
            cluster_keywords: Keywords for the cluster
            
        Returns:
            Dictionary with cluster consensus analysis
        """
        # Check different content columns for diversity
        diversity_indicators = {}
        
        # Check findings diversity
        if 'finding' in group.columns:
            findings = group['finding'].dropna().unique()
            diversity_indicators['finding_diversity'] = len(findings)
        
        # Check methodology diversity (if available)
        if 'method' in group.columns:
            methods = group['method'].dropna().unique()
            diversity_indicators['method_diversity'] = len(methods)
        
        # Check organism diversity
        if 'organism' in group.columns:
            organisms = group['organism'].dropna().unique()
            diversity_indicators['organism_diversity'] = len(organisms)
        
        # Check mission diversity
        if 'mission' in group.columns:
            missions = group['mission'].dropna().unique()
            diversity_indicators['mission_diversity'] = len(missions)
        
        # Check temporal diversity
        if 'year' in group.columns:
            years = group['year'].dropna().unique()
            year_span = len(years)
            diversity_indicators['temporal_span'] = year_span
        
        # Calculate consensus score
        paper_count = len(group)
        consensus_score = 0.0
        
        # Lower diversity = higher consensus
        if 'finding_diversity' in diversity_indicators:
            finding_consensus = 1.0 - min(diversity_indicators['finding_diversity'] / paper_count, 1.0)
            consensus_score += finding_consensus * 0.4
        
        if 'organism_diversity' in diversity_indicators:
            organism_consensus = 1.0 - min(diversity_indicators['organism_diversity'] / paper_count, 1.0)
            consensus_score += organism_consensus * 0.3
        
        if 'mission_diversity' in diversity_indicators:
            mission_consensus = 1.0 - min(diversity_indicators['mission_diversity'] / paper_count, 1.0)
            consensus_score += mission_consensus * 0.2
        
        if 'temporal_span' in diversity_indicators:
            # Recent papers suggest ongoing consensus
            temporal_consensus = 1.0 - min(year_span / 10.0, 1.0) if year_span else 0.5
            consensus_score += temporal_consensus * 0.1
        
        # Normalize consensus score
        weight_sum = sum([0.4, 0.3, 0.2, 0.1])  # Adjust based on available indicators
        consensus_score = consensus_score / weight_sum if weight_sum > 0 else 0.5
        
        # Determine consensus status
        if consensus_score >= config.insights.consensus_threshold:
            consensus_status = "consensus"
        elif consensus_score <= config.insights.disagreement_threshold:
            consensus_status = "disagreement"
        else:
            consensus_status = "mixed"
        
        return {
            'cluster_id': cluster_id,
            'paper_count': paper_count,
            'consensus_status': consensus_status,
            'consensus_score': float(consensus_score),
            'keywords': cluster_keywords,
            **diversity_indicators
        }
    
    def generate_research_opportunities(self, knowledge_gaps: Dict, trends: Dict, 
                                      consensus: Dict) -> Dict[str, Any]:
        """
        Generate research opportunities based on all analyses
        
        Args:
            knowledge_gaps: Knowledge gaps analysis
            trends: Trends analysis
            consensus: Consensus analysis
            
        Returns:
            Dictionary with research opportunities
        """
        self.logger.info("Generating research opportunities")
        
        opportunities = []
        priorities = []
        
        # Knowledge gap opportunities
        gap_clusters = knowledge_gaps['knowledge_gap_clusters']
        if gap_clusters:
            opportunities.append({
                'type': 'knowledge_gap',
                'description': f"Expand research in {len(gap_clusters)} underexplored areas",
                'clusters': gap_clusters,
                'priority': 'high' if knowledge_gaps['percentage_papers_in_gaps'] > 30 else 'medium'
            })
        
        # Declining trend opportunities
        declining_clusters = [cid for cid, data in trends['cluster_trends'].items() 
                            if data['trend_direction'] == 'declining' and data['total_papers'] >= 5]
        if declining_clusters:
            opportunities.append({
                'type': 'revival_opportunity',
                'description': f"Revitalize research in {len(declining_clusters)} declining areas",
                'clusters': declining_clusters,
                'priority': 'medium'
            })
        
        # Disagreement areas (need consensus building)
        disagreement_clusters = consensus['consensus_data'][
            consensus['consensus_data']['consensus_status'] == 'disagreement'
        ]['cluster_id'].tolist()
        
        if disagreement_clusters:
            opportunities.append({
                'type': 'consensus_building',
                'description': f"Build consensus in {len(disagreement_clusters)} areas with conflicting findings",
                'clusters': disagreement_clusters,
                'priority': 'high'
            })
        
        # Growing areas (scaling opportunities)
        growing_clusters = [cid for cid, data in trends['cluster_trends'].items() 
                          if data['trend_direction'] == 'growing']
        if growing_clusters:
            opportunities.append({
                'type': 'scaling_opportunity',
                'description': f"Scale up research in {len(growing_clusters)} rapidly growing areas",
                'clusters': growing_clusters,
                'priority': 'medium'
            })
        
        # Cross-cluster collaboration opportunities
        if len(trends['cluster_trends']) > 3:
            opportunities.append({
                'type': 'interdisciplinary',
                'description': "Foster interdisciplinary collaboration across research clusters",
                'clusters': list(trends['cluster_trends'].keys())[:5],  # Top 5 clusters
                'priority': 'low'
            })
        
        # Prioritize opportunities
        high_priority = [op for op in opportunities if op['priority'] == 'high']
        medium_priority = [op for op in opportunities if op['priority'] == 'medium']
        low_priority = [op for op in opportunities if op['priority'] == 'low']
        
        return {
            'opportunities': opportunities,
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'low_priority': low_priority,
            'total_opportunities': len(opportunities)
        }
    
    def analyze_journal_impact_and_patterns(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze journal-based patterns and publication strategies
        
        Args:
            df: DataFrame with journal information
            
        Returns:
            Dictionary with journal analysis or None if journal field missing/empty
        """
        # Check if journal column exists and has data
        if 'journal' not in df.columns:
            self.logger.info("Journal column not found - skipping journal analysis")
            return None
        
        # Check if journal data is meaningful (not all empty/NaN)
        valid_journals = df['journal'].dropna()
        valid_journals = valid_journals[valid_journals.str.strip() != '']
        
        if len(valid_journals) == 0:
            self.logger.info("No valid journal data found - skipping journal analysis")
            return None
            
        if len(valid_journals) < len(df) * 0.1:  # Less than 10% have journal data
            self.logger.warning(f"Limited journal data: only {len(valid_journals)}/{len(df)} papers have journal info")
            
        self.logger.info(f"Analyzing journal patterns for {len(valid_journals)} papers with journal data")
        
        # Define journal tiers (can be customized based on field)
        journal_tiers = {
            'tier_1': ['Nature', 'Science', 'Cell', 'Nature Biotechnology', 'Nature Medicine'],
            'tier_2': ['PNAS', 'Nature Communications', 'Cell Reports', 'EMBO Journal'],
            'tier_3': ['PLOS Biology', 'eLife', 'Journal of Biological Chemistry'],
            'tier_4': ['PLOS ONE', 'Scientific Reports', 'Frontiers in']  # Frontiers in* journals
        }
        
        # Create working DataFrame with valid journal data
        df_with_journals = df[df['journal'].notna() & (df['journal'].str.strip() != '')].copy()
        
        # Journal distribution analysis
        journal_counts = df_with_journals['journal'].value_counts()
        
        # Cluster-journal analysis
        cluster_journal_analysis = {}
        if 'cluster_id' in df_with_journals.columns:
            for cluster_id in df_with_journals['cluster_id'].unique():
                cluster_data = df_with_journals[df_with_journals['cluster_id'] == cluster_id]
                cluster_journals = cluster_data['journal'].value_counts()
                
                cluster_journal_analysis[int(cluster_id)] = {
                    'primary_journals': cluster_journals.head(3).to_dict(),
                    'journal_diversity': len(cluster_journals),
                    'paper_count': len(cluster_data)
                }
        
        # Journal tier classification
        def classify_journal_tier(journal_name):
            journal_lower = journal_name.lower()
            for tier, journals in journal_tiers.items():
                for tier_journal in journals:
                    if tier_journal.lower() in journal_lower:
                        return tier
            return 'specialized'  # Default for unclassified journals
        
        df_with_journals['journal_tier'] = df_with_journals['journal'].apply(classify_journal_tier)
        tier_distribution = df_with_journals['journal_tier'].value_counts().to_dict()
        
        # Research impact indicators
        impact_analysis = {
            'high_impact_papers': len(df_with_journals[df_with_journals['journal_tier'].isin(['tier_1', 'tier_2'])]),
            'specialized_research': len(df_with_journals[df_with_journals['journal_tier'] == 'specialized']),
            'broad_appeal_papers': len(df_with_journals[df_with_journals['journal_tier'].isin(['tier_3', 'tier_4'])])
        }
        
        # Publication strategy recommendations
        recommendations = []
        
        # Find clusters with high-impact potential but low-tier publications
        for cluster_id, analysis in cluster_journal_analysis.items():
            if analysis['paper_count'] >= 3:  # Only for clusters with sufficient papers
                cluster_papers = df_with_journals[df_with_journals['cluster_id'] == cluster_id]
                high_tier_ratio = len(cluster_papers[cluster_papers['journal_tier'].isin(['tier_1', 'tier_2'])]) / len(cluster_papers)
                
                if high_tier_ratio < 0.3:  # Less than 30% in high-tier journals
                    recommendations.append({
                        'cluster_id': cluster_id,
                        'type': 'publication_upgrade',
                        'description': f'Cluster {cluster_id} has potential for higher-impact journal submissions',
                        'current_tier_distribution': cluster_papers['journal_tier'].value_counts().to_dict(),
                        'recommendation': 'Target tier 1-2 journals for breakthrough findings'
                    })
        
        # Identify publication gaps
        gaps = []
        if len(tier_distribution.get('tier_1', 0)) == 0:
            gaps.append('No publications in top-tier journals (Nature, Science, Cell)')
        if len(tier_distribution.get('specialized', 0)) < len(df_with_journals) * 0.1:
            gaps.append('Limited presence in specialized field journals')
            
        return {
            'journal_distribution': journal_counts.head(10).to_dict(),
            'tier_distribution': tier_distribution,
            'cluster_journal_patterns': cluster_journal_analysis,
            'impact_analysis': impact_analysis,
            'publication_recommendations': recommendations,
            'publication_gaps': gaps,
            'total_journals': len(journal_counts),
            'papers_with_journal_data': len(df_with_journals),
            'journal_coverage': len(df_with_journals) / len(df) * 100
        }

def create_comprehensive_insights(summaries_path: str, clusters_path: str, 
                                cluster_keywords_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create comprehensive insights from all pipeline outputs
    
    Args:
        summaries_path: Path to paper summaries
        clusters_path: Path to cluster assignments
        cluster_keywords_path: Optional path to cluster keywords
        
    Returns:
        Dictionary with all insights
    """
    logger = PipelineLogger("Comprehensive Insights")
    
    # Load data
    summaries_df = load_and_validate_data(summaries_path, ['paper_id'])
    clusters_df = load_and_validate_data(clusters_path, ['paper_id', 'cluster_id'])
    
    # Merge data
    df = summaries_df.merge(clusters_df, on='paper_id', how='inner')
    
    logger.info(f"Loaded {len(df)} papers for insights generation")
    
    # Initialize insights generator
    insights_generator = AdvancedInsightsGenerator()
    
    # Generate all insights
    logger.info("Generating knowledge gaps analysis")
    knowledge_gaps = insights_generator.identify_knowledge_gaps(df)
    
    logger.info("Generating publication trends analysis")
    trends = insights_generator.analyze_publication_trends(df)
    
    logger.info("Generating consensus/disagreement analysis")
    consensus = insights_generator.detect_consensus_disagreement(df, cluster_keywords_path)
    
    logger.info("Generating research opportunities")
    opportunities = insights_generator.generate_research_opportunities(knowledge_gaps, trends, consensus)
    
    # Generate journal analysis (optional - only if journal data is available)
    journal_analysis = insights_generator.analyze_journal_impact_and_patterns(df)
    
    # Create comprehensive summary
    comprehensive_insights = {
        'knowledge_gaps': knowledge_gaps,
        'publication_trends': trends,
        'consensus_analysis': consensus,
        'research_opportunities': opportunities,
        'summary_statistics': {
            'total_papers': len(df),
            'total_clusters': len(df['cluster_id'].unique()),
            'knowledge_gap_clusters': len(knowledge_gaps['knowledge_gap_clusters']),
            'growing_research_areas': len([cid for cid, data in trends['cluster_trends'].items() 
                                         if data['trend_direction'] == 'growing']),
            'consensus_areas': int(consensus['consensus_statistics']['consensus_clusters']),
            'disagreement_areas': int(consensus['consensus_statistics']['disagreement_clusters']),
            'total_opportunities': opportunities['total_opportunities'],
            'high_priority_opportunities': len(opportunities['high_priority'])
        }
    }
    
    # Add journal analysis if available
    if journal_analysis is not None:
        comprehensive_insights['journal_analysis'] = journal_analysis
        comprehensive_insights['summary_statistics']['journal_coverage'] = journal_analysis['journal_coverage']
    
    logger.success("Comprehensive insights generation completed")
    
    return comprehensive_insights

def main():
    """Main execution function for insights pipeline"""
    logger = PipelineLogger("Insights Pipeline")
    
    try:
        # Load data paths
        summaries_path = config.get_output_file_path("paper_summaries.csv", "summaries")
        clusters_path = config.get_output_file_path("paper_clusters.csv", "clusters")
        cluster_keywords_path = config.get_output_file_path("cluster_keywords_summaries.csv", "clusters")
        
        # Generate comprehensive insights
        comprehensive_insights = create_comprehensive_insights(
            summaries_path, clusters_path, cluster_keywords_path
        )
        
        # Save individual insight components
        insights_dir = config.paths.insights_dir
        
        # Knowledge gaps
        gaps_path = config.get_output_file_path("knowledge_gaps.json", "insights")
        save_results_with_metadata(
            comprehensive_insights['knowledge_gaps'], 
            gaps_path,
            {'component': 'knowledge_gaps', 'description': 'Underexplored research areas'}
        )
        
        # Publication trends
        trends_path = config.get_output_file_path("publication_trends.csv", "insights")
        save_results_with_metadata(
            comprehensive_insights['publication_trends']['trends_data'], 
            trends_path,
            {'component': 'publication_trends', 'description': 'Historical publication patterns'}
        )
        
        # Consensus/disagreement
        consensus_path = config.get_output_file_path("consensus_disagreement.csv", "insights")
        save_results_with_metadata(
            comprehensive_insights['consensus_analysis']['consensus_data'], 
            consensus_path,
            {'component': 'consensus_analysis', 'description': 'Research consensus patterns'}
        )
        
        # Research opportunities
        opportunities_path = config.get_output_file_path("research_opportunities.json", "insights")
        save_results_with_metadata(
            comprehensive_insights['research_opportunities'], 
            opportunities_path,
            {'component': 'research_opportunities', 'description': 'Strategic research recommendations'}
        )
        
        # Journal analysis (if available)
        if 'journal_analysis' in comprehensive_insights:
            # Journal analysis summary
            journal_analysis_path = config.get_output_file_path("journal_analysis.json", "insights")
            save_results_with_metadata(
                comprehensive_insights['journal_analysis'], 
                journal_analysis_path,
                {'component': 'journal_analysis', 'description': 'Journal impact and publication strategy analysis'}
            )
            
            # Publication recommendations
            if comprehensive_insights['journal_analysis']['publication_recommendations']:
                recommendations_path = config.get_output_file_path("publication_recommendations.json", "insights")
                save_results_with_metadata(
                    comprehensive_insights['journal_analysis']['publication_recommendations'], 
                    recommendations_path,
                    {'component': 'publication_recommendations', 'description': 'Strategic publication venue recommendations'}
                )
        
        # Comprehensive insights summary
        summary_path = config.get_output_file_path("comprehensive_insights.json", "insights")
        save_results_with_metadata(
            comprehensive_insights, 
            summary_path,
            {'component': 'comprehensive_insights', 'description': 'Complete insights analysis'}
        )
        
        # Log completion
        stats = comprehensive_insights['summary_statistics']
        logger.success("Insights pipeline completed successfully!")
        logger.info(f"Generated insights for {stats['total_papers']} papers across {stats['total_clusters']} clusters")
        logger.info(f"Identified {stats['knowledge_gap_clusters']} knowledge gaps and {stats['total_opportunities']} research opportunities")
        
        return comprehensive_insights
        
    except Exception as e:
        logger.error(f"Insights pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
