"""
Enhanced QA Pipeline for the dashboard chatbot with intelligent responses
"""

import pandas as pd
import re
from typing import Dict, List, Tuple


class QAPipeline:
    """Enhanced QA Pipeline implementation that provides intelligent responses using the papers database"""
    
    def __init__(self):
        self.initialized = True
    
    def answer_question(self, query: str, papers_data: pd.DataFrame) -> str:
        """Generate an intelligent answer using the papers database."""
        try:
            if papers_data is None or papers_data.empty:
                return "Sorry, no research papers are currently available in the database."
            
            # Clean and prepare query
            query_lower = query.lower().strip()
            
            if not query_lower or query_lower == "test":
                return f"I have access to {len(papers_data)} NASA space biology research papers. Ask me about specific topics!"
            
            # Handle different types of queries
            if self._is_counting_query(query_lower):
                return self._handle_counting_query(query_lower, papers_data)
            elif self._is_listing_query(query_lower):
                return self._handle_listing_query(query_lower, papers_data)
            else:
                return self._handle_general_query(query_lower, papers_data)
                
        except Exception as e:
            return f"I encountered an error processing your question: {str(e)}"
    
    def semantic_search(self, query, papers_df=None, top_k=5):
        """Enhanced semantic search using keyword matching"""
        
        if papers_df is None or papers_df.empty:
            return {
                'answer': f"I don't have access to the research papers database to answer: {query}",
                'citations': [],
                'confidence': 0.1
            }
        
        # Use the enhanced answer_question method
        answer = self.answer_question(query, papers_df)
        
        return {
            'answer': answer,
            'citations': [f"Analysis of {len(papers_df)} research papers"],
            'confidence': 0.8
        }
    
    def fact_based_search(self, query, papers_df=None):
        """Enhanced fact-based search for counting and listing queries"""
        
        if papers_df is None or papers_df.empty:
            return {
                'answer': f"I don't have access to the research papers database to answer: {query}",
                'citations': [],
                'confidence': 0.1
            }
        
        # Use the enhanced answer_question method
        answer = self.answer_question(query, papers_df)
        
        return {
            'answer': answer,
            'citations': [f"Database analysis of {len(papers_df)} papers"],
            'confidence': 0.8
        }
    
    def _is_counting_query(self, query: str) -> bool:
        """Check if query is asking for counts."""
        counting_words = ['how many', 'number of', 'count', 'total']
        return any(word in query for word in counting_words)
    
    def _is_listing_query(self, query: str) -> bool:
        """Check if query is asking for lists."""
        listing_words = ['what are', 'list', 'which', 'what organisms', 'what studies']
        return any(word in query for word in listing_words)
    
    def _handle_counting_query(self, query: str, papers_data: pd.DataFrame) -> str:
        """Handle counting-type queries."""
        try:
            if 'paper' in query or 'study' in query or 'research' in query:
                total_papers = len(papers_data)
                relevant_papers = self._find_relevant_papers(query, papers_data)
                
                if len(relevant_papers) > 0:
                    return f"I found **{len(relevant_papers)}** research papers related to your query out of {total_papers} total papers."
                else:
                    return f"The database contains **{total_papers}** research papers in total. Try being more specific."
            
            elif 'organism' in query:
                if 'organisms' in papers_data.columns:
                    unique_organisms = papers_data['organisms'].nunique()
                    top_organisms = papers_data['organisms'].value_counts().head(5)
                    organisms_list = ", ".join(top_organisms.index.tolist())
                    return f"Studies include **{unique_organisms}** different organisms. Most studied: {organisms_list}."
                else:
                    return "Organism data is not specifically categorized in the current database structure."
            
            else:
                return f"I found **{len(papers_data)}** total research papers. Please specify what you'd like me to count."
                
        except Exception as e:
            return f"Error processing counting query: {str(e)}"
    
    def _handle_listing_query(self, query: str, papers_data: pd.DataFrame) -> str:
        """Handle listing-type queries."""
        try:
            relevant_papers = self._find_relevant_papers(query, papers_data)
            
            if len(relevant_papers) == 0:
                return f"No specific matches found. The database contains {len(papers_data)} papers on space biology. Try more specific terms."
            
            response = f"I found **{len(relevant_papers)}** relevant papers:\n\n"
            
            for i, (_, paper) in enumerate(relevant_papers.head(5).iterrows()):
                title = paper.get('title', 'Untitled')
                year = paper.get('publication_year', 'Unknown year')
                response += f"{i+1}. **{title}** ({year})\n"
            
            if len(relevant_papers) > 5:
                response += f"\n... and {len(relevant_papers) - 5} more papers."
            
            return response
            
        except Exception as e:
            return f"Error processing listing query: {str(e)}"
    
    def _handle_general_query(self, query: str, papers_data: pd.DataFrame) -> str:
        """Handle general questions by finding relevant papers."""
        try:
            relevant_papers = self._find_relevant_papers(query, papers_data)
            
            if len(relevant_papers) == 0:
                return self._generate_helpful_fallback(query, papers_data)
            
            response = f"Based on **{len(relevant_papers)}** relevant papers:\n\n"
            
            for i, (_, paper) in enumerate(relevant_papers.head(3).iterrows()):
                title = paper.get('title', 'Untitled')
                year = paper.get('publication_year', 'Unknown year')
                summary = paper.get('summary', '')
                
                response += f"**{i+1}. {title}** ({year})\n"
                
                if summary and len(summary) > 50:
                    relevant_text = self._extract_relevant_sentences(query, summary)
                    if relevant_text:
                        response += f"*Key findings: {relevant_text}*\n\n"
                    else:
                        short_summary = summary[:150] + "..." if len(summary) > 150 else summary
                        response += f"*{short_summary}*\n\n"
            
            if len(relevant_papers) > 3:
                response += f"💡 Found {len(relevant_papers) - 3} additional relevant papers."
            
            return response
            
        except Exception as e:
            return f"Error processing general query: {str(e)}"
    
    def _find_relevant_papers(self, query: str, papers_data: pd.DataFrame) -> pd.DataFrame:
        """Find papers relevant to the query using keyword matching."""
        try:
            keywords = self._extract_keywords(query)
            
            if not keywords:
                return pd.DataFrame()
            
            scores = []
            for _, paper in papers_data.iterrows():
                score = self._calculate_relevance_score(keywords, paper)
                scores.append(score)
            
            papers_data_copy = papers_data.copy()
            papers_data_copy['relevance_score'] = scores
            
            relevant = papers_data_copy[papers_data_copy['relevance_score'] > 0]
            return relevant.sort_values('relevance_score', ascending=False)
            
        except Exception as e:
            return pd.DataFrame()
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from the query."""
        stop_words = {
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'the', 'a', 'an', 
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'about', 'tell', 'me'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _calculate_relevance_score(self, keywords: List[str], paper: Dict) -> float:
        """Calculate relevance score for a paper based on keyword matches."""
        score = 0.0
        
        fields_to_check = {
            'title': 3.0,
            'summary': 2.0,
            'keywords': 2.5,
            'organisms': 1.5,
            'mission_type': 1.0
        }
        
        for field, weight in fields_to_check.items():
            if field in paper and pd.notna(paper[field]):
                field_text = str(paper[field]).lower()
                
                for keyword in keywords:
                    if keyword in field_text:
                        score += weight
        
        return score
    
    def _extract_relevant_sentences(self, query: str, text: str) -> str:
        """Extract sentences from text that are relevant to the query."""
        if not text or pd.isna(text):
            return ""
        
        keywords = self._extract_keywords(query)
        sentences = text.split('.')
        
        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(keyword in sentence.lower() for keyword in keywords):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:2]) + '.'
        return ""
    
    def _generate_helpful_fallback(self, query: str, papers_data: pd.DataFrame) -> str:
        """Generate a helpful response when no specific matches are found."""
        total_papers = len(papers_data)
        
        suggestions = []
        if 'organisms' in papers_data.columns:
            top_organisms = papers_data['organisms'].value_counts().head(3)
            suggestions.extend([f"studies on {org}" for org in top_organisms.index])
        
        response = f"I couldn't find specific matches for your query, but I have **{total_papers}** space biology papers available."
        
        if suggestions:
            response += f"\n\n🔍 **Popular research areas:**\n"
            for suggestion in suggestions[:3]:
                response += f"• {suggestion}\n"
        
        response += "\n💡 **Try asking about:**\n• Microgravity effects\n• Plant growth in space\n• Protein crystallization\n• Astronaut health"
        
        return response