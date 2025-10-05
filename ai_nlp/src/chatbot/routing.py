"""
Query Routing Module for NASA Bioscience Literature Chatbot

This module determines whether a user query should be routed to fact-based
graph queries or semantic search-based question answering.
"""

import re
import logging
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.qa_pipeline import QAPipeline


# Keywords that indicate fact-style queries
FACT_KEYWORDS = [
    'list', 'show', 'which', 'how many', 'count', 'number of',
    'what are', 'who are', 'when was', 'where', 'name',
    'total', 'sum', 'average', 'minimum', 'maximum',
    'first', 'last', 'oldest', 'newest', 'latest'
]

# Keywords that indicate semantic queries
SEMANTIC_KEYWORDS = [
    'explain', 'describe', 'what is', 'how does', 'why',
    'mechanism', 'process', 'relationship', 'effect',
    'impact', 'influence', 'cause', 'result', 'due to'
]


class QueryRouter:
    """
    Routes user queries to appropriate processing pipelines
    """
    
    def __init__(self):
        """Initialize the query router"""
        self.logger = logging.getLogger(__name__)
        self.qa_pipeline = None
    
    def _get_qa_pipeline(self) -> QAPipeline:
        """
        Lazy loading of QA pipeline to avoid initialization overhead
        
        Returns:
            QAPipeline: QA pipeline instance
        """
        if self.qa_pipeline is None:
            self.qa_pipeline = QAPipeline()
        return self.qa_pipeline
    
    def is_fact_query(self, user_input: str) -> bool:
        """
        Determine if a query is fact-style (suitable for graph queries)
        
        Args:
            user_input: User's query string
            
        Returns:
            bool: True if query appears to be fact-style, False otherwise
        """
        user_input_lower = user_input.lower()
        
        # Check for fact keywords
        fact_score = 0
        semantic_score = 0
        
        for keyword in FACT_KEYWORDS:
            if keyword in user_input_lower:
                fact_score += 1
        
        for keyword in SEMANTIC_KEYWORDS:
            if keyword in user_input_lower:
                semantic_score += 1
        
        # Additional patterns for fact queries
        fact_patterns = [
            r'\bhow many\b',
            r'\blist\b.*\bof\b',
            r'\bwhich\b.*\bare\b',
            r'\bshow\b.*\ball\b',
            r'\bcount\b',
            r'\bnumber\b.*\bof\b'
        ]
        
        for pattern in fact_patterns:
            if re.search(pattern, user_input_lower):
                fact_score += 2
        
        # Decision logic
        if fact_score > semantic_score:
            return True
        elif fact_score == semantic_score and fact_score > 0:
            # Tie-breaker: check for question words that suggest fact queries
            if any(word in user_input_lower for word in ['how many', 'which', 'list']):
                return True
        
        return False
    
    def route_query(self, user_input: str) -> Dict:
        """
        Route user query to appropriate processing pipeline
        
        Args:
            user_input: User's query string
            
        Returns:
            Dict: Response with 'answer' and 'citations' keys
        """
        try:
            self.logger.info(f"Routing query: {user_input}")
            
            if self.is_fact_query(user_input):
                self.logger.info("Routing to fact query pipeline")
                # Import here to avoid circular imports
                try:
                    from graph_queries import answer_fact_query
                    return answer_fact_query(user_input)
                except ImportError:
                    self.logger.warning("Graph queries module not available, falling back to semantic search")
                    return self._get_qa_pipeline().answer_query(user_input)
            else:
                self.logger.info("Routing to semantic query pipeline")
                return self._get_qa_pipeline().answer_query(user_input)
                
        except Exception as e:
            self.logger.error(f"Error routing query: {e}")
            return {
                'answer': "I encountered an error while processing your question. Please try again.",
                'citations': []
            }


# Global router instance
_router = None


def get_router() -> QueryRouter:
    """
    Get or create global router instance
    
    Returns:
        QueryRouter: Router instance
    """
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router


def route_query(user_input: str) -> Dict:
    """
    Route a user query to the appropriate processing pipeline
    
    Args:
        user_input: User's question or query string
        
    Returns:
        Dict: Response dictionary with keys:
            - 'answer': Generated answer (2-4 sentences)
            - 'citations': List of citation dictionaries with 'title' and 'doi'
    
    Example:
        >>> response = route_query("How does microgravity affect bone density?")
        >>> print(response['answer'])
        >>> for citation in response['citations']:
        ...     print(f"- {citation['title']}")
    """
    router = get_router()
    return router.route_query(user_input)