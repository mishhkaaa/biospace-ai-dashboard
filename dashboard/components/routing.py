"""
Enhanced Query routing for the dashboard chatbot
"""

class QueryRouter:
    """Enhanced Query Router implementation that categorizes questions intelligently"""
    
    def __init__(self):
        self.initialized = True
    
    def route_query(self, query):
        """Enhanced query routing based on keywords and patterns"""
        query_lower = query.lower().strip()
        
        # Counting queries
        count_patterns = ['how many', 'count', 'number of', 'total', 'amount of']
        if any(pattern in query_lower for pattern in count_patterns):
            return 'counting'
        
        # Listing queries
        list_patterns = ['what are', 'list', 'which', 'show me', 'what organisms', 'what studies', 'name']
        if any(pattern in query_lower for pattern in list_patterns):
            return 'listing'
        
        # Comparison queries
        comparison_patterns = ['compare', 'difference', 'versus', 'vs', 'better', 'worse']
        if any(pattern in query_lower for pattern in comparison_patterns):
            return 'comparison'
        
        # Causal/effect queries
        causal_patterns = ['effect', 'impact', 'influence', 'cause', 'result', 'consequence']
        if any(pattern in query_lower for pattern in causal_patterns):
            return 'causal'
        
        # Explanatory queries
        explanation_patterns = ['how', 'why', 'explain', 'what is', 'what are']
        if any(pattern in query_lower for pattern in explanation_patterns):
            return 'explanatory'
        
        # Factual queries
        fact_patterns = ['when', 'where', 'who', 'which']
        if any(pattern in query_lower for pattern in fact_patterns):
            return 'factual'
        
        # Default to general semantic search
        return 'general'