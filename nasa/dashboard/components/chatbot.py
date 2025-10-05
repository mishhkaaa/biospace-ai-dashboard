"""
Enhanced Chatbot tab component for interactive Q&A about research papers
"""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path
import time
import traceback
from typing import Optional, Dict, Any, List, Tuple

def render_chatbot_tab(data_loader):
    """Render the chatbot tab with Q&A interface"""
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False
        st.session_state.qa_pipeline = None
        st.session_state.query_router = None
        st.session_state.init_message = ""
    
    # Store data_loader in session state for use in other functions
    st.session_state.data_loader = data_loader
    
    st.markdown('<h2 class="tab-header">💬 Research Assistant Chatbot</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Ask questions about NASA's space biology research papers. The chatbot uses semantic search 
    to find relevant information and provides citations for all answers.
    """)
    
    # Initialize chatbot system
    if not st.session_state.chatbot_initialized:
        with st.spinner("Initializing research assistant..."):
            try:
                success, message, qa_pipeline, query_router = initialize_chatbot_system(data_loader.papers_df)
                
                st.session_state.chatbot_initialized = success
                st.session_state.qa_pipeline = qa_pipeline
                st.session_state.query_router = query_router
                st.session_state.init_message = message
            except Exception as e:
                st.error(f"Failed to initialize chatbot: {str(e)}")
                st.session_state.chatbot_initialized = False
                st.session_state.init_message = f"Chatbot system unavailable: {str(e)}"
    
    # Show initialization status
    if st.session_state.chatbot_initialized:
        st.success(st.session_state.init_message)
    else:
        st.error(st.session_state.init_message)
        st.info("💡 You can still browse research papers using the other tabs!")
        return
    
    # Chat interface
    render_chat_interface(data_loader)
    
    # Example queries section
    render_example_queries()
    
    # Chat history management
    render_chat_management()

def initialize_chatbot_system(papers_data: pd.DataFrame) -> Tuple[bool, str, Optional[Any], Optional[Any]]:
    """Initialize the chatbot backend system"""
    
    try:
        # First try to import from AI NLP
        import importlib.util
        import sys
        from pathlib import Path
        
        # Try to load the AI NLP modules dynamically
        ai_nlp_path = Path(__file__).parent.parent.parent / "ai_nlp" / "src"
        qa_pipeline_path = ai_nlp_path / "chatbot" / "qa_pipeline.py"
        routing_path = ai_nlp_path / "chatbot" / "routing.py"
        
        if qa_pipeline_path.exists() and routing_path.exists():
            # Load QA Pipeline module
            spec = importlib.util.spec_from_file_location("qa_pipeline", qa_pipeline_path)
            qa_module = importlib.util.module_from_spec(spec)
            sys.modules["qa_pipeline"] = qa_module
            spec.loader.exec_module(qa_module)
            QAPipeline = qa_module.QAPipeline
            
            # Load Routing module
            spec = importlib.util.spec_from_file_location("routing", routing_path)
            routing_module = importlib.util.module_from_spec(spec)
            sys.modules["routing"] = routing_module
            spec.loader.exec_module(routing_module)
            QueryRouter = routing_module.QueryRouter
            
            # Test initialize the modules to check if they work
            try:
                qa_pipeline = QAPipeline()
                query_router = QueryRouter()
                
                # Check if we have the advanced AI methods
                if hasattr(qa_pipeline, 'answer_question'):
                    # Test with papers data
                    test_response = qa_pipeline.answer_question("test", papers_data)
                    return True, "✅ Advanced AI chatbot system ready!", qa_pipeline, query_router
                else:
                    # This is the fallback system, which is still good
                    return True, "✅ Enhanced chatbot system ready!", qa_pipeline, query_router
                
            except Exception as init_error:
                st.warning(f"AI NLP modules loaded but failed to initialize: {str(init_error)}")
                raise init_error
                
        else:
            raise ImportError("AI NLP modules not found")
            
    except Exception as e:
        # Fallback to local stub modules
        st.info(f"Using enhanced fallback system")
        try:
            # Create enhanced fallback QA pipeline
            qa_pipeline = EnhancedFallbackQA()
            query_router = SimpleQueryRouter()
            
            # Test fallback system
            test_response = qa_pipeline.answer_question("test", papers_data)
            
            return True, "✅ Enhanced fallback chatbot ready!", qa_pipeline, query_router
            
        except Exception as fallback_error:
            st.error(f"Failed to initialize any chatbot system: {str(fallback_error)}")
            return False, f"Chatbot system unavailable: {str(fallback_error)}", None, None

class EnhancedFallbackQA:
    """Enhanced fallback QA pipeline that provides intelligent responses using the papers database."""
    
    def __init__(self):
        """Initialize the QA pipeline."""
        self.initialized = True
    
    def answer_question(self, query: str, papers_data: pd.DataFrame) -> str:
        """Generate an answer using keyword matching and paper analysis."""
        try:
            if papers_data is None or papers_data.empty:
                return "Sorry, no research papers are currently available in the database."
            
            # Clean and prepare query
            query_lower = query.lower().strip()
            
            if not query_lower or query_lower == "test":
                return f"I have access to {len(papers_data)} NASA space biology research papers. Ask me about specific topics, organisms, or research areas!"
            
            # Handle different types of queries
            if self._is_counting_query(query_lower):
                return self._handle_counting_query(query_lower, papers_data)
            elif self._is_listing_query(query_lower):
                return self._handle_listing_query(query_lower, papers_data)
            else:
                return self._handle_general_query(query_lower, papers_data)
                
        except Exception as e:
            return f"I encountered an error processing your question: {str(e)}"
    
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
                
                # Try to find specific topic matches
                relevant_papers = self._find_relevant_papers(query, papers_data)
                
                if len(relevant_papers) > 0:
                    return f"I found **{len(relevant_papers)}** research papers related to your query out of {total_papers} total papers in the database."
                else:
                    return f"The database contains **{total_papers}** research papers in total. Try being more specific about the topic you're interested in."
            
            elif 'organism' in query:
                # Count unique organisms if that column exists
                if 'organisms' in papers_data.columns:
                    unique_organisms = papers_data['organisms'].nunique()
                    organisms_list = papers_data['organisms'].value_counts().head(5)
                    top_organisms = ", ".join(organisms_list.index.tolist())
                    return f"The research database includes studies on **{unique_organisms}** different types of organisms. The most studied include: {top_organisms}."
                else:
                    return "I don't have specific organism count data available, but the database contains diverse biological research."
            
            else:
                return f"I found **{len(papers_data)}** total research papers in the database. Please be more specific about what you'd like me to count."
                
        except Exception as e:
            return f"Error processing counting query: {str(e)}"
    
    def _handle_listing_query(self, query: str, papers_data: pd.DataFrame) -> str:
        """Handle listing-type queries."""
        try:
            relevant_papers = self._find_relevant_papers(query, papers_data)
            
            if len(relevant_papers) == 0:
                return f"I couldn't find papers specifically matching your query. The database contains {len(papers_data)} research papers covering various aspects of space biology. Try asking about 'microgravity', 'plant growth', 'bone density', or 'protein crystallization'."
            
            # Return top relevant papers
            response = f"I found **{len(relevant_papers)}** relevant research papers:\n\n"
            
            for i, (_, paper) in enumerate(relevant_papers.head(5).iterrows()):
                title = paper.get('title', 'Untitled')
                year = paper.get('publication_year', 'Unknown year')
                response += f"{i+1}. **{title}** ({year})\n"
            
            if len(relevant_papers) > 5:
                response += f"\n... and {len(relevant_papers) - 5} more papers. Use the Research Papers tab to explore all results."
            
            return response
            
        except Exception as e:
            return f"Error processing listing query: {str(e)}"
    
    def _handle_general_query(self, query: str, papers_data: pd.DataFrame) -> str:
        """Handle general questions by finding relevant papers."""
        try:
            relevant_papers = self._find_relevant_papers(query, papers_data)
            
            if len(relevant_papers) == 0:
                return self._generate_helpful_fallback(query, papers_data)
            
            # Generate response based on found papers
            response = f"Based on **{len(relevant_papers)}** relevant research papers:\n\n"
            
            # Add top papers as references
            for i, (_, paper) in enumerate(relevant_papers.head(3).iterrows()):
                title = paper.get('title', 'Untitled')
                year = paper.get('publication_year', 'Unknown year')
                summary = paper.get('summary', '')
                
                response += f"**{i+1}. {title}** ({year})\n"
                
                # Extract relevant sentences from summary
                if summary and len(summary) > 50:
                    relevant_sentences = self._extract_relevant_sentences(query, summary)
                    if relevant_sentences:
                        response += f"*Key findings: {relevant_sentences}*\n\n"
                    else:
                        # Use first part of summary
                        short_summary = summary[:150] + "..." if len(summary) > 150 else summary
                        response += f"*{short_summary}*\n\n"
                else:
                    response += "*This paper discusses topics related to your question.*\n\n"
            
            if len(relevant_papers) > 3:
                response += f"💡 Found {len(relevant_papers) - 3} additional relevant papers. Use the Research Papers tab to explore all results."
            
            return response
            
        except Exception as e:
            return f"Error processing general query: {str(e)}"
    
    def _find_relevant_papers(self, query: str, papers_data: pd.DataFrame) -> pd.DataFrame:
        """Find papers relevant to the query using keyword matching."""
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            if not keywords:
                return pd.DataFrame()
            
            # Score papers based on keyword matches
            scores = []
            for _, paper in papers_data.iterrows():
                score = self._calculate_relevance_score(keywords, paper)
                scores.append(score)
            
            papers_data_copy = papers_data.copy()
            papers_data_copy['relevance_score'] = scores
            
            # Return papers with score > 0, sorted by relevance
            relevant = papers_data_copy[papers_data_copy['relevance_score'] > 0]
            return relevant.sort_values('relevance_score', ascending=False)
            
        except Exception as e:
            print(f"Error finding relevant papers: {str(e)}")
            return pd.DataFrame()
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from the query."""
        import re
        
        # Remove common stop words and question words
        stop_words = {
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'the', 'a', 'an', 
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'about', 'tell', 'me'
        }
        
        # Split query into words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _calculate_relevance_score(self, keywords: List[str], paper: Dict[str, Any]) -> float:
        """Calculate relevance score for a paper based on keyword matches."""
        score = 0.0
        
        # Check different fields with different weights
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
        
        # Suggest related topics if available
        suggestions = []
        if 'organisms' in papers_data.columns:
            top_organisms = papers_data['organisms'].value_counts().head(3)
            suggestions.extend([f"studies on {org}" for org in top_organisms.index])
        
        response = f"I couldn't find specific papers matching your exact query, but the database contains **{total_papers}** research papers on space biology."
        
        if suggestions:
            response += f"\n\n🔍 **Popular research areas:**\n"
            for i, suggestion in enumerate(suggestions[:3]):
                response += f"• {suggestion}\n"
        
        response += "\n💡 **Try asking about:**\n• Microgravity effects on plants or animals\n• Protein crystallization in space\n• Bone density changes in astronauts\n• Immune system adaptations to space"
        
        return response

class SimpleQueryRouter:
    """Simple query router for the fallback system."""
    
    def __init__(self):
        """Initialize the query router."""
        self.initialized = True
    
    def route_query(self, query: str) -> str:
        """Route the query to determine its type."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how many', 'count', 'number of']):
            return 'counting'
        elif any(word in query_lower for word in ['what are', 'list', 'which']):
            return 'listing'
        elif any(word in query_lower for word in ['effect', 'impact', 'influence']):
            return 'causal'
        elif any(word in query_lower for word in ['how', 'why', 'explain']):
            return 'explanatory'
        else:
            return 'general'

def render_chat_interface(data_loader):
    """Render the main chat interface"""
    
    # Chat history display
    st.markdown("### 💬 Conversation")
    
    # Create chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                else:
                    with st.chat_message("assistant"):
                        st.write(message['content'])
                        
                        # Show citations if available
                        if 'citations' in message and message['citations']:
                            with st.expander("📚 Sources & Citations"):
                                for j, citation in enumerate(message['citations']):
                                    st.markdown(f"**{j+1}.** {citation}")
        else:
            st.info("👋 Welcome! Ask me anything about NASA's space biology research papers.")
    
    # Query input
    st.markdown("### ❓ Ask a Question")
    
    # Query input form
    with st.form("query_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_query = st.text_input(
                "Your question:",
                placeholder="e.g., What are the effects of microgravity on plant growth?",
                help="Ask about specific topics, organisms, experiments, or research findings"
            )
        
        with col2:
            submit_button = st.form_submit_button("🚀 Ask", type="primary")
    
    # Process query
    if submit_button and user_query:
        process_user_query(user_query, data_loader)

def process_user_query(query, data_loader):
    """Process user query and generate response"""
    
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': query,
        'timestamp': time.time()
    })
    
    # Show thinking indicator
    with st.spinner("🤔 Searching through research papers..."):
        
        # Generate response
        try:
            if st.session_state.get('chatbot_initialized') and st.session_state.qa_pipeline:
                response, citations = generate_ai_response(query, data_loader)
            else:
                response, citations = "Sorry, the chatbot system is not available.", []
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'citations': citations,
                'timestamp': time.time()
            })
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error while processing your question: {str(e)}"
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': error_response,
                'citations': [],
                'timestamp': time.time()
            })
    
    # Rerun to update the chat display
    st.rerun()

def generate_ai_response(query, data_loader):
    """Generate AI-powered response using the QA pipeline"""
    try:
        # Use the QA pipeline
        qa_pipeline = st.session_state.qa_pipeline
        
        # Check if this is the advanced AI pipeline or enhanced fallback
        if hasattr(qa_pipeline, 'answer_question'):
            # Use the answer_question method (both advanced and enhanced fallback have this)
            response = qa_pipeline.answer_question(query, data_loader.papers_df)
            citations = [f"Analysis of {len(data_loader.papers_df)} research papers"]
        else:
            # Legacy fallback system
            query_router = st.session_state.query_router
            query_type = query_router.route_query(query)
            
            # Get response based on query type
            if hasattr(qa_pipeline, 'semantic_search'):
                response_data = qa_pipeline.semantic_search(query, data_loader.papers_df, top_k=5)
                response = response_data.get('answer', 'I could not find relevant information for your query.')
                citations = response_data.get('citations', [])
            else:
                response = "The chatbot system is not properly configured."
                citations = []
        
        return response, citations
        
    except Exception as e:
        st.error(f"Error in AI response generation: {str(e)}")
        return f"I encountered an error while processing your question: {str(e)}", []

def render_example_queries():
    """Render example queries section"""
    
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    
    example_queries = [
        "What are the effects of microgravity on plant growth?",
        "How does spaceflight affect bone density in astronauts?",
        "What organisms have been studied on the International Space Station?",
        "Tell me about protein crystallization experiments in space",
        "How does radiation exposure affect DNA repair mechanisms?",
        "What are the cardiovascular changes during long-duration spaceflight?",
        "Which experiments were conducted on SpaceX missions?",
        "How do immune systems adapt to the space environment?"
    ]
    
    col1, col2 = st.columns(2)
    
    for i, query in enumerate(example_queries):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(query, key=f"example_query_{i}", help="Click to ask this question"):
                # Add the question to chat history and process it
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': query,
                    'timestamp': time.time()
                })
                
                # Generate response
                with st.spinner("🤔 Searching through research papers..."):
                    try:
                        response, citations = generate_ai_response(query, st.session_state.get('data_loader'))
                        
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response,
                            'citations': citations,
                            'timestamp': time.time()
                        })
                    except Exception as e:
                        error_response = f"I apologize, but I encountered an error: {str(e)}"
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': error_response,
                            'citations': [],
                            'timestamp': time.time()
                        })
                
                st.rerun()

def render_chat_management():
    """Render chat history management controls"""
    
    st.markdown("---")
    st.markdown("### 🗂️ Chat Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat History", help="Remove all chat messages"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat", help="Download chat conversation"):
            if st.session_state.chat_history:
                chat_text = ""
                for message in st.session_state.chat_history:
                    role = "You" if message['role'] == 'user' else "Assistant"
                    chat_text += f"{role}: {message['content']}\n\n"
                
                st.download_button(
                    label="📥 Download Chat",
                    data=chat_text,
                    file_name="nasa_chatbot_conversation.txt",
                    mime="text/plain"
                )
            else:
                st.info("No chat history to export")
    
    with col3:
        if st.button("🔄 Reset Chatbot", help="Reinitialize the chatbot system"):
            st.session_state.chatbot_initialized = False
            st.session_state.qa_pipeline = None
            st.session_state.query_router = None
            st.session_state.chat_history = []
            st.success("Chatbot reset! Page will reload...")
            st.rerun()
    
    # Chat statistics
    if st.session_state.chat_history:
        with st.expander("📊 Chat Statistics"):
            user_messages = [msg for msg in st.session_state.chat_history if msg['role'] == 'user']
            assistant_messages = [msg for msg in st.session_state.chat_history if msg['role'] == 'assistant']
            
            st.write(f"""
            **Conversation Statistics:**
            - Total messages: {len(st.session_state.chat_history)}
            - Your questions: {len(user_messages)}
            - Assistant responses: {len(assistant_messages)}
            """)