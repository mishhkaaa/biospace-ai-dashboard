"""
Question Answering Pipeline for NASA Bioscience Literature

This module implements a semantic search and question answering pipeline
that retrieves relevant papers and generates answers using pretrained models.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config


class QAPipeline:
    """
    Question Answering Pipeline for NASA Bioscience Literature
    
    This class handles loading paper summaries and embeddings, embedding queries,
    retrieving similar papers, and generating answers using language models.
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 qa_model: str = "microsoft/DialoGPT-medium",
                 max_papers: int = 5):
        """
        Initialize the QA Pipeline
        
        Args:
            embedding_model: Name of the sentence transformer model for embeddings
            qa_model: Name of the language model for answer generation
            max_papers: Maximum number of papers to retrieve for context
        """
        self.max_papers = max_papers
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model {embedding_model}: {e}")
            raise
            
        try:
            # Use a more suitable model for text generation
            self.qa_pipeline = pipeline(
                "text-generation",
                model="gpt2",
                tokenizer="gpt2",
                device=-1  # Use CPU
            )
            self.logger.info(f"Loaded QA model: gpt2")
        except Exception as e:
            self.logger.error(f"Failed to load QA model: {e}")
            raise
        
        # Initialize data storage
        self.papers_df = None
        self.embeddings = None
        self.metadata_df = None
        
    def load_data(self) -> bool:
        """
        Load paper summaries and embeddings from output files
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            # Load paper summaries
            summaries_path = config.get_output_file_path("paper_summaries.csv", "summaries")
            if not os.path.exists(summaries_path):
                self.logger.error(f"Summaries file not found: {summaries_path}")
                return False
                
            self.papers_df = pd.read_csv(summaries_path)
            self.logger.info(f"Loaded {len(self.papers_df)} paper summaries")
            
            # Load embeddings from JSONL format
            embeddings_path = config.get_output_file_path("paper_embeddings.jsonl", "embeddings")
            if not os.path.exists(embeddings_path):
                self.logger.error(f"Embeddings file not found: {embeddings_path}")
                return False
                
            embeddings_data = []
            paper_ids = []
            
            with open(embeddings_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    paper_ids.append(data['paper_id'])
                    embeddings_data.append(data['embedding'])
            
            self.embeddings = np.array(embeddings_data)
            self.logger.info(f"Loaded embeddings for {len(embeddings_data)} papers")
            
            # Ensure papers and embeddings are aligned
            self.papers_df = self.papers_df[self.papers_df['paper_id'].isin(paper_ids)]
            self.papers_df = self.papers_df.set_index('paper_id').reindex(paper_ids).reset_index()
            
            # Load metadata if available
            metadata_path = config.get_output_file_path("paper_embeddings_metadata.csv", "embeddings")
            if os.path.exists(metadata_path):
                self.metadata_df = pd.read_csv(metadata_path)
                self.logger.info(f"Loaded metadata for {len(self.metadata_df)} papers")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return False
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for user query
        
        Args:
            query: User's question or query string
            
        Returns:
            np.ndarray: Query embedding vector
        """
        try:
            query_embedding = self.embedding_model.encode([query])
            return query_embedding[0]
        except Exception as e:
            self.logger.error(f"Failed to embed query: {e}")
            raise
    
    def retrieve_similar_papers(self, query_embedding: np.ndarray, top_k: int = None) -> List[Tuple[int, float]]:
        """
        Retrieve papers most similar to the query
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of papers to retrieve (defaults to self.max_papers)
            
        Returns:
            List[Tuple[int, float]]: List of (paper_index, similarity_score) tuples
        """
        if top_k is None:
            top_k = self.max_papers
            
        try:
            # Calculate cosine similarity
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            
            # Get top-k most similar papers
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_similarities = similarities[top_indices]
            
            return list(zip(top_indices, top_similarities))
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve similar papers: {e}")
            raise
    
    def generate_answer(self, query: str, context_papers: List[Dict]) -> str:
        """
        Generate answer using retrieved papers as context
        
        Args:
            query: User's original question
            context_papers: List of paper dictionaries with title, summary, etc.
            
        Returns:
            str: Generated answer (2-4 sentences)
        """
        try:
            # Prepare context from papers
            context_text = ""
            for i, paper in enumerate(context_papers[:3]):  # Use top 3 papers for context
                context_text += f"Paper {i+1}: {paper['title']}\n"
                context_text += f"Summary: {paper['summary']}\n\n"
            
            # Create prompt for answer generation
            prompt = f"""Based on the following research papers about space biology and bioscience:

{context_text}

Question: {query}

Answer:"""
            
            # Generate answer
            response = self.qa_pipeline(
                prompt,
                max_length=len(prompt) + 150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            # Extract answer from response
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            # Clean up the answer - take first few sentences
            sentences = answer.split('.')[:3]  # Take first 3 sentences max
            answer = '. '.join(s.strip() for s in sentences if s.strip()) + '.'
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Failed to generate answer: {e}")
            return "I apologize, but I encountered an error while generating an answer to your question."
    
    def format_citations(self, papers: List[Dict]) -> List[Dict]:
        """
        Format paper citations for response
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            List[Dict]: List of citation dictionaries with title and doi
        """
        citations = []
        for paper in papers:
            citation = {
                'title': paper.get('title', 'Unknown Title'),
                'doi': paper.get('paper_id', 'Unknown DOI')  # Using paper_id as DOI placeholder
            }
            citations.append(citation)
        return citations
    
    def answer_query(self, user_input: str) -> Dict:
        """
        Complete pipeline to answer a user query
        
        Args:
            user_input: User's question or query
            
        Returns:
            Dict: Response with 'answer' and 'citations' keys
        """
        try:
            # Ensure data is loaded
            if self.papers_df is None or self.embeddings is None:
                if not self.load_data():
                    return {
                        'answer': "I apologize, but I cannot access the research database at the moment.",
                        'citations': []
                    }
            
            # Embed the query
            query_embedding = self.embed_query(user_input)
            
            # Retrieve similar papers
            similar_papers = self.retrieve_similar_papers(query_embedding)
            
            if not similar_papers:
                return {
                    'answer': "I couldn't find any relevant papers for your question.",
                    'citations': []
                }
            
            # Get paper details
            context_papers = []
            for paper_idx, similarity in similar_papers:
                paper_row = self.papers_df.iloc[paper_idx]
                paper_dict = {
                    'paper_id': paper_row['paper_id'],
                    'title': paper_row['title'],
                    'summary': paper_row['summary'],
                    'year': paper_row.get('year', 'Unknown'),
                    'similarity': similarity
                }
                context_papers.append(paper_dict)
            
            # Generate answer
            answer = self.generate_answer(user_input, context_papers)
            
            # Format citations
            citations = self.format_citations(context_papers)
            
            self.logger.info(f"Successfully answered query with {len(citations)} citations")
            
            return {
                'answer': answer,
                'citations': citations
            }
            
        except Exception as e:
            self.logger.error(f"Failed to answer query: {e}")
            return {
                'answer': "I encountered an error while processing your question. Please try rephrasing your query.",
                'citations': []
            }


def create_qa_pipeline() -> QAPipeline:
    """
    Factory function to create a QA pipeline instance
    
    Returns:
        QAPipeline: Configured QA pipeline instance
    """
    return QAPipeline()