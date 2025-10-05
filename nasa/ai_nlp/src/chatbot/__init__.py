"""
NASA Bioscience Literature Chatbot Module

This module provides a modular chatbot backend for answering questions
about NASA bioscience literature using semantic search and question answering.
"""

from .qa_pipeline import QAPipeline
from .routing import route_query

__all__ = ['QAPipeline', 'route_query']