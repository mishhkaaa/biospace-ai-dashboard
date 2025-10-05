#!/usr/bin/env python3
"""
NASA Bioscience Literature Chatbot - Command Line Interface

This script provides a simple command-line interface for testing
the chatbot backend functionality.
"""

import os
import sys
import logging
from typing import Dict

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatbot.routing import route_query


def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('chatbot.log')
        ]
    )


def print_response(response: Dict):
    """
    Print chatbot response in a formatted way
    
    Args:
        response: Response dictionary with 'answer' and 'citations'
    """
    print("\n" + "="*60)
    print("ANSWER:")
    print("-" * 20)
    print(response.get('answer', 'No answer provided'))
    
    citations = response.get('citations', [])
    if citations:
        print("\nCITATIONS:")
        print("-" * 20)
        for i, citation in enumerate(citations, 1):
            title = citation.get('title', 'Unknown Title')
            doi = citation.get('doi', 'Unknown DOI')
            print(f"{i}. {title}")
            print(f"   DOI/ID: {doi}")
    else:
        print("\nNo citations available.")
    
    print("="*60)


def print_welcome():
    """Print welcome message and instructions"""
    print("\n" + "="*60)
    print("NASA BIOSCIENCE LITERATURE CHATBOT")
    print("="*60)
    print("\nWelcome! I can help you find information about NASA bioscience")
    print("and space biology research papers.")
    print("\nExamples of questions you can ask:")
    print("• How does microgravity affect bone density?")
    print("• What are the effects of space radiation on cells?")
    print("• List papers about mouse experiments in space")
    print("• Explain the mechanisms of muscle atrophy in microgravity")
    print("\nType 'quit', 'exit', or 'bye' to exit the program.")
    print("Type 'help' for more information.")
    print("-" * 60)


def print_help():
    """Print help information"""
    print("\n" + "="*60)
    print("HELP - NASA Bioscience Literature Chatbot")
    print("="*60)
    print("\nThis chatbot can answer two types of questions:")
    print("\n1. SEMANTIC QUERIES (recommended):")
    print("   • Use natural language to ask about concepts, mechanisms, effects")
    print("   • Example: 'How does microgravity affect bone density?'")
    print("   • Example: 'What are the cellular responses to space radiation?'")
    print("\n2. FACT QUERIES (limited functionality):")
    print("   • Ask for specific lists, counts, or factual information")
    print("   • Example: 'How many papers study microgravity?'")
    print("   • Example: 'List all papers about mouse experiments'")
    print("   • Note: Graph database features are still under development")
    print("\nTIPS:")
    print("• Be specific in your questions")
    print("• Use scientific terminology when possible")
    print("• The chatbot searches NASA bioscience literature")
    print("-" * 60)


def main():
    """Main CLI loop"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print_welcome()
    
    try:
        while True:
            try:
                # Get user input
                user_input = input("\nYour question: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nThank you for using the NASA Bioscience Literature Chatbot!")
                    print("Goodbye! 🚀")
                    break
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                elif not user_input:
                    print("Please enter a question, or type 'help' for assistance.")
                    continue
                
                # Process the query
                print("\nProcessing your question...")
                logger.info(f"User query: {user_input}")
                
                response = route_query(user_input)
                print_response(response)
                
            except KeyboardInterrupt:
                print("\n\nExiting chatbot. Goodbye! 🚀")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\nSorry, I encountered an error: {e}")
                print("Please try rephrasing your question.")
                
    except Exception as e:
        logger.error(f"Critical error in main loop: {e}")
        print(f"\nCritical error: {e}")
        print("Please check the logs and try again.")


if __name__ == "__main__":
    main()