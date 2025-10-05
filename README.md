# NASA Space Biology Knowledge Engine | 2025 NASA Space Apps Challenge 

![NASA Space Apps Challenge](https://www.spaceappschallenge.org/assets/img/logo-sm.png)

## Overview

The NASA Space Biology Knowledge Engine is an advanced AI-powered platform that enables researchers, mission planners, and scientists to explore, analyze, and extract actionable insights from NASA's vast collection of bioscience publications. This comprehensive solution helps users understand how humans, plants, and other living systems respond to the space environment, which is critical for planning safe and efficient Moon and Mars exploration missions.

## 🌟 Challenge Addressed

The solution addresses NASA's "Build a Space Biology Knowledge Engine" challenge from the 2025 NASA Space Apps Challenge. The key problem was:

> NASA has been performing biology experiments in space for decades, generating a tremendous amount of information that will need to be considered as humans prepare to revisit the Moon and explore Mars. Although this knowledge is publicly available, it can be difficult for potential users to find information that pertains to their specific interests.

Our solution transforms the way users interact with NASA's bioscience publications by leveraging artificial intelligence, knowledge graphs, and interactive visualizations to make this valuable information accessible, understandable, and actionable.

## 🔬 Solution Architecture

The NASA Space Biology Knowledge Engine consists of three major components:

### 1. AI NLP Pipeline

A sophisticated natural language processing pipeline that processes scientific publications to extract insights, summarize content, identify patterns, and build a semantic understanding of the research:

- **Summarization**: Generates concise, abstractive summaries of research papers using transformer models (BART)
- **Semantic Embeddings**: Creates vector representations of papers for similarity analysis and search
- **Clustering**: Groups papers into thematic areas to identify research domains
- **Keyword Extraction**: Identifies key scientific concepts and terminology
- **Insights Generation**: Analyzes patterns to identify research gaps, consensus areas, and trends

### 2. Knowledge Graph

An interconnected graph database (Neo4j) that represents the relationships between different entities in the research:

- **Entity Recognition**: Extracts organisms, environments, experiments, and outcomes
- **Relationship Mapping**: Establishes connections between entities (e.g., "studied_in", "has_outcome")
- **Graph Database**: Stores and queries these relationships for powerful knowledge discovery
- **Visualization**: Renders the knowledge network graphically for intuitive exploration

### 3. Interactive Dashboard

A user-friendly web interface built with Streamlit that provides multiple ways to explore the data:

- **Research Overview**: Visualizations of key statistics, trends, and paper distribution
- **Paper Browser**: Searchable, sortable list of all publications with summaries
- **Knowledge Graph Explorer**: Interactive visualization of the entity relationships
- **AI Research Assistant**: Chatbot interface for natural language queries about the research

## 💡 Key Features

### Advanced AI Analysis
- **Research Gap Identification**: Highlights underexplored areas needing additional investigation
- **Trend Analysis**: Visualizes how research focus has evolved over time
- **Consensus/Disagreement Detection**: Identifies areas of scientific consensus and controversy
- **Strategic Recommendation**: Suggests promising research directions based on current knowledge

### Interactive Exploration
- **Multi-faceted Search**: Filter by organisms, environments, dates, topics, and more
- **Interactive Visualizations**: Explore data through charts, graphs, and knowledge networks
- **Natural Language Querying**: Ask questions in plain English through the AI assistant
- **Citation Support**: All AI-generated insights include links to source publications

### Research Assistant Chatbot
- **Semantic Search**: Finds the most relevant papers to answer specific questions
- **Context-Aware Responses**: Generates answers based on the content of multiple papers
- **Citation Generation**: Attributes information to the correct source publications
- **Query Routing**: Intelligently routes different types of questions to appropriate answering mechanisms

## 🛠️ Technical Implementation

### AI NLP Pipeline (`ai_nlp/`)
- **Python-based**: Modular, extensible architecture with robust error handling
- **Models**: Leverages state-of-the-art transformer models (BART, SciBERT, SentenceTransformers)
- **Pipelines**: Multi-stage processing pipeline with advanced caching and optimization
- **Output**: Generates structured data (CSV, JSON) for dashboard consumption

### Named Entity Recognition Stack (`my_ner_stack/`)
- **PubMed Integration**: Extracts full text from PubMed Central via BioC API
- **Entity Recognition**: Identifies key biomedical entities in the research text
- **Knowledge Graph Construction**: Builds a Neo4j graph database of interconnected entities
- **Graph Querying**: Enables complex relationship queries across the publication dataset

### Dashboard (`dashboard/`)
- **Streamlit Framework**: Reactive, responsive web interface with minimal boilerplate
- **Interactive Components**: Dynamic filters, search, and visualization tools
- **Multi-tab Interface**: Organized into Overview, Papers, Knowledge Graph, and Chatbot sections
- **Cross-platform**: Works on any modern web browser without specialized hardware

## 📊 Data Sources

The solution processes 608 NASA bioscience publications as listed in the challenge resources, with potential expansion to include:

- **NASA Open Science Data Repository (OSDR)**: Primary data and metadata from studies
- **NASA Space Life Sciences Library**: Additional relevant publications
- **NASA Task Book**: Information on grants that funded the studies

## 👥 Target Users

The Knowledge Engine is designed for multiple audiences:

1. **Scientists**: Researchers generating new hypotheses and exploring existing literature
2. **NASA Managers**: Decision-makers identifying opportunities for investment
3. **Mission Planners**: Engineers and architects designing Moon and Mars missions
4. **Students & Educators**: Those learning about space biology and its challenges

## 🚀 Setup and Installation

### AI NLP Pipeline

```bash
cd ai_nlp
pip install -r requirements.txt
python run_pipeline.py
```

### Dashboard

```bash
cd dashboard
pip install -r requirements.txt
# On Windows
run_dashboard.bat
# On Unix/Linux
./run_dashboard.sh
```

### Knowledge Graph (Neo4j)

See detailed setup instructions in `my_ner_stack/NEO4J_SETUP.md`

## 🔄 Workflow

1. The AI NLP pipeline processes the publications to generate summaries, embeddings, and insights
2. The NER stack extracts entities and relationships to build the knowledge graph
3. The dashboard loads all processed data and presents it through the interactive interface
4. Users can explore papers, visualize trends, navigate the knowledge graph, and query the chatbot

## 💬 Chatbot Integration

A key innovation in our solution is the AI Research Assistant chatbot, which serves as a natural language interface to the entire knowledge base:

- Users can ask questions about any aspect of space biology research
- The chatbot routes queries to either the semantic search system or the knowledge graph
- Responses include cited sources, allowing users to verify information
- Every paper, entity, and knowledge node can be added as context to enhance the chatbot's responses

## 🔮 Future Enhancements

1. **OSDR Data Integration**: Directly link to experimental data from the NASA Open Science Data Repository
2. **Multi-modal Representation**: Add visual and audio representations of complex relationships
3. **Personalized Recommendations**: Suggest relevant papers based on user interests and history
4. **Collaboration Features**: Enable research teams to share insights and annotations

## 🌐 Impact

The NASA Space Biology Knowledge Engine will dramatically accelerate space biology research by:

- **Reducing Redundancy**: Helping researchers quickly understand what's already known
- **Highlighting Opportunities**: Identifying promising areas for new investigation
- **Supporting Planning**: Providing mission architects with comprehensive biological insights
- **Cross-disciplinary Connection**: Revealing unexpected relationships between different research areas

## 🔍 Conclusion

By transforming NASA's extensive bioscience publication archive into an interactive, AI-powered knowledge engine, this solution helps prepare humanity for the next era of space exploration. The comprehensive understanding of how living organisms respond to space environments will be crucial for the success and safety of future missions to the Moon, Mars, and beyond.

## 📚 Resources

- [NASA Space Apps Challenge](https://www.spaceappschallenge.org/)
- [NASA Biological and Physical Sciences Division](https://science.nasa.gov/biological-physical/)
- [NASA Open Science Data Repository](https://osdr.nasa.gov/bio/)
- [NASA Space Life Sciences Repository](https://lsda.jsc.nasa.gov/)
- [NASA Task Book](https://taskbook.nasaprs.com/tbpub/index.cfm)
