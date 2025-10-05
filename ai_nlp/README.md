<<<<<<< HEAD
# NASA Bioscience AI Pipeline# 🚀 NASA Bioscience AI Pipeline



Advanced AI-powered analysis pipeline for NASA bioscience research publications.An advanced AI-powered analysis pipeline for NASA bioscience research publications, transforming metadata into actionable insights for scientific research and knowledge discovery.



## Quick Start## 🎯 Overview



```bashThe NASA Bioscience AI Pipeline is a comprehensive, production-ready system that processes scientific publication metadata to generate structured insights for research planning and knowledge gap identification. It combines state-of-the-art natural language processing, machine learning, and data analysis techniques to deliver:

pip install -r requirements.txt

python run_pipeline.py- **Intelligent Summarization**: Advanced abstractive summarization of research papers

```- **Semantic Understanding**: Deep semantic embeddings for similarity analysis

- **Thematic Clustering**: Automatic grouping of research into thematic areas

## Chatbot API- **Knowledge Discovery**: Extraction of keywords and scientific concepts

- **Strategic Insights**: Identification of research gaps, trends, and opportunities

```python

from src.chatbot.routing import route_query## ✨ Key Features



response = route_query("How does microgravity affect bone density?")### 🧠 Advanced AI Models

print(response['answer'])- **BART-large-CNN** for high-quality summarization

for citation in response['citations']:- **SciBERT/SentenceTransformers** for scientific text embeddings

    print(f"- {citation['title']}")- **KeyBERT** with scientific domain optimization

```- **Multiple clustering algorithms** with automatic selection



## CLI Interface### 📊 Comprehensive Analysis

- **Knowledge Gap Detection**: Identifies underexplored research areas

```bash- **Publication Trend Analysis**: Historical patterns and growth trajectories

cd src- **Consensus/Disagreement Analysis**: Areas of scientific agreement vs. debate

python run_chatbot_cli.py- **Research Opportunity Generation**: Strategic recommendations for future research

```
=======
# 🚀 NASA Bioscience AI Pipeline

An advanced AI-powered analysis pipeline for NASA bioscience research publications, transforming metadata into actionable insights for scientific research and knowledge discovery.

## 🎯 Overview

The NASA Bioscience AI Pipeline is a comprehensive, production-ready system that processes scientific publication metadata to generate structured insights for research planning and knowledge gap identification. It combines state-of-the-art natural language processing, machine learning, and data analysis techniques to deliver:

- **Intelligent Summarization**: Advanced abstractive summarization of research papers
- **Semantic Understanding**: Deep semantic embeddings for similarity analysis
- **Thematic Clustering**: Automatic grouping of research into thematic areas
- **Knowledge Discovery**: Extraction of keywords and scientific concepts
- **Strategic Insights**: Identification of research gaps, trends, and opportunities

## ✨ Key Features

### 🧠 Advanced AI Models
- **BART-large-CNN** for high-quality summarization
- **SciBERT/SentenceTransformers** for scientific text embeddings
- **KeyBERT** with scientific domain optimization
- **Multiple clustering algorithms** with automatic selection

### 📊 Comprehensive Analysis
- **Knowledge Gap Detection**: Identifies underexplored research areas
- **Publication Trend Analysis**: Historical patterns and growth trajectories
- **Consensus/Disagreement Analysis**: Areas of scientific agreement vs. debate
- **Research Opportunity Generation**: Strategic recommendations for future research

>>>>>>> 675f2a402fb638e07614d43eaf9e00688e128708
### 🔧 Production-Ready Features
- **Robust Error Handling**: Graceful degradation and fallback mechanisms
- **Quality Metrics**: Comprehensive evaluation of all outputs
- **Beautiful Logging**: Rich console output with progress tracking
- **Modular Architecture**: Easy to extend and customize
- **Multiple Output Formats**: JSON, CSV, and metadata files

## 📁 Project Structure

```
ai_nlp/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── run_pipeline.py            # Main pipeline orchestrator
├── data/
│   ├── raw/                   # Input data files
│   │   └── person_a_metadata.csv
│   └── processed/             # Cleaned data files
├── src/                       # Source code modules
│   ├── config.py             # Configuration management
│   ├── utils.py              # Shared utilities
│   ├── summarization.py      # Text summarization
<<<<<<< HEAD
│   ├── chatbot/              # Chatbot backend module
│   │   ├── __init__.py       # Module exports
│   │   ├── qa_pipeline.py    # Question answering pipeline
│   │   └── routing.py        # Query routing logic
│   ├── graph_queries.py      # Graph database queries (placeholder)
│   └── run_chatbot_cli.py    # Command-line chatbot interface
=======
>>>>>>> 675f2a402fb638e07614d43eaf9e00688e128708
│   ├── embeddings.py         # Semantic embeddings
│   ├── clustering.py         # Document clustering
│   ├── keywords.py           # Keyword extraction
│   └── insights.py           # Insights generation
├── outputs/                   # Pipeline outputs
│   ├── summaries/            # Paper summaries
│   ├── embeddings/           # Semantic embeddings
│   ├── clusters/             # Clustering results
│   └── insights/             # Analysis insights
└── models/                    # Cached model files
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai_nlp

# Install dependencies
pip install -r requirements.txt

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data

Place your CSV data file in `data/raw/` with the following required columns:
- `paper_id`: Unique identifier for each paper
- `title`: Paper title
- `abstract`: Paper abstract
- `year`: Publication year (optional but recommended)

Optional columns that enhance analysis:
- `authors`, `journal`, `doi_url`, `organism`, `mission`, `finding`

### 3. Run the Pipeline

```bash
# Run complete pipeline
python run_pipeline.py

# Run with verbose logging
python run_pipeline.py --verbose

# Run single component
python run_pipeline.py --component summarization

# Skip specific components
python run_pipeline.py --skip embeddings clustering
```

## 📊 Output Files

### 📝 Summaries
- `paper_summaries.csv`: Individual paper summaries with quality metrics
- `paper_summaries_metadata.json`: Processing metadata

### 🔗 Embeddings
- `paper_embeddings.jsonl`: Semantic embeddings for each paper
- `paper_embeddings_metadata.csv`: Embedding quality metrics
- `paper_embeddings_similarities.json`: Paper similarity information

### 🎯 Clusters
- `paper_clusters.csv`: Cluster assignments for each paper
- `cluster_keywords_summaries.csv`: Keywords and summaries for each cluster
- `cluster_analysis.json`: Detailed cluster analysis

### 💡 Insights
- `knowledge_gaps.json`: Underexplored research areas
- `publication_trends.csv`: Historical publication patterns
- `consensus_disagreement.csv`: Areas of scientific agreement/debate
- `research_opportunities.json`: Strategic research recommendations
- `comprehensive_insights.json`: Complete analysis summary

## ⚙️ Configuration

The pipeline is highly configurable through `src/config.py`:

### Model Configuration
```python
@dataclass
class ModelConfig:
    summarization_model: str = "facebook/bart-large-cnn"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    scientific_embedding_model: str = "allenai/scibert_scivocab_uncased"
    summary_max_length: int = 130
    summary_min_length: int = 30
```

### Clustering Configuration
```python
@dataclass
class ClusterConfig:
    default_n_clusters: int = 8
    min_cluster_size: int = 3
    use_adaptive_clustering: bool = True
    clustering_algorithm: str = "kmeans"  # Options: kmeans, dbscan, agglomerative, auto
```

### Insights Configuration
```python
@dataclass
class InsightsConfig:
    knowledge_gap_threshold: int = 3
    consensus_threshold: float = 0.7
    disagreement_threshold: float = 0.3
```

## 🔬 Pipeline Components

### 1. Summarization Module (`summarization.py`)
- **Purpose**: Generate concise, readable summaries of research abstracts
- **Technology**: Facebook BART-large-CNN transformer model
- **Features**: 
  - Adaptive length parameters based on input text
  - Quality assessment and fallback mechanisms
  - Compression ratio analysis
  - Batch processing for efficiency

### 2. Embeddings Module (`embeddings.py`)
- **Purpose**: Create semantic vector representations for similarity analysis
- **Technology**: SentenceTransformers with scientific domain models
- **Features**:
  - Scientific domain optimization with SciBERT
  - Similarity matrix calculation
  - PCA dimensionality reduction for visualization
  - Multiple output formats (JSONL, NumPy, CSV)

### 3. Clustering Module (`clustering.py`)
- **Purpose**: Group papers into thematic research areas
- **Technology**: Multiple algorithms (K-means, DBSCAN, Agglomerative)
- **Features**:
  - Automatic algorithm selection
  - Optimal cluster number detection
  - Quality metrics (silhouette score, etc.)
  - Knowledge gap identification

### 4. Keywords Module (`keywords.py`)
- **Purpose**: Extract representative keywords and create cluster summaries
- **Technology**: KeyBERT with scientific term enhancement
- **Features**:
  - Multi-technique keyword extraction
  - Scientific terminology detection
  - Cluster-level summarization
  - Quality assessment and scoring

### 5. Insights Module (`insights.py`)
- **Purpose**: Generate actionable research insights and opportunities
- **Technology**: Advanced statistical analysis and pattern detection
- **Features**:
  - Knowledge gap analysis
  - Publication trend detection
  - Consensus/disagreement identification
  - Research opportunity generation

## 📈 Dashboard Integration

The pipeline outputs are designed for easy integration with data visualization dashboards:

### Recommended Visualizations
1. **Knowledge Gap Heatmap**: Cluster sizes vs. research activity
2. **Publication Trends**: Time series of research activity by cluster
3. **Similarity Network**: Paper similarity relationships
4. **Keyword Clouds**: Representative terms for each cluster
5. **Research Opportunity Matrix**: Priority vs. impact analysis

### Output Formats
- **CSV**: For tabular data and time series
- **JSON**: For hierarchical data and metadata
- **JSONL**: For embeddings and large datasets

## 🎛️ Advanced Usage

### Custom Models
```python
# Override models in config.py
config.models.summarization_model = "facebook/bart-large-cnn"
config.models.embedding_model = "allenai/scibert_scivocab_uncased"
```

### Batch Processing
```python
# Process large datasets efficiently
from src.summarization import AdvancedSummarizer
summarizer = AdvancedSummarizer()
summaries = summarizer.summarize_batch(texts, batch_size=16)
```

### Component Integration
```python
# Use individual components programmatically
from src.embeddings import create_enhanced_embeddings
df_embeddings, embeddings, similarity_info = create_enhanced_embeddings(df)
```

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues with Large Models**
   ```bash
   # Use smaller models for limited memory
   python run_pipeline.py --config-model "sshleifer/distilbart-cnn-12-6"
   ```

2. **CUDA/GPU Issues**
   - Models automatically fall back to CPU if GPU unavailable
   - Check CUDA installation: `torch.cuda.is_available()`

3. **Missing Dependencies**
   ```bash
   # Install missing scientific packages
   pip install spacy
   python -m spacy download en_core_web_sm
   ```

4. **Data Format Issues**
   - Ensure CSV has required columns: `paper_id`, `title`, `abstract`
   - Check for encoding issues (use UTF-8)

### Performance Optimization

1. **Use GPU acceleration** for large datasets
2. **Adjust batch sizes** based on available memory
3. **Skip components** for partial analysis
4. **Use smaller models** for faster processing

## 📊 Quality Metrics

The pipeline provides comprehensive quality metrics for all outputs:

### Summarization Quality
- Compression ratio
- Summary length distribution
- Success/failure rates
- Fallback summary usage

### Embedding Quality
- Embedding norms
- Zero embedding detection
- Similarity score distributions
- PCA explained variance

### Clustering Quality
- Silhouette scores
- Cluster size distributions
- Inter-cluster distances
- Knowledge gap identification

### Insights Quality
- Consensus confidence scores
- Trend significance tests
- Opportunity priority rankings
- Coverage statistics

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/
```

## 📚 References

- **BART**: Lewis, M., et al. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." ACL 2020.
- **SciBERT**: Beltagy, I., et al. "SciBERT: A Pretrained Language Model for Scientific Text." EMNLP 2019.
- **KeyBERT**: Grootendorst, M. "KeyBERT: Minimal keyword extraction with BERT." 2020.
- **SentenceTransformers**: Reimers, N. & Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

<<<<<<< HEAD
---

## 🤖 Chatbot Backend API

The NASA Bioscience Literature Chatbot provides a modular backend for answering questions about space biology and bioscience research papers. This section documents the API for integration with user interfaces.

### ✨ Key Features

- **Semantic Search**: Natural language question answering using paper embeddings
- **Query Routing**: Automatic routing between fact-based and semantic queries
- **Citation Support**: Provides source papers for all answers
- **Modular Design**: Easy integration with web interfaces, mobile apps, or other UIs
- **Future Neo4j Support**: Prepared for graph database integration

### 🚀 Quick Start

#### Command Line Interface

```bash
# Run the interactive chatbot CLI
cd src
python run_chatbot_cli.py
```

#### Programmatic Usage

```python
from src.chatbot.routing import route_query

# Answer a semantic question
response = route_query("How does microgravity affect bone density?")
print(response['answer'])
for citation in response['citations']:
    print(f"- {citation['title']} ({citation['doi']})")
```

### 📚 API Reference

#### Main Function: `route_query(user_input: str) -> Dict`

Routes user queries to appropriate processing pipelines and returns structured responses.

**Parameters:**
- `user_input` (str): User's question or query

**Returns:**
- `Dict` with keys:
  - `answer` (str): Generated answer (2-4 sentences)
  - `citations` (List[Dict]): Source papers with `title` and `doi`

**Example:**
```python
response = route_query("What are the effects of space radiation on cells?")
# {
#   'answer': 'Space radiation causes DNA damage and cellular stress...',
#   'citations': [
#     {'title': 'Cellular Responses to Space Radiation', 'doi': 'PMC123456'},
#     {'title': 'DNA Damage in Microgravity', 'doi': 'PMC789012'}
#   ]
# }
```

### 🔀 Query Types

#### Semantic Queries (Recommended)
Best for conceptual questions and explanations:
- "How does microgravity affect bone density?"
- "What are the cellular mechanisms of muscle atrophy in space?"
- "Explain the relationship between radiation and DNA damage"

#### Fact Queries (Limited Functionality)
For specific factual information (currently uses placeholder responses):
- "How many papers study microgravity effects?"
- "List all papers about mouse experiments"
- "Which researchers work on space biology?"

### 🏗️ Architecture

```
User Query
    ↓
Query Router (routing.py)
    ├── Fact Query → Graph Queries (placeholder)
    └── Semantic Query → QA Pipeline
                            ├── Load embeddings
                            ├── Embed query
                            ├── Retrieve similar papers
                            └── Generate answer
```

#### Core Components

**1. QAPipeline (`src/chatbot/qa_pipeline.py`)**
- Loads paper summaries and pre-computed embeddings
- Embeds user queries using Sentence-BERT
- Retrieves top-5 similar papers using cosine similarity
- Generates answers using language models
- Formats citations from source papers

**2. QueryRouter (`src/chatbot/routing.py`)**
- Analyzes query to determine type (fact vs semantic)
- Routes to appropriate processing pipeline
- Handles fallbacks and error cases

**3. GraphQueries (`src/graph_queries.py`)**
- Placeholder for future Neo4j graph database queries
- Currently returns development status messages
- Prepared for fact-based query implementation

### 🔧 Configuration

The chatbot uses configuration from `src/config.py`:

```python
# Default models (configurable)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
qa_model = "gpt2"  # For answer generation
max_papers = 5     # Number of papers to retrieve
```

### 🧪 Testing

Run comprehensive unit tests:

```bash
cd tests
python -m pytest test_chatbot.py -v
```

Test coverage includes:
- QA pipeline functionality
- Query routing logic
- Data loading and processing
- Error handling
- Integration tests

### 🔗 UI Integration Examples

#### Web API Endpoint (Flask Example)

```python
from flask import Flask, request, jsonify
from src.chatbot.routing import route_query

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({'error': 'Query required'}), 400
    
    try:
        response = route_query(user_query)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

#### JavaScript Frontend Integration

```javascript
async function askChatbot(query) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query })
    });
    
    const data = await response.json();
    
    if (response.ok) {
        displayAnswer(data.answer);
        displayCitations(data.citations);
    } else {
        displayError(data.error);
    }
}
```

### 🚧 Future Enhancements

#### Neo4j Graph Database Integration
The system is prepared for graph database integration:

```python
# Future implementation in graph_queries.py
def answer_fact_query(user_input: str) -> Dict:
    # Parse query to extract entities and relationships
    entities = extract_entities(user_input)
    
    # Build Cypher query
    cypher_query = build_cypher_query(user_input, entities)
    
    # Execute against Neo4j
    results = neo4j_session.run(cypher_query)
    
    # Format response
    return format_graph_response(results)
```

#### Advanced Features Roadmap
- **Multi-turn Conversations**: Context-aware follow-up questions
- **Custom Filters**: Filter by date, author, topic, or experiment type
- **Visual Results**: Integration with charts and graphs
- **Real-time Updates**: Live updates as new papers are added
- **Personalization**: User-specific preferences and query history

### 📝 Error Handling

The chatbot includes comprehensive error handling:

```python
# Graceful degradation examples
if not self.load_data():
    return {
        'answer': "I cannot access the research database at the moment.",
        'citations': []
    }

# Fallback responses
except Exception as e:
    logger.error(f"Error processing query: {e}")
    return {
        'answer': "I encountered an error. Please try rephrasing your question.",
        'citations': []
    }
```

### 🔒 Security Considerations

For production deployment:
- Input validation and sanitization
- Rate limiting for API endpoints
- Authentication for sensitive queries
- Logging for audit trails
- HTTPS for all communications

### 📈 Performance

Current performance characteristics:
- **Query Processing**: ~2-5 seconds per query
- **Memory Usage**: ~500MB for loaded models and embeddings
- **Scalability**: Supports concurrent requests with proper deployment
- **Optimization**: Lazy loading of models, embedding caching

### 🆘 Troubleshooting

Common issues and solutions:

**"Cannot access research database"**
- Ensure `outputs/summaries/paper_summaries.csv` exists
- Ensure `outputs/embeddings/paper_embeddings.jsonl` exists
- Run the main pipeline first: `python run_pipeline.py`

**"Model download errors"**
- Check internet connection
- Verify sufficient disk space (~2GB for models)
- Try running with `--no-cache-dir` if using pip

**"Memory errors"**
- Reduce `max_papers` parameter
- Use smaller embedding models
- Consider GPU acceleration for large deployments

---

## 🏷️ Version

**Version 2.0** - Production Ready NASA Bioscience AI Pipeline with Chatbot Backend
=======
## 🏷️ Version

**Version 2.0** - Production Ready NASA Bioscience AI Pipeline
>>>>>>> 675f2a402fb638e07614d43eaf9e00688e128708

**Last Updated**: October 2025

---

**Built with ❤️ for NASA's scientific research community**