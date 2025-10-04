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

## 🏷️ Version

**Version 2.0** - Production Ready NASA Bioscience AI Pipeline

**Last Updated**: October 2025

---

**Built with ❤️ for NASA's scientific research community**