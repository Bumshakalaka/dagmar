# Dagmar

**Local RAG for LLM-based applications**

Dagmar is a powerful document search and retrieval system that enables semantic search across your documents using advanced vector similarity techniques. Built for local deployment, it provides a complete Retrieval-Augmented Generation (RAG) solution with hybrid retrieval capabilities, automatic document indexing, and intelligent reranking.

## Features

- **Hybrid Retrieval**: Combines dense and sparse vector embeddings for superior search accuracy
- **Reranking**: Uses cross-encoder models to rerank results for better relevance
- **Multi-Format Support**: Handles PDF, Markdown, Text, and CSV files automatically
- **Local Vector Database**: Stores embeddings locally using Qdrant for privacy and performance
- **Automatic Indexing**: Documents are processed and indexed on-demand
- **CLI Interface**: Simple command-line interface for easy integration


## Installation

### Prerequisites

- Python 3.11 or higher
- uv package manager (recommended)

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd dagmar

# Install with uv (recommended)
uv sync

```

## CLI Usage

### Basic Search

Search for content in a document using natural language queries:

```bash
dagmar --file document.pdf "What is the main topic discussed?"
```

### Advanced Search Options

```bash
# Specify number of results (default: 4)
dagmar --results 10 --file report.md "Summarize the key findings"

# Short form options
dagmar -k 5 -f data.csv "Show me entries with status=active"
```

### Command Line Options

- `-f, --file`: Path to the document file (required)
- `-k, --results`: Number of search results to return (default: 4)
- `query`: Search query (positional argument)

### Examples

```bash
# Search in a PDF document
dagmar --file manual.pdf "cli usage"

# Search in a Markdown file
dagmar --file notes.md "implementation details"

# Search in a CSV file
dagmar --file data.csv "filter by category"
```

## Technology Stack

### Core Technologies

- **Python 3.11+**: Modern Python runtime
- **Qdrant**: High-performance vector database for similarity search
- **LangChain**: Framework for LLM applications and document processing
- **FastEmbed**: Efficient embedding generation for local use with CPU
- **Cross-Encoders**: Advanced reranking models

### Embedding Models

- **Dense Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
  - 384-dimensional semantic embeddings
  - Cosine similarity distance metric
- **Sparse Embeddings**: `prithivida/Splade_PP_en_v1`
  - SPLADE-based sparse vectors for keyword matching
  - Optimized for lexical search

### Reranking Model

- **Cross-Encoder**: `Xenova/ms-marco-MiniLM-L-6-v2`
  - Fine-tunes relevance scores after initial retrieval
  - Significantly improves search quality

### Document Processing

- **PyPDF**: PDF text extraction
- **LangChain Text Splitters**: Intelligent document chunking
- **CSV Sniffer**: Automatic CSV format detection

## Project Structure

```
dagmar/
├── src/dagmar/
│   ├── cli.py           # Command-line interface
│   ├── store.py         # Vector store and search logic
│   └── splitters.py     # Document processing utilities
├── qdrant_db/           # Local vector database
├── pyproject.toml       # Project configuration
└── README.md           # This file
```

## Configuration

### Vector Database

The system uses a local Qdrant database stored in `./qdrant_db/` by default. Each document gets its own collection named after the file.

### Embedding Configuration

- **Dense Model**: Optimized for semantic understanding
- **Sparse Model**: Optimized for keyword matching
- **Hybrid Fusion**: Uses Reciprocal Rank Fusion (RRF) for combining results

### Text Splitting

- **Chunk Size**: 1000-3000 characters depending on file type
- **Overlap**: 50-150 characters for context preservation
- **Markdown Aware**: Preserves markdown structure when possible

## Advanced Usage

### Programmatic Usage

```python
from dagmar.store import QdrantStore
from pathlib import Path

# Initialize the store
store = QdrantStore("./qdrant_db")

# Search in a document
results = store.search(
    Path("document.pdf"),
    "your search query",
    k=5
)

# Process results
for result in results:
    print(f"Content: {result['content']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Score: {result['score']}")
```

## Acknowledgments

- [Qdrant](https://qdrant.tech/) for the excellent vector database
- [LangChain](https://langchain.com/) for the LLM framework
- [FastEmbed](https://github.com/qdrant/fastembed) for efficient embeddings
- [Sentence Transformers](https://www.sbert.net/) for embedding models

---

