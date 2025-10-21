# Dagmar

![Dagmar Logo](./img/banner.png)

**Local RAG for LLM-based applications**

Dagmar is a document search and retrieval system that enables semantic search across your documents using advanced vector similarity techniques. Built for local deployment, it provides a Retrieval-Augmented Generation (RAG) solution with hybrid retrieval capabilities, automatic document indexing, and reranking.

Each processed document is stored in the database as a separate collection named after the file. For pdf and pptx in addition the converted markdown is stored in `./processed_files/` directory.

## Features

- **Hybrid Retrieval**: Combines dense and sparse vector embeddings for superior search accuracy
- **Reranking**: Uses cross-encoder models to rerank results for better relevance
- **Multi-Format Support**: Handles PDF, Markdown, Text, PPTX and CSV files automatically
- **Multi-Document Search**: Search across multiple documents using regex patterns
- **LLM-Based PDF Processing**: Advanced PDF and PPTX extraction using Azure OpenAI vision models for structured content
- **Local Vector Database**: Stores embeddings locally or on a remote Qdrant server
- **Automatic Indexing**: Documents are processed and indexed on-demand
- **CLI Interface**: Simple command-line interface for testing and development
- **MCP Server**: Model Context Protocol server for integration with AI assistants and IDEs


## Installation

### Prerequisites

- Python 3.11 or higher
- uv package manager

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd dagmar

# Install with uv (allow prerelease packages as langchain 1.0 alpha is used)
uv sync --prerelease=allow

# Set the QDRANT_URL key in .env file to the path to the Qdrant database.
# this is default path for local database
# QDRANT_URL=./qdrant_db
# or
# QDRANT_URL=http://localhost
# or
# QDRANT_URL=:memory:

# to use LLM vision model for document processing, you need to set up Azure OpenAI or OpenAI API keys in .env file
# AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
# AZURE_OPENAI_API_KEY=<your-api-key>
# OPENAI_API_KEY=<your-api-key>

# or

# OPENAI_API_KEY=<your-api-key>

```

## CLI Usage

### Basic Search

Search for content in documents using natural language queries:

### Command Line Options

- `-f, --source`: Path to the document file to search in, or a regex pattern to match multiple file names when searching across documents (required). Can be provided multiple times to search across several sources.
- `-k, --results`: Number of search results to return (default: 4)
- `query`: Search query (positional argument)

```bash
dagmar --source document.pdf "What is the main topic discussed?"

# Specify number of results (default: 4)
dagmar --results 10 --source report.md "Summarize the key findings"

# Search across multiple files
dagmar --source report.md --source data.csv "implementation details"

# Search across multiple files using regex pattern
# regexp patters is used to find already indexed files in the database
dagmar --source "*.manual.*\.pdf" --source "notes*.md" "implementation details"

# Short form options
dagmar -k 5 -s data.csv "Show me entries with status=active"
```

## MCP Server

Dagmar provides an MCP (Model Context Protocol) server that allows AI assistants and IDEs to search documents using Dagmar's retrieval capabilities.

### Install the MCP Server in Cursor

Add to mcp.json in Cursor:

```json
    "dagmar": {
      "command": "uv",
      "args": ["run", "--directory", "path_to_dagmar","dagmar-server"]
    }
```

### MCP Tool: Document Search

The MCP server exposes a `dagmar_doc_search` tool that enables semantic document search with the following capabilities:

- **Hybrid Search**: Combines dense and sparse vector embeddings using Reciprocal Rank Fusion (RRF)
- **Reranking**: Uses cross-encoder models to improve result relevance
- **Multi-Format Support**: Works with PDF, Markdown, Text, and CSV files
- **Flexible Results**: Configurable number of results returned

#### Tool Parameters

- `query`: Natural language search query
- `sources`: List of paths to the local document files to search in, or a regex pattern to match multiple file names when searching across documents
- `limit`: Number of top results to return (default: 4)
- `db_server`: Optional Qdrant server location. Select :memory: for in-memory storage for temporary use, otherwise uses persistent storage (None) as defined in QDRANT_URL environment variable.

## Technology Stack

### Core Technologies

- **Python 3.11+**: Modern Python runtime
- **Qdrant**: High-performance vector database for similarity search
- **LangChain**: Framework for LLM applications and document processing
- **FastEmbed**: Efficient embedding generation for local use with CPU
- **FastMCP**: Model Context Protocol server framework
- **Cross-Encoders**: Advanced reranking models
- **Azure OpenAI**: Vision models for advanced PDF content extraction
- **Pillow**: Image processing for PDF page conversion
- **PyMuPDF**: High-performance PDF rendering and image extraction

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
- **LLM Vision Extraction**: Azure OpenAI vision models for structured PDF content extraction
- **LangChain Text Splitters**: Intelligent document chunking
- **CSV Sniffer**: Automatic CSV format detection
- **Markdown Fixer**: Fixes markdown content to improve search accuracy
- **python-pptx**: PPTX file processing

## Project Structure

```
dagmar/
├── src/dagmar/
│   ├── cli.py                 # Command-line interface
│   ├── server.py              # MCP server implementation
│   ├── store.py               # Vector store and search logic
│   ├── store_document.py      # Document storage and processing utilities
│   ├── parse_filter_string.py # Filter query parsing utilities
│   ├── splitters/             # Document processing utilities
│   ├── my_qdrant_vector_store.py # Qdrant vector store implementation
│   ├── md_fixer.py            # Markdown content fixing utilities
│   └── logging_config.py      # Logging configuration
├── qdrant_db/                 # Local vector database
├── processed_files/           # Processed document storage
```

## Configuration

### Vector Database

The system uses a local Qdrant database stored in `./qdrant_db/` by default. This can be changed by setting the `QDRANT_URL` environment variable.

Each document gets its own collection named after the file.

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
from dagmar.store import Store
from pathlib import Path

# Initialize the store
store = Store()

# Add documents (for single or multiple files)
store.add_to_docs([Path("document1.pdf"), Path("document2.md")])

# Or import from patterns
store.import_docs(".+\.pdf")

# Search in the documents
results = store.search_docs(
    "your search query",
    k=5
)

# Process results
for result in results:
    print(f"Score: {result['score']}")
    print(f"Content: {result['content']}")
    print(f"Metadata: {result['metadata']}")

```

## Acknowledgments

- [Qdrant](https://qdrant.tech/) for the excellent vector database
- [LangChain](https://langchain.com/) for the LLM framework
- [FastEmbed](https://github.com/qdrant/fastembed) for efficient embeddings
- [Sentence Transformers](https://www.sbert.net/) for embedding models

## TODO:

- [ ] Add mcp progress/notifications
- [ ] Add support for markitdown package for markdown processing
- [ ] Add store method to get complete page of document (filter by page field and group and return)
- [ ] Add tests
- [ ] Add searching by payload fields
- [ ] Allow to update payload in collection. E.g. while working with dokument, user would like to add some information to payload for further better search results.
- [ ] Note - the all payload keys goes to metadata dict in Qdrant except page_content