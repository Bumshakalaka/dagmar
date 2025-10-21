# Dagmar

**Local RAG for LLM-based applications**

Dagmar is a powerful document search and retrieval system that enables semantic search across your documents using advanced vector similarity techniques. Built for local deployment, it provides a complete Retrieval-Augmented Generation (RAG) solution with hybrid retrieval capabilities, automatic document indexing, and intelligent reranking.

## Features

- **Hybrid Retrieval**: Combines dense and sparse vector embeddings for superior search accuracy
- **Reranking**: Uses cross-encoder models to rerank results for better relevance
- **Multi-Format Support**: Handles PDF, Markdown, Text, PPTX and CSV files automatically
- **Multi-Document Search**: Search across multiple documents using regex patterns
- **LLM-Based PDF Processing**: Advanced PDF and PPTX extraction using Azure OpenAI vision models for structured content
- **Local Vector Database**: Stores embeddings locally using Qdrant for privacy and performance
- **Automatic Indexing**: Documents are processed and indexed on-demand
- **CLI Interface**: Simple command-line interface for easy integration
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

# Install with uv
uv sync

# Set the QDRANT_URL key in .env file to the path to the Qdrant database.
# QDRANT_URL=./qdrant_db
# or
# QDRANT_URL=http://localhost
# or
# QDRANT_URL=:memory:

# to use LLM vision model for pdf processing, you need to set up Azure OpenAI or OpenAI API keys in .env file
# AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
# AZURE_OPENAI_API_KEY=<your-api-key>
# OPENAI_API_KEY=<your-api-key>

# or

# OPENAI_API_KEY=<your-api-key>

```

## CLI Usage

### Basic Search

Search for content in documents using natural language queries:

```bash
dagmar --source document.pdf "What is the main topic discussed?"
```

### Advanced Search Options

```bash
# Specify number of results (default: 4)
dagmar --results 10 --source report.md "Summarize the key findings"

# Search across multiple files
dagmar --source report.md --source data.csv "implementation details"

# Search across multiple files using regex pattern
dagmar --source "*.manual.*\.pdf" --source "notes*.md" "implementation details"

# Short form options
dagmar -k 5 -s data.csv "Show me entries with status=active"
```

### Command Line Options

- `-f, --source`: Path to the document file to search in, or a regex pattern to match multiple file names when searching across documents (required). Can be provided multiple times to search across several sources.
- `-k, --results`: Number of search results to return (default: 4)
- `query`: Search query (positional argument)

### Examples

```bash
# Search in a PDF document
dagmar --source manual.pdf "cli usage"

# Search in a Markdown file
dagmar --source notes.md "implementation details"

# Search in a CSV file
dagmar --source data.csv "filter by category"

# Search across multiple specific files
dagmar --source manual --source ~\docs\notes.md "implementation details" -k 2

# Search across multiple files using regex patterns
dagmar --source "*.manual.*\.pdf" --source "notes*.md" "implementation details"
```

## MCP Server

Dagmar provides an MCP (Model Context Protocol) server that allows AI assistants and IDEs to search documents using Dagmar's powerful retrieval capabilities.

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
│   │   ├── __init__.py
│   │   ├── base.py            # Base splitter classes
│   │   ├── csv.py             # CSV file processing
│   │   ├── md.py              # Markdown file processing
│   │   ├── pdf.py             # PDF file processing
│   │   ├── pdf_llm.py         # LLM-based PDF processing
│   │   ├── pptx.py            # PPTX file processing
│   │   ├── txt.py             # Text file processing
│   │   ├── llm_base.py        # Base LLM splitter utilities
│   │   └── image_to_md_prompt.md # LLM prompt for image processing
│   ├── my_qdrant_vector_store.py # Qdrant vector store implementation
│   ├── md_fixer.py            # Markdown content fixing utilities
│   └── logging_config.py      # Logging configuration
├── qdrant_db/                 # Local vector database
├── processed_files/           # Processed document storage
├── pyproject.toml             # Project configuration
├── uv.lock                    # Dependency lock file
└── README.md                  # This file
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

- [ ] Add support for pptx and docx files (use markitdown)
- [x] Add MCP server only stdio
- [ ] Add store method to get complete page of document (filter by page field and group and return)
- [x] Add LLM pdf parser (image-to-text)
- [x] Add support for Qdrant server
- [ ] Add tests
- [ ] Allow to update payload in collection. E.g. while working with dokument, user would like to add some information to payload for further better search results.
- [x] Search in multiple files
- [ ] Note - the all payload keys goes to metadata dict in Qdrant except page_content
- [x] Refactor splitters to put every spliter into separate file in src/dagmar/splitters/ directory
- [x] Make LLM splitter more universal and allow to use it for other file types.
- [x] Reduce image size to have one dimantion no longer than 1024px