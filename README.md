# Dagmar

**Local RAG for LLM-based applications**

Dagmar is a powerful document search and retrieval system that enables semantic search across your documents using advanced vector similarity techniques. Built for local deployment, it provides a complete Retrieval-Augmented Generation (RAG) solution with hybrid retrieval capabilities, automatic document indexing, and intelligent reranking.

## Features

- **Hybrid Retrieval**: Combines dense and sparse vector embeddings for superior search accuracy
- **Keyword-Based Filtering**: Support for structured field queries with logical operators
- **Reranking**: Uses cross-encoder models to rerank results for better relevance
- **Multi-Format Support**: Handles PDF, Markdown, Text, and CSV files automatically
- **LLM-Based PDF Processing**: Advanced PDF extraction using Azure OpenAI vision models for structured content
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

Search for content in a document using natural language queries:

```bash
dagmar --file document.pdf "What is the main topic discussed?"
```

### Advanced Search Options

```bash
# Specify number of results (default: 4)
dagmar --results 10 --file report.md "Summarize the key findings"

# Keyword-based field filtering (structured queries)
dagmar --fields --file document.pdf "page_content like 'important keyword'"

# Short form options
dagmar -k 5 -f data.csv "Show me entries with status=active"
```

### Command Line Options

- `-f, --file`: Path to the document file (required)
- `-k, --results`: Number of search results to return (default: 4)
- `--fields`: Use keyword-based field search instead of semantic search (default: false)
- `query`: Search query (positional argument)

### Examples

```bash
# Search in a PDF document
dagmar --file manual.pdf "cli usage"

# Search in a Markdown file
dagmar --file notes.md "implementation details"

# Search in a CSV file
dagmar --file data.csv "filter by category"

# Keyword-based filtering examples
dagmar --fields --file document.pdf "page_content like 'machine learning'"
dagmar --fields --file report.md "page_content like 'error' and page_content like 'handling'"
dagmar --fields --file spec.pdf "(page_content like 'API' or page_content like 'interface') and not page_content like 'deprecated'"
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
- `file_path`: Path to the document file to search
- `limit`: Number of top results to return (default: 4)

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
│   ├── splitters.py           # Document processing utilities
│   └── parse_filter_string.py # Filter query parsing utilities
├── qdrant_db/                 # Local vector database
├── pyproject.toml             # Project configuration
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
from dagmar.store import QdrantStore
from pathlib import Path

# Initialize the store
store = QdrantStore()

# Search in a document
results = store.search_semantic(
    Path("document.pdf"),
    "your search query",
    k=5
)

# Process results
for result in results:
    print(f"Content: {result['content']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Score: {result['score']}")

# Keyword-based field filtering
results = store.search_by_fields(
    Path("document.pdf"),
    "page_content like 'important keyword'",
    k=5
)

# Advanced filtering with logical operators
results = store.search_by_fields(
    Path("report.md"),
    "page_content like 'error' and page_content like 'handling'",
    k=10

)
```

## Filter Query Syntax

When using keyword-based field filtering (`--fields` option), you can use structured queries with the following syntax:

### Supported Operators

- **Text Search**: `page_content like 'search text'` - Search for text within document content
- **Logical AND**: `condition1 and condition2` - Both conditions must be true
- **Logical OR**: `condition1 or condition2` - Either condition must be true
- **Logical NOT**: `not condition` - Negate a condition
- **Grouping**: `(condition1 or condition2) and condition3` - Use parentheses for precedence

### Examples

```bash
# Simple text search
"page_content like 'machine learning'"

# Multiple keywords with AND
"page_content like 'error' and page_content like 'handling'"

# Multiple keywords with OR
"page_content like 'API' or page_content like 'interface'"

# Complex query with grouping
"(page_content like 'bug' or page_content like 'issue') and not page_content like 'resolved'"
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
- [ ] Search in multiple files
- [ ] Note - the all payload keys goes to metadata dict in Qdrant except page_content
- [x] Refactor splitters to put every spliter into separate file in src/dagmar/splitters/ directory
- [ ] Make LLM splitter more universal and allow to use it for other file types.
- [x] Reduce image size to have one dimantion no longer than 1024px