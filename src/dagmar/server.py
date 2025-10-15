"""MCP server for the Dagmar RAG."""

import signal
import sys
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from dagmar.store import QdrantStore

# instantiate an MCP server client
mcp = FastMCP("Dagmar RAG")


@mcp.tool()
def dagmar_doc_search(
    ctx: Context,  # noqa: ARG001,ARG002
    query: str,
    file_path: str,
    limit: int = 4,
):
    """Search for relevant content in a document using vector similarity and hybrid retrieval.

    This tool searches documents using:
    - Hybrid search: Combines vector similarity (dense + sparse) with filtering
    - Reciprocal Rank Fusion (RRF) for combining dense and sparse vector results
    - Reranking: Uses cross-encoder models to rerank results for better relevance

    Args:
        ctx: Context object.
        query: Text query to search for relevant content.
               It should be reformulated to get the most relevant information from Vector Database.
        file_path: Path to the local document file to search in.
        limit: Number of top results to return.

    Returns:
        List of dictionaries containing content, metadata, and relevance scores
        for the top-k most relevant document sections.

    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    if not Path(file_path).is_file():
        raise NotADirectoryError(f"File {file_path} is not a file")
    store = QdrantStore("./qdrant_db")
    results = store.search_semantic(Path(file_path), query, limit)
    return results


def sigint_handler(signum, frame):  # noqa: ARG001
    """Handle SIGINT signal for graceful shutdown."""
    sys.exit(0)


def main():
    """Entry point for the package."""
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        mcp.run(transport="stdio")
    except Exception:
        raise


if __name__ == "__main__":
    main()
