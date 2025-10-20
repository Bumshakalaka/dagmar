"""MCP server for the Dagmar RAG."""

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

from mcp.server.fastmcp import Context, FastMCP

from dagmar.logging_config import setup_logging
from dagmar.store import Store

logger = logging.getLogger(__name__)

# instantiate an MCP server client
mcp = FastMCP("Dagmar RAG")


@mcp.tool()
def dagmar_doc_search(
    ctx: Context,  # noqa: ARG001,ARG002
    query: str,
    sources: List[str | Path],
    limit: int = 4,
    db_server: Optional[str] = None,
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
        sources: List of paths to the local document files to search in, or a regex pattern to match
                 multiple file names when searching across documents.
        limit: Number of top results to return.
        db_server: Optional Qdrant server location. Select :memory: for in-memory storage for temporary use,
            otherwise uses persistent storage (None) as defined in QDRANT_URL environment variable.

    Returns:
        List of dictionaries containing content, metadata, and relevance scores
        for the top-k most relevant document sections.

    """
    logger.info(
        f"Received search request: query='{query}', sources='{sources}', limit={limit}, db_server='{db_server}'"
    )
    try:
        store = Store(db_server)
        for source in sources:
            if Path(source).exists():
                store.add_to_docs([source])
            else:
                store.import_docs(str(source))
        results = store.search_docs(query, limit)
        logger.info(f"Search completed, returning {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Search request failed: {e}")
        raise


def sigint_handler(signum, frame):  # noqa: ARG001
    """Handle SIGINT signal for graceful shutdown."""
    sys.exit(0)


def main():
    """Entry point for the package."""
    parser = argparse.ArgumentParser(description="Dagmar RAG MCP Server")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set the logging level (default: WARNING)",
    )

    args = parser.parse_args()

    # Initialize logging
    setup_logging(args.log_level)

    logger.info("Starting Dagmar RAG MCP Server")
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise


if __name__ == "__main__":
    main()
