"""Command-line interface for Dagmar document search.

This module provides a CLI for searching documents using the QdrantStore
vector database with hybrid retrieval and reranking capabilities.
"""

import argparse
from pathlib import Path

from dagmar.logging_config import setup_logging
from dagmar.store import Store


def main():
    """Entry point for document search functionality."""
    parser = argparse.ArgumentParser(
        prog="dagmar",
        description="Search documents using vector similarity with hybrid retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --results 5 --source document.pdf "What is the main topic?"
  %(prog)s -k 3 -f report.md "Summarize the key findings"
  %(prog)s --source data.csv "Show me entries with status=active"
        """,
    )

    parser.add_argument(
        "--results",
        "-k",
        type=int,
        default=4,
        help="Number of search results to return (default: 4)",
    )

    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        action="append",
        help="""Path to the document file to search in,
        or a regex pattern to match multiple file names when searching across documents.
        Can be provided multiple times to search across several sources.""",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set the logging level (default: WARNING)",
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query to find relevant content",
    )

    args = parser.parse_args()

    # Initialize logging
    setup_logging(args.log_level)

    # Validate that all provided sources exist as files if they're paths.
    # If not, treat as regex patterns (leave as strings).
    sources = []
    for src in args.source:
        src_path = Path(src)
        if src_path.exists():
            if not src_path.is_file():
                parser.error(f"'{src}' exists but is not a file")
            sources.append(str(src_path))
        else:
            # Not a path: treat as pattern/string
            sources.append(src)

    # Initialize store and perform search
    try:
        store = Store()
        for source in sources:
            if Path(source).exists():
                store.add_to_docs([source])
            else:
                store.import_docs(source)
        results = store.search_docs(args.query, args.results)

        if not results:
            print("No results found.")
            return

        # Display results
        print(f"\nSearch Results for: '{args.query}'")
        print("=" * 60)

        for i, result in enumerate(results, 1):
            print(result["score"])
            print(result["content"])
            if result["metadata"]:
                print(f"Metadata: {result['metadata']}")
            print("-" * 60)
    except Exception as e:
        parser.error(f"Search failed: {e}")


if __name__ == "__main__":
    main()
