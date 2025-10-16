"""Command-line interface for Dagmar document search.

This module provides a CLI for searching documents using the QdrantStore
vector database with hybrid retrieval and reranking capabilities.
"""

import argparse
from pathlib import Path

from dagmar.logging_config import setup_logging
from dagmar.store import QdrantStore


def main():
    """Entry point for document search functionality."""
    parser = argparse.ArgumentParser(
        prog="dagmar",
        description="Search documents using vector similarity with hybrid retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --results 5 --file document.pdf "What is the main topic?"
  %(prog)s -k 3 -f report.md "Summarize the key findings"
  %(prog)s --file data.csv "Show me entries with status=active"
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
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to the document file to search in",
    )

    parser.add_argument(
        "--fields",
        action="store_true",
        default=False,
        help="Use keyword-based field search instead of semantic search (default: false)",
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

    # Validate file exists
    file_path = Path(args.file)
    if not file_path.exists():
        parser.error(f"File '{args.file}' does not exist")

    if not file_path.is_file():
        parser.error(f"'{args.file}' is not a file")

    # Initialize store and perform search
    try:
        store = QdrantStore()
        if args.fields:
            results = store.search_by_fields(file_path, args.query, args.results)
        else:
            results = store.search_semantic(file_path, args.query, args.results)

        if not results:
            print("No results found.")
            return

        # Display results
        print(f"\nSearch Results for: '{args.query}'")
        print("=" * 60)

        for i, result in enumerate(results, 1):
            print(result["content"])
            if result["metadata"]:
                print(f"Metadata: {result['metadata']}")
            print("-" * 60)
    except Exception as e:
        parser.error(f"Search failed: {e}")


if __name__ == "__main__":
    main()
