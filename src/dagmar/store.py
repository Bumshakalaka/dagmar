"""Utilities for managing Qdrant-backed document collections.

Provide a `Store` abstraction that connects to a Qdrant instance, tracks
loaded `StoreDocument` objects, imports existing collections, and performs
cross-document search with reranking.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient

from dagmar.store_document import QDRANT_URL, StoreDocument, rerank

logger = logging.getLogger(__name__)


class Store:
    """Manage documents and search operations via a Qdrant client.

    The store maintains an in-memory list of `StoreDocument` instances, provides
    helpers to import collections from Qdrant, and runs searches with optional
    reranking.
    """

    def __init__(self, qdrant_server: Optional[str] = None):
        """Initialize the store and connect to a Qdrant location.

        Sets up the Qdrant client connection. The client can connect to a local
        or remote Qdrant server, or use in-memory storage.

        Args:
            qdrant_server: Qdrant location. If ``None``, uses ``QDRANT_URL``.
                Accepts:
                - ":memory:" for in-memory storage
                - Local path to a Qdrant database directory
                - HTTP/HTTPS URL (e.g., "http://localhost:6333")

        Raises:
            ValueError: If the server URL format is invalid.
            FileNotFoundError: If the path's parent directory does not exist.
            NotADirectoryError: If the provided path is not a directory.

        """
        if not qdrant_server:
            qdrant_server = QDRANT_URL

        logger.info(f"Initializing store with location: {qdrant_server}")
        if qdrant_server == ":memory:":
            self.client = QdrantClient(":memory:")
        elif qdrant_server.startswith("http"):
            # Acceptable formats: http://host[:port]
            match = re.match(r"(https?)://([^:/]+)(?::(\d+))?", qdrant_server)
            if not match:
                raise ValueError(f"Invalid qdrant_server format: {qdrant_server}")

            proto, host, port = match.groups()
            port = int(port) if port else 6333
            self.client = QdrantClient(url=f"{proto}://{host}", port=port)
        else:
            if not Path(qdrant_server).parent.exists():
                raise FileNotFoundError(f"Directory {Path(qdrant_server).parent} does not exist")
            if not Path(qdrant_server).is_dir():
                raise NotADirectoryError(f"Path {qdrant_server} is not a directory")
            self.client = QdrantClient(path=qdrant_server)
        self.qdrant_server = qdrant_server
        self.documents: List[StoreDocument] = []

    @property
    def docs(self):
        """Return the list of loaded documents.

        Returns:
            The in-memory list of loaded `StoreDocument` instances.

        """
        return self.documents

    def clean_docs(self):
        """Clear all loaded documents from memory."""
        for doc in self.documents:
            del doc
        self.documents = []

    def add_to_docs(self, docs: List[Path | str]):
        """Add documents to the store if missing, then track them.

        Each path or collection name is materialized into a `StoreDocument`. If it
        does not exist in Qdrant, it is added, and then referenced in-memory.

        Args:
            docs: Paths or collection names to add and track.

        Returns:
            Nothing.

        """
        for doc in docs:
            d = StoreDocument(doc, self.qdrant_server)
            if not any(existing.doc_exist(str(doc)) for existing in self.documents):
                if not d.doc_exist():
                    d.add_doc()
                self.documents.append(d)

    def del_from_docs(self, docs: List[Path | str]):
        """Remove matching documents from the in-memory list.

        Args:
            docs: Paths or collection names to remove from tracking.

        Returns:
            Nothing.

        """
        for doc in docs:
            for stored in list(self.documents):
                if stored.doc_exist(str(doc)):
                    self.documents.remove(stored)
                    del stored

    def import_docs(self, pattern: Optional[str] = None) -> List[str]:
        """Import collections from Qdrant, optionally filtered by pattern.

        Loads collection names from the connected Qdrant instance. When a pattern
        is provided, only names matching the regular expression are imported. The
        corresponding `StoreDocument` objects are appended to the in-memory list.

        Args:
            pattern: Optional regular expression to filter collection names.

        Returns:
            The list of imported collection names.

        """
        logger.info(f"Getting documents matching pattern: {pattern}")
        ret = []
        collections = self.client.get_collections()
        if not pattern:
            ret = [collection.name for collection in collections.collections]
        else:
            for collection in collections.collections:
                if re.match(pattern, collection.name):
                    ret.append(collection.name)
        for r in ret:
            self.documents.append(StoreDocument(r, self.qdrant_server))
        return ret

    def search_docs(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search across loaded documents and rerank results.

        Executes retrieval against each loaded `StoreDocument`, converts results
        for reranking, and returns the top results.

        Args:
            query: Natural-language query to search for.
            k: Number of results to return after reranking.

        Returns:
            A list of dictionaries with ``content``, ``metadata``, and ``score``.

        """
        if not self.documents:
            logger.error("No documents in the store")
            return []
        results = []
        for doc in self.documents:
            results.extend(doc.search_doc(query, k=k * 2, rerank_results=False))
        converted_results = [(Document(page_content=r["content"], metadata=r["metadata"]), r["score"]) for r in results]
        reranked = rerank(query, converted_results, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata, "score": score} for doc, score in reranked]
