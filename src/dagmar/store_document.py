"""Document storage and retrieval system using Qdrant vector database.

This module provides functionality for storing, indexing, and searching documents
using hybrid dense and sparse vector embeddings with optional reranking capabilities.
It supports various document formats through configurable splitters and integrates
with Qdrant for efficient similarity search operations.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models

from dagmar.my_qdrant_vector_store import MyQdrantVectorStore
from dagmar.splitters import get_splitter

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())
QDRANT_URL = os.getenv("QDRANT_URL", "./qdrant_db")
"""Default Qdrant database URL or path from environment variable."""

dense = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=128)
"""Dense embedding model for semantic similarity search."""

sparse = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1", batch_size=128)
"""Sparse embedding model for keyword-based search."""

reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
"""Cross-encoder model for reranking search results."""


def rerank(query: str, results: List[tuple[Document, float]], k: int) -> List[tuple[Document, float]]:
    """Rerank search results using cross-encoder model.

    Uses a cross-encoder model to improve the relevance ranking of search results
    by computing more accurate similarity scores between the query and each document.

    Args:
        query: Original text query used for searching.
        results: List of document-score tuples from initial search.
        k: Number of top results to return after reranking.

    Returns:
        List of reranked document-score tuples, sorted by reranking scores
        in descending order.

    """
    result_docs = [r.page_content for r, s in results]
    scores = list(reranker.rerank(query, result_docs))
    reranked_results = []
    for doc, score in zip(results, scores):
        reranked_results.append((doc[0], score))

    reranked_results.sort(key=lambda x: x[1], reverse=True)
    return reranked_results[0:k]


class StoreDocument:
    """Document storage and retrieval manager using Qdrant vector database.

    This class provides a high-level interface for storing, indexing, and searching
    documents using hybrid dense and sparse vector embeddings. It supports various
    document formats through configurable splitters and offers optional reranking
    capabilities for improved search relevance.

    The class handles Qdrant client initialization, collection management, document
    processing, and similarity search operations with both dense and sparse vectors.
    """

    def __init__(self, doc_file: Path | str, qdrant_server: Optional[str] = None) -> None:
        """Initialize StoreDocument with Qdrant client and document file.

        Sets up the Qdrant client connection and prepares the document for processing.
        The client can connect to a local Qdrant instance, remote server, or use
        in-memory storage.

        Args:
            doc_file: Path to the document file to process, or collection name
                if the file doesn't exist.
            qdrant_server: Qdrant server location. If None, uses QDRANT_URL
                environment variable. Accepts:
                - ":memory:" for in-memory storage
                - Local path to Qdrant database directory
                - HTTP/HTTPS URL to Qdrant server (e.g., "http://localhost:6333")

        Raises:
            ValueError: If the qdrant_server format is invalid.
            FileNotFoundError: If the qdrant_server is a path to a directory
                that does not exist.
            NotADirectoryError: If the qdrant_server is a path to a file
                that is not a directory.

        """
        if not qdrant_server:
            qdrant_server = QDRANT_URL

        logger.info(f"Initializing document store for {doc_file} with location: {qdrant_server}")
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
        logger.info("Document store initialization completed")
        if (isinstance(doc_file, Path) and doc_file.exists()) or (
            isinstance(doc_file, str) and Path(doc_file).exists()
        ):
            self.doc = doc_file
        else:
            logger.info(f"{doc_file} is not valid file path, it will be treat as collection name")
            self.doc = None
        self.collection = Path(doc_file).name
        self.vector_store: Optional[MyQdrantVectorStore] = None

    def __repr__(self) -> str:
        """Return string representation of StoreDocument instance.

        Returns:
            String representation showing document file and collection name.

        """
        return f"StoreDocument(doc_file={self.doc}, collection={self.collection})"

    def __del__(self):
        """Clean up resources when StoreDocument instance is deleted.

        Logs the deletion and removes references to vector store and client
        to free up memory resources.
        """
        logger.debug(f"{self} deleted")
        del self.vector_store
        del self.client

    def _init_collection(self):
        """Initialize or retrieve existing Qdrant collection.

        Creates a new collection if it doesn't exist, otherwise uses the existing one.
        Sets up both dense and sparse vector configurations for hybrid search.

        Returns:
            None. Initializes self.vector_store with MyQdrantVectorStore instance.

        """
        if self.vector_store:
            return

        if not self.client.collection_exists(self.collection):
            logger.info(f"Creating new collection: {self.collection}")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": models.VectorParams(size=dense.model.embedding_size, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )
        else:
            logger.info(f"Using existing collection: {self.collection}")

        self.vector_store = MyQdrantVectorStore(
            client=self.client,
            collection_name=self.collection,
            embedding=dense,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

    def _add_doc_to_collection(self) -> bool:
        """Add document to the Qdrant collection.

        Processes the document using the appropriate splitter and adds the resulting
        document chunks to the vector store for indexing.

        Returns:
            True if document was successfully added, False otherwise.

        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return False
        if not self.doc:
            logger.error("Doc file not defined, cannot add to database")
            return False
        logger.info(f"Document not indexed, processing: {self.doc}")
        splitter = get_splitter(str(self.doc))
        documents = splitter.split(str(self.doc))
        self.vector_store.add_documents(documents)
        return True

    def _search_collection(self, query: str, k: int, rerank_results: bool) -> List[Dict]:
        """Search the collection for similar documents.

        Performs hybrid search using both dense and sparse vectors, with optional
        reranking for improved relevance. If reranking is enabled, searches for
        more results initially and then reranks to return the top k results.

        Args:
            query: Search query string.
            k: Number of results to return.
            rerank_results: Whether to apply reranking to improve relevance.

        Returns:
            List of dictionaries containing content, metadata, and scores for
            each search result.

        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []

        search_k = k
        if rerank_results:
            search_k = k * 2 if k >= 5 else 10
        logger.info(f"Performing search with {search_k=}")
        results = self.vector_store.similarity_search_with_score(query, search_k)
        logger.info(f"Search returned {len(results)} results")

        if rerank_results:
            logger.info(f"Reranking results to top {k}")
            results = rerank(query, results, k)
            logger.info(f"Search after rerank, returning {len(results)} results")

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
            }
            for doc, score in results
        ]

    ######### Pubic API
    def doc_exist(self, name: Optional[str] = None) -> bool:
        """Check if the document collection exists in the database.

        Returns:
            True if the collection exists, False otherwise.

        """
        if name is None:
            to_find = self.collection
        else:
            to_find = Path(name).name
        ret = self.client.collection_exists(to_find)
        logger.info(f"{to_find} is {'' if ret else 'not '}in database")
        return ret

    def add_doc(self) -> bool:
        """Add document to the database if it doesn't already exist.

        Checks if the document collection already exists in the database.
        If not, initializes the collection and processes the document for indexing.

        Returns:
            True if document was successfully added, False if it already exists.

        Example:
            >>> store = StoreDocument("example.pdf")
            >>> success = store.add_doc()
            >>> print(success)  # True if added, False if exists

        """
        if self.doc_exist():
            return False
        else:
            self._init_collection()
            return self._add_doc_to_collection()

    def search_doc(self, query: str, k: int = 4, rerank_results: bool = True) -> List[Dict]:
        """Search for similar documents in the collection.

        Performs a hybrid search using both dense and sparse vectors to find
        documents similar to the query. Optionally applies reranking for
        improved relevance.

        Args:
            query: Search query string.
            k: Number of results to return (default: 4).
            rerank_results: Whether to apply reranking for better relevance
                (default: True).

        Returns:
            List of dictionaries containing content, metadata, and scores for
            each search result. Returns empty list if collection doesn't exist.

        Example:
            >>> store = StoreDocument("example.pdf")
            >>> store.add_doc()
            >>> results = store.search("relevant query", k=3)
            >>> print(results[0]["content"])

        """
        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []
        if k <= 0:
            raise ValueError(f"Number of results k must be positive, got {k}")
        if not self.doc_exist():
            return []
        else:
            self._init_collection()
            return self._search_collection(query, k, rerank_results)
