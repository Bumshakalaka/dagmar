"""Vector store implementation for document storage and retrieval.

This module provides functionality for storing documents in a Qdrant vector database
and performing similarity searches with hybrid retrieval capabilities including
dense and sparse embeddings with reranking.
"""

import logging
import os
import pprint
import re
from pathlib import Path
from typing import List

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


class QdrantStore:
    """Document storage and retrieval system using Qdrant vector database.

    This class provides a high-level interface for storing documents in a Qdrant
    vector database and performing similarity searches with hybrid retrieval
    capabilities including dense and sparse embeddings with reranking.
    """

    def __init__(self):
        """Initialize QdrantStore with embedding models and reranker.

        Sets up the Qdrant client, dense and sparse embedding models, and
        cross-encoder reranker for document retrieval and ranking.

        Args:
            location: from QDRANT_URL environment variable.
                - Path to Qdrant database or ":memory:" for in-memory storage
                - URL to Qdrant server
                - URL to Qdrant server with port

        Raises:
            ValueError: If the QDRANT_URL format is invalid.
            FileNotFoundError: If the QDRANT_URL is a path to a directory that does not exist.
            NotADirectoryError: If the QDRANT_URL is a path to a file that is not a directory.
            Exception: If there is an error initializing the Qdrant client.

        """
        logger.info(f"Initializing QdrantStore with location: {QDRANT_URL}")
        if QDRANT_URL == ":memory:":
            self.client = QdrantClient(":memory:")
        elif QDRANT_URL.startswith("http"):
            # Acceptable formats: http://host[:port]
            match = re.match(r"(https?)://([^:/]+)(?::(\d+))?", QDRANT_URL)
            if not match:
                raise ValueError(f"Invalid QDRANT_URL format: {QDRANT_URL}")

            proto, host, port = match.groups()
            port = int(port) if port else 6333
            self.client = QdrantClient(url=f"{proto}://{host}", port=port)
        else:
            if not Path(QDRANT_URL).parent.exists():
                raise FileNotFoundError(f"Directory {Path(QDRANT_URL).parent} does not exist")
            if not Path(QDRANT_URL).is_dir():
                raise NotADirectoryError(f"Path {QDRANT_URL} is not a directory")
            self.client = QdrantClient(path=QDRANT_URL)
        logger.debug("Loading dense embedding model: sentence-transformers/all-MiniLM-L6-v2")
        self.dense = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=128)
        logger.debug("Loading sparse embedding model: prithivida/Splade_PP_en_v1")
        self.sparse = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1", batch_size=128)
        logger.debug("Loading reranker model: Xenova/ms-marco-MiniLM-L-6-v2")
        self.reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
        logger.info("QdrantStore initialization completed")

    def _init_colection(self, collection_name: str) -> tuple[MyQdrantVectorStore, bool]:
        """Initialize or retrieve existing Qdrant collection.

        Creates a new collection if it doesn't exist, otherwise returns the existing one.

        Args:
            collection_name: Name of the collection to initialize or retrieve.

        Returns:
            Tuple containing the QdrantVectorStore instance and a boolean indicating
            whether the collection was newly created (True) or already existed (False).

        """
        exists = True
        if not self.client.collection_exists(collection_name):
            logger.info(f"Creating new collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=self.dense.model.embedding_size, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )
            exists = False
        else:
            logger.debug(f"Using existing collection: {collection_name}")
        vector_store = MyQdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.dense,
            sparse_embedding=self.sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        return vector_store, exists

    def _create_index(self, collection_name: str):
        """Create text index on document content for keyword search.

        Payload indexes have no effect in the local Qdrant. Please use server Qdrant if you need payload indexes.

        Args:
            collection_name: Name of the collection to create index for.

        """
        logger.info(f"Creating text index for collection: {collection_name}")
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="page_content",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.PREFIX,
                lowercase=True,
                phrase_matching=True,
                stemmer=models.SnowballParams(type=models.Snowball.SNOWBALL, language=models.SnowballLanguage.ENGLISH),
            ),
            wait=True,  # Wait for index creation to complete
        )
        logger.debug("Text index creation completed")

    def _add_documents(self, v_store: MyQdrantVectorStore, documents: List[Document]):
        """Add documents to the vector store.

        Args:
            v_store: QdrantVectorStore instance to add documents to.
            documents: List of Document objects to be added to the store.

        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        v_store.add_documents(documents)
        logger.debug("Documents added successfully")

    def _search_documents(self, v_store: MyQdrantVectorStore, query: str, k: int = 4):
        """Search for documents similar to the query.

        Args:
            v_store: QdrantVectorStore instance to search in.
            query: Text query to search for similar documents.
            k: Number of top results to return.

        Returns:
            List of tuples containing Document objects and their similarity scores.

        """
        return v_store.similarity_search_with_score(query, k)

    def _rerank(self, query: str, results: List[tuple[Document, float]], k: int):
        """Rerank search results using cross-encoder model.

        Args:
            query: Original text query used for searching.
            results: List of document-score tuples from initial search.
            k: Number of top results to return after reranking.

        Returns:
            List of reranked document-score tuples, sorted by reranking scores.

        """
        result_docs = [r.page_content for r, s in results]
        scores = list(self.reranker.rerank(query, result_docs))
        reranked_results = []
        for doc, score in zip(results, scores):
            reranked_results.append((doc[0], score))

        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results[0:k]

    def search_semantic(self, doc_path: Path, query: str, k: int = 4):
        """Search for relevant content in a document using vector similarity.

        Processes the document if not already indexed, then performs hybrid search
        with dense and sparse embeddings followed by reranking.

        Args:
            doc_path: Path to the document file to search in.
            query: Text query to search for relevant content.
            k: Number of top results to return.

        Returns:
            List of dictionaries containing content, metadata, and relevance scores
            for the top-k most relevant document sections.

        """
        logger.info(f"Starting semantic search for query: '{query}' in file: {doc_path}")
        try:
            v_store, exists = self._init_colection(doc_path.name)
            if not exists:
                logger.info(f"Document not indexed, processing: {doc_path}")
                splitter = get_splitter(str(doc_path))
                documents = splitter.split(str(doc_path))
                self._add_documents(v_store, documents)
                self._create_index(doc_path.name)
            else:
                logger.debug(f"Using existing index for: {doc_path}")

            search_k = k * 2 if k >= 5 else 10
            logger.debug(f"Performing initial search with k={search_k}")
            results = self._search_documents(v_store, query, search_k)
            logger.debug(f"Initial search returned {len(results)} results")

            logger.debug(f"Reranking results to top {k}")
            reranked_results = self._rerank(query, results, k)
            logger.info(f"Semantic search completed, returning {len(reranked_results)} results")

            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
                for doc, score in reranked_results
            ]
        except Exception as e:
            logger.error(f"Semantic search failed for query '{query}' in file {doc_path}: {e}")
            raise

    def search_by_fields(self, doc_path: Path, query: str, k: int = 4):
        """Search documents using keyword-based filtering.

        Args:
            doc_path: Path to the document file to search in.
            query: Filter query string for keyword-based search on page_content. Only page_content can be used.
                - Text search: page_content like 'search text' (for TEXT indexed fields)
                - Logical: and, or, not
                - Grouping: parentheses ()
                - e.g. page_content like 'Langchain' and (page_content like 'Qdrant' or page_content like 'fastembed')
            k: Number of top results to return.

        Returns:
            List of dictionaries containing content, metadata, and relevance scores
            for matching document sections.

        """
        logger.info(f"Starting field-based search for query: '{query}' in file: {doc_path}")
        try:
            v_store, exists = self._init_colection(doc_path.name)
            if not exists:
                logger.info(f"Document not indexed, processing: {doc_path}")
                splitter = get_splitter(str(doc_path))
                documents = splitter.split(str(doc_path))
                self._add_documents(v_store, documents)
                self._create_index(doc_path.name)
            else:
                logger.debug(f"Using existing index for: {doc_path}")

            logger.debug(f"Performing field-based search with k={k}")
            results = v_store.search_by_fields(query, k)
            logger.info(f"Field-based search completed, returning {len(results)} results")

            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Field-based search failed for query '{query}' in file {doc_path}: {e}")
            raise


if __name__ == "__main__":
    store = QdrantStore()
    results = store.search_semantic(
        Path("/home/totyz/Documents/Sidewalk/Amazon_Sidewalk_Test_Specification-1.0-rev-A.4.pdf"),
        "BLE/EP/CONN/DUP/BV/01",
        2,
    )
    pprint.pprint(results)
