"""Vector store implementation for document storage and retrieval.

This module provides functionality for storing documents in a Qdrant vector database
and performing similarity searches with hybrid retrieval capabilities including
dense and sparse embeddings with reranking.
"""

import pprint
from pathlib import Path
from typing import Any, List, Literal, Optional

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from dagmar.splitters import get_splitter


class MyQdrantVectorStore(QdrantVectorStore):
    """Custom QdrantVectorStore with enhanced similarity search functionality.

    This class extends QdrantVectorStore to provide improved similarity search
    capabilities with support for dense, sparse, and hybrid retrieval modes.
    """

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[models.Filter] = None,  # noqa: A002
        search_params: Optional[models.SearchParams] = None,
        offset: int = 0,
        score_threshold: Optional[float] = None,
        consistency: Optional[models.ReadConsistency] = None,
        hybrid_fusion: Optional[models.FusionQuery] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return documents most similar to query with similarity scores.

        Performs similarity search across dense, sparse, or hybrid retrieval modes.

        Args:
            query: Text query to search for similar documents.
            k: Number of top results to return.
            filter: Optional metadata filter for search results.
            search_params: Optional search parameters for fine-tuning.
            offset: Number of results to skip from the beginning.
            score_threshold: Minimum similarity score threshold.
            consistency: Read consistency level for the search.
            hybrid_fusion: Fusion strategy for hybrid retrieval mode.
            **kwargs: Additional search parameters.

        Returns:
            List of tuples containing Document objects and their similarity scores.

        Raises:
            ValueError: If an invalid retrieval mode is specified.

        """
        query_options = {
            "collection_name": self.collection_name,
            "query_filter": filter,
            "search_params": search_params,
            "limit": k,
            "offset": offset,
            "with_payload": True,
            "with_vectors": False,
            "score_threshold": score_threshold,
            "consistency": consistency,
            **kwargs,
        }
        if self.retrieval_mode == RetrievalMode.DENSE:
            embeddings = self._require_embeddings("DENSE mode")
            query_dense_embedding = embeddings.embed_query(query)
            results = self.client.query_points(
                query=query_dense_embedding,
                using=self.vector_name,
                **query_options,
            ).points

        elif self.retrieval_mode == RetrievalMode.SPARSE:
            query_sparse_embedding = self.sparse_embeddings.embed_query(query)
            results = self.client.query_points(
                query=models.SparseVector(
                    indices=query_sparse_embedding.indices,
                    values=query_sparse_embedding.values,
                ),
                using=self.sparse_vector_name,
                **query_options,
            ).points

        elif self.retrieval_mode == RetrievalMode.HYBRID:
            embeddings = self._require_embeddings("HYBRID mode")
            query_dense_embedding = embeddings.embed_query(query)
            query_sparse_embedding = self.sparse_embeddings.embed_query(query)
            results = self.client.query_points(
                prefetch=[
                    models.Prefetch(
                        using=self.vector_name,
                        query=query_dense_embedding,
                        filter=filter,
                        limit=k * 2,
                        params=search_params,
                        score_threshold=0.8,
                    ),
                    models.Prefetch(
                        using=self.sparse_vector_name,
                        query=models.SparseVector(
                            indices=query_sparse_embedding.indices,
                            values=query_sparse_embedding.values,
                        ),
                        filter=filter,
                        limit=k * 2,
                        params=search_params,
                        score_threshold=15.0,
                    ),
                ],
                query=hybrid_fusion or models.FusionQuery(fusion=models.Fusion.RRF),
                **query_options,
            ).points

        else:
            msg = f"Invalid retrieval mode. {self.retrieval_mode}."
            raise ValueError(msg)
        return [
            (
                self._document_from_point(
                    result,
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                result.score,
            )
            for result in results
        ]


class QdrantStore:
    """Document storage and retrieval system using Qdrant vector database.

    This class provides a high-level interface for storing documents in a Qdrant
    vector database and performing similarity searches with hybrid retrieval
    capabilities including dense and sparse embeddings with reranking.
    """

    def __init__(self, location: str | Literal[":memory:"]):
        """Initialize QdrantStore with embedding models and reranker.

        Args:
            location: Path to Qdrant database or ":memory:" for in-memory storage.

        Sets up the Qdrant client, dense and sparse embedding models, and
        cross-encoder reranker for document retrieval and ranking.

        """
        self.client = QdrantClient(":memory:") if location == ":memory:" else QdrantClient(path=location)
        self.dense = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=128)
        self.sparse = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1", batch_size=128)
        self.reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

    def _init_colection(self, collection_name: str) -> tuple[QdrantVectorStore, bool]:
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

    def _add_documents(self, v_store: QdrantVectorStore, documents: List[Document]):
        """Add documents to the vector store.

        Args:
            v_store: QdrantVectorStore instance to add documents to.
            documents: List of Document objects to be added to the store.

        """
        v_store.add_documents(documents)

    def _search_documents(self, v_store: QdrantVectorStore, query: str, k: int = 4):
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

    def search(self, doc_path: Path, query: str, k: int = 4):
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
        v_store, exists = self._init_colection(doc_path.name)
        if not exists:
            splitter = get_splitter(str(doc_path))
            documents = splitter.split(str(doc_path))
            self._add_documents(v_store, documents)
        results = self._search_documents(v_store, query, k * 2 if k >= 5 else 10)
        reranked_results = self._rerank(query, results, k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
            }
            for doc, score in reranked_results
        ]


if __name__ == "__main__":
    store = QdrantStore("./qdrant_db")
    results = store.search(
        Path("/home/totyz/Downloads/Amazon_Sidewalk_Test_Specification-1.0-rev-A.4.md"),
        "BLE/EP/CONN/DUP/BV/04",
        1,
    )
    pprint.pprint(results)
