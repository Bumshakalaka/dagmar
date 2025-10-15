"""Custom QdrantVectorStore with enhanced similarity search functionality and support for keyword-based filtering."""

from typing import Any, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import models

from dagmar.parse_filter_string import parse_filter_string


class MyQdrantVectorStore(QdrantVectorStore):
    """Custom QdrantVectorStore with enhanced similarity search functionality and support for keyword-based filtering.

    This class extends QdrantVectorStore to provide improved similarity search
    capabilities with support for dense, sparse, and hybrid retrieval modes.
    """

    def search_by_fields(
        self,
        filter: str,
        k: int = 4,
    ):
        """Search documents using keyword-based filtering on page_content.

        Args:
            filter: Filter string to parse and apply to document search. Only page_content can be used.
                - Text search: page_content like 'search text'
                - Logical: and, or, not
                - Grouping: parentheses ()
                - e.g. page_content like 'Langchain' and (page_content like 'Qdrant' or page_content like 'fastembed')
            k: Maximum number of documents to return.

        Returns:
            List of tuples containing Document objects and their similarity scores.

        """
        query_filter = parse_filter_string(filter, allow_fields=[self.content_payload_key])
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,  # Don't return vectors for filter-only queries
        )
        return [
            (
                self._document_from_point(
                    result,
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                1,
            )
            for result in results[0]
        ]

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
