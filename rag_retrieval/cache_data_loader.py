"""
Cache-based data loader for RAG retrieval system.
Reads documents and chunks directly from indexes_cache.json without in-memory storage.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .data_models import Document, DocumentChunk

logger = logging.getLogger(__name__)


class CacheDataLoader:
    """
    Cache-based data loader that reads directly from indexes_cache.json.
    Provides document and chunk access without storing everything in memory.
    """

    def __init__(self, cache_file_path: str = "rag_retrieval/indexes_cache.json"):
        """
        Initialize cache data loader.

        Args:
            cache_file_path: Path to the indexes cache file
        """
        self.cache_file_path = Path(cache_file_path)
        self.topic_index: Dict[str, Set[str]] = {}
        self.breadcrumb_index: Dict[str, Set[str]] = {}
        self.title_index: Dict[str, Set[str]] = {}

        # Statistics
        self.total_documents = 0
        self.total_chunks = 0
        self.unique_topics = set()

        # Cache file data (loaded once)
        self._cache_data = None
        self.is_loaded = False

    def initialize(self) -> Dict[str, any]:
        """
        Load indexes from cache file.

        Returns:
            Loading statistics
        """
        if not self.cache_file_path.exists():
            raise FileNotFoundError(f"Cache file not found: {self.cache_file_path}")

        logger.info("Loading indexes from cache...")

        try:
            with open(self.cache_file_path, "r", encoding="utf-8") as f:
                self._cache_data = json.load(f)

            # Load indexes (convert lists back to sets)
            self.topic_index = {
                topic: set(doc_ids)
                for topic, doc_ids in self._cache_data["topic_index"].items()
            }
            self.breadcrumb_index = {
                breadcrumb: set(doc_ids)
                for breadcrumb, doc_ids in self._cache_data["breadcrumb_index"].items()
            }
            self.title_index = {
                title: set(doc_ids)
                for title, doc_ids in self._cache_data["title_index"].items()
            }

            # Load statistics
            self.total_documents = self._cache_data["total_documents"]
            self.total_chunks = self._cache_data["total_chunks"]
            self.unique_topics = set(self._cache_data["unique_topics"])

            self.is_loaded = True

            logger.info(
                f"Loaded from cache: {self.total_documents} documents with {self.total_chunks} chunks"
            )

            return {
                "documents_loaded": self.total_documents,
                "chunks_loaded": self.total_chunks,
                "unique_topics": len(self.unique_topics),
                "topic_index_size": len(self.topic_index),
                "breadcrumb_index_size": len(self.breadcrumb_index),
                "title_index_size": len(self.title_index),
                "initialization_complete": True,
            }

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            raise

    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get a document by ID, reading from cache on-demand.

        Args:
            document_id: Document ID to retrieve

        Returns:
            Document object or None if not found
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Cache data loader not initialized. Call initialize() first."
            )

        if document_id not in self._cache_data["documents"]:
            return None

        from .data_models import dict_to_document

        doc_data = self._cache_data["documents"][document_id]
        return dict_to_document(doc_data)

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get a chunk by ID, reading from cache on-demand.

        Args:
            chunk_id: Chunk ID to retrieve

        Returns:
            DocumentChunk object or None if not found
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Cache data loader not initialized. Call initialize() first."
            )

        if chunk_id not in self._cache_data["chunks"]:
            return None

        from .data_models import dict_to_chunk

        chunk_data = self._cache_data["chunks"][chunk_id]
        return dict_to_chunk(chunk_data)

    def get_documents_by_topics(
        self, topics: List[str], min_matches: int = 1
    ) -> List[str]:
        """
        Get document IDs that match the given topics.

        Args:
            topics: List of topics to search for
            min_matches: Minimum number of topic matches required

        Returns:
            List of document IDs sorted by relevance
        """
        if not topics:
            return []

        # Count topic matches per document
        doc_matches = {}
        for topic in topics:
            topic = topic.lower().strip()
            if topic in self.topic_index:
                for doc_id in self.topic_index[topic]:
                    doc_matches[doc_id] = doc_matches.get(doc_id, 0) + 1

        # Filter by minimum matches and sort by match count
        matching_docs = [
            doc_id for doc_id, count in doc_matches.items() if count >= min_matches
        ]

        # Sort by match count (descending)
        matching_docs.sort(key=lambda doc_id: doc_matches[doc_id], reverse=True)

        return matching_docs

    def get_documents_by_breadcrumb(self, breadcrumb_query: str) -> List[str]:
        """Get document IDs that match breadcrumb patterns."""
        breadcrumb_query = breadcrumb_query.lower().strip()
        matching_docs = set()

        # Exact breadcrumb match
        if breadcrumb_query in self.breadcrumb_index:
            matching_docs.update(self.breadcrumb_index[breadcrumb_query])

        # Partial breadcrumb matches
        for breadcrumb, doc_ids in self.breadcrumb_index.items():
            if breadcrumb_query in breadcrumb or breadcrumb in breadcrumb_query:
                matching_docs.update(doc_ids)

        return list(matching_docs)

    def get_documents_by_title(self, title_query: str) -> List[str]:
        """Get document IDs that match title words."""
        title_words = title_query.lower().split()
        matching_docs = set()

        for word in title_words:
            word = word.strip()
            if word in self.title_index:
                matching_docs.update(self.title_index[word])

        return list(matching_docs)

    def get_all_topics(self) -> List[str]:
        """Get all unique topics in the index."""
        return sorted(list(self.unique_topics))

    def search_content(
        self, query: str, max_results: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Simple content search across all chunks.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of (chunk_id, relevance_score) tuples
        """
        query_lower = query.lower()
        results = []

        # Search through all chunks in cache
        for chunk_id, chunk_data in self._cache_data["chunks"].items():
            content_lower = chunk_data["content"].lower()

            # Simple relevance scoring based on query term frequency
            score = 0.0
            query_words = query_lower.split()

            for word in query_words:
                if word in content_lower:
                    # Count occurrences and normalize by content length
                    count = content_lower.count(word)
                    score += count / len(content_lower.split())

            if score > 0:
                results.append((chunk_id, score))

        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:max_results]

    def get_statistics(self) -> Dict:
        """Get loading and indexing statistics."""
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "unique_topics": len(self.unique_topics),
            "topic_index_size": len(self.topic_index),
            "breadcrumb_index_size": len(self.breadcrumb_index),
            "title_index_size": len(self.title_index),
        }
