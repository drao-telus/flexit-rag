"""
Hybrid topic-enhanced RAG retrieval engine.
Implements two-stage retrieval: topic pre-filtering + vector search.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Add the crawler directory to the path to import topic extractor
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "crawler"))

from crawler.md_pipeline.topic_extractor import TopicExtractor
from rag_retrieval.cache_data_loader import CacheDataLoader, Document, DocumentChunk
from rag_retrieval.retrieval_config import RetrievalConfig
from rag_retrieval.scoring_algorithms import DocumentScore, ScoringAlgorithms

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval process."""

    query: str
    query_topics: Dict[str, Any]
    candidate_documents: List[str]
    scored_documents: List[DocumentScore]
    retrieval_stats: Dict[str, Any]
    processing_time: float


class TopicRetriever:
    """
    Hybrid RAG retrieval engine combining topic filtering and vector search.

    Two-stage approach:
    1. Topic pre-filtering: Use enhanced topic extraction to identify candidate documents
    2. Vector search: Apply semantic similarity within the candidate set
    """

    def __init__(
        self,
        config: RetrievalConfig = None,
        cache_file_path: str = "rag_retrieval/indexes_cache.json",
    ):
        """
        Initialize the hybrid retrieval engine.

        Args:
            config: Retrieval configuration (uses balanced_mode if None)
            cache_file_path: Path to the indexes cache file
        """
        self.config = config or RetrievalConfig.balanced_mode()
        self.cache_loader = CacheDataLoader(cache_file_path)
        self.scoring_algorithms = ScoringAlgorithms(self.config)
        self.topic_extractor = TopicExtractor()

        # Initialize state
        self.is_loaded = False
        self.load_stats = {}

    def initialize(self) -> Dict[str, Any]:
        """
        Load indexes from cache.

        Returns:
            Loading statistics
        """
        logger.info("Initializing hybrid retrieval engine...")

        # Load from cache
        self.load_stats = self.cache_loader.initialize()
        self.load_stats.update(
            {
                "config_mode": self.config.mode,
            }
        )

        self.is_loaded = True

        logger.info(
            f"Initialization complete: {self.load_stats['documents_loaded']} documents, {self.load_stats['chunks_loaded']} chunks"
        )
        logger.info(f"Indexed {self.load_stats['unique_topics']} unique topics")

        return self.load_stats

    def retrieve(
        self,
        query: str,
        max_results: int = None,
        min_score: float = None,
        return_chunks: bool = True,
    ) -> RetrievalResult:
        """
        Perform hybrid retrieval for a query.

        Args:
            query: Search query
            max_results: Maximum number of results (uses config default if None)
            min_score: Minimum relevance score threshold
            return_chunks: Whether to return document chunks or just scores

        Returns:
            RetrievalResult with comprehensive retrieval information
        """
        import time

        start_time = time.time()

        if not self.is_loaded:
            raise RuntimeError(
                "Retrieval engine not initialized. Call initialize() first."
            )

        # Use config defaults if not specified
        max_results = max_results or self.config.max_results
        min_score = min_score or self.config.min_score_threshold

        logger.info(f"Processing query: '{query}'")

        # Stage 1: Enhanced topic extraction
        query_topics = self.topic_extractor.extract_query_topics(query)

        logger.info(f"Extracted {query_topics.get('topic_count', 0)} topics from query")
        logger.info(
            f"High confidence topics: {query_topics.get('high_confidence_topics', [])}"
        )

        # Stage 2: Topic-based candidate filtering
        candidate_docs = self._get_candidate_documents(query_topics, query)

        logger.info(f"Found {len(candidate_docs)} candidate documents")

        # Stage 3: Detailed scoring of candidates
        scored_documents = self._score_candidate_documents(
            candidate_docs, query_topics, query
        )

        # Stage 4: Ranking and filtering
        final_scores = self.scoring_algorithms.rank_documents(
            scored_documents, min_score
        )

        # Limit results
        final_scores = final_scores[:max_results]

        logger.info(f"Final results: {len(final_scores)} documents")

        # Calculate processing time
        processing_time = time.time() - start_time

        # Compile retrieval statistics
        retrieval_stats = {
            "query_topic_count": query_topics.get("topic_count", 0),
            "candidate_count": len(candidate_docs),
            "scored_count": len(scored_documents),
            "final_count": len(final_scores),
            "processing_time_ms": round(processing_time * 1000, 2),
            "avg_score": (
                sum(s.total_score for s in final_scores) / len(final_scores)
                if final_scores
                else 0
            ),
            "top_score": final_scores[0].total_score if final_scores else 0,
            "config_mode": self.config.mode,
        }

        return RetrievalResult(
            query=query,
            query_topics=query_topics,
            candidate_documents=candidate_docs,
            scored_documents=final_scores,
            retrieval_stats=retrieval_stats,
            processing_time=processing_time,
        )

    def _get_candidate_documents(
        self, query_topics: Dict[str, Any], query: str
    ) -> List[str]:
        """
        Stage 1: Get candidate documents using topic filtering.

        Args:
            query_topics: Enhanced query topics from topic extractor
            query: Original query text

        Returns:
            List of candidate document IDs
        """
        candidates = set()

        # Get all query topics
        all_query_topics = query_topics.get("query_topics", [])
        domain_topics = query_topics.get("domain_topics", [])
        entity_topics = query_topics.get("entity_topics", [])
        high_confidence_topics = query_topics.get("high_confidence_topics", [])

        # Priority 1: High-confidence domain topics (most restrictive)
        if domain_topics and self.config.use_domain_filtering:
            domain_candidates = self.cache_loader.get_documents_by_topics(
                domain_topics, min_matches=1
            )
            if domain_candidates:
                candidates.update(domain_candidates[: self.config.max_candidates])
                logger.info(
                    f"Domain filtering found {len(domain_candidates)} candidates"
                )

        # Priority 2: High-confidence topics (any type)
        if high_confidence_topics and len(candidates) < self.config.max_candidates:
            confidence_candidates = self.cache_loader.get_documents_by_topics(
                high_confidence_topics, min_matches=1
            )
            candidates.update(confidence_candidates[: self.config.max_candidates])
            logger.info(
                f"High-confidence filtering added {len(confidence_candidates)} candidates"
            )

        # Priority 3: Entity topics
        if entity_topics and len(candidates) < self.config.max_candidates:
            entity_candidates = self.cache_loader.get_documents_by_topics(
                entity_topics, min_matches=1
            )
            candidates.update(entity_candidates[: self.config.max_candidates])
            logger.info(f"Entity filtering added {len(entity_candidates)} candidates")

        # Priority 4: General topic matching
        if len(candidates) < self.config.max_candidates and all_query_topics:
            # Try multiple topic matches first
            multi_match_candidates = self.cache_loader.get_documents_by_topics(
                all_query_topics, min_matches=2
            )
            candidates.update(multi_match_candidates[: self.config.max_candidates])

            # Then single topic matches if needed
            if len(candidates) < self.config.max_candidates:
                single_match_candidates = self.cache_loader.get_documents_by_topics(
                    all_query_topics, min_matches=1
                )
                candidates.update(single_match_candidates[: self.config.max_candidates])

            logger.info(
                f"General topic filtering added candidates (total: {len(candidates)})"
            )

        # Priority 5: Breadcrumb and title matching as fallback
        if len(candidates) < self.config.max_candidates:
            # Try breadcrumb matching
            breadcrumb_candidates = self.cache_loader.get_documents_by_breadcrumb(query)
            candidates.update(breadcrumb_candidates[: self.config.max_candidates])

            # Try title matching
            title_candidates = self.cache_loader.get_documents_by_title(query)
            candidates.update(title_candidates[: self.config.max_candidates])

            logger.info(
                f"Breadcrumb/title filtering added candidates (total: {len(candidates)})"
            )

        # Priority 6: Content search as last resort
        if len(candidates) < self.config.max_candidates:
            content_results = self.cache_loader.search_content(
                query, max_results=self.config.max_candidates
            )
            content_candidates = [
                chunk_id.split("_")[0] + "_" + chunk_id.split("_")[1]
                for chunk_id, _ in content_results
            ]
            candidates.update(content_candidates)

            logger.info(f"Content search added candidates (total: {len(candidates)})")

        # Convert to list and limit
        candidate_list = list(candidates)[: self.config.max_candidates]

        return candidate_list

    def _score_candidate_documents(
        self, candidate_doc_ids: List[str], query_topics: Dict[str, Any], query: str
    ) -> List[DocumentScore]:
        """
        Stage 2: Score candidate documents using comprehensive algorithms.

        Args:
            candidate_doc_ids: List of candidate document IDs
            query_topics: Enhanced query topics
            query: Original query text

        Returns:
            List of scored documents
        """
        scored_documents = []

        for doc_id in candidate_doc_ids:
            document = self.cache_loader.get_document(doc_id)
            if document:
                score = self.scoring_algorithms.calculate_document_score(
                    document, query_topics, query
                )
                scored_documents.append(score)

        return scored_documents

    def _extract_result_chunks(
        self, scored_documents: List[DocumentScore]
    ) -> List[DocumentChunk]:
        """
        Extract the most relevant chunks from scored documents.

        Args:
            scored_documents: List of scored documents

        Returns:
            List of document chunks
        """
        result_chunks = []

        for doc_score in scored_documents:
            document = self.cache_loader.get_document(doc_score.document_id)
            if document:
                # For now, return all chunks from each document
                # In the future, this could be enhanced with chunk-level scoring
                for chunk in document.chunks:
                    result_chunks.append(chunk)

        return result_chunks

    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retrieval engine statistics."""
        stats = self.load_stats.copy()
        stats.update(
            {
                "is_initialized": self.is_loaded,
                "config": {
                    "mode": self.config.mode,
                    "max_results": self.config.max_results,
                    "max_candidates": self.config.max_candidates,
                    "domain_topic_weight": self.config.domain_topic_weight,
                    "use_domain_filtering": self.config.use_domain_filtering,
                    "normalize_scores": self.config.normalize_scores,
                },
            }
        )
        return stats

    def search_by_topics(
        self, topics: List[str], max_results: int = 10
    ) -> List[Tuple[str, int]]:
        """
        Direct topic-based search without query processing.

        Args:
            topics: List of topics to search for
            max_results: Maximum number of results

        Returns:
            List of (document_id, match_count) tuples
        """
        if not self.is_loaded:
            raise RuntimeError("Retrieval engine not initialized.")

        candidate_docs = self.cache_loader.get_documents_by_topics(
            topics, min_matches=1
        )

        # Count matches for each document
        doc_matches = []
        for doc_id in candidate_docs:
            document = self.cache_loader.get_document(doc_id)
            if document:
                match_count = 0
                doc_topics = set()
                for topic_list in document.topics.values():
                    if isinstance(topic_list, list):
                        doc_topics.update(topic_list)

                for topic in topics:
                    if topic.lower() in doc_topics:
                        match_count += 1

                doc_matches.append((doc_id, match_count))

        # Sort by match count
        doc_matches.sort(key=lambda x: x[1], reverse=True)

        return doc_matches[:max_results]

    def get_document_details(self, document_id: str) -> Optional[Document]:
        """Get full document details by ID."""
        if not self.is_loaded:
            raise RuntimeError("Retrieval engine not initialized.")

        return self.cache_loader.get_document(document_id)

    def get_all_topics(self) -> List[str]:
        """Get all unique topics in the document collection."""
        if not self.is_loaded:
            raise RuntimeError("Retrieval engine not initialized.")

        return self.cache_loader.get_all_topics()
