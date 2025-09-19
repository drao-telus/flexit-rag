"""
Hybrid Retrieval Coordinator for RAG System
Combines topic-based and vector-based retrieval with intelligent fusion.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from rag_retrieval.embeddings import FuelixEmbeddingManager
from rag_retrieval.qdrant_vector_store import QdrantVectorStoreManager
from rag_retrieval.retrieval_config import FusionConfig, RetrievalConfig
from rag_retrieval.topic_retriever import RetrievalResult, TopicRetriever

logger = logging.getLogger(__name__)


@dataclass
class FusedChunk:
    """Represents a chunk with fusion scoring from both retrieval methods."""

    chunk_id: str
    content: str

    # Scores
    fusion_score: float
    topic_score: float
    vector_score: float

    # Flags
    found_by_topic: bool
    found_by_vector: bool
    consensus_match: bool

    # Source information
    document_id: str


@dataclass
class FusedRetrievalResult:
    """Complete result from hybrid retrieval with detailed analytics."""

    query: str
    fused_chunks: List[FusedChunk]

    # Method-specific results
    topic_result: Optional[RetrievalResult]
    vector_results: List[Dict[str, Any]]

    # Fusion analytics
    total_unique_chunks: int
    consensus_chunks: int
    topic_only_chunks: int
    vector_only_chunks: int

    # Performance metrics
    total_processing_time: float
    topic_processing_time: float
    vector_processing_time: float
    fusion_processing_time: float

    # Configuration used
    fusion_config: FusionConfig


class HybridRetrievalCoordinator:
    """
    Coordinates topic-based and vector-based retrieval with intelligent fusion.

    Combines results from TopicRetriever and QdrantVectorStore using chunk IDs
    as the common identifier for alignment and fusion scoring.
    """

    def __init__(
        self,
        retrieval_config: Optional[RetrievalConfig] = None,
        qdrant_manager: Optional[QdrantVectorStoreManager] = None,
        cache_file_path: str = "rag_retrieval/indexes_cache.json",
    ):
        """
        Initialize the hybrid retrieval coordinator.

        Args:
            retrieval_config: Configuration for topic retrieval and fusion
            qdrant_manager: Qdrant vector store manager
            cache_file_path: Path to the indexes cache file
        """
        self.retrieval_config = retrieval_config or RetrievalConfig.balanced_mode()
        self.fusion_config = (
            self.retrieval_config.fusion_config or FusionConfig.balanced_fusion()
        )

        load_dotenv()

        self.api_key = os.getenv("FUELIX_API_KEY")
        self.user_id = os.getenv("FUELIX_USER_ID")

        if not self.api_key:
            raise ValueError("FUELIX_API_KEY not found in environment variables")

        # Initialize components
        self.embedding_manager = FuelixEmbeddingManager(api_key=self.api_key)

        # Initialize retrieval components with cache file path
        self.topic_retriever = TopicRetriever(self.retrieval_config, cache_file_path)

        # Use the populated flexit collection if no qdrant_manager provided
        if qdrant_manager is None:
            from rag_retrieval.qdrant_vector_store import QdrantConfig

            qdrant_config = QdrantConfig(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name="flexit_rag_collection",  # Use the populated collection
            )
            self.qdrant_manager = QdrantVectorStoreManager(qdrant_config)
        else:
            self.qdrant_manager = qdrant_manager

        # Initialize state
        self.is_initialized = False

        logger.info(
            f"HybridRetrievalCoordinator initialized with {self.retrieval_config.mode} mode"
        )
        logger.info(f"Fusion enabled: {self.fusion_config.enable_fusion}")

    def initialize(self) -> Dict[str, Any]:
        """
        Initialize both retrieval systems.

        Returns:
            Initialization statistics
        """
        logger.info("Initializing hybrid retrieval coordinator...")

        start_time = time.time()

        # Initialize topic retriever
        topic_stats = self.topic_retriever.initialize()

        # Test Qdrant connection
        qdrant_test = self.qdrant_manager.test_connection()

        # Test embedding API
        embedding_test = self.embedding_manager.test_embedding_api()

        initialization_time = time.time() - start_time

        self.is_initialized = (
            topic_stats.get("initialization_complete", False)
            and qdrant_test.get("status") == "success"
            and embedding_test.get("status") == "success"
        )

        init_stats = {
            "hybrid_initialization_complete": self.is_initialized,
            "initialization_time": initialization_time,
            "topic_retriever_stats": topic_stats,
            "qdrant_status": qdrant_test.get("status"),
            "embedding_status": embedding_test.get("status"),
            "fusion_enabled": self.fusion_config.enable_fusion,
        }

        if self.is_initialized:
            logger.info(
                f"✅ Hybrid retrieval coordinator initialized successfully in {initialization_time:.2f}s"
            )
        else:
            logger.error("❌ Failed to initialize hybrid retrieval coordinator")

        return init_stats

    def retrieve(
        self,
        query: str,
        max_results: Optional[int] = None,
        enable_parallel: bool = True,
    ) -> FusedRetrievalResult:
        """
        Perform hybrid retrieval combining topic and vector methods.

        Args:
            query: Search query
            max_results: Maximum results to return (uses fusion config if None)
            enable_parallel: Whether to run retrievals in parallel

        Returns:
            FusedRetrievalResult with combined and ranked results
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Hybrid retrieval coordinator not initialized. Call initialize() first."
            )

        max_results = max_results or self.fusion_config.max_fusion_results

        logger.info(f"Starting hybrid retrieval for query: '{query}'")
        start_time = time.time()

        # Perform both retrievals
        if enable_parallel:
            topic_result, vector_results = self._parallel_retrieval(query)
        else:
            topic_result, vector_results = self._sequential_retrieval(query)

        # Fusion processing
        fusion_start = time.time()
        fused_chunks = self._fuse_results(topic_result, vector_results, query)

        # Limit and rank results
        final_chunks = self._rank_and_limit_results(fused_chunks, max_results)

        fusion_time = time.time() - fusion_start
        total_time = time.time() - start_time

        # Calculate analytics
        analytics = self._calculate_fusion_analytics(
            final_chunks, topic_result, vector_results
        )

        result = FusedRetrievalResult(
            query=query,
            fused_chunks=final_chunks,
            topic_result=topic_result,
            vector_results=vector_results,
            total_unique_chunks=analytics["total_unique_chunks"],
            consensus_chunks=analytics["consensus_chunks"],
            topic_only_chunks=analytics["topic_only_chunks"],
            vector_only_chunks=analytics["vector_only_chunks"],
            total_processing_time=total_time,
            topic_processing_time=topic_result.processing_time if topic_result else 0.0,
            vector_processing_time=analytics["vector_processing_time"],
            fusion_processing_time=fusion_time,
            fusion_config=self.fusion_config,
        )

        logger.info(
            f"Hybrid retrieval completed: {len(final_chunks)} results in {total_time:.2f}s"
        )
        logger.info(f"Consensus matches: {analytics['consensus_chunks']}")

        return result

    def _parallel_retrieval(
        self, query: str
    ) -> Tuple[Optional[RetrievalResult], List[Dict[str, Any]]]:
        """Perform topic and vector retrieval in parallel."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            topic_future = executor.submit(self._perform_topic_retrieval, query)
            vector_future = executor.submit(self._perform_vector_retrieval, query)

            # Get results
            topic_result = topic_future.result()
            vector_results = vector_future.result()

            return topic_result, vector_results

    def _sequential_retrieval(
        self, query: str
    ) -> Tuple[Optional[RetrievalResult], List[Dict[str, Any]]]:
        """Perform topic and vector retrieval sequentially."""
        topic_result = self._perform_topic_retrieval(query)
        vector_results = self._perform_vector_retrieval(query)
        return topic_result, vector_results

    def _perform_topic_retrieval(self, query: str) -> Optional[RetrievalResult]:
        """Perform topic-based retrieval."""
        try:
            return self.topic_retriever.retrieve(
                query=query,
                max_results=self.retrieval_config.max_results,
                return_chunks=True,
            )
        except Exception as e:
            logger.error(f"Topic retrieval failed: {e}")
            return None

    def _perform_vector_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Perform vector-based retrieval."""
        try:
            # Generate query embedding
            embedding_result = self.embedding_manager.generate_embedding(query)

            # Search Qdrant with appropriate threshold for cosine distance
            # With cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
            # Use a permissive threshold (0.0 allows all results, let fusion handle ranking)
            search_results = self.qdrant_manager.search_documents(
                query_vector=embedding_result.embedding,
                limit=self.retrieval_config.max_candidates,
                score_threshold=0.0,  # Permissive threshold - let fusion handle quality filtering
            )

            return search_results

        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []

    def _fuse_results(
        self,
        topic_result: Optional[RetrievalResult],
        vector_results: List[Dict[str, Any]],
        query: str,
    ) -> List[FusedChunk]:
        """
        Fuse results from both retrieval methods using chunk IDs.

        Args:
            topic_result: Results from topic retrieval
            vector_results: Results from vector retrieval
            query: Original query for context

        Returns:
            List of fused chunks with combined scoring
        """
        fused_chunks = {}

        # Process topic results - extract chunks from scored documents
        if topic_result and topic_result.scored_documents:
            for doc_score in topic_result.scored_documents:
                # Get document and extract chunks
                document = self.topic_retriever.get_document_details(
                    doc_score.document_id
                )
                if document and document.chunks:
                    for chunk in document.chunks:
                        chunk_id = chunk.chunk_id

                        fused_chunks[chunk_id] = FusedChunk(
                            chunk_id=chunk_id,
                            content=chunk.content,
                            fusion_score=0.0,  # Will be calculated later
                            topic_score=doc_score.total_score,
                            vector_score=0.0,
                            found_by_topic=True,
                            found_by_vector=False,
                            consensus_match=False,
                            document_id=chunk.document_id,
                        )

        # Process vector results
        for vector_result in vector_results:
            payload = vector_result.get("payload", {})
            chunk_id = payload.get("chunk_id")

            if not chunk_id:
                continue

            vector_score = vector_result.get("score", 0.0)

            if chunk_id in fused_chunks:
                # Update existing chunk
                fused_chunks[chunk_id].vector_score = vector_score
                fused_chunks[chunk_id].found_by_vector = True
                fused_chunks[chunk_id].consensus_match = True
            else:
                # Create new chunk from vector result
                fused_chunks[chunk_id] = FusedChunk(
                    chunk_id=chunk_id,
                    content=payload.get("content", ""),
                    fusion_score=0.0,  # Will be calculated later
                    topic_score=0.0,
                    vector_score=vector_score,
                    found_by_topic=False,
                    found_by_vector=True,
                    consensus_match=False,
                    document_id=payload.get("document_id", ""),
                )

        # Calculate fusion scores
        for chunk in fused_chunks.values():
            chunk.fusion_score = self._calculate_fusion_score(chunk)

        return list(fused_chunks.values())

    def _calculate_fusion_score(self, chunk: FusedChunk) -> float:
        """
        Calculate fusion score for a chunk based on topic and vector scores.

        Args:
            chunk: FusedChunk to score

        Returns:
            Fusion score combining both methods
        """
        # Normalize scores to 0-1 range
        normalized_topic = min(chunk.topic_score, 1.0)
        normalized_vector = min(chunk.vector_score, 1.0)

        # Apply weights
        weighted_score = (
            normalized_topic * self.fusion_config.topic_weight
            + normalized_vector * self.fusion_config.vector_weight
        )

        # Apply consensus boost
        if chunk.consensus_match:
            weighted_score *= self.fusion_config.consensus_boost

        return min(weighted_score, 1.0)  # Cap at 1.0

    def _rank_and_limit_results(
        self, fused_chunks: List[FusedChunk], max_results: int
    ) -> List[FusedChunk]:
        """Rank fused chunks by fusion score and limit results."""
        # Sort by fusion score (descending)
        ranked_chunks = sorted(fused_chunks, key=lambda x: x.fusion_score, reverse=True)

        # Limit results
        return ranked_chunks[:max_results]

    def _calculate_fusion_analytics(
        self,
        fused_chunks: List[FusedChunk],
        topic_result: Optional[RetrievalResult],
        vector_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate analytics for fusion results."""
        consensus_chunks = sum(1 for chunk in fused_chunks if chunk.consensus_match)
        topic_only_chunks = sum(
            1
            for chunk in fused_chunks
            if chunk.found_by_topic and not chunk.found_by_vector
        )
        vector_only_chunks = sum(
            1
            for chunk in fused_chunks
            if chunk.found_by_vector and not chunk.found_by_topic
        )

        return {
            "total_unique_chunks": len(fused_chunks),
            "consensus_chunks": consensus_chunks,
            "topic_only_chunks": topic_only_chunks,
            "vector_only_chunks": vector_only_chunks,
            "vector_processing_time": 0.0,  # Will be calculated from vector results timing
        }

    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fusion system statistics."""
        topic_stats = (
            self.topic_retriever.get_retrieval_statistics()
            if self.is_initialized
            else {}
        )

        return {
            "is_initialized": self.is_initialized,
            "fusion_enabled": self.fusion_config.enable_fusion,
            "fusion_config": {
                "topic_weight": self.fusion_config.topic_weight,
                "vector_weight": self.fusion_config.vector_weight,
                "consensus_boost": self.fusion_config.consensus_boost,
                "max_fusion_results": self.fusion_config.max_fusion_results,
            },
            "retrieval_config": {
                "mode": self.retrieval_config.mode,
                "max_results": self.retrieval_config.max_results,
                "max_candidates": self.retrieval_config.max_candidates,
            },
            "topic_retriever_stats": topic_stats,
        }


def demonstrate_hybrid_retrieval():
    """
    Demonstrate hybrid retrieval functionality.
    """
    print("Hybrid Retrieval Coordinator Demonstration")
    print("=" * 50)

    try:
        # Initialize coordinator
        print("\n1. Initializing Hybrid Retrieval Coordinator")
        print("-" * 40)

        coordinator = HybridRetrievalCoordinator()
        init_stats = coordinator.initialize()

        if init_stats["hybrid_initialization_complete"]:
            print("✅ Hybrid coordinator initialized successfully")
            print(
                f"✅ Topic retriever: {init_stats['topic_retriever_stats']['documents_loaded']} documents"
            )
            print(f"✅ Qdrant status: {init_stats['qdrant_status']}")
            print(f"✅ Embedding status: {init_stats['embedding_status']}")
        else:
            print("❌ Initialization failed")
            return

        # Test retrieval
        print("\n2. Testing Hybrid Retrieval")
        print("-" * 40)

        test_query = "How do I set up annual enrollment?"
        print(f"Query: '{test_query}'")

        result = coordinator.retrieve(test_query, max_results=5)

        print(f"✅ Total processing time: {result.total_processing_time:.2f}s")
        print(f"✅ Results found: {len(result.fused_chunks)}")
        print(f"✅ Consensus matches: {result.consensus_chunks}")
        print(f"✅ Topic-only matches: {result.topic_only_chunks}")
        print(f"✅ Vector-only matches: {result.vector_only_chunks}")

        # Show top results
        print("\n3. Top Results")
        print("-" * 40)

        for i, chunk in enumerate(result.fused_chunks[:3], 1):
            print(f"\nResult {i}:")
            print(f"  Chunk ID: {chunk.chunk_id}")
            print(f"  Fusion Score: {chunk.fusion_score:.3f}")
            print(f"  Topic Score: {chunk.topic_score:.3f}")
            print(f"  Vector Score: {chunk.vector_score:.3f}")
            print(f"  Consensus: {chunk.consensus_match}")
            print(f"  Content: {chunk.content[:100]}...")

        print("\n🎉 Hybrid retrieval demonstration completed successfully!")

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")


if __name__ == "__main__":
    demonstrate_hybrid_retrieval()
