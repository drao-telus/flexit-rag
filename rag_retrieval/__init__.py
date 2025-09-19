"""
RAG Retrieval System
Hybrid topic-filtered vector search with Qdrant vector store for enhanced document retrieval
"""

from rag_retrieval.topic_retriever import TopicRetriever
from rag_retrieval.document_loader import DocumentLoader
from rag_retrieval.scoring_algorithms import ScoringAlgorithms
from rag_retrieval.retrieval_config import RetrievalConfig
from rag_retrieval.qdrant_vector_store import (
    QdrantVectorStoreManager,
    QdrantConfig,
    DocumentPoint,
)
from rag_retrieval.embedding_indexer import QdrantRAGPipeline
from rag_retrieval.embeddings import FuelixEmbeddingManager, EmbeddingConfig

__version__ = "2.0.0"
__all__ = [
    "TopicRetriever",
    "DocumentLoader",
    "ScoringAlgorithms",
    "RetrievalConfig",
    "QdrantVectorStoreManager",
    "QdrantConfig",
    "DocumentPoint",
    "QdrantRAGPipeline",
    "FuelixEmbeddingManager",
    "EmbeddingConfig",
]
