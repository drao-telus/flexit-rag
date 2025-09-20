"""
RAG Retrieval System
Hybrid topic-filtered vector search with Qdrant vector store for enhanced document retrieval
"""

from data_processor.document_loader import DocumentLoader
from data_processor.qdrant_pipeline import QdrantPipeline

__version__ = "2.0.0"
__all__ = [
    "DocumentLoader",
    "QdrantPipeline",
]
