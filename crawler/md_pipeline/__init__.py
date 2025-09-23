"""
Direct MD Pipeline - Optimized HTML to Markdown processing for RAG applications.

This module provides a streamlined approach to convert HTML content directly to
Markdown format with intelligent chunking strategies based on content size and type.
"""

__version__ = "1.0.0"

from .html_to_md_converter import HTMLToMDConverter
from .md_chunking_strategy import (
    ChunkingStrategy,
    NoChunkingStrategy,
    SemanticSectionStrategy,
    TableSectionStrategy,
    ChunkingStrategyManager,
)
from .md_rag_processor import (
    MDRAGProcessor,
    RAGDocument,
    RAGChunk,
    get_file_processing_stats,
)

from .md_pipeline_main import MDPipelineOrchestrator, PipelineConfig, PipelineStats

__all__ = [
    # Core converters
    "HTMLToMDConverter",
    # Chunking strategies
    "ChunkingStrategy",
    "NoChunkingStrategy",
    "SemanticSectionStrategy",
    "TableSectionStrategy",
    "ChunkingStrategyManager",
    # RAG processing
    "MDRAGProcessor",
    "RAGDocument",
    "RAGChunk",
    "get_file_processing_stats",
    # Pipeline orchestration
    "MDPipelineOrchestrator",
    "PipelineConfig",
    "PipelineStats",
]
