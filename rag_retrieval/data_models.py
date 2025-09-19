"""
Shared data models and serialization utilities for RAG retrieval system.
Contains Document and DocumentChunk classes with serialization helpers.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DocumentChunk:
    """Represents a single chunk from a RAG document."""

    chunk_id: str
    document_id: str
    content: str
    source_file: str
    title: str
    breadcrumb: str


@dataclass
class Document:
    """Represents a complete RAG document with all chunks."""

    document_id: str
    source_file: str
    title: str
    breadcrumb: str
    chunks: List[DocumentChunk]
    topics: Dict[str, List[str]]
    metadata: Dict
    created_at: str


def document_to_dict(document: Document) -> Dict:
    """Convert Document object to dictionary for JSON serialization."""
    return {
        "document_id": document.document_id,
        "source_file": document.source_file,
        "title": document.title,
        "breadcrumb": document.breadcrumb,
        "chunks": [chunk_to_dict(chunk) for chunk in document.chunks],
        "topics": document.topics,
        "metadata": document.metadata,
        "created_at": document.created_at,
    }


def chunk_to_dict(chunk: DocumentChunk) -> Dict:
    """Convert DocumentChunk object to dictionary for JSON serialization."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "source_file": chunk.source_file,
        "title": chunk.title,
        "breadcrumb": chunk.breadcrumb,
    }


def dict_to_document(doc_data: Dict) -> Document:
    """Convert dictionary to Document object."""
    chunks = [dict_to_chunk(chunk_data) for chunk_data in doc_data["chunks"]]
    return Document(
        document_id=doc_data["document_id"],
        source_file=doc_data["source_file"],
        title=doc_data["title"],
        breadcrumb=doc_data["breadcrumb"],
        chunks=chunks,
        topics=doc_data["topics"],
        metadata=doc_data["metadata"],
        created_at=doc_data["created_at"],
    )


def dict_to_chunk(chunk_data: Dict) -> DocumentChunk:
    """Convert dictionary to DocumentChunk object."""
    return DocumentChunk(
        chunk_id=chunk_data["chunk_id"],
        document_id=chunk_data["document_id"],
        content=chunk_data["content"],
        source_file=chunk_data["source_file"],
        title=chunk_data["title"],
        breadcrumb=chunk_data["breadcrumb"],
    )
