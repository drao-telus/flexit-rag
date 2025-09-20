"""
Document loader for RAG JSON files with topic-enhanced indexing.
Loads and indexes documents from crawler/result_data/rag_output/ with comprehensive topic metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from flexit_llm.utility.data_models import (
    Document,
    DocumentChunk,
    chunk_to_dict,
    document_to_dict,
)

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads and indexes RAG JSON documents from crawler output.

    NOTE: This class is primarily used for initial data processing and cache generation.
    For retrieval operations, use CacheDataLoader which provides better performance.
    """

    def __init__(self, rag_output_dir: str = "crawler/result_data/rag_output"):
        """
        Initialize document loader.

        Args:
            rag_output_dir: Directory containing RAG JSON files
        """
        self.rag_output_dir = Path(rag_output_dir)
        self.topic_index: Dict[str, Set[str]] = {}  # topic -> set of document_ids
        self.breadcrumb_index: Dict[str, Set[str]] = (
            {}
        )  # breadcrumb -> set of document_ids
        self.title_index: Dict[str, Set[str]] = {}  # title words -> set of document_ids

        # Statistics
        self.total_documents = 0
        self.total_chunks = 0
        self.unique_topics = set()

    def load_all_documents(self) -> Tuple[int, int]:
        """
        Load all RAG JSON files from the output directory and generate cache.

        This method is primarily used for initial cache generation.
        For retrieval operations, use CacheDataLoader instead.

        Returns:
            Tuple of (documents_loaded, chunks_loaded)
        """
        cache_file = Path("flexit_llm/utility/indexes_cache.json")

        # Check if cache already exists
        if cache_file.exists():
            logger.info(
                "Cache file already exists. Use CacheDataLoader for retrieval operations."
            )
            # Load basic stats from cache for return value
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                return cache_data["total_documents"], cache_data["total_chunks"]
            except Exception as e:
                logger.warning(f"Failed to read cache stats: {e}")

        # Perform full indexing and cache generation
        if not self.rag_output_dir.exists():
            raise FileNotFoundError(
                f"RAG output directory not found: {self.rag_output_dir}"
            )

        json_files = list(self.rag_output_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to load")

        # Temporary storage for cache generation
        documents = {}
        chunks = {}
        documents_loaded = 0
        chunks_loaded = 0

        for json_file in json_files:
            try:
                document = self._load_document(json_file)
                if document:
                    documents[document.document_id] = document
                    documents_loaded += 1

                    # Index chunks
                    for chunk in document.chunks:
                        chunks[chunk.chunk_id] = chunk
                        chunks_loaded += 1

                    # Build topic indexes
                    self._index_document(document)

            except Exception as e:
                logger.error(f"Error loading document {json_file}: {e}")
                continue

        self.total_documents = documents_loaded
        self.total_chunks = chunks_loaded
        self.unique_topics = set(self.topic_index.keys())

        logger.info(f"Loaded {documents_loaded} documents with {chunks_loaded} chunks")
        logger.info(f"Indexed {len(self.unique_topics)} unique topics")

        # Save to cache
        try:
            self._save_to_cache(documents, chunks)
            logger.info("Indexes saved to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        return documents_loaded, chunks_loaded

    def _load_document(self, json_file: Path) -> Optional[Document]:
        """Load a single RAG JSON document."""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Create document chunks
            chunks = []
            for chunk_data in data.get("chunks", []):
                chunk = DocumentChunk(
                    chunk_id=chunk_data["chunk_id"],
                    document_id=data["document_id"],
                    content=chunk_data["content"],
                    # chunk_type=chunk_data['chunk_type'],
                    # chunk_index=chunk_data['chunk_index'],
                    # total_chunks=chunk_data['total_chunks'],
                    # topics=chunk_data['metadata'].get('topics', {}),
                    # metadata=chunk_data['metadata'],
                    source_file=data["source_file"],
                    title=data["title"],
                    breadcrumb=data["breadcrumb"],
                )
                chunks.append(chunk)

            # Create document
            document = Document(
                document_id=data["document_id"],
                source_file=data["source_file"],
                title=data["title"],
                breadcrumb=data["breadcrumb"],
                chunks=chunks,
                topics=data["metadata"].get("topics", {}),
                metadata=data["metadata"],
                created_at=data["created_at"],
            )

            return document

        except Exception as e:
            logger.error(f"Error parsing JSON file {json_file}: {e}")
            return None

    def _index_document(self, document: Document):
        """Build topic and metadata indexes for a document."""
        doc_id = document.document_id

        # Index document-level topics
        topics = document.topics
        for topic_type, topic_list in topics.items():
            # Skip non-list values like topic_count
            if topic_type == "topic_count":
                continue

            if isinstance(topic_list, list):
                for topic in topic_list:
                    if isinstance(topic, str):
                        topic = topic.lower().strip()
                        if topic:
                            if topic not in self.topic_index:
                                self.topic_index[topic] = set()
                            self.topic_index[topic].add(doc_id)
            elif isinstance(topic_list, (str, int)):
                # Handle case where topic_list is a single value instead of list
                topic = str(topic_list).lower().strip()
                if topic:
                    if topic not in self.topic_index:
                        self.topic_index[topic] = set()
                    self.topic_index[topic].add(doc_id)

        # Index breadcrumb
        breadcrumb = document.breadcrumb.lower().strip()
        if breadcrumb:
            if breadcrumb not in self.breadcrumb_index:
                self.breadcrumb_index[breadcrumb] = set()
            self.breadcrumb_index[breadcrumb].add(doc_id)

            # Also index individual breadcrumb parts
            breadcrumb_parts = [part.strip().lower() for part in breadcrumb.split(">")]
            for part in breadcrumb_parts:
                if part:
                    if part not in self.breadcrumb_index:
                        self.breadcrumb_index[part] = set()
                    self.breadcrumb_index[part].add(doc_id)

        # Index title words
        title_words = document.title.lower().split()
        for word in title_words:
            word = word.strip()
            if word and len(word) > 2:  # Skip very short words
                if word not in self.title_index:
                    self.title_index[word] = set()
                self.title_index[word].add(doc_id)

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

    def _save_to_cache(
        self, documents: Dict[str, Document], chunks: Dict[str, DocumentChunk]
    ):
        """Save all indexes and data to cache file."""
        cache_file = Path("flexit_llm/utility/indexes_cache.json")

        # Convert sets to lists for JSON serialization
        cache_data = {
            "topic_index": {
                topic: list(doc_ids) for topic, doc_ids in self.topic_index.items()
            },
            "breadcrumb_index": {
                breadcrumb: list(doc_ids)
                for breadcrumb, doc_ids in self.breadcrumb_index.items()
            },
            "title_index": {
                title: list(doc_ids) for title, doc_ids in self.title_index.items()
            },
            "documents": {
                doc_id: document_to_dict(doc) for doc_id, doc in documents.items()
            },
            "chunks": {
                chunk_id: chunk_to_dict(chunk) for chunk_id, chunk in chunks.items()
            },
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "unique_topics": list(self.unique_topics),
        }

        # Ensure directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
