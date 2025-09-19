"""
Qdrant Vector Store Manager for RAG System
Simplified implementation for Qdrant Cloud integration with text-embedding-3-large embeddings.
"""

import os
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    """Configuration for Qdrant connection and collection."""

    url: str
    api_key: str
    collection_name: str = "flexit_rag_collection"
    vector_size: int = 3072  # text-embedding-3-large dimensions
    distance_metric: Distance = Distance.COSINE


@dataclass
class DocumentPoint:
    """Represents a document chunk point for Qdrant storage."""

    id: str
    vector: List[float]
    payload: Dict[str, Any]


class QdrantVectorStoreManager:
    """
    Simplified Qdrant vector store manager for RAG system.
    Handles collection management, document storage, and search operations.
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        """
        Initialize Qdrant vector store manager.

        Args:
            config: Optional Qdrant configuration. If None, loads from environment.
        """
        # Load environment variables
        load_dotenv()

        if config is None:
            config = QdrantConfig(
                url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
            )

        if not config.url or not config.api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be provided")

        self.config = config
        self.client = QdrantClient(url=config.url, api_key=config.api_key)

        logger.info(f"Qdrant client initialized for: {config.url}")
        logger.info(f"Collection: {config.collection_name}")

    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create or recreate the Qdrant collection.

        Args:
            recreate: If True, delete existing collection and create new one

        Returns:
            True if collection was created/recreated, False if already exists
        """
        collection_name = self.config.collection_name

        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == collection_name for col in collections.collections
            )

            if collection_exists and not recreate:
                logger.info(f"Collection '{collection_name}' already exists")
                return False

            if recreate and collection_exists:
                logger.info(f"Deleting existing collection '{collection_name}'")
                self.client.delete_collection(collection_name)

            # Create collection with proper vector configuration
            logger.info(
                f"Creating collection '{collection_name}' with {self.config.vector_size} dimensions"
            )

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size, distance=self.config.distance_metric
                ),
            )

            logger.info(f"✅ Collection '{collection_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create collection: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Collection information dictionary
        """
        try:
            collection_info = self.client.get_collection(self.config.collection_name)

            return {
                "name": collection_info.config.params.vectors.size,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }

        except Exception as e:
            logger.error(f"❌ Failed to get collection info: {e}")
            raise

    def add_documents(
        self, documents: List[DocumentPoint], batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Add documents to the collection in batches.

        Args:
            documents: List of DocumentPoint objects to add
            batch_size: Number of documents to process per batch

        Returns:
            Dictionary with insertion results
        """
        collection_name = self.config.collection_name
        total_docs = len(documents)

        logger.info(f"Adding {total_docs} documents to collection '{collection_name}'")

        start_time = time.time()
        successful_count = 0
        failed_count = 0

        try:
            # Process documents in batches
            for i in range(0, total_docs, batch_size):
                batch = documents[i : i + batch_size]

                # Convert to PointStruct objects
                points = [
                    PointStruct(id=doc.id, vector=doc.vector, payload=doc.payload)
                    for doc in batch
                ]

                try:
                    # Upsert batch
                    self.client.upsert(
                        collection_name=collection_name, wait=True, points=points
                    )

                    successful_count += len(batch)
                    logger.info(
                        f"Batch {i // batch_size + 1}: Added {len(batch)} documents"
                    )

                except Exception as e:
                    logger.error(f"Failed to add batch {i // batch_size + 1}: {e}")
                    failed_count += len(batch)

            processing_time = time.time() - start_time

            results = {
                "total_documents": total_docs,
                "successful_count": successful_count,
                "failed_count": failed_count,
                "processing_time_seconds": processing_time,
                "average_time_per_doc_ms": (processing_time * 1000) / total_docs
                if total_docs > 0
                else 0,
            }

            logger.info(f"✅ Document insertion completed:")
            logger.info(f"   Successful: {successful_count}")
            logger.info(f"   Failed: {failed_count}")
            logger.info(f"   Total time: {processing_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            raise

    def search_documents(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_conditions: Optional metadata filters

        Returns:
            List of search results with scores and payloads
        """
        collection_name = self.config.collection_name

        try:
            # Build filter if provided
            search_filter = None
            if filter_conditions:
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )
                search_filter = Filter(must=conditions)

            # Perform search
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
            )

            # Format results
            results = []
            for hit in search_results:
                results.append(
                    {"id": hit.id, "score": hit.score, "payload": hit.payload}
                )

            logger.info(f"Search completed: {len(results)} results found")
            return results

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            raise

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Alias for search_documents to maintain compatibility.
        Returns raw search results from Qdrant client.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_conditions: Optional metadata filters

        Returns:
            List of raw Qdrant search results
        """
        collection_name = self.config.collection_name

        try:
            # Build filter if provided
            search_filter = None
            if filter_conditions:
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )
                search_filter = Filter(must=conditions)

            # Perform search and return raw results
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
            )

            logger.info(f"Search completed: {len(search_results)} results found")
            return search_results

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            raise

    def delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Delete documents by their IDs.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Deletion results
        """
        collection_name = self.config.collection_name

        try:
            self.client.delete(
                collection_name=collection_name, points_selector=document_ids, wait=True
            )

            logger.info(f"✅ Deleted {len(document_ids)} documents")

            return {"deleted_count": len(document_ids), "document_ids": document_ids}

        except Exception as e:
            logger.error(f"❌ Failed to delete documents: {e}")
            raise

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if successful
        """
        try:
            # Get collection info first
            info = self.get_collection_info()
            points_count = info.get("points_count", 0)

            if points_count == 0:
                logger.info("Collection is already empty")
                return True

            # Delete collection and recreate
            self.client.delete_collection(self.config.collection_name)
            self.create_collection()

            logger.info(f"✅ Cleared collection: {points_count} documents removed")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
            raise

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the Qdrant connection and basic operations.

        Returns:
            Test results
        """
        try:
            # Test 1: Get collections
            collections = self.client.get_collections()

            # Test 2: Check if our collection exists
            collection_exists = any(
                col.name == self.config.collection_name
                for col in collections.collections
            )

            # Test 3: Get collection info if exists
            collection_info = None
            if collection_exists:
                collection_info = self.get_collection_info()

            return {
                "status": "success",
                "connection": "✅ Connected",
                "collections_count": len(collections.collections),
                "target_collection_exists": collection_exists,
                "collection_info": collection_info,
            }

        except Exception as e:
            return {"status": "failed", "error": str(e), "connection": "❌ Failed"}


def create_document_point(
    chunk_id: str, embedding: List[float], content: str, metadata: Dict[str, Any]
) -> DocumentPoint:
    """
    Create a DocumentPoint from chunk data.

    Args:
        chunk_id: Unique chunk identifier (will be stored in payload)
        embedding: Vector embedding
        content: Text content
        metadata: Additional metadata

    Returns:
        DocumentPoint ready for Qdrant storage
    """
    # Prepare payload with all metadata
    payload = {
        "content": content,
        "chunk_id": chunk_id,  # Store original chunk_id in payload
        **metadata,
    }

    # Use UUID for Qdrant point ID (required by Qdrant)
    return DocumentPoint(id=str(uuid.uuid4()), vector=embedding, payload=payload)


def demonstrate_qdrant_functionality():
    """
    Demonstrate basic Qdrant functionality.
    """
    print("Qdrant Vector Store Demonstration")
    print("=" * 50)

    try:
        # Initialize manager
        manager = QdrantVectorStoreManager()

        # Test connection
        print("\n1. Testing Connection")
        print("-" * 30)
        test_result = manager.test_connection()

        if test_result["status"] == "success":
            print(f"✅ {test_result['connection']}")
            print(f"✅ Collections available: {test_result['collections_count']}")
            print(
                f"✅ Target collection exists: {test_result['target_collection_exists']}"
            )
        else:
            print(f"❌ Connection failed: {test_result['error']}")
            return

        # Create collection
        print("\n2. Creating Collection")
        print("-" * 30)
        created = manager.create_collection(recreate=True)
        if created:
            print("✅ Collection created")

        # Get collection info
        print("\n3. Collection Information")
        print("-" * 30)
        info = manager.get_collection_info()
        print(f"✅ Vector size: {info['vector_size']}")
        print(f"✅ Distance metric: {info['distance_metric']}")
        print(f"✅ Points count: {info['points_count']}")

        print("\n🎉 Qdrant demonstration completed successfully!")

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")


if __name__ == "__main__":
    demonstrate_qdrant_functionality()
