#!/usr/bin/env python3
"""
Simplified RAG Pipeline with Qdrant Vector Store
Complete RAG solution: document loading, embedding generation, and vector storage/retrieval.
"""

import logging
import os
import time
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from rag_retrieval.document_loader import DocumentLoader
from rag_retrieval.embeddings import (
    EmbeddingConfig,
    FuelixEmbeddingManager,
)
from rag_retrieval.qdrant_vector_store import (
    QdrantConfig,
    QdrantVectorStoreManager,
    create_document_point,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantRAGPipeline:
    """
    Complete RAG pipeline using Qdrant for vector storage.
    Handles document loading, embedding generation, vector storage
    """

    def __init__(self, collection_name: str = "flexit_rag_collection"):
        """
        Initialize the Qdrant RAG pipeline.

        Args:
            collection_name: Name for the Qdrant collection
        """
        # Load environment variables
        load_dotenv()

        self.collection_name = collection_name
        self.api_key = os.getenv("FUELIX_API_KEY")
        self.user_id = os.getenv("FUELIX_USER_ID")

        if not self.api_key:
            raise ValueError("FUELIX_API_KEY not found in environment variables")

        # Initialize components
        self.document_loader = DocumentLoader()
        self.embedding_manager = FuelixEmbeddingManager(api_key=self.api_key)

        # Initialize Qdrant vector store
        qdrant_config = QdrantConfig(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=collection_name,
            vector_size=3072,  # text-embedding-3-large dimensions
        )
        self.qdrant_manager = QdrantVectorStoreManager(qdrant_config)

        # Pipeline state
        self.documents_loaded = False
        self.embeddings_generated = False
        self.documents_stored = False
        self.processed_chunks: List[Dict[str, Any]] = []

        logger.info("Qdrant RAG Pipeline initialized")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"User: {self.user_id}")

    def setup_vector_store(self, recreate: bool = False) -> Dict[str, Any]:
        """
        Set up the Qdrant vector store collection.

        Args:
            recreate: Whether to recreate the collection if it exists

        Returns:
            Setup results
        """
        logger.info("Setting up vector store...")

        try:
            # Test connection
            connection_test = self.qdrant_manager.test_connection()
            if connection_test["status"] != "success":
                raise Exception(f"Connection failed: {connection_test.get('error')}")

            # Create collection
            created = self.qdrant_manager.create_collection(recreate=recreate)

            # Get collection info
            collection_info = self.qdrant_manager.get_collection_info()

            results = {
                "status": "success",
                "collection_created": created,
                "collection_info": collection_info,
                "connection_test": connection_test,
            }

            logger.info("✅ Vector store setup completed")
            return results

        except Exception as e:
            logger.error(f"❌ Vector store setup failed: {e}")
            raise

    def load_documents(self) -> Tuple[int, int]:
        """
        Load all RAG documents from the output directory.

        Returns:
            Tuple of (documents_loaded, chunks_loaded)
        """
        logger.info("Loading RAG documents...")

        try:
            docs_loaded, chunks_loaded = self.document_loader.load_all_documents()
            self.documents_loaded = True

            logger.info(
                f"✅ Loaded {docs_loaded} documents with {chunks_loaded} chunks"
            )
            return docs_loaded, chunks_loaded

        except Exception as e:
            logger.error(f"❌ Failed to load documents: {e}")
            raise

    def generate_embeddings(self, batch_size: int = 20) -> Dict[str, Any]:
        """
        Generate embeddings for all loaded document chunks.

        Args:
            batch_size: Number of chunks to process per batch

        Returns:
            Embedding generation results
        """
        if not self.documents_loaded:
            raise ValueError("Documents must be loaded first. Call load_documents()")

        logger.info("Generating embeddings for document chunks...")

        # Collect chunks and their content
        chunks_to_process = []
        chunk_metadata = {}

        for chunk_id, chunk in self.document_loader.chunks.items():
            content = chunk.content.strip()
            if content:
                chunks_to_process.append(content)
                chunk_metadata[len(chunks_to_process) - 1] = {
                    "chunk_id": chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "breadcrumb": chunk.breadcrumb,
                    "source_file": chunk.source_file,
                }

        logger.info(
            f"Processing {len(chunks_to_process)} chunks in batches of {batch_size}"
        )

        # Configure embedding generation
        config = EmbeddingConfig(model="text-embedding-3-large", user=self.user_id)

        start_time = time.time()

        try:
            # Generate embeddings in batches
            batch_result = self.embedding_manager.generate_batch_embeddings(
                texts=chunks_to_process, config=config, batch_size=batch_size
            )

            processing_time = time.time() - start_time

            # Process results and prepare for storage
            self.processed_chunks = []

            for i, embedding in enumerate(batch_result.embeddings):
                if i in chunk_metadata and embedding:  # Skip failed embeddings
                    chunk_info = chunk_metadata[i]

                    processed_chunk = {
                        "chunk_id": chunk_info["chunk_id"],
                        "embedding": embedding,
                        "content": chunks_to_process[i],
                        "document_id": chunk_info["document_id"],
                        "title": chunk_info["title"],
                        "breadcrumb": chunk_info["breadcrumb"],
                        "source_file": chunk_info["source_file"],
                    }

                    self.processed_chunks.append(processed_chunk)

            self.embeddings_generated = True

            # Results summary
            results = {
                "total_chunks": len(chunks_to_process),
                "successful_embeddings": len(self.processed_chunks),
                "failed_embeddings": batch_result.failed_count,
                "processing_time_seconds": processing_time,
                "model": batch_result.model,
                "dimensions": batch_result.dimensions,
                "total_tokens": batch_result.usage.get("total_tokens", 0),
            }

            logger.info("✅ Embedding generation completed:")
            logger.info(f"   Successful: {results['successful_embeddings']}")
            logger.info(f"   Failed: {results['failed_embeddings']}")
            logger.info(f"   Total time: {results['processing_time_seconds']:.2f}s")

            return results

        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise

    def store_embeddings(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Store generated embeddings in Qdrant vector store.

        Args:
            batch_size: Number of documents to process per batch

        Returns:
            Storage results
        """
        if not self.embeddings_generated:
            raise ValueError(
                "Embeddings must be generated first. Call generate_embeddings()"
            )

        logger.info("Storing embeddings in Qdrant vector store...")

        try:
            # Convert chunks to DocumentPoint objects
            document_points = []

            for chunk_data in self.processed_chunks:
                # Prepare metadata
                metadata = {
                    "document_id": chunk_data.get("document_id"),
                    "title": chunk_data.get("title"),
                    "breadcrumb": chunk_data.get("breadcrumb"),
                    "source_file": chunk_data.get("source_file"),
                    "user_id": self.user_id,
                    "stored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                # Create document point
                doc_point = create_document_point(
                    chunk_id=chunk_data["chunk_id"],
                    embedding=chunk_data["embedding"],
                    content=chunk_data["content"],
                    metadata=metadata,
                )

                document_points.append(doc_point)

            # Store in Qdrant
            storage_results = self.qdrant_manager.add_documents(
                documents=document_points, batch_size=batch_size
            )

            self.documents_stored = True

            logger.info("✅ Embeddings stored successfully:")
            logger.info(f"   Total chunks: {len(self.processed_chunks)}")
            logger.info(
                f"   Stored successfully: {storage_results['successful_count']}"
            )

            return {
                "status": "success",
                "total_chunks": len(self.processed_chunks),
                "valid_chunks": len(document_points),
                "storage_results": storage_results,
            }

        except Exception as e:
            logger.error(f"❌ Failed to store embeddings: {e}")
            raise

    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.

        Returns:
            Collection statistics
        """
        try:
            collection_info = self.qdrant_manager.get_collection_info()

            stats = {
                "collection_name": self.collection_name,
                "total_documents": collection_info.get("points_count", 0),
                "vector_size": collection_info.get("vector_size"),
                "distance_metric": str(collection_info.get("distance_metric")),
                "status": collection_info.get("status"),
                "user_id": self.user_id,
                "pipeline_status": {
                    "documents_loaded": self.documents_loaded,
                    "embeddings_generated": self.embeddings_generated,
                    "documents_stored": self.documents_stored,
                },
            }

            return stats

        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            raise

    def run_complete_pipeline(
        self,
        recreate_collection: bool = False,
        embedding_batch_size: int = 50,
        storage_batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline from start to finish.

        Args:
            recreate_collection: Whether to recreate the Qdrant collection
            embedding_batch_size: Batch size for embedding generation
            storage_batch_size: Batch size for vector storage

        Returns:
            Complete pipeline results
        """
        logger.info("🚀 Starting Complete Qdrant RAG Pipeline")
        logger.info("=" * 50)

        pipeline_start = time.time()
        results = {}

        try:
            # Step 1: Setup vector store
            logger.info("\n🗄️ Step 1: Setting up Vector Store")
            setup_results = self.setup_vector_store(recreate=recreate_collection)
            results["vector_store_setup"] = setup_results

            # Step 2: Load documents
            logger.info("\n📚 Step 2: Loading Documents")
            docs_loaded, chunks_loaded = self.load_documents()
            results["documents_loaded"] = docs_loaded
            results["chunks_loaded"] = chunks_loaded

            # Step 3: Generate embeddings
            logger.info("\n🧠 Step 3: Generating Embeddings")
            embedding_results = self.generate_embeddings(
                batch_size=embedding_batch_size
            )
            results["embedding_results"] = embedding_results

            # Step 4: Store embeddings
            logger.info("\n💾 Step 4: Storing Embeddings in Qdrant")
            storage_results = self.store_embeddings(batch_size=storage_batch_size)
            results["storage_results"] = storage_results

            # Step 5: Get final statistics
            logger.info("\n📊 Step 5: Final Statistics")
            pipeline_stats = self.get_collection_statistics()
            results["pipeline_statistics"] = pipeline_stats

            # Pipeline summary
            total_time = time.time() - pipeline_start
            results["total_pipeline_time_seconds"] = total_time

            logger.info("\n✅ Complete Qdrant RAG Pipeline Finished!")
            logger.info(f"   Total time: {total_time:.2f}s")
            logger.info(f"   Documents processed: {results['documents_loaded']}")
            logger.info(
                f"   Embeddings generated: {embedding_results['successful_embeddings']}"
            )
            logger.info(
                f"   Documents stored: {storage_results['storage_results']['successful_count']}"
            )
            logger.info(f"   Collection: {self.collection_name}")

            return results

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise


def run_qdrant_rag_demo():
    """
    Demonstrate the complete Qdrant RAG pipeline.
    """
    print("Qdrant RAG Pipeline Demonstration")
    print("=" * 50)

    try:
        # Initialize pipeline
        pipeline = QdrantRAGPipeline()

        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            recreate_collection=True, embedding_batch_size=50
        )

        print("\n🎉 Qdrant RAG Pipeline Demo Completed!")
        print("📊 Results Summary:")
        print(f"   Documents: {results['documents_loaded']}")
        print(f"   Chunks: {results['chunks_loaded']}")
        print(f"   Embeddings: {results['embedding_results']['successful_embeddings']}")
        print(
            f"   Stored: {results['storage_results']['storage_results']['successful_count']}"
        )
        print(f"   Total time: {results['total_pipeline_time_seconds']:.2f}s")

        # Test search functionality
        print("\n🔍 Testing Search Functionality")
        print("-" * 30)

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    run_qdrant_rag_demo()
