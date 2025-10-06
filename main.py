import crawler
import asyncio
from vector_processor.qdrant_pipeline import QdrantPipeline


# asyncio.run(crawler.execute_crawler())


# Initialize the complete pipeline
pipeline = QdrantPipeline(collection_name="flexit_rag_collection")

# Run the complete setup process
# results = pipeline.run_complete_pipeline(
#     recreate_collection=True,  # Create fresh collection
#     embedding_batch_size=50,  # Process 50 chunks per batch
#     storage_batch_size=100,  # Store 100 vectors per batch
# )
