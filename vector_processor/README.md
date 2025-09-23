# Data Processor Module

A comprehensive data processing module for the Flexit RAG system that handles document loading, vector store management, and provides examples for hybrid retrieval operations.

## 🚀 Quick Start

Get started with the data processor in 3 simple steps:

```python
from data_processor.qdrant_pipeline import QdrantPipeline

# 1. Initialize the complete pipeline
pipeline = QdrantPipeline(collection_name="flexit_rag_collection")

# 2. Run the complete setup process
results = pipeline.run_complete_pipeline(
    recreate_collection=True,
    embedding_batch_size=50,
    storage_batch_size=100
)

# 3. Use the hybrid retrieval system
from flexit_llm.utility.hybrid_retrieval_coordinator import HybridRetrievalCoordinator

coordinator = HybridRetrievalCoordinator(cache_file_path="flexit_llm/utility/indexes_cache.json")
coordinator.initialize()
result = coordinator.retrieve("How do I set up annual enrollment?")
```

## 📁 Current Architecture

```
data_processor/
├── __init__.py                    # Module initialization
├── document_loader.py             # Document loading and cache generation
├── qdrant_pipeline.py            # Complete RAG pipeline with Qdrant
├── example_retrieval.py          # Usage examples and demonstrations
├── README.md                     # This comprehensive guide
└── tests/                        # Test files
```

## 🔧 Core Components

### DocumentLoader (`document_loader.py`)

Handles loading and indexing of RAG JSON documents from crawler output with comprehensive topic metadata.

**Key Features:**
- Loads documents from `crawler/result_data/rag_output/`
- Generates optimized cache at `flexit_llm/utility/indexes_cache.json`
- Creates topic, breadcrumb, and title indexes
- Primarily used for initial data processing and cache generation

**Usage:**
```python
from data_processor.document_loader import DocumentLoader

# Initialize with crawler output directory
loader = DocumentLoader("crawler/result_data/rag_output")

# Process all JSON files and generate cache
documents_loaded, chunks_loaded = loader.load_all_documents()

print(f"✅ Processed {documents_loaded} documents with {chunks_loaded} chunks")
print(f"✅ Cache saved to: flexit_llm/utility/indexes_cache.json")
```

**Cache Structure:**
The DocumentLoader creates an optimized cache with multiple indexes:

```json
{
  "topic_index": {
    "setup": ["doc1_id", "doc2_id"],
    "benefits_enrollment": ["doc2_id", "doc4_id"]
  },
  "breadcrumb_index": {
    "benefits > enrollment": ["doc1_id", "doc3_id"]
  },
  "title_index": {
    "annual": ["doc1_id", "doc2_id"],
    "enrollment": ["doc2_id", "doc3_id"]
  },
  "documents": {
    "doc1_id": { /* full document data */ }
  },
  "chunks": {
    "chunk1_id": { /* full chunk data */ }
  },
  "total_documents": 479,
  "total_chunks": 533,
  "unique_topics": ["setup", "benefits_enrollment", ...]
}
```

### QdrantPipeline (`qdrant_pipeline.py`)

Complete RAG pipeline using Qdrant for vector storage, handling document loading, embedding generation, and vector storage.

**Key Features:**
- Complete end-to-end RAG pipeline
- Qdrant vector store integration
- Batch processing for embeddings and storage
- Comprehensive error handling and logging
- Performance monitoring and statistics

**Environment Setup:**
```bash
# Required environment variables
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
FUELIX_API_KEY=your_fuelix_api_key
FUELIX_USER_ID=your_user_id
```

**Complete Pipeline Usage:**
```python
from data_processor.qdrant_pipeline import QdrantPipeline

# Initialize pipeline
pipeline = QdrantPipeline(collection_name="flexit_rag_collection")

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    recreate_collection=True,    # Create fresh collection
    embedding_batch_size=50,     # Process 50 chunks per batch
    storage_batch_size=100       # Store 100 vectors per batch
)

print(f"✅ Pipeline completed!")
print(f"   Documents processed: {results['documents_loaded']}")
print(f"   Embeddings generated: {results['embedding_results']['successful_embeddings']}")
print(f"   Vectors stored: {results['storage_results']['storage_results']['successful_count']}")
print(f"   Total time: {results['total_pipeline_time_seconds']:.2f}s")
```

**Step-by-Step Pipeline Operations:**

1. **Vector Store Setup:**
```python
# Setup vector store collection
setup_results = pipeline.setup_vector_store(recreate=True)
```

2. **Document Loading:**
```python
# Load documents and generate cache
docs_loaded, chunks_loaded = pipeline.load_documents()
```

3. **Embedding Generation:**
```python
# Generate embeddings for all chunks
embedding_results = pipeline.generate_embeddings(batch_size=50)
```

4. **Vector Storage:**
```python
# Store embeddings in Qdrant
storage_results = pipeline.store_embeddings(batch_size=100)
```

5. **Statistics:**
```python
# Get collection statistics
stats = pipeline.get_collection_statistics()
```

### Example Usage (`example_retrieval.py`)

Comprehensive examples demonstrating various usage patterns and configuration options for the hybrid retrieval system.

**Available Examples:**

1. **Basic Usage:** Standard hybrid retrieval with balanced configuration
2. **Custom Configuration:** Custom fusion weights and parameters
3. **Mode Comparison:** Comparing fast, balanced, and accurate modes
4. **Topic-Only Fallback:** Using topic-based retrieval without vector search
5. **Error Handling:** Graceful degradation and error management

**Running Examples:**
```python
# Run all examples
python data_processor/example_retrieval.py

# Or import specific examples
from data_processor.example_retrieval import example_basic_usage
example_basic_usage()
```

## 🔄 Integration with Flexit LLM Utility

The data processor integrates seamlessly with the `flexit_llm/utility/` components:

### Key Integration Points

**Cache Data Loader:**
```python
from flexit_llm.utility.cache_data_loader import CacheDataLoader

# Load from generated cache
loader = CacheDataLoader("flexit_llm/utility/indexes_cache.json")
stats = loader.initialize()
```

**Hybrid Retrieval Coordinator:**
```python
from flexit_llm.utility.hybrid_retrieval_coordinator import HybridRetrievalCoordinator

# Initialize with cache
coordinator = HybridRetrievalCoordinator(
    cache_file_path="flexit_llm/utility/indexes_cache.json"
)
coordinator.initialize()
```

**Vector Store Manager:**
```python
from flexit_llm.utility.qdrant_vector_store import QdrantVectorStoreManager, QdrantConfig

# Configure and use vector store
config = QdrantConfig(
    url="https://your-cluster-url.qdrant.io",
    api_key="your_qdrant_api_key",
    collection_name="flexit_rag_collection"
)
vector_store = QdrantVectorStoreManager(config)
```

**Embedding Manager:**
```python
from flexit_llm.utility.embeddings import FuelixEmbeddingManager, EmbeddingConfig

# Generate embeddings
embedding_manager = FuelixEmbeddingManager(api_key="your_api_key")
config = EmbeddingConfig(model="text-embedding-3-large", user="your_user_id")
```

## 📊 Performance Metrics

### Document Loading Performance
| Operation | Documents | Chunks | Time | Memory |
|-----------|-----------|--------|------|--------|
| **Initial Loading** | 479 | 533 | ~2-5s | ~50MB |
| **Cache Generation** | 479 | 533 | ~1-3s | ~25MB |
| **Cache Loading** | 479 | 533 | ~100ms | ~50MB |

### Pipeline Performance
| Operation | Batch Size | Processing Time | Throughput |
|-----------|------------|-----------------|------------|
| **Embedding Generation** | 50 chunks | ~15-30s | 2-3 chunks/sec |
| **Vector Storage** | 100 vectors | ~5-10s | 10-20 vectors/sec |
| **Complete Pipeline** | 533 chunks | ~45-90s | 6-12 chunks/sec |

### Retrieval Performance
| Mode | Max Results | Avg Response Time | Accuracy |
|------|-------------|-------------------|----------|
| **Fast Mode** | 10 | ~5-8ms | Good |
| **Balanced Mode** | 20 | ~8-12ms | Better |
| **Accurate Mode** | 30 | ~12-18ms | Best |

## 🛠️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Qdrant Configuration
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# Fuelix Embedding Service
FUELIX_API_KEY=your_fuelix_api_key
FUELIX_USER_ID=your_user_id
```

### Pipeline Configuration

```python
from data_processor.qdrant_pipeline import QdrantPipeline

# Custom configuration
pipeline = QdrantPipeline(collection_name="custom_collection")

# Run with custom parameters
results = pipeline.run_complete_pipeline(
    recreate_collection=False,      # Keep existing collection
    embedding_batch_size=25,        # Smaller batches
    storage_batch_size=50          # Smaller storage batches
)
```

### Retrieval Configuration

```python
from flexit_llm.utility.retrieval_config import RetrievalConfig, FusionConfig

# Custom fusion configuration
custom_fusion = FusionConfig(
    topic_weight=0.6,
    vector_weight=0.4,
    consensus_boost=1.3,
    max_fusion_results=15
)

config = RetrievalConfig.balanced_mode()
config.fusion_config = custom_fusion
```

## 🔍 API Reference

### DocumentLoader

```python
class DocumentLoader:
    def __init__(self, rag_output_dir: str = "crawler/result_data/rag_output")
    def load_all_documents(self) -> Tuple[int, int]
    def get_statistics(self) -> Dict
```

### QdrantPipeline

```python
class QdrantPipeline:
    def __init__(self, collection_name: str = "flexit_rag_collection")
    def setup_vector_store(self, recreate: bool = False) -> Dict[str, Any]
    def load_documents(self) -> Tuple[int, int]
    def generate_embeddings(self, batch_size: int = 20) -> Dict[str, Any]
    def store_embeddings(self, batch_size: int = 100) -> Dict[str, Any]
    def get_collection_statistics(self) -> Dict[str, Any]
    def run_complete_pipeline(self, recreate_collection: bool = False, 
                            embedding_batch_size: int = 50,
                            storage_batch_size: int = 100) -> Dict[str, Any]
```

## 🚀 Getting Started

### Prerequisites

1. **Python Environment:** Python 3.8+
2. **Dependencies:** Install from `requirements.txt`
3. **Qdrant Cloud Account:** Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
4. **Fuelix API Access:** For embedding generation
5. **Crawler Data:** RAG JSON files in `crawler/result_data/rag_output/`

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Quick Setup

```bash
# 1. Generate document cache (if not exists)
python -c "from data_processor.document_loader import DocumentLoader; DocumentLoader().load_all_documents()"

# 2. Run complete pipeline
python -c "from data_processor.qdrant_pipeline import QdrantPipeline; QdrantPipeline().run_complete_pipeline()"

# 3. Test retrieval examples
python data_processor/example_retrieval.py
```

## 🛠️ Troubleshooting

### Common Issues

#### Cache File Not Found
```python
# Check if cache exists
import os
cache_path = "flexit_llm/utility/indexes_cache.json"
if not os.path.exists(cache_path):
    print("❌ Cache file not found. Run DocumentLoader.load_all_documents()")
```

#### Qdrant Connection Issues
```python
# Test Qdrant connection
from data_processor.qdrant_pipeline import QdrantPipeline
pipeline = QdrantPipeline()
setup_results = pipeline.setup_vector_store()
print(f"Connection status: {setup_results['connection_test']['status']}")
```

#### Embedding Generation Failures
```python
# Check environment variables
import os
required_vars = ["FUELIX_API_KEY", "FUELIX_USER_ID"]
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f"❌ Missing environment variables: {missing}")
```

### Performance Issues

- **Slow embedding generation:** Reduce `embedding_batch_size`
- **Memory issues:** Reduce `storage_batch_size`
- **Connection timeouts:** Check network and API limits
- **Cache loading slow:** Regenerate cache if corrupted

## 📈 Migration Guide

### From Old Structure

If migrating from previous versions:

1. **Update Import Paths:**
```python
# Old
from rag_retrieval.document_loader import DocumentLoader

# New
from data_processor.document_loader import DocumentLoader
```

2. **Update Cache Paths:**
```python
# Old
cache_file_path="rag_retrieval/indexes_cache.json"

# New
cache_file_path="flexit_llm/utility/indexes_cache.json"
```

3. **Use New Pipeline:**
```python
# Old approach (multiple steps)
# loader = DocumentLoader()
# embedding_manager = EmbeddingManager()
# vector_store = VectorStore()

# New approach (single pipeline)
from data_processor.qdrant_pipeline import QdrantPipeline
pipeline = QdrantPipeline()
results = pipeline.run_complete_pipeline()
```

## 📝 Development

### Running Tests

```bash
# Run all tests
python -m pytest data_processor/tests/

# Run specific test
python -m pytest data_processor/tests/test_document_loader.py
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Include error handling and logging

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📚 Related Documentation

For comprehensive information about the hybrid retrieval system and utility functions, see:

- **[FLEXIT_LLM_UTILITY.md](../FLEXIT_LLM_UTILITY.md)** - Complete guide to the hybrid RAG retrieval system
- **[flexit_llm/utility/](../flexit_llm/utility/)** - Core utility modules for retrieval and vector operations

---

**🎯 Ready to get started?** Initialize your pipeline and start processing!

```python
from data_processor.qdrant_pipeline import QdrantPipeline

pipeline = QdrantPipeline()
results = pipeline.run_complete_pipeline()
print(f"✅ Pipeline completed! Processed {results['documents_loaded']} documents")
```
