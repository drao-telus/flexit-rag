"""
Example Usage of Hybrid Fusion Retrieval System
Demonstrates basic usage patterns and configuration options.
"""

import logging
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_retrieval.hybrid_retrieval_coordinator import HybridRetrievalCoordinator
from rag_retrieval.retrieval_config import FusionConfig, RetrievalConfig


def get_cache_file_path():
    """Get the correct path to the indexes_cache.json file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(current_dir, "indexes_cache.json")
    return cache_file


def setup_logging():
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def example_basic_usage():
    """Example 1: Basic hybrid retrieval usage."""
    print("Example 1: Basic Hybrid Retrieval")
    print("=" * 40)

    # Initialize with default balanced mode and correct cache path
    config = RetrievalConfig.balanced_mode()
    coordinator = HybridRetrievalCoordinator(
        config, cache_file_path=get_cache_file_path()
    )

    # Initialize the system
    print("Initializing hybrid retrieval system...")
    init_stats = coordinator.initialize()

    if not init_stats["hybrid_initialization_complete"]:
        print("❌ Initialization failed. Please check your configuration.")
        return

    print("✅ System initialized successfully!")
    print(
        f"   Documents loaded: {init_stats['topic_retriever_stats'].get('documents_loaded', 0)}"
    )
    print(f"   Qdrant status: {init_stats['qdrant_status']}")

    # Perform a search
    query = "How do I set up annual enrollment?"
    print(f"\nSearching for: '{query}'")

    result = coordinator.retrieve(query, max_results=5)

    # Display results
    print("\n📊 Results Summary:")
    print(f"   Total results: {len(result.fused_chunks)}")
    print(f"   Consensus matches: {result.consensus_chunks}")
    print(f"   Topic-only matches: {result.topic_only_chunks}")
    print(f"   Vector-only matches: {result.vector_only_chunks}")
    print(f"   Processing time: {result.total_processing_time:.2f}s")

    # Show top 3 results
    print("\n🔍 Top Results:")
    for i, chunk in enumerate(result.fused_chunks[:3], 1):
        print(f"\n   Result {i}:")
        print(f"   📄 Chunk ID: {chunk.chunk_id}")
        print(f"   ⭐ Fusion Score: {chunk.fusion_score:.3f}")
        print(f"   🎯 Topic Score: {chunk.topic_score:.3f}")
        print(f"   🔍 Vector Score: {chunk.vector_score:.3f}")
        print(f"   🤝 Consensus: {'Yes' if chunk.consensus_match else 'No'}")
        print(f"   📝 Content: {chunk.content[:100]}...")


def example_custom_configuration():
    """Example 2: Custom fusion configuration."""
    print("\n\nExample 2: Custom Fusion Configuration")
    print("=" * 40)

    # Create a custom fusion configuration that favors topic-based results
    custom_fusion = FusionConfig(
        topic_weight=0.7,  # Favor topic-based results
        vector_weight=0.3,  # Lower weight for vector results
        consensus_boost=1.5,  # Strong boost for consensus
        max_fusion_results=10,  # Limit results
    )

    # Apply to retrieval configuration
    config = RetrievalConfig.balanced_mode()
    config.fusion_config = custom_fusion

    # Initialize coordinator with custom config and correct cache path
    coordinator = HybridRetrievalCoordinator(
        config, cache_file_path=get_cache_file_path()
    )

    print("Initializing with custom configuration...")
    print(f"   Topic weight: {custom_fusion.topic_weight}")
    print(f"   Vector weight: {custom_fusion.vector_weight}")
    print(f"   Consensus boost: {custom_fusion.consensus_boost}")

    init_stats = coordinator.initialize()

    if not init_stats["hybrid_initialization_complete"]:
        print("❌ Initialization failed.")
        return

    # Test with a domain-specific query
    query = "Employee benefits enrollment process"
    print(f"\nTesting topic-heavy configuration with: '{query}'")

    result = coordinator.retrieve(query, max_results=3)

    print("\n📊 Custom Configuration Results:")
    print(f"   Results found: {len(result.fused_chunks)}")
    print(f"   Consensus matches: {result.consensus_chunks}")

    # Show how the custom weighting affects scores
    if result.fused_chunks:
        top_chunk = result.fused_chunks[0]
        print("\n   Top Result Analysis:")
        print(f"   📊 Raw topic score: {top_chunk.topic_score:.3f}")
        print(f"   📊 Raw vector score: {top_chunk.vector_score:.3f}")
        print("   ⚖️  Weighted contribution:")
        print(f"      Topic: {top_chunk.topic_score * custom_fusion.topic_weight:.3f}")
        print(
            f"      Vector: {top_chunk.vector_score * custom_fusion.vector_weight:.3f}"
        )
        print(f"   🎯 Final fusion score: {top_chunk.fusion_score:.3f}")


def example_comparison_modes():
    """Example 3: Compare different retrieval modes."""
    print("\n\nExample 3: Comparing Retrieval Modes")
    print("=" * 40)

    modes = [
        ("Fast Mode", RetrievalConfig.fast_mode()),
        ("Balanced Mode", RetrievalConfig.balanced_mode()),
        ("Accurate Mode", RetrievalConfig.accurate_mode()),
    ]

    query = "How to add beneficiaries?"
    print(f"Comparing modes with query: '{query}'")

    for mode_name, config in modes:
        print(f"\n🔧 {mode_name}:")
        print(f"   Max results: {config.fusion_config.max_fusion_results}")
        print(f"   Consensus boost: {config.fusion_config.consensus_boost}")

        try:
            coordinator = HybridRetrievalCoordinator(
                config, cache_file_path=get_cache_file_path()
            )
            init_stats = coordinator.initialize()

            if init_stats["hybrid_initialization_complete"]:
                result = coordinator.retrieve(query, max_results=3)
                print(f"   ✅ Results: {len(result.fused_chunks)}")
                print(f"   ✅ Time: {result.total_processing_time:.2f}s")
                print(f"   ✅ Consensus: {result.consensus_chunks}")

                if result.fused_chunks:
                    print(f"   ✅ Top score: {result.fused_chunks[0].fusion_score:.3f}")
            else:
                print("   ❌ Initialization failed")

        except Exception as e:
            print(f"   ❌ Error: {e}")


def example_topic_only_fallback():
    """Example 4: Topic-only fallback mode."""
    print("\n\nExample 4: Topic-Only Fallback")
    print("=" * 40)

    # Create configuration with fusion disabled
    config = RetrievalConfig.balanced_mode()
    config.fusion_config.enable_fusion = False

    coordinator = HybridRetrievalCoordinator(
        config, cache_file_path=get_cache_file_path()
    )

    print("Initializing with fusion disabled (topic-only mode)...")
    init_stats = coordinator.initialize()

    if not init_stats["hybrid_initialization_complete"]:
        print("❌ Initialization failed.")
        return

    query = "Payroll setup configuration"
    print(f"Testing topic-only retrieval: '{query}'")

    result = coordinator.retrieve(query, max_results=5)

    print("\n📊 Topic-Only Results:")
    print(f"   Results found: {len(result.fused_chunks)}")
    print(
        f"   All results are topic-only: {result.topic_only_chunks == len(result.fused_chunks)}"
    )
    print(f"   Vector results: {result.vector_only_chunks} (should be 0)")
    print(f"   Processing time: {result.total_processing_time:.2f}s")

    # Show that fusion scores equal topic scores in this mode
    if result.fused_chunks:
        chunk = result.fused_chunks[0]
        print("\n   Score verification:")
        print(f"   Topic score: {chunk.topic_score:.3f}")
        print(f"   Fusion score: {chunk.fusion_score:.3f}")
        print(f"   Scores match: {abs(chunk.topic_score - chunk.fusion_score) < 0.001}")


def example_error_handling():
    """Example 5: Error handling and graceful degradation."""
    print("\n\nExample 5: Error Handling")
    print("=" * 40)

    # Test with potentially problematic configuration
    coordinator = HybridRetrievalCoordinator(cache_file_path=get_cache_file_path())

    # Try to retrieve without initialization
    print("Testing retrieval without initialization...")
    try:
        result = coordinator.retrieve("test query")
        print("❌ Should have failed!")
    except RuntimeError as e:
        print(f"✅ Correctly caught error: {e}")

    # Test initialization
    print("\nTesting initialization...")
    init_stats = coordinator.initialize()

    if init_stats["hybrid_initialization_complete"]:
        print("✅ Initialization successful")

        # Test with empty query
        print("\nTesting with empty query...")
        try:
            result = coordinator.retrieve("", max_results=1)
            print(f"✅ Handled empty query: {len(result.fused_chunks)} results")
        except Exception as e:
            print(f"⚠️  Empty query error: {e}")

        # Test with very specific query
        print("\nTesting with very specific query...")
        result = coordinator.retrieve("xyz123nonexistentquery", max_results=1)
        print(f"✅ Handled specific query: {len(result.fused_chunks)} results")

    else:
        print("❌ Initialization failed - some components may not be available")
        print("   This is expected if Qdrant or embedding services are not configured")


def main():
    """Run all examples."""
    setup_logging()

    print("🚀 Hybrid Fusion Retrieval System - Examples")
    print("=" * 60)
    print(
        "This script demonstrates various usage patterns of the hybrid retrieval system."
    )
    print(
        "Some examples may fail if external services (Qdrant, embeddings) are not configured."
    )
    print("=" * 60)

    try:
        example_basic_usage()
        example_custom_configuration()
        example_comparison_modes()
        example_topic_only_fallback()
        example_error_handling()

        print("\n" + "=" * 60)
        print("🎉 All examples completed!")
        print("\nNext steps:")
        print("1. Run the comprehensive test suite: python test_hybrid_retrieval.py")
        print("2. Check the documentation: FUSION_RETRIEVAL_GUIDE.md")
        print("3. Configure Qdrant and embedding services for full functionality")

    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        print("This may be due to missing configuration or external services.")


if __name__ == "__main__":
    main()
