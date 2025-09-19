"""
Configuration settings for the hybrid RAG retrieval system.
Provides different modes optimized for speed, accuracy, or balanced performance.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FusionConfig:
    """Minimalistic configuration for dual-retrieval fusion system."""

    # Core fusion settings
    enable_fusion: bool = True
    topic_weight: float = 0.5
    vector_weight: float = 0.5

    # Consensus boosting
    consensus_boost: float = 1.2

    # Result limits
    max_fusion_results: int = 20

    @classmethod
    def fast_fusion(cls) -> "FusionConfig":
        """Fast fusion - balanced weights, minimal boosting."""
        return cls(
            topic_weight=0.5,
            vector_weight=0.5,
            consensus_boost=1.1,
            max_fusion_results=10,
        )

    @classmethod
    def balanced_fusion(cls) -> "FusionConfig":
        """Balanced fusion - default settings."""
        return cls()

    @classmethod
    def accurate_fusion(cls) -> "FusionConfig":
        """Accurate fusion - higher consensus boost, more results."""
        return cls(consensus_boost=1.3, max_fusion_results=30)

    @classmethod
    def semantic_fusion(cls) -> "FusionConfig":
        """Semantic fusion - vector-heavy weighting for semantic queries."""
        return cls(
            topic_weight=0.3,
            vector_weight=0.7,
            consensus_boost=1.2,
            max_fusion_results=25,
        )


@dataclass
class RetrievalConfig:
    """Configuration for the hybrid topic-enhanced retrieval system."""

    # Mode identifier
    mode: str = "balanced"

    # Result limits
    max_results: int = 10
    max_candidates: int = 50
    min_score_threshold: float = 0.1

    # Topic scoring weights
    domain_topic_weight: float = 3.0
    primary_topic_weight: float = 2.0
    entity_topic_weight: float = 1.5
    breadcrumb_weight: float = 1.5
    title_weight: float = 1.0
    content_weight: float = 0.8
    confidence_bonus_weight: float = 1.0
    topic_weight: float = 1.0

    # Filtering settings
    use_domain_filtering: bool = True
    confidence_threshold: float = 0.5
    normalize_scores: bool = True

    # Performance settings
    enable_caching: bool = True
    parallel_processing: bool = False

    # Fusion configuration
    fusion_config: Optional[FusionConfig] = None

    @classmethod
    def fast_mode(cls) -> "RetrievalConfig":
        """Fast retrieval configuration - prioritizes speed over accuracy."""
        return cls(
            mode="fast",
            max_results=5,
            max_candidates=20,
            min_score_threshold=0.2,
            domain_topic_weight=2.0,
            primary_topic_weight=1.5,
            entity_topic_weight=1.0,
            breadcrumb_weight=1.0,
            title_weight=0.8,
            content_weight=0.5,
            confidence_bonus_weight=0.5,
            topic_weight=1.0,
            normalize_scores=False,
            use_domain_filtering=True,
            confidence_threshold=0.6,
            fusion_config=FusionConfig.fast_fusion(),
        )

    @classmethod
    def balanced_mode(cls) -> "RetrievalConfig":
        """Balanced retrieval configuration - good speed and accuracy."""
        return cls(
            mode="balanced",
            max_results=10,
            max_candidates=50,
            min_score_threshold=0.1,
            domain_topic_weight=3.0,
            primary_topic_weight=2.0,
            entity_topic_weight=1.5,
            breadcrumb_weight=1.5,
            title_weight=1.0,
            content_weight=0.8,
            confidence_bonus_weight=1.0,
            topic_weight=1.0,
            normalize_scores=True,
            use_domain_filtering=True,
            confidence_threshold=0.5,
            fusion_config=FusionConfig.balanced_fusion(),
        )

    @classmethod
    def accurate_mode(cls) -> "RetrievalConfig":
        """Accurate retrieval configuration - prioritizes accuracy over speed."""
        return cls(
            mode="accurate",
            max_results=15,
            max_candidates=100,
            min_score_threshold=0.05,
            domain_topic_weight=4.0,
            primary_topic_weight=2.5,
            entity_topic_weight=2.0,
            breadcrumb_weight=2.0,
            title_weight=1.5,
            content_weight=1.0,
            confidence_bonus_weight=1.5,
            topic_weight=1.2,
            normalize_scores=True,
            use_domain_filtering=True,
            confidence_threshold=0.3,
            fusion_config=FusionConfig.accurate_fusion(),
        )

    @classmethod
    def semantic_mode(cls) -> "RetrievalConfig":
        """Semantic retrieval configuration - optimized for vector-based semantic search."""
        return cls(
            mode="semantic",
            max_results=12,
            max_candidates=75,
            min_score_threshold=0.08,
            domain_topic_weight=2.0,
            primary_topic_weight=1.5,
            entity_topic_weight=1.0,
            breadcrumb_weight=1.0,
            title_weight=0.8,
            content_weight=0.6,
            confidence_bonus_weight=0.8,
            topic_weight=0.8,
            normalize_scores=True,
            use_domain_filtering=False,
            confidence_threshold=0.4,
            fusion_config=FusionConfig.semantic_fusion(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RetrievalConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def get_summary(self) -> str:
        """Get a summary of the configuration."""
        return (
            f"RetrievalConfig(mode={self.mode}, "
            f"max_results={self.max_results}, "
            f"max_candidates={self.max_candidates}, "
            f"domain_weight={self.domain_topic_weight}, "
            f"use_domain_filtering={self.use_domain_filtering})"
        )


# Default configurations
DEFAULT_CONFIG = RetrievalConfig.balanced_mode()
FAST_CONFIG = RetrievalConfig.fast_mode()
BALANCED_CONFIG = RetrievalConfig.balanced_mode()
ACCURATE_CONFIG = RetrievalConfig.accurate_mode()
SEMANTIC_CONFIG = RetrievalConfig.semantic_mode()
