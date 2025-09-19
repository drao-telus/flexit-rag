"""
Topic-based scoring algorithms for hybrid RAG retrieval.
Implements confidence-weighted scoring for topic relevance calculation.
"""

from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import math
import logging

from rag_retrieval.retrieval_config import RetrievalConfig
from rag_retrieval.document_loader import Document, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class TopicMatch:
    """Represents a topic match between query and document."""

    topic: str
    topic_type: str  # domain, entity, content, breadcrumb, etc.
    confidence: float
    weight: float
    score: float


@dataclass
class DocumentScore:
    """Comprehensive scoring for a document."""

    document_id: str
    total_score: float
    topic_matches: List[TopicMatch]
    breadcrumb_score: float
    title_score: float
    content_score: float
    confidence_bonus: float
    match_count: int
    explanation: str


class ScoringAlgorithms:
    """Advanced scoring algorithms for topic-enhanced retrieval."""

    def __init__(self, config: RetrievalConfig):
        """
        Initialize scoring algorithms with configuration.

        Args:
            config: Retrieval configuration with weights and thresholds
        """
        self.config = config

    def calculate_document_score(
        self, document: Document, query_topics: Dict[str, any], query_text: str = ""
    ) -> DocumentScore:
        """
        Calculate comprehensive relevance score for a document.

        Args:
            document: Document to score
            query_topics: Enhanced query topics from topic extractor
            query_text: Original query text for content matching

        Returns:
            DocumentScore with detailed scoring breakdown
        """
        # Extract query topic data - handle both query format and document format
        # Query format (from extract_query_topics): query_topics, domain_topics, entity_topics, high_confidence_topics, confidence_scores
        # Document format (from RAG files): all_topics, keyword_topics, entity_topics, primary_topics

        query_topic_list = query_topics.get(
            "query_topics", query_topics.get("all_topics", [])
        )
        domain_topics = query_topics.get(
            "domain_topics", query_topics.get("keyword_topics", [])
        )
        entity_topics = query_topics.get("entity_topics", [])
        high_confidence_topics = query_topics.get(
            "high_confidence_topics", query_topics.get("primary_topics", [])
        )
        confidence_scores = query_topics.get("confidence_scores", {})

        # Calculate different scoring components
        topic_matches = self._calculate_topic_matches(
            document, query_topic_list, domain_topics, entity_topics, confidence_scores
        )

        breadcrumb_score = self._calculate_breadcrumb_score(
            document.breadcrumb, query_topic_list, domain_topics
        )

        title_score = self._calculate_title_score(
            document.title, query_topic_list, query_text
        )

        content_score = self._calculate_content_score(document, query_text)

        confidence_bonus = self._calculate_confidence_bonus(
            topic_matches, high_confidence_topics
        )

        # Combine scores with weights
        topic_score = sum(match.score for match in topic_matches)

        total_score = (
            topic_score * self.config.topic_weight
            + breadcrumb_score * self.config.breadcrumb_weight
            + title_score * self.config.title_weight
            + content_score * self.config.content_weight
            + confidence_bonus * self.config.confidence_bonus_weight
        )

        # Apply normalization
        if self.config.normalize_scores:
            total_score = self._normalize_score(total_score)

        # Generate explanation
        explanation = self._generate_score_explanation(
            topic_score, breadcrumb_score, title_score, content_score, confidence_bonus
        )

        return DocumentScore(
            document_id=document.document_id,
            total_score=total_score,
            topic_matches=topic_matches,
            breadcrumb_score=breadcrumb_score,
            title_score=title_score,
            content_score=content_score,
            confidence_bonus=confidence_bonus,
            match_count=len(topic_matches),
            explanation=explanation,
        )

    def _calculate_topic_matches(
        self,
        document: Document,
        query_topics: List[str],
        domain_topics: List[str],
        entity_topics: List[str],
        confidence_scores: Dict[str, float],
    ) -> List[TopicMatch]:
        """Calculate topic matches with confidence weighting."""
        matches = []
        doc_topics = document.topics

        # Get document topics using actual RAG output fields and normalize case
        doc_breadcrumb_topics = set(
            topic.lower().strip()
            for topic in doc_topics.get("breadcrumb_topics", [])
            if isinstance(topic, str)
        )
        doc_content_topics = set(
            topic.lower().strip()
            for topic in doc_topics.get("content_topics", [])
            if isinstance(topic, str)
        )
        doc_heading_topics = set(
            topic.lower().strip()
            for topic in doc_topics.get("heading_topics", [])
            if isinstance(topic, str)
        )
        doc_keyword_topics = set(
            topic.lower().strip()
            for topic in doc_topics.get("keyword_topics", [])
            if isinstance(topic, str)
        )
        doc_entity_topics = set(
            topic.lower().strip()
            for topic in doc_topics.get("entity_topics", [])
            if isinstance(topic, str)
        )
        doc_all_topics = set(
            topic.lower().strip()
            for topic in doc_topics.get("all_topics", [])
            if isinstance(topic, str)
        )
        doc_primary_topics = set(
            topic.lower().strip()
            for topic in doc_topics.get("primary_topics", [])
            if isinstance(topic, str)
        )

        # Combine all document topics for matching
        all_doc_topics = set()
        all_doc_topics.update(doc_breadcrumb_topics)
        all_doc_topics.update(doc_content_topics)
        all_doc_topics.update(doc_heading_topics)
        all_doc_topics.update(doc_keyword_topics)
        all_doc_topics.update(doc_entity_topics)
        all_doc_topics.update(doc_all_topics)
        all_doc_topics.update(doc_primary_topics)

        # Check domain topic matches (highest priority)
        for topic in domain_topics:
            topic_normalized = topic.lower().strip()
            if topic_normalized in all_doc_topics:
                confidence = confidence_scores.get(
                    topic, 0.8
                )  # High confidence for domain topics
                weight = self.config.domain_topic_weight
                score = confidence * weight

                # Determine topic type in document
                topic_type = self._determine_topic_type(
                    topic_normalized,
                    doc_breadcrumb_topics,
                    doc_content_topics,
                    doc_heading_topics,
                    doc_keyword_topics,
                    doc_entity_topics,
                    doc_primary_topics,
                )

                matches.append(
                    TopicMatch(
                        topic=topic,
                        topic_type=f"domain_{topic_type}",
                        confidence=confidence,
                        weight=weight,
                        score=score,
                    )
                )

        # Check entity topic matches
        for topic in entity_topics:
            topic_normalized = topic.lower().strip()
            domain_topics_normalized = [t.lower().strip() for t in domain_topics]
            if (
                topic_normalized in all_doc_topics
                and topic_normalized not in domain_topics_normalized
            ):
                confidence = confidence_scores.get(
                    topic, 0.6
                )  # Medium confidence for entity topics
                weight = self.config.entity_topic_weight
                score = confidence * weight

                topic_type = self._determine_topic_type(
                    topic_normalized,
                    doc_breadcrumb_topics,
                    doc_content_topics,
                    doc_heading_topics,
                    doc_keyword_topics,
                    doc_entity_topics,
                    doc_primary_topics,
                )

                matches.append(
                    TopicMatch(
                        topic=topic,
                        topic_type=f"entity_{topic_type}",
                        confidence=confidence,
                        weight=weight,
                        score=score,
                    )
                )

        # Check other query topic matches
        for topic in query_topics:
            topic_normalized = topic.lower().strip()
            domain_topics_normalized = [t.lower().strip() for t in domain_topics]
            entity_topics_normalized = [t.lower().strip() for t in entity_topics]
            if (
                topic_normalized in all_doc_topics
                and topic_normalized not in domain_topics_normalized
                and topic_normalized not in entity_topics_normalized
            ):
                confidence = confidence_scores.get(
                    topic, 0.5
                )  # Medium confidence for query topics
                weight = self.config.primary_topic_weight

                # Boost score for primary topics
                if topic_normalized in doc_primary_topics:
                    weight *= 1.5

                score = confidence * weight

                topic_type = self._determine_topic_type(
                    topic_normalized,
                    doc_breadcrumb_topics,
                    doc_content_topics,
                    doc_heading_topics,
                    doc_keyword_topics,
                    doc_entity_topics,
                    doc_primary_topics,
                )

                matches.append(
                    TopicMatch(
                        topic=topic,
                        topic_type=topic_type,
                        confidence=confidence,
                        weight=weight,
                        score=score,
                    )
                )

        return matches

    def _determine_topic_type(
        self,
        topic: str,
        breadcrumb_topics: Set[str],
        content_topics: Set[str],
        heading_topics: Set[str],
        keyword_topics: Set[str],
        entity_topics: Set[str],
        primary_topics: Set[str],
    ) -> str:
        """Determine the type of topic match in the document using actual RAG fields."""
        if topic in primary_topics:
            return "primary"
        elif topic in keyword_topics:
            return "keyword"
        elif topic in heading_topics:
            return "heading"
        elif topic in entity_topics:
            return "entity"
        elif topic in breadcrumb_topics:
            return "breadcrumb"
        elif topic in content_topics:
            return "content"
        else:
            return "general"

    def _calculate_breadcrumb_score(
        self, breadcrumb: str, query_topics: List[str], domain_topics: List[str]
    ) -> float:
        """Calculate breadcrumb relevance score."""
        if not breadcrumb:
            return 0.0

        breadcrumb_lower = breadcrumb.lower()
        score = 0.0

        # Check for domain topic matches in breadcrumb (high value)
        for topic in domain_topics:
            if topic.lower() in breadcrumb_lower:
                score += 2.0

        # Check for other query topic matches
        for topic in query_topics:
            if topic.lower() in breadcrumb_lower and topic not in domain_topics:
                score += 1.0

        return min(score, 5.0)  # Cap at 5.0

    def _calculate_title_score(
        self, title: str, query_topics: List[str], query_text: str
    ) -> float:
        """Calculate title relevance score."""
        if not title:
            return 0.0

        title_lower = title.lower()
        score = 0.0

        # Check for query topic matches in title
        for topic in query_topics:
            if topic.lower() in title_lower:
                score += 1.5

        # Check for direct query text matches
        if query_text:
            query_words = query_text.lower().split()
            for word in query_words:
                if len(word) > 2 and word in title_lower:
                    score += 1.0

        return min(score, 4.0)  # Cap at 4.0

    def _calculate_content_score(self, document: Document, query_text: str) -> float:
        """Calculate content relevance score."""
        if not query_text:
            return 0.0

        query_lower = query_text.lower()
        query_words = query_lower.split()
        score = 0.0
        total_content_length = 0

        for chunk in document.chunks:
            content_lower = chunk.content.lower()
            content_words = content_lower.split()
            total_content_length += len(content_words)

            # Calculate term frequency
            for word in query_words:
                if len(word) > 2:  # Skip short words
                    count = content_lower.count(word)
                    if count > 0:
                        # TF-IDF-like scoring
                        tf = count / len(content_words)
                        score += tf * 10  # Scale up for visibility

        # Normalize by total content length
        if total_content_length > 0:
            score = score / math.log(total_content_length + 1)

        return min(score, 3.0)  # Cap at 3.0

    def _calculate_confidence_bonus(
        self, topic_matches: List[TopicMatch], high_confidence_topics: List[str]
    ) -> float:
        """Calculate bonus score for high-confidence matches."""
        bonus = 0.0

        # Bonus for high-confidence topic matches
        high_confidence_matches = [
            match for match in topic_matches if match.topic in high_confidence_topics
        ]

        bonus += len(high_confidence_matches) * 0.5

        # Bonus for multiple topic matches (indicates strong relevance)
        if len(topic_matches) >= 3:
            bonus += 1.0
        elif len(topic_matches) >= 2:
            bonus += 0.5

        # Bonus for high average confidence
        if topic_matches:
            avg_confidence = sum(match.confidence for match in topic_matches) / len(
                topic_matches
            )
            if avg_confidence > 0.8:
                bonus += 1.0
            elif avg_confidence > 0.6:
                bonus += 0.5

        return min(bonus, 3.0)  # Cap at 3.0

    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-1 range using sigmoid function."""
        return 1 / (1 + math.exp(-score + 5))  # Sigmoid with offset

    def _generate_score_explanation(
        self,
        topic_score: float,
        breadcrumb_score: float,
        title_score: float,
        content_score: float,
        confidence_bonus: float,
    ) -> str:
        """Generate human-readable scoring explanation."""
        components = []

        if topic_score > 0:
            components.append(f"Topic: {topic_score:.2f}")
        if breadcrumb_score > 0:
            components.append(f"Breadcrumb: {breadcrumb_score:.2f}")
        if title_score > 0:
            components.append(f"Title: {title_score:.2f}")
        if content_score > 0:
            components.append(f"Content: {content_score:.2f}")
        if confidence_bonus > 0:
            components.append(f"Confidence: {confidence_bonus:.2f}")

        return " | ".join(components) if components else "No matches"

    def rank_documents(
        self, document_scores: List[DocumentScore], min_score: float = None
    ) -> List[DocumentScore]:
        """
        Rank documents by score with optional filtering.

        Args:
            document_scores: List of document scores
            min_score: Minimum score threshold for filtering

        Returns:
            Sorted list of document scores
        """
        # Filter by minimum score if specified
        if min_score is not None:
            document_scores = [
                score for score in document_scores if score.total_score >= min_score
            ]

        # Sort by total score (descending)
        document_scores.sort(key=lambda x: x.total_score, reverse=True)

        return document_scores

    def get_top_matches(
        self, document_scores: List[DocumentScore], max_results: int = None
    ) -> List[DocumentScore]:
        """Get top N document matches."""
        ranked_scores = self.rank_documents(document_scores)

        if max_results is not None:
            return ranked_scores[:max_results]

        return ranked_scores
