"""
Topic Extraction Module for RAG Pipeline
Extracts topics from both breadcrumbs and content for enhanced semantic understanding
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TopicExtractor:
    """
    Extracts topics from breadcrumbs and markdown content to enhance RAG semantic understanding
    """

    def __init__(self):
        # Domain-specific keywords derived from actual document corpus analysis
        # Generated from 479 RAG files with data-driven analysis
        self.domain_keywords = {
            "employee_management": [
                "employee",
                "employee admin",
                "employee s",
                "employee site",
                "employees",
                "manage employees page",
                "update employee data",
            ],
            "benefits_enrollment": [
                "annual enrollment",
                "annual enrollment setup",
                "annual enrollment timeline",
                "benefit",
                "benefits",
                "client re enrollment checklist",
                "coverage",
                "current plan",
                "enrollment",
                "future plan",
                "plan",
                "plan setup",
                "plan year",
                "plan years",
            ],
            "financial": [
                "payroll",
                "payroll code",
                "flex",
                "flex credits",
                "excess flex",
                "sales tax",
            ],
            "dates_deadlines": ["date", "effective date", "year", "year s", "end date"],
            "beneficiary_management": [
                "adding a beneficiary",
                "beneficiaries",
                "beneficiary",
                "beneficiary designation",
                "electronic beneficiary",
                "life event",
            ],
            "forms_documents": [
                "file",
                "information",
                "manage files",
                "template",
                "upload files",
            ],
            "administration": [
                "admin ref",
                "administration",
                "set",
                "setup",
                "configure account contributions",
            ],
            "life_insurance": ["life", "life event", "life insurance"],
            "new_hire_process": ["new hire", "new hire enrollment", "new hire welcome"],
            "system_technical": [
                "hris",
                "client specific",
                "configuration",
                "data",
                "system",
            ],
            "workflow_process": [
                "change",
                "current",
                "following",
                "options",
                "reduction",
                "status",
            ],
        }

        # Common stop words to exclude from topics
        self.stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "cannot",
            "a",
            "an",
        }

        # Extended stop words based on analysis of irrelevant topics
        self.extended_stop_words = {
            # Generic/Stop Words
            "then",
            "now",
            "here",
            "there",
            "all",
            "any",
            "some",
            "many",
            "most",
            "more",
            "other",
            "another",
            "each",
            "every",
            "both",
            "either",
            "neither",
            "one",
            "two",
            "three",
            "first",
            "second",
            "next",
            "last",
            "new",
            "old",
            "only",
            "also",
            "already",
            "always",
            "never",
            "usually",
            "normally",
            "typically",
            "often",
            "sometimes",
            "however",
            "therefore",
            "additionally",
            "furthermore",
            "although",
            "because",
            "since",
            "until",
            "while",
            "during",
            "within",
            "across",
            "over",
            "under",
            "above",
            "below",
            "before",
            "after",
            "again",
            "once",
            "twice",
            # UI/Technical Interface Terms
            "click",
            "button",
            "screen",
            "window",
            "page",
            "menu",
            "tab",
            "field",
            "form",
            "dialog",
            "browse",
            "scroll",
            "navigate",
            "apply",
            "reset",
            "refresh",
            "view",
            "edit",
            "choose",
            "enter",
            "fill",
            "check",
            "uncheck",
            "checked",
            "unchecked",
            "display",
            "show",
            "hide",
            "appear",
            "disappear",
            "icon",
            "image",
            "link",
            "url",
            "browser",
            "chrome",
            "firefox",
            "safari",
            "mobile",
            "desktop",
            "tablet",
            "app",
            "application",
            "platform",
            "system",
            "server",
            "database",
            "api",
            "web",
        }

        # Broken/Incomplete words to filter out
        self.broken_words = {
            "add-yea",
            "healtha",
            "healthd",
            "dependen",
            "continge",
            "peut",
            "aucun",
            "xxxx",
            "xxxxx",
            "testlastname",
            "john-test",
            "fname",
            "lname",
            "webst",
            "webs",
            "websincstage",
            "stdpay",
            "premiumadj",
            "ltdtb",
            "nemm",
            "wsf",
            "udb",
            "opa",
            "sked",
        }

        # Technical artifacts/IDs patterns
        self.technical_patterns = [
            r"^[a-f0-9]{8,}$",  # Hash IDs like "b974121b", "d2a27cb4"
            r"^chunk_\d+_[a-f0-9]+$",  # Chunk IDs like "chunk_0_b974121b"
            r"^[a-z]{2,4}\d+$",  # Technical codes
            r"^[cdmy]{4,8}$",  # Date format codes like "ccyymmdd", "ddmmyyyy"
            r"^record\s+format.*length\s+\d+$",  # Technical specifications
            r"^header\s+file\s+format.*length\s+\d+$",  # Header specifications
        ]

        # Combine all stop words
        self.all_stop_words = self.stop_words.union(self.extended_stop_words)

        # Patterns for extracting important terms
        self.important_patterns = [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Proper nouns
            r"\b\d{3}[kK]\b",  # 401k variations
            r"\b[A-Z]{2,}\b",  # Acronyms
            r"\b(?:Plan|Setup|Management|Service|System|Window|Page)\b",  # Important nouns
        ]

    def extract_topics_from_breadcrumb(self, breadcrumb: str) -> Dict[str, List[str]]:
        """
        Extract topics from breadcrumb navigation

        Args:
            breadcrumb: Breadcrumb string like "Setup > Plan Setup > Add Year"

        Returns:
            Dictionary with breadcrumb topics
        """
        if not breadcrumb or breadcrumb.strip() == "":
            return {"breadcrumb_topics": [], "breadcrumb_hierarchy": []}

        try:
            # Split breadcrumb into levels
            levels = [level.strip() for level in breadcrumb.split(">")]

            # Normalize each level to create topics
            breadcrumb_topics = []
            for level in levels:
                normalized = self._normalize_topic(level)
                if normalized:
                    breadcrumb_topics.append(normalized)

            return {
                "breadcrumb_topics": breadcrumb_topics,
                "breadcrumb_hierarchy": levels,
            }

        except Exception as e:
            logger.warning(
                f"Error extracting breadcrumb topics from '{breadcrumb}': {e}"
            )
            return {"breadcrumb_topics": [], "breadcrumb_hierarchy": []}

    def extract_topics_from_content(
        self, markdown_content: str
    ) -> Dict[str, List[str]]:
        """
        Extract topics from markdown content optimized for vector store semantic search
        Focuses on domain categories and meaningful content terms rather than exhaustive extraction

        Args:
            markdown_content: The markdown content to analyze

        Returns:
            Dictionary with content-based topics optimized for metadata filtering
        """
        if not markdown_content:
            return {
                "content_topics": [],
                "heading_topics": [],
                "keyword_topics": [],
                "entity_topics": [],
                "content_terms": [],
            }

        try:
            # Extract topics from different sources
            heading_topics = self._extract_heading_topics(markdown_content)
            keyword_topics = self._extract_keyword_topics(markdown_content)
            entity_topics = self._extract_entity_topics(markdown_content)

            # NEW: Extract meaningful content terms for better chunk understanding
            content_terms = self._extract_content_terms(markdown_content)

            # Combine content topics with improved filtering
            all_content_topics = self._combine_and_filter_topics(
                heading_topics, keyword_topics, entity_topics, content_terms
            )

            return {
                "content_topics": all_content_topics,
                "heading_topics": heading_topics,
                "keyword_topics": keyword_topics,
                "entity_topics": entity_topics,
                "content_terms": content_terms,
            }

        except Exception as e:
            logger.warning(f"Error extracting content topics: {e}")
            return {
                "content_topics": [],
                "heading_topics": [],
                "keyword_topics": [],
                "entity_topics": [],
                "content_terms": [],
            }

    def extract_all_topics(
        self, breadcrumb: str, markdown_content: str
    ) -> Dict[str, any]:
        """
        Extract topics from both breadcrumb and content

        Args:
            breadcrumb: Breadcrumb navigation string
            markdown_content: Markdown content to analyze

        Returns:
            Complete topic analysis
        """
        # Extract from both sources
        breadcrumb_data = self.extract_topics_from_breadcrumb(breadcrumb)
        content_data = self.extract_topics_from_content(markdown_content)

        # Combine topics
        all_topics = list(
            set(breadcrumb_data["breadcrumb_topics"] + content_data["content_topics"])
        )

        # Determine primary topics (most relevant)
        primary_topics = self._get_primary_topics(
            breadcrumb_data["breadcrumb_topics"], content_data["content_topics"]
        )

        return {
            "breadcrumb_topics": breadcrumb_data["breadcrumb_topics"],
            "breadcrumb_hierarchy": breadcrumb_data["breadcrumb_hierarchy"],
            "content_topics": content_data["content_topics"],
            "heading_topics": content_data["heading_topics"],
            "keyword_topics": content_data["keyword_topics"],
            "entity_topics": content_data["entity_topics"],
            "all_topics": all_topics,
            "primary_topics": primary_topics,
            "topic_count": len(all_topics),
        }

    def _extract_heading_topics(self, markdown_content: str) -> List[str]:
        """Extract topics from markdown headings"""
        headings = []

        # Find all markdown headings
        heading_pattern = r"^#+\s+(.+)$"
        matches = re.findall(heading_pattern, markdown_content, re.MULTILINE)

        for heading in matches:
            # Clean heading text
            clean_heading = re.sub(r"[^\w\s]", " ", heading)
            words = clean_heading.lower().split()

            # Extract meaningful words
            for word in words:
                if len(word) > 2 and word not in self.stop_words:
                    normalized = self._normalize_topic(word)
                    if normalized:
                        headings.append(normalized)

        return list(set(headings))

    def _extract_keyword_topics(self, markdown_content: str) -> List[str]:
        """Extract topics based on domain keywords"""
        content_lower = markdown_content.lower()
        found_topics = []

        for topic, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    found_topics.append(topic)
                    break  # Found this topic, move to next

        return found_topics

    def _extract_entity_topics(self, markdown_content: str) -> List[str]:
        """Extract important entities and terms"""
        entities = []

        for pattern in self.important_patterns:
            matches = re.findall(pattern, markdown_content)
            for match in matches:
                normalized = self._normalize_topic(match)
                if normalized and len(normalized) > 2:
                    entities.append(normalized)

        # Remove duplicates and return most common
        entity_counts = Counter(entities)
        return [entity for entity, count in entity_counts.most_common(10)]

    def _normalize_topic(self, topic: str) -> str:
        """Normalize topic string to consistent format with comprehensive filtering"""
        if not topic:
            return ""

        # Clean the topic
        cleaned = re.sub(r"[^\w\s]", " ", topic)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()

        # Apply comprehensive filtering
        if not self._is_valid_topic(cleaned):
            return ""

        # Convert to kebab-case for consistency
        normalized = cleaned.replace(" ", "-")

        return normalized

    def _is_valid_topic(self, topic: str) -> bool:
        """
        Comprehensive validation to filter out irrelevant topics

        Args:
            topic: The topic string to validate

        Returns:
            True if topic is valid and relevant, False otherwise
        """
        if not topic or len(topic.strip()) <= 2:
            return False

        topic_clean = topic.strip().lower()

        # Filter out stop words (including extended set)
        if topic_clean in self.all_stop_words:
            return False

        # Filter out broken/incomplete words
        if topic_clean in self.broken_words:
            return False

        # Filter out technical artifacts using patterns
        for pattern in self.technical_patterns:
            if re.match(pattern, topic_clean):
                return False

        # Filter out very short words (less than 3 characters)
        if len(topic_clean) < 3:
            return False

        # Filter out purely numeric strings
        if topic_clean.isdigit():
            return False

        # Filter out single characters or very short abbreviations that aren't meaningful
        if len(topic_clean) <= 3 and not self._is_meaningful_abbreviation(topic_clean):
            return False

        # Filter out topics that are mostly punctuation or special characters
        if len(re.sub(r"[a-zA-Z0-9]", "", topic_clean)) > len(topic_clean) * 0.5:
            return False

        return True

    def _is_meaningful_abbreviation(self, abbrev: str) -> bool:
        """
        Check if a short string is a meaningful business abbreviation

        Args:
            abbrev: The abbreviation to check

        Returns:
            True if it's a meaningful business abbreviation
        """
        meaningful_abbrevs = {
            "api",
            "csv",
            "xml",
            "pdf",
            "rtf",
            "hsa",
            "gst",
            "pst",
            "hst",
            "ytd",
            "ltd",
            "std",
            "eoi",
            "cob",
            "nem",
            "aws",
            "sso",
            "pin",
            "udb",
            "tfsa",
            "rrsp",
            "hris",
            "adp",
            "dfs",
        }
        return abbrev.lower() in meaningful_abbrevs

    def _get_primary_topics(
        self, breadcrumb_topics: List[str], content_topics: List[str]
    ) -> List[str]:
        """Determine the most relevant topics from both sources"""
        # Breadcrumb topics get higher priority as they represent navigation context
        primary = []

        # Add all breadcrumb topics (they're usually most relevant)
        primary.extend(breadcrumb_topics)

        # Add top content topics that aren't already covered
        for topic in content_topics[:5]:  # Top 5 content topics
            if topic not in primary:
                primary.append(topic)

        return primary[:8]  # Limit to 8 primary topics

    def extract_query_topics(self, user_query: str) -> Dict[str, Any]:
        """
        Enhanced query topic extraction using comprehensive methods with confidence scoring

        Args:
            user_query: User's question or query

        Returns:
            Dictionary with extracted topics and confidence scores
        """
        if not user_query:
            return {
                "query_topics": [],
                "domain_topics": [],
                "entity_topics": [],
                "normalized_topics": [],
                "high_confidence_topics": [],
                "confidence_scores": {},
                "topic_count": 0,
            }

        # Method 1: Domain keyword matching (high confidence)
        domain_topics = self._extract_domain_topics_with_confidence(user_query)

        # Method 2: Entity extraction using existing patterns
        entity_topics = self._extract_query_entities(user_query)

        # Method 3: Normalized important terms with basic filtering
        normalized_topics = self._extract_normalized_query_terms(user_query)

        # Method 4: Combine with confidence scoring
        result = self._combine_query_topics_with_confidence(
            domain_topics, entity_topics, normalized_topics
        )

        return result

    def _extract_domain_topics_with_confidence(
        self, user_query: str
    ) -> Dict[str, float]:
        """Extract domain topics with confidence scores"""
        domain_matches = {}
        query_lower = user_query.lower()

        for topic, keywords in self.domain_keywords.items():
            max_confidence = 0.0
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    # Higher confidence for longer, more specific keywords
                    confidence = 0.9 if len(keyword) > 5 else 0.8
                    # Boost confidence for exact word boundaries
                    if re.search(
                        r"\b" + re.escape(keyword.lower()) + r"\b", query_lower
                    ):
                        confidence += 0.1
                    max_confidence = max(max_confidence, confidence)

            if max_confidence > 0:
                domain_matches[topic] = min(max_confidence, 1.0)

        return domain_matches

    def _extract_query_entities(self, user_query: str) -> List[str]:
        """Extract entities from query using existing important patterns"""
        entities = []

        # Use the same patterns as document processing
        for pattern in self.important_patterns:
            matches = re.findall(pattern, user_query)
            for match in matches:
                normalized = self._normalize_topic(match)
                if normalized and len(normalized) > 2:
                    entities.append(normalized)

        # Remove duplicates while preserving order and limit to prevent noise
        seen = set()
        filtered_entities = []
        for entity in entities:
            if entity not in seen and entity not in self.stop_words:
                seen.add(entity)
                filtered_entities.append(entity)

        return filtered_entities[:8]  # Limit to top 8 entities

    def _extract_normalized_query_terms(self, user_query: str) -> List[str]:
        """Extract normalized terms from query with basic filtering"""
        words = re.findall(r"\b\w+\b", user_query.lower())
        normalized_terms = []

        # Basic query stop words (common question words)
        query_stop_words = self.stop_words.union(
            {
                "how",
                "what",
                "where",
                "when",
                "why",
                "who",
                "which",
                "can",
                "could",
                "should",
                "would",
                "do",
                "does",
                "did",
                "will",
                "need",
                "want",
                "help",
            }
        )

        for word in words:
            if (
                len(word) > 3 and word not in query_stop_words and not word.isdigit()
            ):  # Simple filters only
                normalized = self._normalize_topic(word)
                if normalized:
                    normalized_terms.append(normalized)

        return list(set(normalized_terms))

    def _combine_query_topics_with_confidence(
        self,
        domain_topics: Dict[str, float],
        entity_topics: List[str],
        normalized_topics: List[str],
    ) -> Dict[str, Any]:
        """Combine all topics with confidence scoring"""

        # High confidence topics (domain matches with high scores)
        high_confidence = [
            topic for topic, score in domain_topics.items() if score >= 0.8
        ]

        # All domain topics
        all_domain_topics = list(domain_topics.keys())

        # Combine all unique topics
        all_topics = list(set(all_domain_topics + entity_topics + normalized_topics))

        # Create confidence scores for all topics
        confidence_scores = domain_topics.copy()
        for topic in entity_topics:
            confidence_scores[topic] = 0.7  # Medium confidence for entities
        for topic in normalized_topics:
            confidence_scores[topic] = 0.5  # Lower confidence for normalized terms

        # Sort by confidence score (highest first) and limit to prevent noise
        prioritized_topics = sorted(
            all_topics, key=lambda t: confidence_scores.get(t, 0.0), reverse=True
        )[
            :10
        ]  # Limit to top 10 topics

        return {
            "query_topics": prioritized_topics,
            "domain_topics": all_domain_topics,
            "entity_topics": entity_topics,
            "normalized_topics": normalized_topics,
            "high_confidence_topics": high_confidence,
            "confidence_scores": confidence_scores,
            "topic_count": len(all_topics),
        }

    def get_topic_statistics(self, all_documents_topics: List[Dict]) -> Dict:
        """
        Generate statistics about topics across all documents

        Args:
            all_documents_topics: List of topic dictionaries from all documents

        Returns:
            Topic statistics and insights
        """
        all_topics = []
        breadcrumb_topics = []
        content_topics = []

        for doc_topics in all_documents_topics:
            all_topics.extend(doc_topics.get("all_topics", []))
            breadcrumb_topics.extend(doc_topics.get("breadcrumb_topics", []))
            content_topics.extend(doc_topics.get("content_topics", []))

        return {
            "total_unique_topics": len(set(all_topics)),
            "most_common_topics": Counter(all_topics).most_common(20),
            "most_common_breadcrumb_topics": Counter(breadcrumb_topics).most_common(10),
            "most_common_content_topics": Counter(content_topics).most_common(10),
            "topic_distribution": {
                "total_topics": len(all_topics),
                "breadcrumb_topics": len(breadcrumb_topics),
                "content_topics": len(content_topics),
            },
        }

    def get_chunk_topics(self, chunk_content: str, breadcrumb: str) -> Dict[str, any]:
        """
        Get topics for a specific chunk based on its content and breadcrumb
        Enhanced with content-based filtering to ensure topic relevance

        Args:
            chunk_content: The markdown content of the specific chunk
            breadcrumb: Breadcrumb navigation string for the document

        Returns:
            Topics dictionary for the chunk with filtered relevant topics
        """
        try:
            if not chunk_content:
                logger.warning("No content provided for chunk topic extraction")
                return {
                    "breadcrumb_topics": [],
                    "breadcrumb_hierarchy": [],
                    "content_topics": [],
                    "heading_topics": [],
                    "keyword_topics": [],
                    "entity_topics": [],
                    "all_topics": [],
                    "primary_topics": [],
                    "topic_count": 0,
                }

            # Extract topics specifically for this chunk's content
            raw_topics = self.extract_all_topics(breadcrumb, chunk_content)

            # Apply content-based filtering to ensure relevance
            filtered_topics = self._filter_topics_by_content_relevance(
                raw_topics, chunk_content
            )

            logger.info(
                f"Extracted {filtered_topics.get('topic_count', 0)} relevant topics for chunk"
            )
            return filtered_topics

        except Exception as e:
            logger.error(f"Error extracting topics for chunk: {e}")
            return {
                "breadcrumb_topics": [],
                "breadcrumb_hierarchy": [],
                "content_topics": [],
                "heading_topics": [],
                "keyword_topics": [],
                "entity_topics": [],
                "all_topics": [],
                "primary_topics": [],
                "topic_count": 0,
            }

    def _filter_topics_by_content_relevance(
        self, topics_dict: Dict[str, any], chunk_content: str
    ) -> Dict[str, any]:
        """
        Filter topics to only include those that are actually relevant to the chunk content

        Args:
            topics_dict: Raw topics extracted from breadcrumb and content
            chunk_content: The actual chunk content to validate against

        Returns:
            Filtered topics dictionary with only relevant topics
        """
        if not chunk_content:
            return topics_dict

        content_lower = chunk_content.lower()

        # Helper function to check if a topic is mentioned in content
        def is_topic_relevant(topic: str) -> bool:
            if not topic:
                return False

            # Clean topic for matching (remove hyphens, underscores)
            clean_topic = topic.replace("-", " ").replace("_", " ")

            # Check for exact matches
            if clean_topic.lower() in content_lower:
                return True

            # Check for partial word matches for compound topics
            topic_words = clean_topic.split()
            if len(topic_words) > 1:
                # For compound topics, require at least 2 words to be present
                matches = sum(
                    1 for word in topic_words if word.lower() in content_lower
                )
                return matches >= 2
            else:
                # For single words, check word boundaries to avoid false positives
                return bool(
                    re.search(
                        r"\b" + re.escape(clean_topic.lower()) + r"\b", content_lower
                    )
                )

        # Filter each topic category
        filtered_breadcrumb_topics = [
            t for t in topics_dict.get("breadcrumb_topics", []) if is_topic_relevant(t)
        ]
        filtered_content_topics = [
            t for t in topics_dict.get("content_topics", []) if is_topic_relevant(t)
        ]
        filtered_heading_topics = [
            t for t in topics_dict.get("heading_topics", []) if is_topic_relevant(t)
        ]
        filtered_keyword_topics = [
            t for t in topics_dict.get("keyword_topics", []) if is_topic_relevant(t)
        ]
        filtered_entity_topics = [
            t for t in topics_dict.get("entity_topics", []) if is_topic_relevant(t)
        ]

        # Combine all filtered topics
        all_filtered_topics = list(
            set(filtered_breadcrumb_topics + filtered_content_topics)
        )

        # Re-determine primary topics from filtered set
        filtered_primary_topics = self._get_primary_topics(
            filtered_breadcrumb_topics, filtered_content_topics
        )

        # If we have very few relevant topics, add some high-confidence breadcrumb topics
        # to maintain context, but only if they're semantically related
        if len(all_filtered_topics) < 3:
            for topic in topics_dict.get("breadcrumb_topics", []):
                if topic not in all_filtered_topics and len(all_filtered_topics) < 5:
                    # Add breadcrumb topics even if not directly mentioned
                    # as they provide important navigational context
                    all_filtered_topics.append(topic)
                    if topic not in filtered_primary_topics:
                        filtered_primary_topics.append(topic)

        return {
            "breadcrumb_topics": filtered_breadcrumb_topics,
            "breadcrumb_hierarchy": topics_dict.get("breadcrumb_hierarchy", []),
            "content_topics": filtered_content_topics,
            "heading_topics": filtered_heading_topics,
            "keyword_topics": filtered_keyword_topics,
            "entity_topics": filtered_entity_topics,
            "all_topics": all_filtered_topics,
            "primary_topics": filtered_primary_topics,
            "topic_count": len(all_filtered_topics),
        }

    def _calculate_topic_relevance_score(self, topic: str, content: str) -> float:
        """
        Calculate a relevance score for a topic based on its presence in content

        Args:
            topic: The topic to score
            content: The content to check against

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not topic or not content:
            return 0.0

        content_lower = content.lower()
        clean_topic = topic.replace("-", " ").replace("_", " ").lower()

        # Base score for exact match
        if clean_topic in content_lower:
            # Count occurrences for frequency scoring
            occurrences = content_lower.count(clean_topic)
            base_score = min(0.8 + (occurrences * 0.05), 1.0)

            # Bonus for word boundary matches (more precise)
            if re.search(r"\b" + re.escape(clean_topic) + r"\b", content_lower):
                base_score = min(base_score + 0.1, 1.0)

            return base_score

        # Partial scoring for compound topics
        topic_words = clean_topic.split()
        if len(topic_words) > 1:
            matches = sum(1 for word in topic_words if word in content_lower)
            return (matches / len(topic_words)) * 0.6  # Max 0.6 for partial matches

        return 0.0

    def _extract_content_terms(self, markdown_content: str) -> List[str]:
        """
        Extract meaningful content terms optimized for vector store metadata
        Focuses on domain-relevant terms that complement semantic search

        Args:
            markdown_content: The markdown content to analyze

        Returns:
            List of meaningful content terms for metadata filtering
        """
        if not markdown_content:
            return []

        # Remove markdown formatting and clean content
        clean_content = self._clean_content_for_analysis(markdown_content)

        # Extract meaningful terms using frequency and domain relevance
        content_terms = []

        # 1. Extract domain-relevant multi-word phrases
        domain_phrases = self._extract_domain_phrases(clean_content)
        content_terms.extend(domain_phrases)

        # 2. Extract important single terms with frequency filtering
        important_terms = self._extract_important_terms(clean_content)
        content_terms.extend(important_terms)

        # 3. Remove duplicates and filter by relevance
        filtered_terms = self._filter_content_terms(content_terms, clean_content)

        return filtered_terms[:10]  # Limit to top 10 most relevant terms

    def _clean_content_for_analysis(self, markdown_content: str) -> str:
        """Clean markdown content for term extraction"""
        # Remove markdown formatting
        clean_content = re.sub(r"[#*_`\[\]()]", " ", markdown_content)
        # Remove image references and links
        clean_content = re.sub(r"\[Image:.*?\]", " ", clean_content)
        clean_content = re.sub(r"https?://\S+", " ", clean_content)
        # Normalize whitespace
        clean_content = re.sub(r"\s+", " ", clean_content).strip()
        return clean_content

    def _extract_domain_phrases(self, content: str) -> List[str]:
        """Extract domain-specific multi-word phrases"""
        content_lower = content.lower()
        phrases = []

        # Define domain-specific phrase patterns
        domain_patterns = [
            r"\b(?:payroll|benefit|enrollment|employee)\s+(?:setup|configuration|management|process|data|file|export)\b",
            r"\b(?:annual|re-enrollment|plan)\s+(?:timeline|setup|year|period)\b",
            r"\b(?:life|health|dental)\s+(?:insurance|coverage|benefit|plan)\b",
            r"\b(?:flex|salary|deduction)\s+(?:credits|amounts|codes)\b",
            r"\b(?:data|file)\s+(?:formatting|export|import|processing)\b",
        ]

        for pattern in domain_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                normalized = self._normalize_topic(match)
                if normalized and len(normalized) > 3:
                    phrases.append(normalized)

        return list(set(phrases))

    def _extract_important_terms(self, content: str) -> List[str]:
        """Extract important single terms with frequency filtering"""
        words = re.findall(r"\b[a-zA-Z]{4,}\b", content.lower())

        # Filter out common UI and generic terms
        ui_terms = {
            "click",
            "button",
            "page",
            "window",
            "menu",
            "option",
            "field",
            "form",
            "table",
            "column",
            "section",
            "display",
            "show",
            "view",
            "open",
            "close",
            "save",
            "cancel",
            "next",
            "back",
            "previous",
        }

        # Count word frequencies
        word_counts = Counter(
            word
            for word in words
            if word not in self.stop_words and word not in ui_terms and len(word) > 3
        )

        # Extract terms that appear multiple times or are domain-relevant
        important_terms = []
        for word, count in word_counts.items():
            if count >= 2 or self._is_domain_relevant_term(word):
                normalized = self._normalize_topic(word)
                if normalized:
                    important_terms.append(normalized)

        return important_terms[:15]  # Limit to prevent noise

    def _is_domain_relevant_term(self, term: str) -> bool:
        """Check if a term is relevant to the business domain"""
        domain_terms = {
            "payroll",
            "employee",
            "benefit",
            "enrollment",
            "plan",
            "coverage",
            "insurance",
            "deduction",
            "salary",
            "flex",
            "credits",
            "timeline",
            "configuration",
            "setup",
            "administration",
            "management",
            "export",
            "import",
            "file",
            "data",
            "system",
            "hris",
            "carrier",
            "provider",
        }
        return term.lower() in domain_terms

    def _filter_content_terms(self, terms: List[str], content: str) -> List[str]:
        """Filter content terms by relevance and avoid duplication"""
        if not terms:
            return []

        # Score terms by frequency and domain relevance
        term_scores = {}
        content_lower = content.lower()

        for term in terms:
            score = 0.0
            clean_term = term.replace("-", " ").replace("_", " ")

            # Frequency score
            frequency = content_lower.count(clean_term.lower())
            score += min(frequency * 0.2, 1.0)

            # Domain relevance score
            if self._is_domain_relevant_term(clean_term):
                score += 0.5

            # Length bonus for compound terms
            if len(clean_term.split()) > 1:
                score += 0.3

            term_scores[term] = score

        # Sort by score and return top terms
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        return [term for term, score in sorted_terms if score > 0.3]

    def _combine_and_filter_topics(
        self,
        heading_topics: List[str],
        keyword_topics: List[str],
        entity_topics: List[str],
        content_terms: List[str],
    ) -> List[str]:
        """
        Combine and filter topics from different sources with priority weighting
        Optimized for vector store metadata rather than exhaustive extraction
        """
        # Priority weighting: domain keywords > content terms > headings > entities
        combined_topics = []

        # Add domain keywords (highest priority for metadata filtering)
        combined_topics.extend(keyword_topics)

        # Add meaningful content terms
        for term in content_terms:
            if term not in combined_topics:
                combined_topics.append(term)

        # Add heading topics if not already covered
        for topic in heading_topics:
            if topic not in combined_topics and len(combined_topics) < 15:
                combined_topics.append(topic)

        # Add selective entity topics (avoid generic terms)
        filtered_entities = [
            e
            for e in entity_topics
            if self._is_domain_relevant_term(e) and e not in combined_topics
        ]
        combined_topics.extend(filtered_entities[:5])  # Limit entity topics

        return self._post_process_topics(
            combined_topics[:12]
        )  # Reasonable limit for metadata

    def _post_process_topics(self, topics: List[str]) -> List[str]:
        """
        Final post-processing to remove any remaining irrelevant topics

        Args:
            topics: List of topics to clean

        Returns:
            Cleaned list of relevant topics
        """
        if not topics:
            return []

        cleaned_topics = []

        for topic in topics:
            # Apply final validation
            if self._is_valid_topic(topic):
                # Additional checks for compound topics
                if self._is_meaningful_compound_topic(topic):
                    cleaned_topics.append(topic)

        # Remove duplicates while preserving order
        seen = set()
        final_topics = []
        for topic in cleaned_topics:
            if topic not in seen:
                seen.add(topic)
                final_topics.append(topic)

        return final_topics

    def _is_meaningful_compound_topic(self, topic: str) -> bool:
        """
        Check if a compound topic (with hyphens) is meaningful

        Args:
            topic: The topic to check

        Returns:
            True if the compound topic is meaningful
        """
        if not topic or "-" not in topic:
            return True  # Single word topics already validated

        # Split compound topic and check each part
        parts = topic.split("-")

        # Filter out compound topics where any part is irrelevant
        for part in parts:
            if part in self.all_stop_words or part in self.broken_words:
                return False

        # Require at least one part to be domain-relevant for compound topics
        domain_relevant_parts = sum(
            1 for part in parts if self._is_domain_relevant_term(part)
        )

        # For compound topics, require at least one domain-relevant part
        return domain_relevant_parts > 0 or len(parts) <= 2
