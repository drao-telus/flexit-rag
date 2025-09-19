"""
Domain Keyword Analysis Tool
Analyzes the actual document corpus to derive domain-specific keywords
instead of using hardcoded assumptions
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import argparse


class DomainKeywordAnalyzer:
    """
    Analyzes actual document content to extract domain-specific keywords
    """

    def __init__(self):
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
            "you",
            "your",
            "we",
            "our",
            "they",
            "their",
            "it",
            "its",
            "he",
            "his",
            "she",
            "her",
            "i",
            "my",
            "me",
            "us",
            "them",
            "him",
            "if",
            "when",
            "where",
            "how",
            "what",
            "why",
            "who",
            "which",
            "than",
            "then",
            "now",
            "here",
            "there",
            "all",
            "any",
            "each",
            "every",
            "some",
            "many",
            "much",
            "more",
            "most",
            "other",
            "another",
            "such",
            "only",
            "just",
            "also",
            "even",
            "still",
            "back",
            "out",
            "down",
            "off",
            "over",
            "under",
            "again",
            "further",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
        }

        # Common technical/UI words to filter out
        self.technical_words = {
            "click",
            "button",
            "page",
            "screen",
            "window",
            "tab",
            "menu",
            "link",
            "field",
            "box",
            "dropdown",
            "checkbox",
            "radio",
            "select",
            "option",
            "text",
            "enter",
            "type",
            "save",
            "cancel",
            "ok",
            "yes",
            "no",
            "next",
            "previous",
            "back",
            "forward",
            "home",
            "help",
            "search",
            "filter",
            "sort",
            "view",
            "edit",
            "delete",
            "add",
            "remove",
            "update",
            "refresh",
            "load",
            "loading",
            "submit",
            "form",
            "table",
            "row",
            "column",
            "cell",
            "header",
            "footer",
            "sidebar",
            "content",
            "section",
            "div",
            "span",
            "image",
            "icon",
            "logo",
            "banner",
            "navigation",
            "nav",
            "breadcrumb",
        }

    def analyze_rag_files(self, rag_directory: str) -> Dict:
        """
        Analyze all RAG JSON files to extract domain keywords

        Args:
            rag_directory: Path to directory containing RAG JSON files

        Returns:
            Analysis results with extracted keywords and patterns
        """
        rag_path = Path(rag_directory)
        if not rag_path.exists():
            raise FileNotFoundError(f"RAG directory not found: {rag_directory}")

        print(f"Analyzing RAG files in: {rag_path}")

        # Collect all content
        all_content = []
        all_breadcrumbs = []
        file_count = 0

        for rag_file in rag_path.glob("*_rag.json"):
            try:
                with open(rag_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract content from chunks
                for chunk in data.get("chunks", []):
                    content = chunk.get("content", "")
                    if content:
                        all_content.append(content)

                # Extract breadcrumb
                breadcrumb = data.get("breadcrumb", "")
                if breadcrumb:
                    all_breadcrumbs.append(breadcrumb)

                file_count += 1

            except Exception as e:
                print(f"Error processing {rag_file}: {e}")

        print(f"Processed {file_count} RAG files")

        # Analyze the content
        results = {
            "file_count": file_count,
            "content_analysis": self._analyze_content(all_content),
            "breadcrumb_analysis": self._analyze_breadcrumbs(all_breadcrumbs),
            "domain_keywords": {},
            "suggested_categories": [],
        }

        # Generate domain keywords based on analysis
        results["domain_keywords"] = self._generate_domain_keywords(results)
        results["suggested_categories"] = self._suggest_categories(results)

        return results

    def _analyze_content(self, all_content: List[str]) -> Dict:
        """Analyze content to find common terms and patterns"""

        # Combine all content
        combined_content = " ".join(all_content).lower()

        # Extract words and phrases
        words = re.findall(r"\b[a-z]+\b", combined_content)

        # Filter out stop words and technical words
        meaningful_words = [
            word
            for word in words
            if len(word) > 2
            and word not in self.stop_words
            and word not in self.technical_words
        ]

        # Count word frequencies
        word_counts = Counter(meaningful_words)

        # Extract multi-word phrases (2-3 words)
        phrases_2 = self._extract_phrases(combined_content, 2)
        phrases_3 = self._extract_phrases(combined_content, 3)

        # Find domain-specific patterns
        patterns = self._find_domain_patterns(combined_content)

        return {
            "total_words": len(words),
            "unique_words": len(set(words)),
            "meaningful_words": len(meaningful_words),
            "top_words": word_counts.most_common(50),
            "top_2word_phrases": phrases_2.most_common(30),
            "top_3word_phrases": phrases_3.most_common(20),
            "domain_patterns": patterns,
        }

    def _analyze_breadcrumbs(self, all_breadcrumbs: List[str]) -> Dict:
        """Analyze breadcrumbs to understand navigation patterns"""

        # Extract all breadcrumb levels
        all_levels = []
        breadcrumb_patterns = []

        for breadcrumb in all_breadcrumbs:
            levels = [level.strip() for level in breadcrumb.split(">")]
            all_levels.extend(levels)
            breadcrumb_patterns.append(" > ".join(levels))

        # Normalize levels
        normalized_levels = []
        for level in all_levels:
            # Clean and normalize
            clean_level = re.sub(r"[^\w\s]", " ", level.lower())
            clean_level = re.sub(r"\s+", " ", clean_level).strip()
            if clean_level and len(clean_level) > 2:
                normalized_levels.append(clean_level)

        level_counts = Counter(normalized_levels)
        pattern_counts = Counter(breadcrumb_patterns)

        return {
            "total_breadcrumbs": len(all_breadcrumbs),
            "total_levels": len(all_levels),
            "unique_levels": len(set(normalized_levels)),
            "top_levels": level_counts.most_common(30),
            "top_patterns": pattern_counts.most_common(20),
        }

    def _extract_phrases(self, text: str, word_count: int) -> Counter:
        """Extract n-word phrases from text"""
        words = re.findall(r"\b[a-z]+\b", text)
        phrases = []

        for i in range(len(words) - word_count + 1):
            phrase_words = words[i : i + word_count]

            # Skip if contains stop words or technical words
            if any(
                word in self.stop_words or word in self.technical_words
                for word in phrase_words
            ):
                continue

            phrase = " ".join(phrase_words)
            if len(phrase) > 5:  # Minimum phrase length
                phrases.append(phrase)

        return Counter(phrases)

    def _find_domain_patterns(self, text: str) -> Dict:
        """Find domain-specific patterns in the text"""
        patterns = {}

        # Find numbers with context (like 401k, percentages, etc.)
        number_patterns = re.findall(r"\b\d+[a-z]*\b|\b\d+%\b|\b\$\d+\b", text)
        patterns["numbers"] = Counter(number_patterns).most_common(20)

        # Find capitalized terms (likely proper nouns or important terms)
        cap_terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        patterns["capitalized_terms"] = Counter(cap_terms).most_common(30)

        # Find acronyms
        acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
        patterns["acronyms"] = Counter(acronyms).most_common(20)

        # Find terms ending in common suffixes
        suffixes = ["ment", "tion", "sion", "ance", "ence", "ity", "ing", "ed"]
        suffix_terms = []
        for suffix in suffixes:
            terms = re.findall(rf"\b\w+{suffix}\b", text)
            suffix_terms.extend(terms)
        patterns["suffix_terms"] = Counter(suffix_terms).most_common(30)

        return patterns

    def _generate_domain_keywords(self, analysis_results: Dict) -> Dict:
        """Generate domain keyword categories based on analysis"""

        content_analysis = analysis_results["content_analysis"]
        breadcrumb_analysis = analysis_results["breadcrumb_analysis"]

        # Get top words and phrases
        top_words = [word for word, count in content_analysis["top_words"][:30]]
        top_phrases = [
            phrase for phrase, count in content_analysis["top_2word_phrases"][:20]
        ]
        top_levels = [level for level, count in breadcrumb_analysis["top_levels"][:20]]

        # Group related terms into categories
        domain_keywords = defaultdict(list)

        # Analyze and categorize terms
        all_terms = top_words + top_phrases + top_levels

        # Define semantic groups based on actual data
        semantic_groups = self._create_semantic_groups(all_terms)

        return dict(semantic_groups)

    def _create_semantic_groups(self, terms: List[str]) -> Dict[str, List[str]]:
        """Create semantic groups from actual terms found in data"""

        groups = defaultdict(set)

        # Define patterns for grouping
        grouping_patterns = {
            "employee_management": [
                "employee",
                "staff",
                "personnel",
                "worker",
                "user",
                "member",
            ],
            "benefits_enrollment": [
                "enrollment",
                "enroll",
                "benefit",
                "plan",
                "coverage",
                "insurance",
            ],
            "administration": [
                "admin",
                "administrator",
                "management",
                "setup",
                "configuration",
            ],
            "forms_documents": ["form", "document", "application", "template", "file"],
            "dates_deadlines": ["date", "deadline", "year", "annual", "period", "time"],
            "financial": [
                "401k",
                "retirement",
                "savings",
                "pension",
                "payroll",
                "cost",
            ],
            "health_medical": [
                "health",
                "medical",
                "healthcare",
                "wellness",
                "dependent",
            ],
            "system_technical": ["system", "data", "report", "screen", "page", "tab"],
            "process_workflow": ["process", "workflow", "step", "guide", "instruction"],
            "compliance_audit": [
                "audit",
                "compliance",
                "tracking",
                "log",
                "record",
                "history",
            ],
        }

        # Categorize terms
        for term in terms:
            term_lower = term.lower()
            categorized = False

            for category, keywords in grouping_patterns.items():
                for keyword in keywords:
                    if keyword in term_lower or term_lower in keyword:
                        groups[category].add(term)
                        categorized = True
                        break
                if categorized:
                    break

            # If not categorized, create a general category
            if not categorized and len(term) > 3:
                groups["general"].add(term)

        # Convert sets to lists and filter small groups
        result = {}
        for category, terms_set in groups.items():
            if len(terms_set) >= 2:  # Only keep categories with at least 2 terms
                result[category] = sorted(list(terms_set))

        return result

    def _suggest_categories(self, analysis_results: Dict) -> List[str]:
        """Suggest main categories based on analysis"""

        domain_keywords = analysis_results["domain_keywords"]

        # Rank categories by number of terms and frequency
        category_scores = {}
        for category, terms in domain_keywords.items():
            category_scores[category] = len(terms)

        # Sort by score and return top categories
        sorted_categories = sorted(
            category_scores.items(), key=lambda x: x[1], reverse=True
        )

        return [category for category, score in sorted_categories[:10]]

    def save_analysis(self, results: Dict, output_file: str):
        """Save analysis results to JSON file"""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Analysis saved to: {output_file}")

    def print_summary(self, results: Dict):
        """Print a summary of the analysis"""
        print("\n" + "=" * 60)
        print("DOMAIN KEYWORD ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"Files analyzed: {results['file_count']}")
        print(f"Total unique words: {results['content_analysis']['unique_words']:,}")
        print(f"Meaningful words: {results['content_analysis']['meaningful_words']:,}")
        print(
            f"Unique breadcrumb levels: {results['breadcrumb_analysis']['unique_levels']}"
        )

        print("\nTOP DOMAIN CATEGORIES:")
        for i, category in enumerate(results["suggested_categories"][:5], 1):
            terms = results["domain_keywords"].get(category, [])
            print(f"{i}. {category}: {len(terms)} terms")
            print(f"   Examples: {', '.join(terms[:5])}")

        print("\nTOP WORDS FROM ACTUAL DATA:")
        for word, count in results["content_analysis"]["top_words"][:10]:
            print(f"  {word}: {count}")

        print("\nTOP PHRASES FROM ACTUAL DATA:")
        for phrase, count in results["content_analysis"]["top_2word_phrases"][:10]:
            print(f"  '{phrase}': {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze domain keywords from RAG files"
    )
    parser.add_argument(
        "--rag-dir",
        default="crawler/result_data/rag_output",
        help="Directory containing RAG JSON files",
    )
    parser.add_argument(
        "--output",
        default="crawler/domain_keyword_analysis.json",
        help="Output file for analysis results",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, do not save detailed results",
    )

    args = parser.parse_args()

    try:
        # Create analyzer
        analyzer = DomainKeywordAnalyzer()

        # Run analysis
        print("Starting domain keyword analysis...")
        results = analyzer.analyze_rag_files(args.rag_dir)

        # Print summary
        analyzer.print_summary(results)

        # Save results if requested
        if not args.summary_only:
            analyzer.save_analysis(results, args.output)
            print(f"\nDetailed analysis saved to: {args.output}")
            print(
                "You can use this data to update the domain_keywords in topic_extractor.py"
            )

        # Print suggested domain keywords for easy copy-paste
        print("\n" + "=" * 60)
        print("SUGGESTED DOMAIN KEYWORDS FOR topic_extractor.py:")
        print("=" * 60)
        print("Replace the hardcoded domain_keywords with:")
        print()
        print("self.domain_keywords = {")
        for category, terms in results["domain_keywords"].items():
            if len(terms) > 1:  # Only show categories with multiple terms
                print(f'    "{category}": {terms},')
        print("}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
