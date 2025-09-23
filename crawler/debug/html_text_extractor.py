"""
Text Extraction Utility for RAG Validation

This module provides robust text extraction from HTML files and RAG JSON outputs
for comparison and validation purposes.
"""

import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple


class TextExtractor:
    """Handles text extraction from HTML and RAG JSON files."""

    def __init__(self):
        self.html_parser = "html.parser"

    def extract_html_text(self, html_content: str) -> str:
        """
        Extract clean text from HTML content using BeautifulSoup.
        Specifically targets the div with id="mc-main-content" role="main".

        Args:
            html_content (str): Raw HTML content

        Returns:
            str: Cleaned text content from main content div
        """
        try:
            soup = BeautifulSoup(html_content, self.html_parser)

            # Find the main content div
            main_content_div = soup.find(
                "div", {"id": "mc-main-content", "role": "main"}
            )

            if not main_content_div:
                # Fallback: try to find div with just id="mc-main-content"
                main_content_div = soup.find("div", {"id": "mc-main-content"})

            if not main_content_div:
                print(
                    "Warning: Could not find main content div, extracting from entire document"
                )
                main_content_div = soup

            # Remove script and style elements from the main content
            for script in main_content_div(["script", "style"]):
                script.decompose()

            for img_caption in main_content_div.find_all("p", class_="figure"):
                img_caption.decompose()

            # Extract text with proper spacing preservation
            text = main_content_div.get_text(separator=" ", strip=True)

            # Normalize whitespace but preserve word boundaries
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

            return text

        except Exception as e:
            print(f"Error extracting HTML text: {e}")
            return ""

    def extract_rag_text(self, rag_json_content: Dict[str, Any]) -> str:
        """
        Extract and combine text from RAG JSON chunks.

        Args:
            rag_json_content (dict): Parsed RAG JSON content

        Returns:
            str: Combined text from all chunks
        """
        try:
            combined_text = ""

            # Handle chunks array
            if "chunks" in rag_json_content and isinstance(
                rag_json_content["chunks"], list
            ):
                for chunk in rag_json_content["chunks"]:
                    if "content" in chunk:
                        chunk_content = chunk["content"]

                        # Remove markdown formatting for comparison
                        chunk_content = self._clean_markdown(chunk_content)
                        combined_text += " " + chunk_content

            # Normalize whitespace
            combined_text = re.sub(r"\s+", " ", combined_text)
            combined_text = combined_text.strip()

            return combined_text

        except Exception as e:
            print(f"Error extracting RAG text: {e}")
            return ""

    def _clean_markdown(self, text: str) -> str:
        """
        Remove markdown formatting from text for comparison.

        Args:
            text (str): Text with markdown formatting

        Returns:
            str: Clean text without markdown
        """
        # Remove markdown headers
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

        # Remove markdown bold/italic
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)

        # Remove markdown links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove markdown images
        text = re.sub(r"\[Image:[^\]]+\]", "", text)

        # Remove markdown tables formatting
        text = re.sub(r"\|", " ", text)
        text = re.sub(r"^-+\s*$", "", text, flags=re.MULTILINE)

        # Remove markdown list markers
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def compare_texts(self, html_text: str, rag_text: str) -> Dict[str, Any]:
        """
        Compare HTML and RAG texts to find missing content.

        Args:
            html_text (str): Text extracted from HTML
            rag_text (str): Text extracted from RAG JSON

        Returns:
            dict: Comparison results with missing content analysis
        """
        # Convert to lowercase for case-insensitive comparison
        html_lower = html_text.lower()
        rag_lower = rag_text.lower()

        # Split into sentences for detailed analysis
        html_sentences = self._split_into_sentences(html_lower)
        rag_sentences = self._split_into_sentences(rag_lower)

        # Find missing sentences
        missing_sentences = []
        for sentence in html_sentences:
            if sentence.strip() and not self._is_sentence_covered(sentence, rag_lower):
                missing_sentences.append(sentence)

        # Calculate coverage metrics
        total_html_sentences = len([s for s in html_sentences if s.strip()])
        missing_count = len(missing_sentences)
        coverage_percentage = (
            ((total_html_sentences - missing_count) / total_html_sentences * 100)
            if total_html_sentences > 0
            else 0
        )

        return {
            "html_text_length": len(html_text),
            "rag_text_length": len(rag_text),
            "html_sentences_count": total_html_sentences,
            "missing_sentences_count": missing_count,
            "missing_sentences": missing_sentences,
            "coverage_percentage": round(coverage_percentage, 2),
            "is_complete": missing_count == 0,
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for detailed analysis."""
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_sentence_covered(self, sentence: str, rag_text: str) -> bool:
        """Check if a sentence is covered in the RAG text."""
        # Remove punctuation and extra spaces for better matching
        clean_sentence = re.sub(r"[^\w\s]", " ", sentence)
        clean_sentence = re.sub(r"\s+", " ", clean_sentence).strip()

        # Check if the core content is present
        words = clean_sentence.split()
        if len(words) < 3:  # Skip very short sentences
            return True

        # Check if most words are present in the RAG text
        word_matches = sum(1 for word in words if word in rag_text)
        match_ratio = word_matches / len(words)

        return match_ratio >= 0.7  # 70% of words should match
