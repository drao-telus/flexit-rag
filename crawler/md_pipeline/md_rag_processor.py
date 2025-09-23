"""
MD RAG Processor - Generates RAG documents from markdown content using chunking strategies.
Handles efficient processing without reading complete long files, recognizing common patterns.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flexit_llm.utility.topic_extractor import TopicExtractor

from crawler.md_pipeline.md_chunking_strategy import ChunkingStrategyManager
from crawler.url.url_mapper import URLMapper
from crawler.url.image_mapper import ImageMapper


@dataclass
class RAGChunk:
    """Represents a single RAG chunk with metadata."""

    chunk_id: str
    content: str  # Original markdown content
    cleaned_content: str  # Cleaned markdown content optimized for embeddings
    # chunk_type: str  # 'complete', 'section', 'table_section'
    # chunk_index: int
    # total_chunks: int
    # metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RAGDocument:
    """Represents a complete RAG document with all chunks."""

    document_id: str
    source_file: str
    title: str
    breadcrumb: str
    total_chunks: int
    processing_strategy: str
    chunks: List[RAGChunk]
    images: List[Dict[str, str]]
    metadata: Dict[str, Any]
    created_at: str
    page_url: str = ""  # Original page URL from page_url.py

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["chunks"] = [chunk.to_dict() for chunk in self.chunks]
        return result


class MDRAGProcessor:
    """
    Processes markdown content into RAG documents using intelligent chunking strategies.
    Designed to handle files efficiently without reading complete long files.
    """

    def __init__(
        self,
        enable_url_mapping: bool = True,
        enable_image_mapping: bool = True,
        base_url: str = "",
    ):
        self.chunking_manager = ChunkingStrategyManager()
        self.topic_extractor = TopicExtractor()
        self.url_mapper = None
        self.image_mapper = None

        # Initialize URL mapper if enabled
        if enable_url_mapping:
            try:
                self.url_mapper = URLMapper(
                    "crawler/url/page_url.py", base_url=base_url
                )
                # Build mapping cache to ensure fresh mappings
                result = self.url_mapper.build_mapping_cache()
                if result["success"]:
                    print(
                        f"URL mapping cache built: {result['unique_filenames']} mappings created"
                    )
                    if base_url:
                        print(f"Base URL configured: {base_url}")
                else:
                    print(
                        f"Warning: Failed to build URL mapping cache: {result.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                print(f"Warning: Could not initialize URL mapper: {e}")
                self.url_mapper = None

        # Initialize Image mapper if enabled
        if enable_image_mapping:
            try:
                self.image_mapper = ImageMapper(
                    "crawler/url/image_mapping_cache.json", base_url=base_url
                )
                # Build mapping cache to ensure fresh mappings
                result = self.image_mapper.build_mapping_cache()
                if result["success"]:
                    print(
                        f"Image mapping cache built: {result['total_images']} images mapped"
                    )
                else:
                    print(
                        f"Warning: Failed to build image mapping cache: {result.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                print(f"Warning: Could not initialize image mapper: {e}")
                self.image_mapper = None

    def process_markdown_to_rag(
        self,
        markdown_content: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RAGDocument:
        """
        Process markdown content into a RAG document.

        Args:
            markdown_content: The markdown content to process
            source_file: Path to the source file
            metadata: Optional metadata to include

        Returns:
            RAGDocument with processed chunks
        """
        if metadata is None:
            metadata = {}

        # Extract basic document info efficiently (pattern recognition)
        title, breadcrumb = self._extract_document_info(
            markdown_content, metadata.get("html_metadata")
        )

        # Extract topics from breadcrumb and content
        document_topics = self.topic_extractor.extract_all_topics(
            breadcrumb, markdown_content
        )

        # Get appropriate chunking strategy based on content size and patterns
        strategy = self.chunking_manager.get_strategy(markdown_content)

        # Process chunks using the selected strategy with markdown content directly
        # Use markdown content for both content and text processing
        chunk_dicts = strategy.chunk_content(
            markdown_content, markdown_content, metadata
        )

        # Resolve image information using image mapper
        images = self._resolve_image_info(source_file)

        # Generate document ID
        document_id = self._generate_document_id(source_file, markdown_content)

        # Add topic information to metadata
        metadata_with_topics = metadata.copy()
        metadata_with_topics["topics"] = document_topics

        # Convert chunk dictionaries to RAG format
        rag_chunks = []
        for i, chunk_dict in enumerate(chunk_dicts):
            rag_chunk = self._process_structured_chunk(
                chunk_dict,
                i,
                len(chunk_dicts),
                strategy.__class__.__name__,
                metadata_with_topics,
                document_topics,
            )
            rag_chunks.append(rag_chunk)

        # Resolve page URL using URL mapper
        page_url = self._resolve_page_url(source_file)

        # Create RAG document
        rag_document = RAGDocument(
            document_id=document_id,
            source_file=source_file,
            title=title,
            breadcrumb=breadcrumb,
            total_chunks=len(rag_chunks),
            processing_strategy=strategy.__class__.__name__,
            chunks=rag_chunks,
            images=images,
            metadata=metadata_with_topics,
            created_at=datetime.now().isoformat(),
            page_url=page_url,
        )

        return rag_document

    def _extract_document_info(
        self, markdown_content: str, html_metadata: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        Extract title and breadcrumb efficiently using pattern recognition.
        Prioritizes HTML metadata over markdown content patterns.
        """
        title = "Untitled"
        breadcrumb = ""

        # First, try to get breadcrumb from HTML metadata (most reliable)
        if html_metadata and "breadcrumbs" in html_metadata:
            breadcrumbs_list = html_metadata["breadcrumbs"]
            if breadcrumbs_list and isinstance(breadcrumbs_list, list):
                breadcrumb = " > ".join(breadcrumbs_list)

        # Try to get title from HTML metadata
        if html_metadata and "page_title" in html_metadata:
            page_title = html_metadata["page_title"]
            if (
                page_title
                and page_title.strip()
                and page_title.lower() not in ["untitled", "document", ""]
            ):
                title = page_title.strip()

        # Fallback to markdown content patterns if HTML metadata didn't provide good results
        if title == "Untitled" or not breadcrumb:
            lines = markdown_content.split("\n")

            # Common pattern: title is usually in first few lines
            for i, line in enumerate(lines[:10]):  # Only check first 10 lines
                line = line.strip()

                # Pattern: Breadcrumb usually starts with specific markers (fallback)
                if not breadcrumb and (
                    line.startswith("**Breadcrumb:**") or line.startswith("Breadcrumb:")
                ):
                    breadcrumb = line.split(":", 1)[1].strip()
                    continue

                # Pattern: Title is usually the first heading (fallback)
                if line.startswith("# ") and (not title or title == "Untitled"):
                    title = line[2:].strip()
                    continue

                # Pattern: Sometimes title is in bold at the start (fallback)
                if line.startswith("**") and line.endswith("**") and len(line) > 4:
                    if not title or title == "Untitled":
                        title = line[2:-2].strip()

                # Stop early if we found both
                if title != "Untitled" and breadcrumb:
                    break

        return title, breadcrumb

    def _process_structured_chunk(
        self,
        chunk_dict: Dict[str, Any],
        chunk_index: int,
        total_chunks: int,
        strategy_name: str,
        metadata: Dict[str, Any],
        document_topics: Optional[Dict[str, Any]] = None,
    ) -> RAGChunk:
        """Process a structured chunk dictionary into RAG format with dual content."""

        # Extract both original and cleaned content from chunk dictionary
        original_content = chunk_dict.get("md_content", "")
        cleaned_content = chunk_dict.get("text", original_content)
        chunk_type = chunk_dict.get("chunk_type", "section")

        # Generate chunk ID based on cleaned content (for consistency)
        chunk_id = self._generate_chunk_id(cleaned_content, chunk_index)

        # Merge chunk metadata with provided metadata
        chunk_metadata = {}
        chunk_metadata.update(metadata)

        # Extract chunk-specific topics using the chunk's actual content
        breadcrumb = metadata.get("breadcrumb", "")
        if (
            not breadcrumb
            and "html_metadata" in metadata
            and "breadcrumbs" in metadata["html_metadata"]
        ):
            breadcrumbs_list = metadata["html_metadata"]["breadcrumbs"]
            if breadcrumbs_list and isinstance(breadcrumbs_list, list):
                breadcrumb = " > ".join(breadcrumbs_list)

        return RAGChunk(
            chunk_id=chunk_id,
            content=original_content.strip(),  # Original markdown content
            cleaned_content=cleaned_content.strip(),  # Cleaned content optimized for embeddings
            # chunk_type=chunk_type,
            # chunk_index=chunk_index,
            # total_chunks=total_chunks,
            # metadata=chunk_metadata
        )

    def _determine_chunk_type(self, chunk: str, strategy_name: str) -> str:
        """Determine chunk type based on strategy and content patterns."""
        if strategy_name == "NoChunkingStrategy":
            return "complete"
        elif strategy_name == "TableSectionStrategy":
            # Pattern recognition for table content
            if "|" in chunk and chunk.count("|") > 5:
                return "table_section"
            return "section"
        else:
            return "section"

    def _extract_chunk_metadata(self, chunk: str) -> Dict[str, Any]:
        """Extract metadata from chunk using common patterns."""
        metadata = {}

        # Pattern: Count headings
        heading_count = chunk.count("\n#")
        if chunk.startswith("#"):
            heading_count += 1
        metadata["heading_count"] = heading_count

        # Pattern: Check for tables
        has_tables = "|" in chunk and chunk.count("|") > 2
        metadata["has_tables"] = has_tables

        # Pattern: Check for lists
        has_lists = any(
            line.strip().startswith(("- ", "* ", "+ ")) for line in chunk.split("\n")
        )
        metadata["has_lists"] = has_lists

        # Pattern: Check for images
        has_images = "[Image:" in chunk
        metadata["has_images"] = has_images

        # Pattern: Estimate reading time (common pattern)
        word_count = len(chunk.split())
        reading_time_minutes = max(1, word_count // 200)  # ~200 words per minute
        metadata["word_count"] = word_count
        metadata["estimated_reading_time"] = reading_time_minutes

        return metadata

    def _generate_document_id(self, source_file: str, content: str) -> str:
        """Generate a unique document ID."""
        # Use file path and content hash for uniqueness
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        file_name = Path(source_file).stem
        return f"{file_name}_{content_hash}"

    def _generate_chunk_id(self, chunk: str, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
        return f"chunk_{chunk_index}_{chunk_hash}"

    def _resolve_page_url(self, source_file: str) -> str:
        """
        Resolve the original page URL from the source file using URL mapper.

        Args:
            source_file: Path to the source HTML file

        Returns:
            Full URL (if base_url configured) or relative URL from page_url.py, or empty string if not found
        """
        if not self.url_mapper:
            return ""

        try:
            # Extract filename from source file path (without extension)
            source_path = Path(source_file)
            filename = source_path.stem

            # Get the full URL (includes base_url if configured) or relative URL
            page_url = self.url_mapper.get_full_url(filename)

            return page_url

        except Exception as e:
            print(f"Warning: Could not resolve page URL for {source_file}: {e}")
            return ""

    def _resolve_image_info(self, source_file: str) -> List[Dict[str, str]]:
        """
        Resolve image information from the source file using image mapper.

        Args:
            source_file: Path to the source HTML file

        Returns:
            List of image dictionaries with metadata, or empty list if not found
        """
        if not self.image_mapper:
            return []

        try:
            # Extract filename from source file path (without extension)
            source_path = Path(source_file)
            filename = source_path.stem

            # Get images for this filename from the image mapper
            images = self.image_mapper.get_images_for_filename(filename)

            return images

        except Exception as e:
            print(f"Warning: Could not resolve image info for {source_file}: {e}")
            return []

    def save_rag_document(self, rag_document: RAGDocument, output_path: str) -> None:
        """Save RAG document to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(rag_document.to_dict(), f, indent=2, ensure_ascii=False)

    def process_file_to_rag(
        self,
        markdown_file: str,
        output_file: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RAGDocument:
        """
        Process a markdown file to RAG document.
        Efficiently handles large files by reading in chunks if needed.
        """
        markdown_path = Path(markdown_file)

        # Check file size to determine reading strategy
        file_size = markdown_path.stat().st_size

        if file_size > 1024 * 1024:  # > 1MB
            # For very large files, read in chunks (pattern recognition)
            content = self._read_large_file_efficiently(markdown_path)
        else:
            # For normal files, read completely
            with open(markdown_path, "r", encoding="utf-8") as f:
                content = f.read()

        # Process to RAG
        rag_document = self.process_markdown_to_rag(
            content, str(markdown_file), metadata
        )

        # Save result
        self.save_rag_document(rag_document, output_file)

        return rag_document

    def _read_large_file_efficiently(self, file_path: Path) -> str:
        """
        Read large files efficiently by recognizing patterns.
        This avoids loading the complete file into memory at once.
        """
        content_parts = []

        with open(file_path, "r", encoding="utf-8") as f:
            # Read in chunks to avoid memory issues
            chunk_size = 8192  # 8KB chunks
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                content_parts.append(chunk)

        return "".join(content_parts)

    def batch_process_files(
        self, input_dir: str, output_dir: str, file_pattern: str = "*.md"
    ) -> List[RAGDocument]:
        """
        Batch process multiple markdown files.
        Efficiently handles multiple files using pattern recognition.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        processed_documents = []

        # Find all matching files
        markdown_files = list(input_path.glob(file_pattern))

        for md_file in markdown_files:
            try:
                # Generate output filename
                output_file = output_path / f"{md_file.stem}_rag.json"

                # Process file
                rag_doc = self.process_file_to_rag(str(md_file), str(output_file))

                processed_documents.append(rag_doc)

                print(f"Processed: {md_file.name} -> {output_file.name}")

            except Exception as e:
                print(f"Error processing {md_file.name}: {str(e)}")
                continue

        return processed_documents


# Utility functions for common patterns
def get_file_processing_stats(rag_document: RAGDocument) -> Dict[str, Any]:
    """Get processing statistics for a RAG document."""
    total_content_length = sum(len(chunk.content) for chunk in rag_document.chunks)

    return {
        "document_id": rag_document.document_id,
        "total_chunks": rag_document.total_chunks,
        "processing_strategy": rag_document.processing_strategy,
        "total_content_length": total_content_length,
        "average_chunk_size": (
            total_content_length // rag_document.total_chunks
            if rag_document.total_chunks > 0
            else 0
        ),
        "has_images": len(rag_document.images) > 0,
    }
