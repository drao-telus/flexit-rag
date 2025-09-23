"""Extensible chunking framework with size-based strategies and overlap support."""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk_content(
        self, markdown_content: str, semantic_text: str, metadata: Dict
    ) -> List[Dict]:
        """Chunk content according to strategy."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class NoChunkingStrategy(ChunkingStrategy):
    """Strategy for small files that don't need chunking."""

    def chunk_content(
        self, markdown_content: str, semantic_text: str, metadata: Dict
    ) -> List[Dict]:
        """Return single chunk with complete content."""
        return [
            {
                "chunk_index": 0,
                "total_chunks": 1,
                "text": semantic_text,
                "md_content": markdown_content,
                "chunk_type": "complete_document",
                "section_title": metadata.get("page_title", ""),
                "overlap_info": {"has_overlap": False},
            }
        ]

    def get_strategy_name(self) -> str:
        return "no_chunking"


class SemanticSectionStrategy(ChunkingStrategy):
    """Strategy for medium files - chunk by semantic sections with overlap."""

    def __init__(
        self,
        max_chunk_size: int = 800,
        overlap_size: int = 100,
        min_chunk_size: int = 100,
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = (
            min_chunk_size  # Minimum chunk size to prevent tiny chunks
        )

    def chunk_content(
        self, markdown_content: str, semantic_text: str, metadata: Dict
    ) -> List[Dict]:
        """Chunk by semantic sections with contextual overlap."""
        # Use raw markdown as text for text-embedding-3-large
        clean_text = self._clean_markdown_for_embedding(markdown_content)

        # Extract heading structure inline for section boundaries
        headings = self._extract_headings_inline(markdown_content)

        if not headings:
            # No headings - use text-based chunking
            return self._chunk_by_text_blocks(markdown_content, clean_text, metadata)

        # Chunk by sections defined by headings
        return self._chunk_by_sections(markdown_content, clean_text, metadata, headings)

    def _clean_markdown_for_embedding(self, markdown_content: str) -> str:
        """Clean markdown content for text-embedding-3-large (minimal processing)."""
        # text-embedding-3-large handles markdown well, so minimal cleaning
        # Just normalize whitespace and remove excessive line breaks
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", markdown_content.strip())
        return text

    def _extract_headings_inline(self, markdown_content: str) -> List[Dict]:
        """Extract heading structure inline using regex."""
        lines = markdown_content.split("\n")
        headings = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith("#"):
                match = re.match(r"^(#+)\s*(.+)", line_stripped)
                if match:
                    level = len(match.group(1))
                    text = match.group(2).strip()
                    headings.append({"level": level, "text": text, "line_number": i})

        return headings

    def _chunk_by_sections(
        self,
        markdown_content: str,
        clean_text: str,
        metadata: Dict,
        headings: List[Dict],
    ) -> List[Dict]:
        """Chunk content by heading sections with semantic merging for tiny chunks."""
        lines = markdown_content.split("\n")
        raw_chunks = []

        # First pass: Create initial chunks based on headings
        for i, heading in enumerate(headings):
            # Determine section boundaries
            start_line = heading["line_number"]
            end_line = (
                headings[i + 1]["line_number"] if i + 1 < len(headings) else len(lines)
            )

            # Extract section content
            section_lines = lines[start_line:end_line]
            section_md = "\n".join(section_lines).strip()

            if not section_md:
                continue

            # Use cleaned markdown as text for embedding (optimized for text-embedding-3-large)
            section_text = self._clean_markdown_for_embedding(section_md)

            # Check if section needs further chunking based on word count
            word_count = len(section_text.split())
            if word_count <= self.max_chunk_size:
                # Section fits in one chunk
                chunk = self._create_section_chunk(
                    section_md, section_text, heading, i, len(headings), metadata
                )
                raw_chunks.append(chunk)
            else:
                # Section too large - split further with improved logic
                sub_chunks = self._split_large_section_improved(
                    section_md, section_text, heading, metadata
                )
                raw_chunks.extend(sub_chunks)

        # Second pass: Apply semantic merging for tiny chunks
        merged_chunks = self._apply_semantic_merging(raw_chunks)

        # Third pass: Add overlaps and finalize
        final_chunks = self._add_overlaps_and_finalize(merged_chunks)

        return final_chunks

    def _chunk_by_text_blocks(
        self, markdown_content: str, semantic_text: str, metadata: Dict
    ) -> List[Dict]:
        """Fallback chunking by text blocks when no clear sections."""
        words = semantic_text.split()
        chunks = []

        i = 0
        while i < len(words):
            # Determine chunk size
            chunk_end = min(i + self.max_chunk_size, len(words))
            chunk_words = words[i:chunk_end]

            # Add overlap from previous chunk
            if i > 0:
                overlap_words = words[max(0, i - self.overlap_size) : i]
                chunk_words = overlap_words + chunk_words

            chunk_text = " ".join(chunk_words)

            # Find corresponding markdown content (approximate)
            chunk_md = self._extract_corresponding_md(
                markdown_content, chunk_text, i == 0
            )

            chunk = {
                "chunk_index": len(chunks),
                "total_chunks": 0,  # Will be updated later
                "text": chunk_text,
                "md_content": chunk_md,
                "chunk_type": "text_block",
                "section_title": f"Block {len(chunks) + 1}",
                "overlap_info": {
                    "has_overlap": i > 0,
                    "overlap_source": "previous_block" if i > 0 else None,
                },
            }

            chunks.append(chunk)
            i += self.max_chunk_size

        # Update total chunks
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)

        return chunks

    def _create_section_chunk(
        self,
        section_md: str,
        section_semantic: str,
        heading: Dict,
        section_index: int,
        total_sections: int,
        metadata: Dict,
    ) -> Dict:
        """Create a chunk for a complete section."""
        return {
            "chunk_index": section_index,
            "total_chunks": total_sections,
            "text": section_semantic,
            "md_content": section_md,
            "chunk_type": "section_based",
            "section_title": heading["text"],
            "section_level": heading["level"],
            "overlap_info": {"has_overlap": False},
        }

    def _split_large_section(
        self, section_md: str, section_semantic: str, heading: Dict, metadata: Dict
    ) -> List[Dict]:
        """Split a large section into smaller chunks while preserving complete content."""
        # For large sections, especially tables, preserve the complete markdown content
        # instead of splitting it artificially

        # Check if this is a table-heavy section
        if self._is_section_table_heavy(section_md):
            # For table sections, keep the complete content in one chunk
            # The table structure should not be broken
            chunk = {
                "chunk_index": 0,
                "total_chunks": 1,
                "text": section_semantic,
                "md_content": section_md,
                "chunk_type": "large_table_section",
                "section_title": heading["text"],
                "section_level": heading["level"],
                "overlap_info": {"has_overlap": False},
            }
            return [chunk]

        # For non-table large sections, split by semantic boundaries
        words = section_semantic.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk_end = min(i + self.max_chunk_size, len(words))
            chunk_words = words[i:chunk_end]

            # Add overlap between sub-chunks
            if i > 0:
                overlap_words = words[max(0, i - self.overlap_size) : i]
                chunk_words = overlap_words + chunk_words

            chunk_text = " ".join(chunk_words)

            # For large sections, include the complete markdown with heading
            # to maintain context, rather than splitting the markdown
            chunk_md = f"{'#' * heading['level']} {heading['text']}\n\n{section_md}"

            chunk = {
                "chunk_index": len(chunks),
                "total_chunks": 0,  # Will be updated
                "text": chunk_text,
                "md_content": chunk_md,
                "chunk_type": "section_subsection",
                "section_title": f"{heading['text']} (Part {len(chunks) + 1})",
                "section_level": heading["level"],
                "overlap_info": {
                    "has_overlap": i > 0,
                    "overlap_source": "previous_subsection" if i > 0 else None,
                },
            }

            chunks.append(chunk)
            i += self.max_chunk_size

        return chunks

    def _is_section_table_heavy(self, section_md: str) -> bool:
        """Check if a section is table-heavy."""
        lines = section_md.split("\n")
        table_lines = [line for line in lines if line.strip().startswith("|")]
        return len(table_lines) > len(lines) * 0.5  # 50% table content

    def _get_overlap_text(self, previous_text: str, overlap_size: int) -> str:
        """Get overlap text from previous chunk."""
        words = previous_text.split()
        if len(words) <= overlap_size:
            return previous_text
        return " ".join(words[-overlap_size:])

    def _get_smart_overlap(self, previous_text: str, overlap_size: int) -> str:
        """Get smart overlap text that respects sentence boundaries."""
        if not previous_text:
            return ""

        # Try to find sentence boundaries for better overlap
        sentences = re.split(r"[.!?]+\s+", previous_text)
        if len(sentences) <= 1:
            # Fallback to word-based overlap
            return self._get_overlap_text(previous_text, overlap_size)

        # Get last few sentences that fit within overlap size
        overlap_sentences = []
        word_count = 0

        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= overlap_size:
                overlap_sentences.insert(0, sentence.strip())
                word_count += sentence_words
            else:
                break

        if overlap_sentences:
            return ". ".join(overlap_sentences) + "."
        else:
            # Fallback to word-based overlap if sentences are too long
            return self._get_overlap_text(previous_text, overlap_size)

    def _split_large_section_improved(
        self, section_md: str, section_text: str, heading: Dict, metadata: Dict
    ) -> List[Dict]:
        """Split a large section with improved logic for better boundaries."""
        # Check if this is a table-heavy section
        if self._is_section_table_heavy(section_md):
            # For table sections, keep the complete content in one chunk
            chunk = {
                "chunk_index": 0,
                "total_chunks": 1,
                "text": section_text,
                "md_content": section_md,
                "chunk_type": "large_table_section",
                "section_title": heading["text"],
                "section_level": heading["level"],
                "overlap_info": {"has_overlap": False},
            }
            return [chunk]

        # For non-table sections, try to split at logical boundaries
        chunks = []

        # Try to split by sub-headings first
        sub_headings = self._find_sub_headings(section_md, heading["level"])

        if sub_headings:
            # Split by sub-headings
            chunks = self._split_by_sub_headings(
                section_md, section_text, heading, sub_headings, metadata
            )
        else:
            # Split by paragraphs or sentences
            chunks = self._split_by_content_boundaries(
                section_md, section_text, heading, metadata
            )

        return chunks

    def _find_sub_headings(self, section_md: str, parent_level: int) -> List[Dict]:
        """Find sub-headings within a section."""
        lines = section_md.split("\n")
        sub_headings = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith("#"):
                match = re.match(r"^(#+)\s*(.+)", line_stripped)
                if match:
                    level = len(match.group(1))
                    if level > parent_level:  # Only sub-headings
                        text = match.group(2).strip()
                        sub_headings.append(
                            {"level": level, "text": text, "line_number": i}
                        )

        return sub_headings

    def _split_by_sub_headings(
        self,
        section_md: str,
        section_text: str,
        heading: Dict,
        sub_headings: List[Dict],
        metadata: Dict,
    ) -> List[Dict]:
        """Split section by sub-headings."""
        lines = section_md.split("\n")
        chunks = []

        # Add the main heading as the first boundary
        boundaries = [
            {"line_number": 0, "text": heading["text"], "level": heading["level"]}
        ] + sub_headings

        for i, boundary in enumerate(boundaries):
            start_line = boundary["line_number"]
            end_line = (
                boundaries[i + 1]["line_number"]
                if i + 1 < len(boundaries)
                else len(lines)
            )

            # Extract sub-section content
            sub_section_lines = lines[start_line:end_line]
            sub_section_md = "\n".join(sub_section_lines).strip()

            if not sub_section_md:
                continue

            sub_section_text = self._clean_markdown_for_embedding(sub_section_md)
            word_count = len(sub_section_text.split())

            if word_count <= self.max_chunk_size:
                # Sub-section fits in one chunk
                chunk = {
                    "chunk_index": len(chunks),
                    "total_chunks": 0,  # Will be updated
                    "text": sub_section_text,
                    "md_content": sub_section_md,
                    "chunk_type": "section_subsection",
                    "section_title": boundary["text"],
                    "section_level": boundary["level"],
                    "overlap_info": {"has_overlap": False},
                }
                chunks.append(chunk)
            else:
                # Sub-section still too large, split by content boundaries
                sub_chunks = self._split_by_content_boundaries(
                    sub_section_md, sub_section_text, boundary, metadata
                )
                chunks.extend(sub_chunks)

        # Update chunk indices and add overlaps
        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i
            chunk["total_chunks"] = len(chunks)

            # Add overlap from previous chunk
            if i > 0:
                overlap_text = self._get_smart_overlap(
                    chunks[i - 1]["text"], self.overlap_size
                )
                if overlap_text:
                    chunk["text"] = f"{overlap_text}\n\n{chunk['text']}"
                    chunk["overlap_info"]["has_overlap"] = True
                    chunk["overlap_info"]["overlap_source"] = "previous_subsection"

        return chunks

    def _split_by_content_boundaries(
        self, section_md: str, section_text: str, heading: Dict, metadata: Dict
    ) -> List[Dict]:
        """Split section by content boundaries (paragraphs, lists, etc.)."""
        # Split by double line breaks (paragraphs)
        paragraphs = re.split(r"\n\s*\n", section_md)
        chunks = []
        current_chunk_paras = []
        current_word_count = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_word_count = len(para.split())

            # Check if adding this paragraph would exceed chunk size
            if (
                current_word_count + para_word_count > self.max_chunk_size
                and current_chunk_paras
            ):
                # Create chunk from current paragraphs
                chunk_md = "\n\n".join(current_chunk_paras)
                chunk_text = self._clean_markdown_for_embedding(chunk_md)

                chunk = {
                    "chunk_index": len(chunks),
                    "total_chunks": 0,  # Will be updated
                    "text": chunk_text,
                    "md_content": chunk_md,
                    "chunk_type": "section_subsection",
                    "section_title": f"{heading['text']} (Part {len(chunks) + 1})",
                    "section_level": heading["level"],
                    "overlap_info": {"has_overlap": False},
                }
                chunks.append(chunk)

                # Start new chunk
                current_chunk_paras = [para]
                current_word_count = para_word_count
            else:
                # Add paragraph to current chunk
                current_chunk_paras.append(para)
                current_word_count += para_word_count

        # Add final chunk if there are remaining paragraphs
        if current_chunk_paras:
            chunk_md = "\n\n".join(current_chunk_paras)
            chunk_text = self._clean_markdown_for_embedding(chunk_md)

            chunk = {
                "chunk_index": len(chunks),
                "total_chunks": 0,  # Will be updated
                "text": chunk_text,
                "md_content": chunk_md,
                "chunk_type": "section_subsection",
                "section_title": f"{heading['text']} (Part {len(chunks) + 1})",
                "section_level": heading["level"],
                "overlap_info": {"has_overlap": False},
            }
            chunks.append(chunk)

        # Update chunk indices and add overlaps
        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i
            chunk["total_chunks"] = len(chunks)

            # Add overlap from previous chunk
            if i > 0:
                overlap_text = self._get_smart_overlap(
                    chunks[i - 1]["text"], self.overlap_size
                )
                if overlap_text:
                    chunk["text"] = f"{overlap_text}\n\n{chunk['text']}"
                    chunk["overlap_info"]["has_overlap"] = True
                    chunk["overlap_info"]["overlap_source"] = "previous_subsection"

        return chunks

    def _extract_corresponding_md(
        self, markdown_content: str, chunk_text: str, is_first: bool
    ) -> str:
        """Extract corresponding markdown content for a text chunk (approximate)."""
        # This is a simplified approach - in practice, you might want more sophisticated mapping
        lines = markdown_content.split("\n")

        if is_first:
            # Return first portion of markdown
            return "\n".join(lines[: len(lines) // 3])
        else:
            # Return middle or end portion
            return "\n".join(lines[len(lines) // 3 :])

    def _apply_semantic_merging(self, raw_chunks: List[Dict]) -> List[Dict]:
        """Apply semantic merging to combine tiny chunks with adjacent content."""
        if not raw_chunks:
            return raw_chunks

        merged_chunks = []
        i = 0

        while i < len(raw_chunks):
            current_chunk = raw_chunks[i].copy()
            current_word_count = len(current_chunk["text"].split())

            # Check if current chunk is too small
            if current_word_count < self.min_chunk_size:
                # Try to merge with next chunk(s) while maintaining semantic coherence
                merged_chunk = self._merge_with_next_chunks(raw_chunks, i)
                merged_chunks.append(merged_chunk)

                # Skip the chunks that were merged
                i += merged_chunk.get("_merged_count", 1)
            else:
                # Chunk is large enough, keep as is
                merged_chunks.append(current_chunk)
                i += 1

        return merged_chunks

    def _merge_with_next_chunks(self, chunks: List[Dict], start_idx: int) -> Dict:
        """Merge small chunk with semantically related next chunks."""
        merged_chunk = chunks[start_idx].copy()
        merged_text = merged_chunk["text"]
        merged_md = merged_chunk["md_content"]
        merged_count = 1
        current_word_count = len(merged_text.split())

        # Look ahead to find chunks to merge
        for i in range(start_idx + 1, len(chunks)):
            next_chunk = chunks[i]
            next_word_count = len(next_chunk["text"].split())

            # Check if merging would exceed max chunk size
            if current_word_count + next_word_count > self.max_chunk_size:
                break

            # Check semantic compatibility
            if self._are_chunks_semantically_compatible(merged_chunk, next_chunk):
                # Merge the chunks
                merged_text += f"\n\n{next_chunk['text']}"
                merged_md += f"\n\n{next_chunk['md_content']}"
                current_word_count += next_word_count
                merged_count += 1

                # Update section title to reflect merged content
                if merged_chunk["section_title"] != next_chunk["section_title"]:
                    merged_chunk["section_title"] = (
                        f"{merged_chunk['section_title']} & {next_chunk['section_title']}"
                    )

                # If we've reached a good size, stop merging
                if current_word_count >= self.min_chunk_size:
                    break
            else:
                # Not semantically compatible, stop merging
                break

        # Update merged chunk
        merged_chunk["text"] = merged_text
        merged_chunk["md_content"] = merged_md
        merged_chunk["chunk_type"] = (
            "merged_section" if merged_count > 1 else merged_chunk["chunk_type"]
        )
        merged_chunk["_merged_count"] = merged_count

        # Add merge info to overlap_info
        if merged_count > 1:
            merged_chunk["overlap_info"]["has_merge"] = True
            merged_chunk["overlap_info"]["merged_sections"] = merged_count

        return merged_chunk

    def _are_chunks_semantically_compatible(self, chunk1: Dict, chunk2: Dict) -> bool:
        """Check if two chunks are semantically compatible for merging."""
        # Check heading levels - prefer merging chunks at similar levels
        level1 = chunk1.get("section_level", 1)
        level2 = chunk2.get("section_level", 1)

        # Allow merging if:
        # 1. Same heading level (sibling sections)
        # 2. Second chunk is a sub-section of first (level2 > level1)
        # 3. Level difference is not too large (max 2 levels)
        if abs(level1 - level2) > 2:
            return False

        # Check chunk types - prefer merging similar types
        type1 = chunk1.get("chunk_type", "")
        type2 = chunk2.get("chunk_type", "")

        # Compatible chunk types
        compatible_types = {
            "section_based": ["section_based", "section_subsection"],
            "section_subsection": ["section_based", "section_subsection"],
            "large_table_section": [
                "large_table_section"
            ],  # Tables should stay separate
        }

        if type1 in compatible_types:
            if type2 not in compatible_types[type1]:
                return False

        # Additional semantic checks could be added here
        # For now, we use structural compatibility
        return True

    def _add_overlaps_and_finalize(self, merged_chunks: List[Dict]) -> List[Dict]:
        """Add overlaps between chunks and finalize chunk indices."""
        if not merged_chunks:
            return merged_chunks

        final_chunks = []

        for i, chunk in enumerate(merged_chunks):
            final_chunk = chunk.copy()

            # Add overlap from previous chunk if appropriate
            if i > 0 and not final_chunk["overlap_info"].get("has_overlap", False):
                prev_chunk = final_chunks[i - 1]

                # Only add overlap if current chunk has room and isn't already merged
                current_word_count = len(final_chunk["text"].split())
                if current_word_count < self.max_chunk_size - self.overlap_size:
                    overlap_text = self._get_smart_overlap(
                        prev_chunk["text"], self.overlap_size
                    )
                    if overlap_text:
                        final_chunk["text"] = f"{overlap_text}\n\n{final_chunk['text']}"
                        final_chunk["overlap_info"]["has_overlap"] = True
                        final_chunk["overlap_info"][
                            "overlap_source"
                        ] = "previous_section"

            # Update chunk indices
            final_chunk["chunk_index"] = i
            final_chunk["total_chunks"] = len(merged_chunks)

            # Clean up temporary merge tracking
            if "_merged_count" in final_chunk:
                del final_chunk["_merged_count"]

            final_chunks.append(final_chunk)

        return final_chunks

    def _create_contextual_md(
        self, section_md: str, heading: Dict, part_index: int
    ) -> str:
        """Create contextual markdown that includes heading and relevant content."""
        lines = section_md.split("\n")

        # Always include the section heading
        heading_line = f"{'#' * heading['level']} {heading['text']}"

        # Include relevant portion of content
        content_lines = [line for line in lines if not line.strip().startswith("#")]

        # Approximate content distribution
        total_parts = max(
            1, len(" ".join(content_lines).split()) // self.max_chunk_size + 1
        )
        lines_per_part = max(1, len(content_lines) // total_parts)

        start_idx = part_index * lines_per_part
        end_idx = min((part_index + 1) * lines_per_part, len(content_lines))

        relevant_content = content_lines[start_idx:end_idx]

        return f"{heading_line}\n\n" + "\n".join(relevant_content)

    def get_strategy_name(self) -> str:
        return "semantic_section"


class TableSectionStrategy(ChunkingStrategy):
    """Strategy for large files with heavy table content."""

    def __init__(self, max_rows_per_chunk: int = 20, overlap_rows: int = 3):
        self.max_rows_per_chunk = max_rows_per_chunk
        self.overlap_rows = overlap_rows

    def chunk_content(
        self, markdown_content: str, semantic_text: str, metadata: Dict
    ) -> List[Dict]:
        """Chunk large tables by sections with header preservation."""
        # Detect if content is table-heavy
        if not self._is_table_heavy(markdown_content):
            # Fallback to section strategy
            fallback = SemanticSectionStrategy()
            return fallback.chunk_content(markdown_content, semantic_text, metadata)

        return self._chunk_table_content(markdown_content, semantic_text, metadata)

    def _is_table_heavy(self, markdown_content: str) -> bool:
        """Check if content is primarily tables."""
        lines = markdown_content.split("\n")
        table_lines = [line for line in lines if line.strip().startswith("|")]
        return len(table_lines) > len(lines) * 0.6  # 60% table content

    def _chunk_table_content(
        self, markdown_content: str, semantic_text: str, metadata: Dict
    ) -> List[Dict]:
        """Chunk table content preserving headers and structure."""
        lines = markdown_content.split("\n")
        chunks = []

        # Extract table sections
        table_sections = self._extract_table_sections(lines)

        for section in table_sections:
            if section["type"] == "table":
                table_chunks = self._chunk_single_table(section, metadata)
                chunks.extend(table_chunks)
            else:
                # Non-table content - use cleaned markdown as text
                section_md = "\n".join(section["lines"])
                section_text = self._clean_markdown_for_embedding(section_md)

                chunk = {
                    "chunk_index": len(chunks),
                    "total_chunks": 0,
                    "text": section_text,
                    "md_content": section_md,
                    "chunk_type": "content_section",
                    "section_title": section.get("title", "Content"),
                    "overlap_info": {"has_overlap": False},
                }
                chunks.append(chunk)

        # Update indices and add overlaps
        self._add_table_overlaps(chunks)

        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i
            chunk["total_chunks"] = len(chunks)

        return chunks

    def _extract_table_sections(self, lines: List[str]) -> List[Dict]:
        """Extract table and non-table sections."""
        sections = []
        current_section = {"type": "content", "lines": [], "title": ""}

        in_table = False
        table_headers = []

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith("|") and not self._is_table_separator(
                line_stripped
            ):
                if not in_table:
                    # Starting new table
                    if current_section["lines"]:
                        sections.append(current_section)

                    current_section = {"type": "table", "lines": [], "headers": []}
                    in_table = True

                    # Extract headers
                    table_headers = [
                        cell.strip() for cell in line_stripped.split("|")[1:-1]
                    ]
                    current_section["headers"] = table_headers

                current_section["lines"].append(line)

            elif self._is_table_separator(line_stripped):
                if in_table:
                    current_section["lines"].append(line)

            else:
                if in_table:
                    # End of table
                    sections.append(current_section)
                    current_section = {"type": "content", "lines": [], "title": ""}
                    in_table = False

                if line_stripped:
                    current_section["lines"].append(line)

                    # Extract title from headings
                    if line_stripped.startswith("#") and not current_section["title"]:
                        current_section["title"] = line_stripped.lstrip("#").strip()

        # Add final section
        if current_section["lines"]:
            sections.append(current_section)

        return sections

    def _chunk_single_table(self, table_section: Dict, metadata: Dict) -> List[Dict]:
        """Chunk a single large table."""
        lines = table_section["lines"]
        headers = table_section.get("headers", [])

        # Find header and separator lines
        header_lines = []
        data_lines = []

        found_separator = False
        for line in lines:
            if self._is_table_separator(line.strip()):
                found_separator = True
                header_lines.append(line)
            elif not found_separator:
                header_lines.append(line)
            else:
                data_lines.append(line)

        # Chunk data rows
        chunks = []
        i = 0
        while i < len(data_lines):
            chunk_end = min(i + self.max_rows_per_chunk, len(data_lines))
            chunk_data = data_lines[i:chunk_end]

            # Add overlap from previous chunk
            if i > 0:
                overlap_start = max(0, i - self.overlap_rows)
                overlap_data = data_lines[overlap_start:i]
                chunk_data = overlap_data + chunk_data

            # Create chunk with headers
            chunk_lines = header_lines + chunk_data
            chunk_md = "\n".join(chunk_lines)
            chunk_text = self._clean_markdown_for_embedding(chunk_md)

            chunk = {
                "chunk_index": len(chunks),
                "total_chunks": 0,
                "text": chunk_text,
                "md_content": chunk_md,
                "chunk_type": "table_section",
                "section_title": f"Table Section {len(chunks) + 1}",
                "table_info": {
                    "start_row": i + 1,
                    "end_row": chunk_end,
                    "total_rows": len(data_lines),
                    "headers": headers,
                },
                "overlap_info": {
                    "has_overlap": i > 0,
                    "overlap_source": "previous_table_section" if i > 0 else None,
                },
            }

            chunks.append(chunk)
            i += self.max_rows_per_chunk

        return chunks

    def _is_table_separator(self, line: str) -> bool:
        """Check if line is a table separator."""
        return bool(re.match(r"^\|\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$", line))

    def _add_table_overlaps(self, chunks: List[Dict]) -> None:
        """Add overlaps between different table chunks."""
        for i in range(1, len(chunks)):
            if chunks[i - 1]["chunk_type"] != chunks[i]["chunk_type"]:
                # Different chunk types - add contextual overlap
                if len(chunks[i]["text"].split()) < 600:  # Only if there's room
                    overlap_text = self._get_overlap_text(chunks[i - 1]["text"], 50)
                    chunks[i]["text"] = f"{overlap_text} {chunks[i]['text']}"
                    chunks[i]["overlap_info"]["has_overlap"] = True
                    chunks[i]["overlap_info"]["overlap_source"] = "previous_section"

    def _clean_markdown_for_embedding(self, markdown_content: str) -> str:
        """Clean markdown content for text-embedding-3-large (minimal processing)."""
        # text-embedding-3-large handles markdown well, so minimal cleaning
        # Just normalize whitespace and remove excessive line breaks
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", markdown_content.strip())
        return text

    def _get_overlap_text(self, previous_text: str, overlap_size: int) -> str:
        """Get overlap text from previous chunk."""
        words = previous_text.split()
        if len(words) <= overlap_size:
            return previous_text
        return " ".join(words[-overlap_size:])

    def get_strategy_name(self) -> str:
        return "table_section"


class ChunkingStrategyManager:
    """Manager for selecting and applying chunking strategies."""

    def __init__(self):
        self.strategies = {
            "small": NoChunkingStrategy(),
            "medium": SemanticSectionStrategy(),
            "large_table": TableSectionStrategy(),
            "large_content": SemanticSectionStrategy(
                max_chunk_size=1000, overlap_size=150
            ),
        }

        # Size thresholds in bytes
        self.size_thresholds = {
            "small": 2048,  # 2KB
            "medium": 15360,  # 15KB
            "large": 30720,  # 30KB
        }

    def add_custom_strategy(self, name: str, strategy: ChunkingStrategy) -> None:
        """Add a custom chunking strategy."""
        self.strategies[name] = strategy

    def determine_strategy(self, content_size: int, markdown_content: str) -> str:
        """Determine the best chunking strategy based on content."""
        if content_size <= self.size_thresholds["small"]:
            return "small"
        elif content_size <= self.size_thresholds["medium"]:
            return "medium"
        else:
            # Check if it's table-heavy for large files
            if self._is_table_heavy(markdown_content):
                return "large_table"
            else:
                return "large_content"

    def _is_table_heavy(self, markdown_content: str) -> bool:
        """Check if content is table-heavy."""
        lines = markdown_content.split("\n")
        table_lines = [line for line in lines if line.strip().startswith("|")]
        return len(table_lines) > len(lines) * 0.6

    def get_strategy(self, markdown_content: str) -> ChunkingStrategy:
        """Get the appropriate chunking strategy for the content."""
        content_size = len(markdown_content.encode("utf-8"))
        strategy_name = self.determine_strategy(content_size, markdown_content)
        return self.strategies[strategy_name]

    def chunk_content(
        self, markdown_content: str, semantic_text: str, metadata: Dict
    ) -> List[Dict]:
        """Chunk content using the appropriate strategy."""
        content_size = len(markdown_content.encode("utf-8"))
        strategy_name = self.determine_strategy(content_size, markdown_content)

        strategy = self.strategies[strategy_name]
        chunks = strategy.chunk_content(markdown_content, semantic_text, metadata)

        # Add strategy info to chunks
        for chunk in chunks:
            chunk["chunking_strategy"] = strategy_name
            chunk["content_size"] = content_size

        return chunks

    def get_strategy_info(self) -> Dict:
        """Get information about available strategies."""
        return {
            "strategies": {
                name: strategy.get_strategy_name()
                for name, strategy in self.strategies.items()
            },
            "size_thresholds": self.size_thresholds,
        }
