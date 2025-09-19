"""Markdown semantic text processor for extracting clean text suitable for embeddings."""

import re
from typing import Dict, List, Tuple


class MDSemanticProcessor:
    """Processes Markdown content to extract semantic text for embeddings."""

    def __init__(self):
        self.semantic_markers = {
            "heading": "HEADING",
            "list_item": "ITEM",
            "ordered_item": "STEP",
            "table_header": "TABLE_HEADER",
            "table_cell": "TABLE_CELL",
            "emphasis": "EMPHASIS",
            "code": "CODE",
        }

    def extract_semantic_text(self, markdown_content: str) -> str:
        """Extract clean semantic text from Markdown content."""
        if not markdown_content:
            return ""

        # Process line by line to maintain structure
        lines = markdown_content.split("\n")
        semantic_parts = []

        in_table = False
        table_headers = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check if this line contains table headers (look for table pattern before separator)
            if not in_table and self._contains_table_header(line):
                # Extract headers from this line
                table_headers = self._extract_table_headers_from_line(line)

            # Track table state
            if self._is_table_separator(line):
                in_table = True
                # If we haven't found headers yet, look in the previous line
                if not table_headers and i > 0:
                    prev_line = lines[i - 1].strip()
                    if self._contains_table_header(prev_line):
                        table_headers = self._extract_table_headers_from_line(prev_line)
            elif in_table and not line.startswith("|"):
                in_table = False
                table_headers = []

            # Process different markdown elements
            processed_line = self._process_markdown_line(line, in_table, table_headers)

            if processed_line:
                semantic_parts.append(processed_line)

        return " ".join(semantic_parts)

    def _process_markdown_line(
        self, line: str, in_table: bool, table_headers: List[str]
    ) -> str:
        """Process a single markdown line to extract semantic meaning."""
        # Skip table separator lines
        if self._is_table_separator(line):
            return ""

        # Headings
        if line.startswith("#"):
            heading_text = self._process_heading(line)
            if "HEADING_LEVEL_1:" in heading_text:
                return heading_text + ". CONTENT:"

        # List items
        elif re.match(r"^\d+\.", line):
            return self._process_ordered_list_item(line)
        elif line.startswith("-") or line.startswith("*"):
            return self._process_unordered_list_item(line)

        # Table rows
        elif line.startswith("|"):
            return self._process_table_row(line, table_headers)

        # Regular text
        else:
            return self._process_regular_text(line)

    def _process_heading(self, line: str) -> str:
        """Process heading line."""
        # Extract heading level and text
        match = re.match(r"^(#+)\s*(.+)", line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            # Clean any remaining markdown
            clean_text = self._clean_markdown_formatting(text)
            return f"{self.semantic_markers['heading']}_LEVEL_{level}: {clean_text}"
        return ""

    def _process_ordered_list_item(self, line: str) -> str:
        """Process ordered list item."""
        # Remove number and extract content
        match = re.match(r"^\d+\.\s*(.+)", line)
        if match:
            content = match.group(1).strip()
            clean_content = self._clean_markdown_formatting(content)
            return f"{self.semantic_markers['ordered_item']}: {clean_content}"
        return ""

    def _process_unordered_list_item(self, line: str) -> str:
        """Process unordered list item."""
        # Remove bullet and extract content
        match = re.match(r"^[-*]\s*(.+)", line)
        if match:
            content = match.group(1).strip()

            # Check if this list item contains table headers
            if self._contains_table_header(content):
                # Extract the table headers and process them separately
                headers = self._extract_table_headers_from_line(content)

                # Split content into text part and table part
                table_match = re.search(r"(\|\s*[^|]+\s*\|\s*[^|]+\s*\|)", content)
                if table_match:
                    # Get text before the table
                    text_before = content[: table_match.start()].strip()

                    # Process headers
                    header_parts = []
                    for header in headers:
                        if header:
                            header_parts.append(
                                f"{self.semantic_markers['table_header']}: {header}"
                            )

                    # Combine text and headers
                    parts = []
                    if text_before:
                        clean_text = self._clean_markdown_formatting(text_before)
                        parts.append(
                            f"{self.semantic_markers['list_item']}: {clean_text}"
                        )

                    if header_parts:
                        parts.extend(header_parts)

                    return " ".join(parts)

            # Regular list item processing
            clean_content = self._clean_markdown_formatting(content)
            return f"{self.semantic_markers['list_item']}: {clean_content}"
        return ""

    def _process_table_row(self, line: str, table_headers: List[str]) -> str:
        """Process table row."""
        # Extract cells
        cells = [
            cell.strip() for cell in line.split("|")[1:-1]
        ]  # Remove empty first/last

        if not cells:
            return ""

        # If this is the first row and no headers set, treat as headers
        if not table_headers:
            processed_cells = []
            for cell in cells:
                clean_cell = self._clean_markdown_formatting(cell)
                if clean_cell:
                    processed_cells.append(
                        f"{self.semantic_markers['table_header']}: {clean_cell}"
                    )
            return " ".join(processed_cells)
        else:
            # Regular table row - all cells are data cells
            processed_cells = []
            for cell in cells:
                clean_cell = self._clean_markdown_formatting(cell)
                if clean_cell:
                    processed_cells.append(
                        f"{self.semantic_markers['table_cell']}: {clean_cell}"
                    )
            return " ".join(processed_cells)

    def _process_regular_text(self, line: str) -> str:
        """Process regular text line."""
        # Clean markdown formatting but preserve semantic meaning
        clean_text = self._clean_markdown_formatting(line)
        return clean_text if clean_text else ""

    def _clean_markdown_formatting(self, text: str) -> str:
        """Clean markdown formatting while preserving semantic content."""
        if not text:
            return ""

        # Remove image references but keep alt text
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"Image: \1", text)

        # Convert links to text with context
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

        # Remove bold/italic markers but keep text
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)

        # Remove code markers but keep content
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _is_table_separator(self, line: str) -> bool:
        """Check if line is a table separator (|---|---|)."""
        return bool(re.match(r"^\|\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$", line))

    def _extract_table_headers(self, line: str) -> List[str]:
        """Extract table headers from first table row."""
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        return [self._clean_markdown_formatting(cell) for cell in cells if cell.strip()]

    def _contains_table_header(self, line: str) -> bool:
        """Check if line contains table headers (| Header1 | Header2 |)."""
        # Skip table separator lines
        if self._is_table_separator(line):
            return False
        # Look for table pattern at the end of a line or as a standalone line
        return bool(re.search(r"\|\s*[^|]+\s*\|\s*[^|]+\s*\|", line))

    def _extract_table_headers_from_line(self, line: str) -> List[str]:
        """Extract table headers from any line that contains them."""
        # Find the table pattern in the line
        match = re.search(r"(\|\s*[^|]+\s*\|\s*[^|]+\s*\|)", line)
        if match:
            table_part = match.group(1)
            cells = [cell.strip() for cell in table_part.split("|")[1:-1]]
            return [
                self._clean_markdown_formatting(cell) for cell in cells if cell.strip()
            ]
        return []

    def get_content_statistics(self, markdown_content: str) -> Dict:
        """Get statistics about the markdown content."""
        if not markdown_content:
            return {
                "word_count": 0,
                "heading_count": 0,
                "list_item_count": 0,
                "table_count": 0,
                "image_count": 0,
                "link_count": 0,
            }

        lines = markdown_content.split("\n")

        stats = {
            "word_count": len(markdown_content.split()),
            "heading_count": len([l for l in lines if l.strip().startswith("#")]),
            "list_item_count": len(
                [
                    l
                    for l in lines
                    if re.match(r"^\s*[-*]\s+", l.strip())
                    or re.match(r"^\s*\d+\.\s+", l.strip())
                ]
            ),
            "table_count": len(
                [l for l in lines if self._is_table_separator(l.strip())]
            ),
            "image_count": len(re.findall(r"!\[([^\]]*)\]\([^)]*\)", markdown_content)),
            "link_count": len(re.findall(r"\[([^\]]+)\]\([^)]*\)", markdown_content)),
        }

        return stats

    def extract_headings_structure(self, markdown_content: str) -> List[Dict]:
        """Extract heading structure for navigation and chunking."""
        if not markdown_content:
            return []

        headings = []
        lines = markdown_content.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("#"):
                match = re.match(r"^(#+)\s*(.+)", line)
                if match:
                    level = len(match.group(1))
                    text = self._clean_markdown_formatting(match.group(2).strip())
                    headings.append({"level": level, "text": text, "line_number": i})

        return headings

    def extract_images_info(self, markdown_content: str) -> List[Dict]:
        """Extract image information for retrieval using ImageExtractorUtility."""
        if not markdown_content:
            return []

        # Use ImageExtractorUtility for proper image detection and mapping
        from .image_extractor_utility import ImageExtractorUtility

        extractor = ImageExtractorUtility()

        # Extract images using the utility (handles both markdown and HTML img tags)
        extracted_images = extractor.extract_images_from_markdown(markdown_content)

        # Convert to the format expected by the RAG processor
        images = []
        for img_info in extracted_images:
            images.append(
                {
                    "alt_text": img_info.get("alt_text", ""),
                    "path": img_info.get("original_src", ""),
                    "local_path": img_info.get("local_path", ""),
                    "exists": img_info.get("exists", False),
                    "type": img_info.get("type", "unknown"),
                    "filename": img_info.get("original_src", "").split("/")[-1]
                    if img_info.get("original_src")
                    else "",
                }
            )

        return images
