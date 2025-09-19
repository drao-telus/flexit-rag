"""HTML to Markdown converter with image path mapping and semantic structure preservation."""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup, Tag


class HTMLToMDConverter:
    """Converts filtered HTML content to clean Markdown format."""

    def __init__(self):
        self.image_base_path = "crawler/process-images"

    def convert_html_to_md(
        self, html_content: str, source_url: str, md_handler=None
    ) -> Dict:
        """Convert HTML content to Markdown format with metadata."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract metadata
        metadata = self._extract_metadata(soup, source_url)

        # Convert main content to markdown
        main_content = soup.find("div", {"role": "main", "id": "mc-main-content"})
        if not main_content:
            main_content = soup.find("body")

        if md_handler and callable(md_handler):
            markdown_content = md_handler(main_content)
        else:
            markdown_content = (
                self._convert_element_to_md(main_content) if main_content else ""
            )

        return {
            "markdown_content": markdown_content.strip(),
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }

    def _extract_metadata(self, soup: BeautifulSoup, source_url: str) -> Dict:
        """Extract metadata from HTML."""
        # Extract breadcrumbs
        breadcrumbs = self._extract_breadcrumbs(soup)

        # Get meaningful title
        page_title = self._extract_page_title(soup, "")

        return {
            "page_title": page_title,
            "breadcrumbs": breadcrumbs,
            "url": source_url,
            "language": (
                soup.find("html").get("lang", "en") if soup.find("html") else "en"
            ),
            "has_images": bool(soup.find("img")),
            "has_tables": bool(soup.find("table")),
        }

    def _extract_breadcrumbs(self, soup: BeautifulSoup) -> List[str]:
        """Extract breadcrumb navigation."""
        breadcrumbs_div = soup.find("div", class_="MCBreadcrumbsBox_0")
        if breadcrumbs_div:
            crumbs = breadcrumbs_div.find_all("span", class_="MCBreadcrumbsSelf")
            return [
                self._clean_text(crumb) for crumb in crumbs if self._clean_text(crumb)
            ]
        return []

    def _extract_page_title(self, soup: BeautifulSoup, fallback_title: str) -> str:
        """Extract meaningful page title."""
        # Try H1 first
        h1_element = soup.find("h1")
        if h1_element:
            h1_text = self._clean_text(h1_element)
            if h1_text and len(h1_text.strip()) > 0:
                return h1_text

        # Try breadcrumbs
        breadcrumbs = self._extract_breadcrumbs(soup)
        if breadcrumbs and len(breadcrumbs) > 1:
            return breadcrumbs[-1]

        # Fallback
        if fallback_title and fallback_title.lower() not in [
            "document",
            "untitled",
            "",
        ]:
            return fallback_title

        return breadcrumbs[0] if breadcrumbs else "Untitled Page"

    def _convert_element_to_md(self, element: Tag) -> str:
        """Convert HTML element to Markdown recursively."""
        if not element:
            return ""

        markdown_parts = []

        for child in element.children:
            if isinstance(child, Tag):
                md_content = self._convert_tag_to_md(child)
                if md_content:
                    markdown_parts.append(md_content)
            elif hasattr(child, "strip") and child.strip():
                # Text node
                clean_text = self._clean_text_content(child.strip())
                if clean_text:
                    markdown_parts.append(clean_text)

        return "\n\n".join(markdown_parts)

    def _convert_tag_to_md(self, tag: Tag) -> str:
        """Convert specific HTML tags to Markdown."""
        tag_name = tag.name.lower()

        # Skip breadcrumbs and navigation
        if "breadcrumb" in tag.get("class", []):
            return ""

        if "figure" in tag.get("class", []):
            return ""

        # Headings
        if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(tag_name[1])
            text = self._clean_text(tag)
            return f"{'#' * level} {text}"

        # Paragraphs
        elif tag_name == "p":
            text = self._process_paragraph(tag)
            return text if text else ""

        # Lists
        elif tag_name in ["ul", "ol"]:
            return self._convert_list_to_md(tag)

        # Tables
        elif tag_name == "table":
            return self._convert_table_to_md(tag)

        # Images
        elif tag_name == "img":
            return self._convert_image_to_md(tag)

        # Links
        elif tag_name == "a":
            return self._convert_link_to_md(tag)

        # Emphasis
        elif tag_name in ["strong", "b"]:
            text = self._clean_text(tag)
            return f"**{text}**" if text else ""

        elif tag_name in ["em", "i"]:
            text = self._clean_text(tag)
            return f"*{text}*" if text else ""

        # Code
        elif tag_name == "code":
            text = self._clean_text(tag)
            return f"`{text}`" if text else ""

        # Span tags (inline content)
        elif tag_name == "span":
            text = self._clean_text(tag)
            return text if text else ""

        # Divs and other containers
        elif tag_name in ["div", "section", "article", "li"]:
            return self._convert_element_to_md(tag)

        # Default: extract text content
        else:
            text = self._clean_text(tag)
            return text if text else ""

    def _process_paragraph(self, p_tag: Tag) -> str:
        """Process paragraph with inline elements."""
        parts = []

        for child in p_tag.children:
            if isinstance(child, Tag):
                if child.name == "img":
                    parts.append(self._convert_image_to_md(child))
                elif child.name == "a":
                    parts.append(self._convert_link_to_md(child))
                elif child.name in ["strong", "b"]:
                    text = self._clean_text(child)
                    parts.append(f"**{text}**" if text else "")
                elif child.name in ["em", "i"]:
                    text = self._clean_text(child)
                    parts.append(f"*{text}*" if text else "")
                else:
                    text = self._clean_text(child)
                    parts.append(text if text else "")
            elif hasattr(child, "strip"):
                clean_text = self._clean_text_content(child.strip())
                parts.append(clean_text if clean_text else "")

        return " ".join(parts).strip()

    def _convert_list_to_md(self, list_tag: Tag) -> str:
        """Convert HTML list to Markdown."""
        is_ordered = list_tag.name == "ol"
        items = []

        # Process all direct children (both li and nested ul/ol)
        for child in list_tag.children:
            if isinstance(child, Tag):
                if child.name == "li":
                    # Get the value attribute for ordered lists (handles non-sequential numbering)
                    if is_ordered and child.get("value"):
                        marker = f"{child.get('value')}."
                    else:
                        # Count existing items for proper numbering
                        marker = f"{len(items) + 1}." if is_ordered else "-"

                    item_content = self._process_list_item(child)
                    if item_content:
                        items.append(f"{marker} {item_content}")

                elif child.name in ["ul", "ol"]:
                    # Handle direct nested lists (malformed HTML but we need to support it)
                    nested_content = self._convert_list_to_md(child)
                    if nested_content:
                        # Add nested content with proper indentation
                        indented = "\n".join(
                            f"  {line}" for line in nested_content.split("\n")
                        )
                        items.append(indented)

                elif child.name == "table":
                    # Handle tables directly inside lists (malformed HTML but common)
                    table_content = self._convert_table_to_md(child)
                    if table_content:
                        # Add table content with proper indentation
                        indented = "\n".join(
                            f"  {line}" for line in table_content.split("\n")
                        )
                        items.append(f"\n{indented}\n")

                elif child.name == "p":
                    # Handle paragraphs directly inside lists (malformed HTML but common)
                    p_content = self._process_paragraph(child)
                    if p_content:
                        # Don't add a marker for paragraphs inside lists - they're continuation content
                        # Instead, add them as indented content under the previous item
                        if items:
                            # Add as continuation of previous item
                            items.append(f"  {p_content}")
                        else:
                            # If no previous item, create a new item
                            marker = f"{len(items) + 1}." if is_ordered else "-"
                            items.append(f"{marker} {p_content}")

        return "\n".join(items)

    def _process_list_item(self, li_tag: Tag) -> str:
        """Process list item with potential nested content."""
        parts = []

        for child in li_tag.children:
            if isinstance(child, Tag):
                if child.name in ["ul", "ol"]:
                    # Nested list
                    nested_md = self._convert_list_to_md(child)
                    if nested_md:
                        # Indent nested list
                        indented = "\n".join(
                            f"  {line}" for line in nested_md.split("\n")
                        )
                        parts.append(f"\n{indented}")
                elif child.name == "p":
                    # Process paragraphs within list items
                    p_content = self._process_paragraph(child)
                    if p_content:
                        parts.append(p_content)
                else:
                    content = self._convert_tag_to_md(child)
                    if content:
                        parts.append(content)
            elif hasattr(child, "strip"):
                text = self._clean_text_content(child.strip())
                if text:
                    parts.append(text)

        return " ".join(parts).strip()

    def _convert_table_to_md(self, table_tag: Tag) -> str:
        """Convert HTML table to Markdown table."""
        parts = []

        # Process table caption if it exists
        caption = table_tag.find("caption")
        if caption:
            caption_text = self._clean_text(caption)
            if caption_text:
                parts.append(caption_text)

        rows = []
        has_header = False

        # Process headers
        thead = table_tag.find("thead")
        if thead:
            # Process all header rows, not just the first one
            for header_row in thead.find_all("tr"):
                headers = []
                for th in header_row.find_all(["th", "td"]):
                    # Handle colspan by repeating the header
                    colspan = int(th.get("colspan", 1))
                    header_text = self._process_table_cell(th)
                    headers.extend([header_text] * colspan)

                if headers:
                    rows.append("| " + " | ".join(headers) + " |")
                    # Only add separator after the first header row
                    if len(rows) == 1:
                        rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
                        has_header = True

        # Process ALL tbody elements, not just the first one
        tbody_elements = table_tag.find_all("tbody")
        if not tbody_elements:
            # If no tbody, process table directly for rows
            tbody_elements = [table_tag]

        data_rows = []
        for tbody in tbody_elements:
            for tr in tbody.find_all("tr", recursive=False):  # Only direct children
                cells = []
                for td in tr.find_all(["td", "th"]):
                    # Handle colspan by repeating the cell content
                    colspan = int(td.get("colspan", 1))
                    cell_text = self._process_table_cell(td)
                    cells.extend([cell_text] * colspan)

                if cells:
                    data_rows.append("| " + " | ".join(cells) + " |")

        # If no header was found but we have data rows, create a generic header
        if not has_header and data_rows:
            # Determine number of columns from first data row
            first_row_cells = (
                len(data_rows[0].split("|")) - 2
            )  # Subtract 2 for leading/trailing empty parts
            if first_row_cells > 0:
                # Create generic headers
                generic_headers = [f"Column {i + 1}" for i in range(first_row_cells)]
                rows.append("| " + " | ".join(generic_headers) + " |")
                rows.append("| " + " | ".join(["---"] * first_row_cells) + " |")

        # Add data rows
        rows.extend(data_rows)

        if rows:
            parts.append("\n".join(rows))

        return "\n\n".join(parts)

    def _process_table_cell(self, cell_tag: Tag) -> str:
        """Process table cell content, handling nested elements properly."""
        if not cell_tag:
            return ""

        parts = []

        for child in cell_tag.children:
            if isinstance(child, Tag):
                if child.name == "p":
                    # Process paragraphs within cells
                    p_content = self._process_paragraph(child)
                    if p_content:
                        parts.append(p_content)
                elif child.name in ["ul", "ol"]:
                    # Process lists within cells
                    list_content = self._convert_list_to_md(child)
                    if list_content:
                        # Convert list to inline format for table cells
                        list_items = []
                        for line in list_content.split("\n"):
                            if line.strip().startswith(
                                (
                                    "-",
                                    "1.",
                                    "2.",
                                    "3.",
                                    "4.",
                                    "5.",
                                    "6.",
                                    "7.",
                                    "8.",
                                    "9.",
                                )
                            ):
                                # Extract the content after the marker
                                content = line.strip()
                                if content.startswith("-"):
                                    content = content[1:].strip()
                                elif "." in content:
                                    content = content.split(".", 1)[1].strip()
                                list_items.append(content)
                        if list_items:
                            parts.append(" ".join(list_items))
                elif child.name == "a":
                    # Process links within cells
                    link_content = self._convert_link_to_md(child)
                    if link_content:
                        parts.append(link_content)
                elif child.name in ["strong", "b"]:
                    text = self._clean_text(child)
                    parts.append(f"**{text}**" if text else "")
                elif child.name in ["em", "i"]:
                    text = self._clean_text(child)
                    parts.append(f"*{text}*" if text else "")
                elif child.name == "span":
                    text = self._clean_text(child)
                    if text:
                        parts.append(text)
                else:
                    # For other tags, extract text content
                    text = self._clean_text(child)
                    if text:
                        parts.append(text)
            elif hasattr(child, "strip"):
                # Text node
                clean_text = self._clean_text_content(child.strip())
                if clean_text:
                    parts.append(clean_text)

        # Join parts and clean up for table cell format
        result = " ".join(parts).strip()

        # Replace newlines with spaces for table format
        result = re.sub(r"\s+", " ", result)

        # Escape pipe characters that would break table format
        result = result.replace("|", "\\|")

        return result

    def _convert_image_to_md(self, img_tag: Tag) -> str:
        """Convert image to Markdown with proper path mapping."""
        src = img_tag.get("src", "")
        alt = img_tag.get("alt", "")

        # Look for figure caption in next sibling (paragraph-specific logic)
        caption = ""
        try:
            next_sibling = img_tag.find_parent("p").find_next_sibling(
                "p", class_="figure"
            )
            if next_sibling:
                caption = next_sibling.get_text().strip()
        except:
            pass

        alt = caption if caption else alt

        # Map image path to process-images structure
        if src:
            # Use the image extractor utility for proper mapping
            from .image_extractor_utility import ImageExtractorUtility

            extractor = ImageExtractorUtility()

            # Process the image source to get proper mapping
            image_info = extractor._process_image_src(src, alt)
            if image_info and image_info["exists"]:
                # Use a more readable format for RAG processing
                image_name = Path(image_info["local_path"]).stem
                if alt:
                    return f"[Image: {alt} - {image_name}]"
                else:
                    return f"[Image: {image_name}]"
            else:
                # Keep original path if not found
                if alt:
                    return f"[Image: {alt}]"
                else:
                    return f"[Image: {Path(src).stem}]"

        return f"[Image: {alt}]" if alt else "[Image]"

    def _map_image_path(self, image_name: str) -> str:
        """Map image name to the correct path in process-images structure."""
        # This method is now deprecated in favor of using ImageExtractorUtility
        # Keeping for backward compatibility
        return f"Images/{image_name}"

    def _convert_link_to_md(self, a_tag: Tag) -> str:
        """Convert link to Markdown."""
        href = a_tag.get("href", "")

        # Check if this link contains an image (common pattern for image links)
        img_tag = a_tag.find("img")
        if img_tag:
            # This is an image link, process the image instead
            return self._convert_image_to_md(img_tag)

        text = self._clean_text(a_tag)

        if href and text:
            return f"[{text}]({href})"
        elif text:
            return text
        else:
            return ""

    def _clean_text(self, element: Tag) -> str:
        """Clean text content removing extra whitespace."""
        if not element:
            return ""
        text = element.get_text(separator=" ", strip=True)
        return self._clean_text_content(text)

    def _clean_text_content(self, text: str) -> str:
        """Clean text content."""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters that might interfere with markdown
        text = text.replace("\u00a0", " ")  # Non-breaking space
        return text.strip()
