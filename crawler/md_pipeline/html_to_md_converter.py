"""HTML to Markdown converter with image path mapping and semantic structure preservation."""

import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup, Tag


class HTMLToMDConverter:
    """Converts filtered HTML content to clean Markdown format."""

    def __init__(self, url_mapper=None, current_url=None):
        self.image_base_path = "crawler/process-images"
        self.url_mapper = url_mapper
        self.current_url = current_url  # Current page URL for resolving relative links
        self.images_metadata = []  # Store images during conversion
        self.content_position = 0  # Track position for unique IDs

    def convert_html_to_md(
        self, html_content: str, source_url: str, md_handler=None
    ) -> Dict:
        """Convert HTML content to Markdown format with separate image metadata."""
        # Reset for each conversion
        self.images_metadata = []
        self.content_position = 0

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
            "images_metadata": self.images_metadata,  # Separate image metadata
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
            # Skip figure captions as they are handled by image processing
            if "figure" in tag.get("class", []):
                return ""
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
        """Process paragraph with inline elements, handling nested content recursively."""
        parts = []

        for child in p_tag.children:
            if isinstance(child, Tag):
                if child.name == "img":
                    parts.append(self._convert_image_to_md(child))
                elif child.name == "a":
                    parts.append(self._convert_link_to_md(child))
                elif child.name in ["strong", "b"]:
                    # Check if this element contains nested links or other elements
                    nested_content = self._process_inline_element_content(child)
                    parts.append(f"**{nested_content}**" if nested_content else "")
                elif child.name in ["em", "i"]:
                    # Check if this element contains nested links or other elements
                    nested_content = self._process_inline_element_content(child)
                    parts.append(f"*{nested_content}*" if nested_content else "")
                elif child.name in ["span", "div", "small", "u", "sup", "sub"]:
                    # Process inline elements that might contain nested links
                    nested_content = self._process_inline_element_content(child)
                    parts.append(nested_content if nested_content else "")
                else:
                    # For other elements, try to process nested content first
                    nested_content = self._process_inline_element_content(child)
                    parts.append(nested_content if nested_content else "")
            elif hasattr(child, "strip"):
                clean_text = self._clean_text_content(child.strip())
                parts.append(clean_text if clean_text else "")

        return " ".join(parts).strip()

    def _process_inline_element_content(self, element: Tag) -> str:
        """Process content within inline elements, handling nested links and formatting."""
        if not element:
            return ""

        parts = []

        for child in element.children:
            if isinstance(child, Tag):
                if child.name == "a":
                    # Process nested links
                    parts.append(self._convert_link_to_md(child))
                elif child.name == "img":
                    # Process nested images
                    parts.append(self._convert_image_to_md(child))
                elif child.name in ["strong", "b"]:
                    # Recursively process nested formatting
                    nested_content = self._process_inline_element_content(child)
                    parts.append(f"**{nested_content}**" if nested_content else "")
                elif child.name in ["em", "i"]:
                    # Recursively process nested formatting
                    nested_content = self._process_inline_element_content(child)
                    parts.append(f"*{nested_content}*" if nested_content else "")
                else:
                    # For other nested elements, recursively process their content
                    nested_content = self._process_inline_element_content(child)
                    parts.append(nested_content if nested_content else "")
            elif hasattr(child, "strip"):
                # Text node
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
        """Convert image to clean Markdown and store metadata separately."""
        src = img_tag.get("src", "")
        alt = img_tag.get("alt", "")

        # Look for figure caption in next sibling
        caption = ""
        try:
            next_sibling = img_tag.find_parent("p").find_next_sibling(
                "p", class_="figure"
            )
            if next_sibling:
                caption = next_sibling.get_text().strip()
        except:
            pass

        # Generate description: use caption, alt text, or filename-based description
        description = self._generate_image_description(caption, alt, src)

        # Track position for unique ID generation
        self.content_position += 1

        # Map image path to process-images structure
        if src:
            # Use the image extractor utility for proper mapping
            from .image_extractor_utility import ImageExtractorUtility

            extractor = ImageExtractorUtility(
                url_mapper=self.url_mapper, current_url=self.current_url
            )

            # First, try to resolve the image src to get the proper URL
            resolved_src = extractor._resolve_image_src(src)

            # Process the image source to get proper mapping
            image_info = extractor._process_image_src(src, alt)
            if image_info:
                # Generate unique image ID
                image_id = str(self._generate_image_id(src, alt, self.content_position))

                # Generate description using caption, alt text, or fallback to "Image"
                description = self._generate_image_description(caption, alt, src)

                # Create complete metadata object
                image_metadata = {
                    "image_id": image_id,
                    "type": image_info.get("type", "missing_image"),
                    "description": description,
                    "filename": (
                        Path(image_info["local_path"]).name
                        if image_info["local_path"]
                        else ""
                    ),
                    "local_path": image_info["local_path"],
                    "category": image_info.get("category", ""),
                    "exists": image_info["exists"],
                    "image_url": image_info.get("image_url", ""),
                    "enhanced_image_url": image_info.get("enhanced_image_url", ""),
                    "original_src": src,
                    "position_in_content": self.content_position,
                }

                # Store metadata separately
                self.images_metadata.append(image_metadata)

                # Return proper markdown image syntax with resolved URL
                image_url = image_info.get("enhanced_image_url", "") or image_info.get(
                    "image_url", ""
                )
                if image_url:
                    return f"![{description}]({image_url}){{image_id: {image_id}}}"
                elif resolved_src and resolved_src != src:
                    # Use resolved URL even if image doesn't exist locally
                    return f"![{description}]({resolved_src}){{image_id: {image_id}}}"
                else:
                    # Fallback to RAG-friendly format only if no URL can be constructed
                    return (
                        f"[Image: {description}]{{image_id: {image_id}}}"
                        if description
                        else f"[Image: {Path(src).stem}]{{image_id: {image_id}}}"
                    )
            else:
                # Handle case where image_info is None
                image_id = str(self._generate_image_id(src, alt, self.content_position))

                image_metadata = {
                    "image_id": image_id,
                    "type": "missing_image",
                    "description": alt,
                    "filename": "",
                    "local_path": "",
                    "category": "",
                    "exists": False,
                    "image_url": "",
                    "enhanced_image_url": "",
                    "original_src": src,
                    "position_in_content": self.content_position,
                }

                self.images_metadata.append(image_metadata)

                # Try to resolve the URL even if image doesn't exist locally
                if resolved_src and resolved_src != src:
                    return f"![{alt}]({resolved_src}){{image_id:{image_id}}}"
                else:
                    return (
                        f"[Image: {alt}]{{image_id:{image_id}}}"
                        if alt
                        else f"[Image: {Path(src).stem}]{{image_id:{image_id}}}"
                    )

        return (
            f"[Image: {alt}]{{image_id:{image_id}}}"
            if alt
            else f"[Image]{{image_id:{image_id}}}"
        )

    def _generate_image_description(self, caption: str, alt_text: str, src: str) -> str:
        """Generate image description: use caption, alt text, or static 'Image' fallback."""
        if caption and caption.strip():
            return caption.strip()

        if alt_text and alt_text.strip():
            return alt_text.strip()

        # Static fallback instead of using src
        return "Image"

    def _generate_image_description_with_url(
        self, alt_text: str, enhanced_image_url: str
    ) -> str:
        """Generate image description: use alt text, enhanced_image_url, or static 'Image' fallback."""
        if alt_text and alt_text.strip():
            return alt_text.strip()

        if enhanced_image_url and enhanced_image_url.strip():
            return enhanced_image_url.strip()

        # Static fallback
        return "Image"

    def _generate_image_id(self, src: str, alt_text: str, position: int) -> str:
        """Generate unique image ID based on source, alt text, and position"""
        content = f"{src}_{alt_text}_{position}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def _convert_link_to_md(self, a_tag: Tag) -> str:
        """Convert link to Markdown with href resolution."""
        href = a_tag.get("href", "")

        # Check if this link contains an image (common pattern for image links)
        img_tag = a_tag.find("img")
        if img_tag:
            # This is an image link, process the image instead
            return self._convert_image_to_md(img_tag)

        text = self._clean_text(a_tag)

        if href and text:
            # Resolve href using URL mapper if available
            resolved_href = self._resolve_href(href)
            return f"[{text}]({resolved_href})"
        elif text:
            return text
        else:
            return ""

    def _resolve_href(self, href: str) -> str:
        """
        Resolve href using the page_url extracted at root level.

        Handles various relative URL patterns:
        - href="../Plan Setup Reference/PlanSetup/Benefit Plans.htm" (parent directory)
        - href="Annual Enrollment Timeline.htm" (same directory)
        - href="/root/path.htm" (root-relative)

        Args:
            href: The href attribute value from the HTML link

        Returns:
            Resolved full URL if possible, otherwise original href
        """
        if not self.url_mapper or not href:
            return href

        # Clean href - remove fragments and whitespace
        clean_href = href.strip().split("#")[0]

        # If href is already absolute, return as is
        if clean_href.startswith(("http://", "https://")):
            return clean_href

        # If we have a base_url, construct the full URL
        if self.url_mapper.base_url:
            if clean_href.startswith("/"):
                # Root-relative URL: href="/root/path.htm"
                return f"{self.url_mapper.base_url}{clean_href}"
            elif clean_href.startswith("../"):
                # Parent directory relative: href="../Plan Setup Reference/PlanSetup/Benefit Plans.htm"
                # Get current page URL to determine current directory
                current_url = self._get_current_page_url()
                if current_url:
                    resolved_url = self._resolve_relative_path(current_url, clean_href)
                    return f"{self.url_mapper.base_url}{resolved_url}"
                else:
                    # Fallback: treat as relative to base
                    return f"{self.url_mapper.base_url}/{clean_href}"
            else:
                # Same directory relative: href="Annual Enrollment Timeline.htm"
                # Get current page URL to determine current directory
                current_url = self._get_current_page_url()
                if current_url:
                    # Extract directory from current URL
                    current_dir = "/".join(current_url.split("/")[:-1])
                    if current_dir:
                        return f"{self.url_mapper.base_url}{current_dir}/{clean_href}"
                    else:
                        return f"{self.url_mapper.base_url}/{clean_href}"
                else:
                    # Fallback: treat as relative to base
                    return f"{self.url_mapper.base_url}/{clean_href}"

        # Fallback: return original href
        return href

    def _get_current_page_url(self) -> str:
        """Get the current page URL from the URL mapper."""
        if not self.current_url or not self.url_mapper:
            return ""

        # If current_url is a filename, get the corresponding URL
        if not self.current_url.startswith(("http://", "https://", "/")):
            # It's a filename, get the relative URL
            return self.url_mapper.get_relative_url(self.current_url)

        # It's already a URL
        return self.current_url

    def _resolve_relative_path(self, current_url: str, relative_href: str) -> str:
        """
        Resolve relative path like '../Plan Setup Reference/PlanSetup/Benefit Plans.htm'
        against current URL.

        Args:
            current_url: Current page URL (relative, like '/Annual Enrollment/page.htm')
            relative_href: Relative href (like '../Plan Setup Reference/PlanSetup/Benefit Plans.htm')

        Returns:
            Resolved relative URL
        """
        # Split current URL into parts
        current_parts = [part for part in current_url.split("/") if part]

        # Split relative href into parts
        href_parts = [part for part in relative_href.split("/") if part]

        # Start with current directory (remove filename)
        if current_parts:
            current_parts = current_parts[:-1]  # Remove filename, keep directory

        # Process each part of the relative href
        for part in href_parts:
            if part == "..":
                # Go up one directory
                if current_parts:
                    current_parts.pop()
            elif part == ".":
                # Stay in current directory
                continue
            else:
                # Add this part to the path
                current_parts.append(part)

        # Reconstruct the path
        resolved_path = "/" + "/".join(current_parts) if current_parts else "/"
        return resolved_path

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
