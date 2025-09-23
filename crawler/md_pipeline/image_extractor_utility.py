"""
Image Extraction Utility - Handles image extraction and mapping for the MD pipeline.
Works with existing process-images structure and recognizes common image patterns.
"""

import re
import json
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse, unquote
import hashlib


class ImageExtractorUtility:
    """
    Utility for extracting and managing images in the MD pipeline.
    Integrates with existing crawler/process-images structure.
    """

    def __init__(
        self,
        process_images_dir: str = "crawler/process-images",
        url_mapper=None,
        current_url=None,
    ):
        self.process_images_dir = Path(process_images_dir)
        self.url_mapper = url_mapper
        self.current_url = current_url  # Current page URL for resolving relative paths
        self.image_patterns = [
            r"!\[([^\]]*)\]\(([^)]+)\)",  # Standard markdown images
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',  # HTML img tags
            r'src=["\']([^"\']+\.(?:png|jpg|jpeg|gif|svg|webp))["\']',  # Direct src attributes
        ]

        # Load image base URL from page_url.py
        self.image_base_url = self._load_image_base_url()

    def _load_image_base_url(self) -> str:
        """Load image base URL from page_url.py"""
        try:
            import json

            page_url_file = "crawler/url/page_url.py"
            with open(page_url_file, "r") as f:
                content = f.read()
                url_data = json.loads(content)
                return url_data.get(
                    "image_base_url",
                    url_data.get("base_url", "https://flexit.telus.com"),
                )
        except Exception as e:
            print(f"Warning: Could not load image base URL from page_url.py: {e}")
            return "https://flexit.telus.com"

    def extract_images_from_markdown(
        self, markdown_content: str
    ) -> List[Dict[str, str]]:
        """
        Extract all images from markdown content using pattern recognition.
        Returns list of image info dictionaries.
        """
        images = []

        # Pattern 1: Standard markdown images ![alt](src)
        markdown_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
        for match in re.finditer(markdown_pattern, markdown_content):
            alt_text = match.group(1).strip()
            src = match.group(2).strip()

            image_info = self._process_image_src(src, alt_text)
            if image_info:
                images.append(image_info)

        # Pattern 2: HTML img tags
        html_pattern = (
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*(?:alt=["\']([^"\']*)["\'])?[^>]*>'
        )
        for match in re.finditer(html_pattern, markdown_content, re.IGNORECASE):
            src = match.group(1).strip()
            alt_text = match.group(2) if match.group(2) else ""

            image_info = self._process_image_src(src, alt_text)
            if image_info:
                images.append(image_info)

        # Pattern 3: RAG-friendly image format [Image: name]
        rag_pattern = r"\[Image: ([^\]]+)\]"
        for match in re.finditer(rag_pattern, markdown_content):
            image_name = match.group(1).strip()

            # Convert RAG-friendly name to potential filename
            # Try common extensions
            for ext in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]:
                potential_src = f"{image_name}{ext}"
                image_info = self._process_image_src(potential_src, image_name)
                if image_info and image_info["exists"]:
                    # Update to show it was from RAG format
                    image_info["rag_format"] = True
                    image_info["rag_name"] = image_name
                    images.append(image_info)
                    break

        return self._deduplicate_images(images)

    def _process_image_src(
        self, src: str, alt_text: str = ""
    ) -> Optional[Dict[str, str]]:
        """Process image source and return standardized image info."""
        if not src:
            return None

        # Clean up the source path
        src = unquote(src).strip()

        # Store original src for reference
        original_src = src

        # Resolve relative paths using URL mapper if available
        resolved_src = self._resolve_image_src(src)

        # Generate image_url by prefixing resolved src with base_image_url
        image_url = ""
        if resolved_src.startswith(("http://", "https://")):
            # Already a complete URL
            image_url = resolved_src
        elif resolved_src.startswith("data:"):
            # Data URL, keep as is
            image_url = ""
        else:
            # Relative path, prefix with base_image_url
            # Remove leading slash if present to avoid double slashes
            clean_src = resolved_src.lstrip("/")
            image_url = f"{self.image_base_url}/{clean_src}"

        # Handle data URLs separately
        if resolved_src.startswith("data:"):
            return {
                "alt_text": alt_text,
                "original_src": original_src,
                "local_path": "",
                "exists": False,
                "type": "data_url",
                "image_url": "",
                "enhanced_image_url": "",
            }

        # For HTTP/HTTPS URLs, try to map to local files first
        # This handles cases where URLs are resolved but files exist locally
        local_path = None
        if resolved_src.startswith(("http://", "https://")):
            # Try to map using the original src first (before URL resolution)
            local_path = self._map_to_process_images_path(original_src)

            # If that doesn't work, try extracting the path from the resolved URL
            if not (local_path and local_path.exists()):
                # Extract path after base URL for local mapping
                if self.image_base_url and resolved_src.startswith(self.image_base_url):
                    relative_path = resolved_src[len(self.image_base_url) :].lstrip("/")
                    local_path = self._map_to_process_images_path(relative_path)
                elif (
                    self.url_mapper
                    and self.url_mapper.base_url
                    and resolved_src.startswith(self.url_mapper.base_url)
                ):
                    relative_path = resolved_src[
                        len(self.url_mapper.base_url) :
                    ].lstrip("/")
                    local_path = self._map_to_process_images_path(relative_path)
        else:
            # For relative paths, map directly
            local_path = self._map_to_process_images_path(resolved_src)

        # Generate enhanced_image_url
        enhanced_image_url = ""
        if local_path and local_path.exists() and image_url:
            # Local file exists - use the actual local filename
            image_filename = local_path.name
            # Get complete path except filename from image_url
            path_without_filename = image_url.rsplit("/", 1)[0]
            enhanced_image_url = f"{path_without_filename}/{image_filename}"
        elif image_url:
            # No local file but we have a valid image_url - use it as enhanced_image_url
            enhanced_image_url = image_url

        # Determine the type based on whether we found a local file
        if local_path and local_path.exists():
            image_type = "retrieved_image"
        elif resolved_src.startswith(("http://", "https://")):
            image_type = "external_image"  # URL resolved but no local file found
        else:
            image_type = "missing_image"  # Local path but file doesn't exist

        return {
            "alt_text": alt_text,
            "original_src": original_src,
            "local_path": str(local_path) if local_path else "",
            "exists": local_path.exists() if local_path else False,
            "type": image_type,
            "image_url": image_url,
            "enhanced_image_url": enhanced_image_url,
        }

    def _map_to_process_images_path(self, src: str) -> Optional[Path]:
        """
        Map image source to process-images directory structure.
        Recognizes common patterns in the existing structure.
        """
        # Remove leading slashes and normalize
        src = src.lstrip("/")

        # Extract filename from path
        filename = Path(src).name

        # Handle thumbnail pattern: filename_thumb_0_200.png -> filename.png
        if "_thumb_" in filename:
            # Remove thumbnail suffix pattern
            base_name = re.sub(r"_thumb_\d+_\d+", "", filename)
            # Try to find the base image file
            direct_path = self.process_images_dir / base_name
            if direct_path.exists():
                return direct_path

            # Also try the original filename in case it exists
            original_path = self.process_images_dir / filename
            if original_path.exists():
                return original_path

        # Common patterns in the existing structure
        if src.lower().startswith("images/"):
            # Pattern: Images/file.png -> process-images/file.png
            relative_path = src[7:]  # Remove 'images/' prefix

            # Handle thumbnail pattern in relative path
            if "_thumb_" in relative_path:
                base_name = re.sub(r"_thumb_\d+_\d+", "", relative_path)
                direct_path = self.process_images_dir / base_name
                if direct_path.exists():
                    return direct_path

            # Try direct mapping
            direct_path = self.process_images_dir / relative_path
            if direct_path.exists():
                return direct_path

            # Try finding in subdirectories
            return self._find_image_in_subdirs(Path(relative_path).name)

        # Pattern: Direct file reference
        if "/" not in src:
            # Look for the file in process-images root and subdirectories
            direct_path = self.process_images_dir / filename
            if direct_path.exists():
                return direct_path
            return self._find_image_in_subdirs(filename)

        # Pattern: Relative path from document
        # Try to map based on document structure
        path_parts = Path(src).parts

        # Common mapping patterns
        if len(path_parts) >= 2:
            # Try: folder/file.png -> process-images/file.png (flattened)
            filename = path_parts[-1]

            # Handle thumbnail pattern
            if "_thumb_" in filename:
                base_name = re.sub(r"_thumb_\d+_\d+", "", filename)
                direct_path = self.process_images_dir / base_name
                if direct_path.exists():
                    return direct_path

            # Try direct file in root
            direct_path = self.process_images_dir / filename
            if direct_path.exists():
                return direct_path

            # Try preserving structure
            return self.process_images_dir / Path(*path_parts)

        return self.process_images_dir / src

    def _find_image_in_subdirs(self, filename: str) -> Optional[Path]:
        """Find image file in process-images subdirectories."""
        if not self.process_images_dir.exists():
            return None

        # Handle thumbnail pattern: filename_thumb_0_200.png -> filename.png
        search_names = [filename]
        if "_thumb_" in filename:
            base_name = re.sub(r"_thumb_\d+_\d+", "", filename)
            search_names.append(base_name)

        # Search in root directory first
        for name in search_names:
            root_path = self.process_images_dir / name
            if root_path.exists():
                return root_path

        # Search in all subdirectories
        for subdir in self.process_images_dir.iterdir():
            if subdir.is_dir():
                for name in search_names:
                    image_path = subdir / name
                    if image_path.exists():
                        return image_path

        return None

    def _deduplicate_images(self, images: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate images based on source path."""
        seen = set()
        unique_images = []

        for image in images:
            key = image["original_src"]
            if key not in seen:
                seen.add(key)
                unique_images.append(image)

        return unique_images

    def verify_image_paths(self, images: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Verify image paths and return statistics.
        Useful for debugging and validation.
        """
        stats = {
            "total_images": len(images),
            "existing_images": 0,
            "missing_images": 0,
            "external_images": 0,
            "data_url_images": 0,
            "missing_files": [],
        }

        for image in images:
            if image["type"] == "external":
                stats["external_images"] += 1
            elif image["type"] == "data_url":
                stats["data_url_images"] += 1
            elif image["exists"]:
                stats["existing_images"] += 1
            else:
                stats["missing_images"] += 1
                stats["missing_files"].append(
                    {
                        "original_src": image["original_src"],
                        "expected_path": image["local_path"],
                        "alt_text": image["alt_text"],
                    }
                )

        return stats

    def copy_images_to_output(
        self,
        images: List[Dict[str, str]],
        output_dir: str,
        preserve_structure: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Copy images to output directory for deployment.
        Returns updated image list with new paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        copied_images = []

        for image in images:
            if not image["exists"] or image["type"] != "local":
                # Keep external/missing images as-is
                copied_images.append(image.copy())
                continue

            source_path = Path(image["local_path"])

            if preserve_structure:
                # Preserve directory structure
                relative_path = source_path.relative_to(self.process_images_dir)
                dest_path = output_path / relative_path
            else:
                # Flatten structure
                dest_path = output_path / source_path.name

            # Create destination directory
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            try:
                shutil.copy2(source_path, dest_path)

                # Update image info
                updated_image = image.copy()
                updated_image["copied_path"] = str(dest_path)
                updated_image["copied_relative"] = str(
                    dest_path.relative_to(output_path)
                )
                copied_images.append(updated_image)

            except Exception as e:
                # Keep original if copy fails
                error_image = image.copy()
                error_image["copy_error"] = str(e)
                copied_images.append(error_image)

        return copied_images

    def update_markdown_image_paths(
        self, markdown_content: str, path_mapping: Dict[str, str]
    ) -> str:
        """
        Update image paths in markdown content based on mapping.
        Useful for deployment or restructuring.
        """
        updated_content = markdown_content

        # Update markdown image syntax
        def replace_markdown_image(match):
            alt_text = match.group(1)
            original_src = match.group(2)
            new_src = path_mapping.get(original_src, original_src)
            return f"![{alt_text}]({new_src})"

        updated_content = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)", replace_markdown_image, updated_content
        )

        # Update HTML img tags
        def replace_html_image(match):
            full_tag = match.group(0)
            original_src = match.group(1)
            new_src = path_mapping.get(original_src, original_src)
            return full_tag.replace(
                f'src="{original_src}"', f'src="{new_src}"'
            ).replace(f"src='{original_src}'", f"src='{new_src}'")

        updated_content = re.sub(
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',
            replace_html_image,
            updated_content,
            flags=re.IGNORECASE,
        )

        return updated_content

    def _resolve_image_src(self, src: str) -> str:
        """
        Resolve image src using the page_url extracted at root level.

        Handles various relative URL patterns:
        - src="../Plan Setup Reference/PlanSetup/image.png" (parent directory)
        - src="image.png" (same directory)
        - src="/root/path/image.png" (root-relative)

        Args:
            src: The src attribute value from the HTML image tag

        Returns:
            Resolved full URL if possible, otherwise original src
        """
        if not self.url_mapper or not src:
            return src

        # Clean src - remove fragments and whitespace
        clean_src = src.strip().split("#")[0]

        # If src is already absolute, return as is
        if clean_src.startswith(("http://", "https://")):
            return clean_src

        # If we have a base_url, construct the full URL
        if self.url_mapper.base_url:
            if clean_src.startswith("/"):
                # Root-relative URL: src="/root/path/image.png"
                return f"{self.url_mapper.base_url}{clean_src}"
            elif clean_src.startswith("../"):
                # Parent directory relative: src="../Images/image.png"
                # Get current page URL to determine current directory
                current_url = self._get_current_page_url()
                if current_url:
                    resolved_url = self._resolve_relative_path(current_url, clean_src)
                    return f"{self.url_mapper.base_url}{resolved_url}"
                else:
                    # Fallback: treat as relative to base
                    return f"{self.url_mapper.base_url}/{clean_src.lstrip('../')}"
            else:
                # Same directory relative: src="image.png"
                # Get current page URL to determine current directory
                current_url = self._get_current_page_url()
                if current_url:
                    # Extract directory from current URL
                    current_dir = "/".join(current_url.split("/")[:-1])
                    if current_dir:
                        return f"{self.url_mapper.base_url}{current_dir}/{clean_src}"
                    else:
                        return f"{self.url_mapper.base_url}/{clean_src}"
                else:
                    # Fallback: treat as relative to base
                    return f"{self.url_mapper.base_url}/{clean_src}"

        # Fallback: return original src
        return src

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

    def _resolve_relative_path(self, current_url: str, relative_src: str) -> str:
        """
        Resolve relative path like '../Plan Setup Reference/PlanSetup/image.png'
        against current URL.

        Args:
            current_url: Current page URL (relative, like '/Annual Enrollment/page.htm')
            relative_src: Relative src (like '../Plan Setup Reference/PlanSetup/image.png')

        Returns:
            Resolved relative URL
        """
        # Split current URL into parts
        current_parts = [part for part in current_url.split("/") if part]

        # Split relative src into parts
        src_parts = [part for part in relative_src.split("/") if part]

        # Start with current directory (remove filename)
        if current_parts:
            current_parts = current_parts[:-1]  # Remove filename, keep directory

        # Process each part of the relative src
        for part in src_parts:
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

    def scan_process_images_directory(self) -> Dict[str, Any]:
        """
        Scan the process-images directory to understand structure.
        Useful for debugging and mapping validation.
        """
        if not self.process_images_dir.exists():
            return {"error": f"Directory not found: {self.process_images_dir}"}

        structure = {
            "base_dir": str(self.process_images_dir),
            "subdirectories": [],
            "total_images": 0,
            "image_extensions": set(),
            "files_by_extension": {},
        }

        for item in self.process_images_dir.rglob("*"):
            if item.is_file():
                extension = item.suffix.lower()
                if extension in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
                    structure["total_images"] += 1
                    structure["image_extensions"].add(extension)

                    if extension not in structure["files_by_extension"]:
                        structure["files_by_extension"][extension] = []
                    structure["files_by_extension"][extension].append(
                        str(item.relative_to(self.process_images_dir))
                    )

            elif item.is_dir() and item != self.process_images_dir:
                rel_path = str(item.relative_to(self.process_images_dir))
                if "/" not in rel_path:  # Only direct subdirectories
                    structure["subdirectories"].append(rel_path)

        # Convert set to list for JSON serialization
        structure["image_extensions"] = list(structure["image_extensions"])

        return structure


# Utility functions for common image operations
def extract_and_verify_images(
    markdown_content: str, process_images_dir: str = None
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Extract images from markdown and verify their existence.
    Returns tuple of (images_list, verification_stats).
    """
    extractor = (
        ImageExtractorUtility(process_images_dir)
        if process_images_dir
        else ImageExtractorUtility()
    )

    images = extractor.extract_images_from_markdown(markdown_content)
    stats = extractor.verify_image_paths(images)

    return images, stats


def create_image_deployment_package(
    markdown_content: str, output_dir: str, process_images_dir: str = None
) -> Dict[str, Any]:
    """
    Create a complete image deployment package.
    Returns deployment information and statistics.
    """
    extractor = (
        ImageExtractorUtility(process_images_dir)
        if process_images_dir
        else ImageExtractorUtility()
    )

    # Extract images
    images = extractor.extract_images_from_markdown(markdown_content)

    # Copy images to output
    copied_images = extractor.copy_images_to_output(images, output_dir)

    # Create path mapping for markdown updates
    path_mapping = {}
    for image in copied_images:
        if "copied_relative" in image:
            path_mapping[image["original_src"]] = image["copied_relative"]

    # Update markdown content
    updated_markdown = extractor.update_markdown_image_paths(
        markdown_content, path_mapping
    )

    return {
        "images": copied_images,
        "path_mapping": path_mapping,
        "updated_markdown": updated_markdown,
        "statistics": extractor.verify_image_paths(copied_images),
    }
