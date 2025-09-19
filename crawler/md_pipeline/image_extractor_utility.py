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

    def __init__(self, process_images_dir: str = "crawler/process-images"):
        self.process_images_dir = Path(process_images_dir)
        self.image_patterns = [
            r"!\[([^\]]*)\]\(([^)]+)\)",  # Standard markdown images
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',  # HTML img tags
            r'src=["\']([^"\']+\.(?:png|jpg|jpeg|gif|svg|webp))["\']',  # Direct src attributes
        ]

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

        # Skip data URLs and external URLs for now
        if src.startswith(("data:", "http://", "https://")):
            return {
                "alt_text": alt_text,
                "original_src": src,
                "local_path": "",
                "exists": False,
                "type": "external" if src.startswith("http") else "data_url",
            }

        # Map to process-images structure
        local_path = self._map_to_process_images_path(src)

        return {
            "alt_text": alt_text,
            "original_src": src,
            "local_path": str(local_path) if local_path else "",
            "exists": local_path.exists() if local_path else False,
            "type": "local",
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

    def generate_image_manifest(
        self, images: List[Dict[str, str]], output_file: str
    ) -> None:
        """Generate a manifest file with all image information."""
        manifest = {
            "generated_at": str(Path().cwd()),
            "total_images": len(images),
            "images": images,
            "statistics": self.verify_image_paths(images),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

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

    # Generate manifest
    manifest_path = Path(output_dir) / "image_manifest.json"
    extractor.generate_image_manifest(copied_images, str(manifest_path))

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
        "manifest_path": str(manifest_path),
        "statistics": extractor.verify_image_paths(copied_images),
    }
