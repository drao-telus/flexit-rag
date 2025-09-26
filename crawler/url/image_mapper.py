#!/usr/bin/env python3
"""
Image Mapper for Flexit RAG Project

This script crawls all pages in crawler/result_data/filtered_content and extracts
images with related captions, generating an image_mapping_cache.json file.

Features:
- Processes all HTML files in filtered_content directory
- Normalizes image URLs using base_url and document context
- Extracts captions using multiple strategies
- Generates structured JSON output with actual filenames
- Includes both document URL and filename for better tracking
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote
from typing import Dict, List, Optional, Tuple
import re

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup4 is required. Install with: pip install beautifulsoup4")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImageMapper:
    """
    Image Mapper that creates mapping between filtered content filenames
    and their associated images with metadata from the image cache.
    """

    def __init__(
        self,
        image_cache_file: str = "crawler/url/image_mapping_cache.json",
        base_url: str = "",
    ):
        """
        Initialize Image mapper.

        Args:
            image_cache_file: Path to image mapping cache JSON file
            base_url: Base URL to prepend to relative URLs (e.g., "https://example.com").
        """
        self.image_cache_file = image_cache_file
        self.base_url = (
            base_url.rstrip("/")
            if base_url
            else "https://d3u2d4xznamk2r.cloudfront.net"
        )
        self.image_base_url = "https://d3u2d4xznamk2r.cloudfront.net"
        self.filename_to_images_map: Dict[str, List[Dict]] = {}
        self.cache_metadata: Dict = {}

        # For cache building functionality
        self.url_mapping = self.load_url_mapping()
        self.image_cache = {}
        self.processed_files = 0
        self.total_images = 0
        self.errors = []

        # Try to load existing cache
        self._load_image_cache()

    def reload_existing_cache(self) -> bool:
        """Load existing image mapping cache from file."""
        return self._load_image_cache()

    def get_images_for_filename(self, filename: str) -> List[Dict[str, str]]:
        """
        Get all images associated with a filename.

        Args:
            filename: Filtered content filename (with .html extension)

        Returns:
            List of image dictionaries with metadata, or empty list if not found
        """
        return self.filename_to_images_map.get(filename, [])

    def get_image_by_src(self, filename: str, src: str) -> Optional[Dict[str, str]]:
        """
        Get specific image by filename and src attribute.

        Args:
            filename: Filtered content filename (with .html extension)
            src: Image src attribute to match

        Returns:
            Image dictionary with metadata, or None if not found
        """
        images = self.get_images_for_filename(filename)
        for image in images:
            if image.get("src") == src:
                return image
        return None

    def build_mapping_cache(self) -> Dict[str, any]:
        """
        Build image mapping cache by processing filtered content files.
        Similar to URLMapper's build_mapping_cache - processes source files to build cache.

        Returns:
            Dictionary with mapping statistics and results
        """
        logger.info("Building image mapping cache from filtered content files")

        try:
            # Clear existing mappings and reset counters
            self.filename_to_images_map.clear()
            self.image_cache.clear()
            self.processed_files = 0
            self.total_images = 0
            self.errors = []

            # Process all files to build the cache
            output = self.process_all_files()

            # Extract results from output
            # metadata = output.get("metadata", {})
            document_images = output.get("document_images", {})

            # Build filename-based mapping from processed data
            processed_count = 0
            total_images = 0

            for document_url, doc_data in document_images.items():
                filename = doc_data.get("filename")
                images = doc_data.get("images", [])

                if not filename:
                    logger.warning(f"No filename found for document: {document_url}")
                    continue

                if images:
                    self.filename_to_images_map[filename] = images
                    total_images += len(images)
                    processed_count += 1

            # Update metadata for optimized cache
            self.cache_metadata = {
                "generated_at": datetime.now().isoformat(),
                "source_directory": "crawler/result_data/filtered_content",
                "total_documents": processed_count,
                "total_images": total_images,
                "description": "Filename-based image mapping cache optimized for RAG processing",
                "schema_version": "3.0",
            }

            # Save both the original format and optimized cache
            self.save_cache(output)  # Save original format
            self._save_optimized_cache()  # Save optimized format

            result = {
                "success": True,
                "total_documents": processed_count,
                "total_images": total_images,
                "cache_file": self.image_cache_file,
            }

            logger.info(
                f"Image mapping cache built successfully: {processed_count} documents, {total_images} images"
            )

            return result

        except Exception as e:
            logger.error(f"Error building image mapping cache: {e}")
            return {"success": False, "error": str(e)}

    def validate_mapping(self, filtered_content_dir: str) -> Dict[str, any]:
        """
        Validate mapping against actual filtered content files.

        Args:
            filtered_content_dir: Directory containing filtered HTML files

        Returns:
            Validation results with statistics and missing files
        """
        logger.info(f"Validating image mapping against {filtered_content_dir}")

        filtered_path = Path(filtered_content_dir)
        if not filtered_path.exists():
            return {"success": False, "error": f"Directory not found: {filtered_path}"}

        # Get all HTML files in filtered content directory
        html_files = list(filtered_path.glob("*.html"))
        existing_filenames = {f.name for f in html_files}

        # Check which mappings have corresponding files
        mapped_filenames = set(self.filename_to_images_map.keys())

        found_files = mapped_filenames & existing_filenames
        missing_files = mapped_filenames - existing_filenames
        unmapped_files = existing_filenames - mapped_filenames

        # Count images
        found_images = sum(
            len(self.filename_to_images_map[filename]) for filename in found_files
        )

        validation_result = {
            "success": True,
            "total_mapped_files": len(mapped_filenames),
            "total_filtered_files": len(existing_filenames),
            "found_files": len(found_files),
            "found_images": found_images,
            "missing_files": list(missing_files),
            "unmapped_files": list(unmapped_files),
            "coverage_percentage": (
                (len(found_files) / len(mapped_filenames) * 100)
                if mapped_filenames
                else 0
            ),
        }

        logger.info(
            f"Validation complete: {len(found_files)}/{len(mapped_filenames)} mapped files have corresponding content"
        )

        if missing_files:
            logger.warning(
                f"Missing files: {len(missing_files)} mapped files have no corresponding filtered content"
            )

        if unmapped_files:
            logger.info(
                f"Unmapped files: {len(unmapped_files)} filtered files have no image mapping"
            )

        return validation_result

    def get_mapping_stats(self) -> Dict[str, any]:
        """Get statistics about current mapping."""
        total_images = sum(
            len(images) for images in self.filename_to_images_map.values()
        )

        return {
            "total_files_with_images": len(self.filename_to_images_map),
            "total_images": total_images,
            "cache_file": self.image_cache_file,
            "cache_exists": Path(self.image_cache_file).exists(),
            "metadata": self.cache_metadata,
            "sample_mappings": dict(
                list(self.filename_to_images_map.items())[:3]
            ),  # First 3 for preview
        }

    def _load_image_cache(self) -> bool:
        """Load existing image mapping cache from file."""
        cache_path = Path(self.image_cache_file)

        if not cache_path.exists():
            logger.info(f"No existing image cache found at {cache_path}")
            return False

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Check if it's the optimized format (schema version 3.0)
            metadata = cache_data.get("metadata", {})
            self.filename_to_images_map = cache_data.get("filename_to_images", {})
            self.cache_metadata = metadata
            logger.info(
                f"Loaded optimized image cache: {len(self.filename_to_images_map)} files with images"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading image cache: {e}")
            return False

    def _load_raw_cache(self) -> Dict:
        """Load raw image cache data from file."""
        cache_path = Path(self.image_cache_file)

        if not cache_path.exists():
            logger.warning(f"Image cache file not found: {cache_path}")
            return {}

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading raw image cache: {e}")
            return {}

    def _save_optimized_cache(self) -> None:
        """Save optimized mapping to cache file."""
        cache_data = {
            "metadata": self.cache_metadata,
            "filename_to_images": self.filename_to_images_map,
        }

        # Ensure cache directory exists
        cache_path = Path(self.image_cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Optimized image cache saved to {cache_path}")

        except Exception as e:
            logger.error(f"Error saving optimized image cache: {e}")

    def load_url_mapping(self) -> Dict:
        """Load the URL mapping cache from JSON file."""
        try:
            mapping_path = Path("crawler/url/url_mapping_cache.json")
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("filename_to_url", {})
        except Exception as e:
            logger.error(f"Failed to load URL mapping: {e}")
            return {}

    def get_document_url(self, filename: str) -> Optional[str]:
        """Get the full document URL from filename."""
        # Remove .html extension and get base filename
        base_filename = filename.replace(".html", "")

        if base_filename in self.url_mapping:
            relative_url = self.url_mapping[base_filename]
            return urljoin(self.base_url, relative_url)

        logger.warning(f"No URL mapping found for: {base_filename}")
        return None

    def _encode_url_path(self, url_path: str) -> str:
        """
        Encode URL path components to handle spaces and special characters.

        Args:
            url_path: URL path that may contain spaces and special characters

        Returns:
            URL-encoded path with spaces converted to %20
        """
        if not url_path:
            return url_path

        # Split the path into components to preserve the directory structure
        parts = url_path.split("/")
        encoded_parts = []

        for part in parts:
            if part:  # Skip empty parts
                # Encode each part individually, preserving forward slashes
                # Use quote with safe='' to encode spaces and special chars but preserve basic URL structure
                encoded_part = quote(part, safe="")
                encoded_parts.append(encoded_part)
            else:
                encoded_parts.append(
                    part
                )  # Preserve empty parts (leading/trailing slashes)

        return "/".join(encoded_parts)

    def normalize_image_url(self, img_src: str, document_url: str) -> str:
        """
        Normalize image URL based on document context with proper URL encoding.

        Handles various formats:
        - images/logo.png
        - /images/logo.png
        - ../images/logo.png
        - ../../images/logo.png

        Also removes thumbnail suffixes to get actual full-size image URLs:
        - filename_thumb_0_200.png -> filename.png

        Properly encodes spaces and special characters in URLs.
        """
        if not img_src:
            return ""

        # If already absolute URL, encode and return
        if img_src.startswith(("http://", "https://")):
            # For absolute URLs, we need to be more careful about encoding
            # Split into base and path components
            if "://" in img_src:
                protocol_and_domain, path = img_src.split("://", 1)
                if "/" in path:
                    domain, url_path = path.split("/", 1)
                    encoded_path = self._encode_url_path("/" + url_path)
                    return f"{protocol_and_domain}://{domain}{encoded_path}"
                else:
                    return img_src
            return self._encode_url_path(img_src)

        # Parse document URL to get base path
        parsed_doc = urlparse(document_url)
        doc_path = parsed_doc.path

        # Handle absolute paths (starting with /)
        if img_src.startswith("/"):
            encoded_path = self._encode_url_path(img_src)
            normalized_url = urljoin(self.image_base_url, encoded_path)
            return normalized_url

        # Handle relative paths
        if img_src.startswith("../"):
            # Count number of ../ to determine how many directories to go up
            up_count = 1
            remaining_path = img_src
            while remaining_path.startswith("../"):
                up_count += 1
                remaining_path = remaining_path[3:]

            # Split document path and go up the required number of directories
            path_parts = doc_path.strip("/").split("/")
            if len(path_parts) > up_count:
                base_parts = path_parts[:-up_count] if up_count > 0 else path_parts
                base_path = "/".join(base_parts)
                normalized_path = (
                    f"/{base_path}/{remaining_path}"
                    if base_path
                    else f"/{remaining_path}"
                )
            else:
                normalized_path = f"/{remaining_path}"

            encoded_path = self._encode_url_path(normalized_path)
            normalized_url = urljoin(self.image_base_url, encoded_path)
            return normalized_url

        # Handle simple relative paths (no ../)
        doc_dir = "/".join(doc_path.strip("/").split("/")[:-1])
        if doc_dir:
            normalized_path = f"/{doc_dir}/{img_src}"
        else:
            normalized_path = f"/{img_src}"

        encoded_path = self._encode_url_path(normalized_path)
        normalized_url = urljoin(self.image_base_url, encoded_path)
        return normalized_url

    def remove_thumbnail_suffix(self, url: str) -> str:
        """
        Remove thumbnail suffix from image URL to get actual full-size image URL.

        Converts: filename_thumb_0_200.png -> filename.png
        Strategy: Remove everything after "_thumb_" including "_thumb_" itself
        """
        if not url:
            return url

        # Extract the path part from URL
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split("/")

        if not path_parts:
            return url

        # Get the filename (last part of path)
        filename = path_parts[-1]

        # Check if filename contains thumbnail pattern
        if "_thumb_" in filename:
            # Find the position of "_thumb_" and get everything before it
            thumb_index = filename.find("_thumb_")
            base_name = filename[:thumb_index]

            # Get the file extension from the original filename
            original_ext = Path(filename).suffix

            # Reconstruct the filename: base_name + extension
            new_filename = base_name + original_ext

            # Replace the filename in the path
            path_parts[-1] = new_filename

            # Reconstruct the URL
            new_path = "/".join(path_parts)
            return f"{parsed_url.scheme}://{parsed_url.netloc}{new_path}"

        return url

    def extract_caption(self, img_tag) -> str:
        """
        Extract caption for an image using the specific strategy.

        Only looks for next sibling paragraph with class "figure".
        Returns empty string if no caption found (image will be skipped).
        """
        caption = ""

        # Look for next sibling paragraph with class "figure" (as specified)
        try:
            parent_p = img_tag.find_parent("p")
            if parent_p:
                next_sibling = parent_p.find_next_sibling("p", class_="figure")
                if next_sibling:
                    caption = next_sibling.get_text().strip()
                    if caption:
                        return caption
        except Exception:
            pass

        return caption

    def extract_images_from_file(self, file_path: Path) -> List[Dict]:
        """Extract all images and their captions from an HTML file.

        Only processes images that meet both criteria:
        1. Have a parent <a> tag with href attribute
        2. Have a caption
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            soup = BeautifulSoup(content, "html.parser")
            images = []

            # Find all img tags
            img_tags = soup.find_all("img")

            # Get document URL for this file
            filename = file_path.name
            document_url = self.get_document_url(filename)

            if not document_url:
                logger.warning(f"Skipping {filename} - no URL mapping found")
                return []

            for img_tag in img_tags:
                try:
                    # Check 1: Must have parent <a> tag with href
                    parent_a = img_tag.find_parent("a")
                    if not parent_a or not parent_a.get("href"):
                        continue  # Skip this image - no parent <a> tag or no href

                    # Check 2: Must have caption
                    caption = self.extract_caption(img_tag)
                    if not caption:
                        continue  # Skip this image - no caption found

                    # Both conditions met - process the image
                    img_src = img_tag.get("src", "").strip()
                    if not img_src:
                        continue

                    # Get enhance_url from parent <a> tag href
                    href = parent_a.get("href").strip()
                    enhance_url = self.normalize_image_url(href, document_url)

                    # Normalize the image URL for image_url
                    normalized_url = self.normalize_image_url(img_src, document_url)

                    # Create image entry
                    image_entry = {
                        "image_url": normalized_url,
                        "enhance_url": enhance_url,
                        "description": caption,
                        "src": img_src,
                    }

                    images.append(image_entry)
                    self.total_images += 1

                except Exception as e:
                    logger.error(f"Error processing image in {filename}: {e}")
                    self.errors.append(f"Image processing error in {filename}: {e}")

            return images

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.errors.append(f"File processing error {file_path}: {e}")
            return []

    def process_all_files(self) -> Dict:
        """Process all HTML files in the filtered_content directory."""
        logger.info("Starting image extraction from filtered content...")

        # Path to filtered content directory
        filtered_content_dir = Path("crawler/result_data/filtered_content")

        if not filtered_content_dir.exists():
            raise FileNotFoundError(
                f"Filtered content directory not found: {filtered_content_dir}"
            )

        # Get all HTML files
        html_files = list(filtered_content_dir.glob("*.html"))
        total_files = len(html_files)

        logger.info(f"Found {total_files} HTML files to process")

        # Process each file
        for i, file_path in enumerate(html_files, 1):
            logger.info(f"Processing {i}/{total_files}: {file_path.name}")

            # Get document URL
            document_url = self.get_document_url(file_path.name)
            if not document_url:
                continue

            # Extract images from file
            images = self.extract_images_from_file(file_path)

            # Store in cache if images found - now including filename
            if images:
                # Create document entry with both URL and filename
                document_key = document_url
                self.image_cache[document_key] = {
                    "filename": file_path.name,
                    "document_url": document_url,
                    "images": images,
                }
                logger.info(f"  Found {len(images)} images")

            self.processed_files += 1

        # Generate final output structure
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_documents": self.processed_files,
                "total_images": self.total_images,
                "documents_with_images": len(self.image_cache),
                "description": "Image mapping cache with URLs, filenames and captions",
                "errors_count": len(self.errors),
                "schema_version": "2.0",
                "changes": [
                    "Added filename field to each document entry",
                    "Restructured document_images to include metadata per document",
                    "Maintained backward compatibility with image structure",
                ],
            },
            "document_images": self.image_cache,
        }

        if self.errors:
            output["processing_errors"] = self.errors[:50]  # Limit error list

        return output

    def save_cache(
        self, output: Dict, output_path: str = "crawler/url/image_mapping_cache.json"
    ):
        """Save the image cache to JSON file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"Image mapping cache saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            raise

    def migrate_from_old_format(
        self, old_cache_path: str = "crawler/url/image_mapping_cache.json"
    ) -> Dict:
        """
        Migrate from old cache format to new format that includes filenames.
        This method can be used to upgrade existing cache files.
        """
        try:
            with open(old_cache_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)

            # Check if it's already in new format
            if "schema_version" in old_data.get("metadata", {}):
                logger.info("Cache is already in new format")
                return old_data

            logger.info("Migrating cache from old format to new format...")

            # Create new structure
            new_cache = {}

            old_document_images = old_data.get("document_images", {})

            for document_url, images in old_document_images.items():
                # Try to find the filename from URL mapping
                filename = None
                for fname, url in self.url_mapping.items():
                    full_url = urljoin(self.base_url, url)
                    if full_url == document_url:
                        filename = fname + ".html"
                        break

                if not filename:
                    # Extract filename from URL as fallback
                    parsed_url = urlparse(document_url)
                    path_parts = parsed_url.path.strip("/").split("/")
                    if path_parts and path_parts[-1]:
                        filename = path_parts[-1]
                        if not filename.endswith(".html"):
                            filename += ".html"
                    else:
                        filename = "unknown.html"

                # Create new format entry
                new_cache[document_url] = {
                    "filename": filename,
                    "document_url": document_url,
                    "images": images,
                }

            # Update metadata
            old_metadata = old_data.get("metadata", {})
            new_metadata = old_metadata.copy()
            new_metadata.update(
                {
                    "schema_version": "2.0",
                    "migrated_at": datetime.now().isoformat(),
                    "description": "Image mapping cache with URLs, filenames and captions",
                    "changes": [
                        "Added filename field to each document entry",
                        "Restructured document_images to include metadata per document",
                        "Maintained backward compatibility with image structure",
                    ],
                }
            )

            return {
                "metadata": new_metadata,
                "document_images": new_cache,
                "processing_errors": old_data.get("processing_errors", []),
            }

        except Exception as e:
            logger.error(f"Failed to migrate cache: {e}")
            raise


# Convenience functions for easy usage
def create_image_mapper(
    image_cache_file: str = "crawler/url/image_mapping_cache.json",
) -> ImageMapper:
    """Create image mapper with default settings."""
    return ImageMapper(image_cache_file)


def build_image_mapping_cache(
    image_cache_file: str = "crawler/url/image_mapping_cache.json",
) -> Dict[str, any]:
    """Build image mapping cache with default settings."""
    mapper = create_image_mapper(image_cache_file)
    return mapper.build_mapping_cache()


def validate_image_mapping(
    filtered_content_dir: str = "crawler/result_data/filtered_content",
) -> Dict[str, any]:
    """Validate image mapping against filtered content directory."""
    mapper = create_image_mapper()
    return mapper.validate_mapping(filtered_content_dir)


def main():
    """Main function to run the image mapper."""
    try:
        # Create image mapper instance
        mapper = ImageMapper()

        # Process all files
        output = mapper.process_all_files()

        # Save results
        mapper.save_cache(output)

        # Print summary
        metadata = output["metadata"]
        print("\n" + "=" * 50)
        print("IMAGE MAPPING COMPLETE")
        print("=" * 50)
        print(f"Documents processed: {metadata['total_documents']}")
        print(f"Documents with images: {metadata['documents_with_images']}")
        print(f"Total images found: {metadata['total_images']}")
        print(f"Processing errors: {metadata['errors_count']}")
        print(f"Schema version: {metadata.get('schema_version', '1.0')}")
        print(f"Output saved to: crawler/url/image_mapping_cache.json")

        if metadata["errors_count"] > 0:
            print(f"\nFirst few errors:")
            for error in output.get("processing_errors", [])[:5]:
                print(f"  - {error}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    # CLI interface for testing
    import argparse

    parser = argparse.ArgumentParser(description="Image Mapper Utility")
    parser.add_argument(
        "--build-cache", action="store_true", help="Build image mapping cache"
    )
    parser.add_argument("--validate", action="store_true", help="Validate mapping")
    parser.add_argument("--stats", action="store_true", help="Show mapping statistics")
    parser.add_argument(
        "--cache-file",
        default="crawler/url/image_mapping_cache.json",
        help="Image cache file",
    )
    parser.add_argument(
        "--filtered-dir",
        default="crawler/result_data/filtered_content",
        help="Filtered content directory",
    )

    args = parser.parse_args()

    mapper = ImageMapper(args.cache_file)

    if args.build_cache:
        result = mapper.build_mapping_cache()
        print(f"Cache build result: {result}")

    if args.validate:
        result = mapper.validate_mapping(args.filtered_dir)
        print(f"Validation result: {result}")

    if args.stats:
        stats = mapper.get_mapping_stats()
        print(f"Mapping statistics: {stats}")

    # If no specific action requested, run the main image extraction
    if not any([args.build_cache, args.validate, args.stats]):
        main()
