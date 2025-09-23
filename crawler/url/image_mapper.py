#!/usr/bin/env python3
"""
Image Mapper for Flexit RAG Project

This script crawls all pages in crawler/result_data/filtered_content and extracts
images with related captions, generating an image_mapping_cache.json file.

Features:
- Processes all HTML files in filtered_content directory
- Normalizes image URLs using base_url and document context
- Extracts captions using multiple strategies
- Generates structured JSON output
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
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
    """Main class for extracting images and captions from HTML files."""

    def __init__(self):
        """Initialize the ImageMapper with configuration."""
        self.base_url = "https://d3u2d4xznamk2r.cloudfront.net"
        self.image_base_url = "https://d3u2d4xznamk2r.cloudfront.net"
        self.url_mapping = self.load_url_mapping()
        self.image_cache = {}
        self.processed_files = 0
        self.total_images = 0
        self.errors = []

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

    def normalize_image_url(self, img_src: str, document_url: str) -> str:
        """
        Normalize image URL based on document context.

        Handles various formats:
        - images/logo.png
        - /images/logo.png
        - ../images/logo.png
        - ../../images/logo.png

        Also removes thumbnail suffixes to get actual full-size image URLs:
        - filename_thumb_0_200.png -> filename.png
        """
        if not img_src:
            return ""

        # If already absolute URL, process it for thumbnail removal
        if img_src.startswith(("http://", "https://")):
            return self.remove_thumbnail_suffix(img_src)

        # Parse document URL to get base path
        parsed_doc = urlparse(document_url)
        doc_path = parsed_doc.path

        # Handle absolute paths (starting with /)
        if img_src.startswith("/"):
            normalized_url = urljoin(self.image_base_url, img_src)
            return self.remove_thumbnail_suffix(normalized_url)

        # Handle relative paths
        if img_src.startswith("../"):
            # Count number of ../ to determine how many directories to go up
            up_count = 0
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

            normalized_url = urljoin(self.image_base_url, normalized_path)
            return normalized_url

        # Handle simple relative paths (no ../)
        doc_dir = "/".join(doc_path.strip("/").split("/")[:-1])
        if doc_dir:
            normalized_path = f"/{doc_dir}/{img_src}"
        else:
            normalized_path = f"/{img_src}"

        normalized_url = urljoin(self.image_base_url, normalized_path)
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
        Extract caption for an image using multiple strategies.

        Primary strategy: Look for next sibling paragraph with class "figure"
        Fallback strategies: alt text, figcaption, title, nearby text
        """
        caption = ""

        # Strategy 1: Next sibling paragraph with class "figure" (as specified)
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

        # Strategy 2: Check alt attribute
        try:
            alt_text = img_tag.get("alt", "").strip()
            if alt_text and len(alt_text) > 3:  # Avoid very short alt texts
                return alt_text
        except Exception:
            pass

        # Strategy 3: Check title attribute
        try:
            title_text = img_tag.get("title", "").strip()
            if title_text and len(title_text) > 3:
                return title_text
        except Exception:
            pass

        return caption

    def extract_images_from_file(self, file_path: Path) -> List[Dict]:
        """Extract all images and their captions from an HTML file."""
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
                    # Get image source
                    img_src = img_tag.get("src", "").strip()
                    if not img_src:
                        continue

                    # Normalize the image URL
                    normalized_url = self.normalize_image_url(img_src, document_url)

                    # Extract caption
                    caption = self.extract_caption(img_tag)

                    # Create image entry
                    image_entry = {
                        "image_url": normalized_url,
                        "enhance_url": self.remove_thumbnail_suffix(normalized_url),
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

            # Store in cache if images found
            if images:
                self.image_cache[document_url] = images
                logger.info(f"  Found {len(images)} images")

            self.processed_files += 1

        # Generate final output structure
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_documents": self.processed_files,
                "total_images": self.total_images,
                "documents_with_images": len(self.image_cache),
                "description": "Image mapping cache with URLs and captions",
                "errors_count": len(self.errors),
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
        print(f"Output saved to: crawler/image_mapping_cache.json")

        if metadata["errors_count"] > 0:
            print(f"\nFirst few errors:")
            for error in output.get("processing_errors", [])[:5]:
                print(f"  - {error}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
