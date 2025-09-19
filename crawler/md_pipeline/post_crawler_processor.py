"""
Post-Crawler RAG Processor - Processes filtered pages after crawler completion.
This module provides a simple interface to convert crawler output to RAG JSON files.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from .md_pipeline_main import (
    MDPipelineOrchestrator,
    RAGProcessingConfig,
    PipelineConfig,
)
from logger_config import get_logger

logger = get_logger(__name__)


class PostCrawlerRAGProcessor:
    """
    Processes filtered HTML pages from crawler and generates RAG JSON outputs.
    This class provides a simple interface to be called after crawler completion.
    """

    def __init__(
        self,
        filtered_content_dir: str = "crawler/result_data/filtered_content",
        rag_output_dir: str = "crawler/result_data/rag_output",
    ):
        """
        Initialize the post-crawler RAG processor.

        Args:
            filtered_content_dir: Directory where crawler saves filtered HTML pages
            rag_output_dir: Directory where RAG JSON files will be saved
        """
        self.filtered_content_dir = filtered_content_dir
        self.rag_output_dir = rag_output_dir

        # Initialize the MD pipeline orchestrator
        pipeline_config = PipelineConfig(
            input_dir=filtered_content_dir,
            output_dir=rag_output_dir,
            process_images_dir="crawler/process-images",
            file_pattern="*.html",
            preserve_image_structure=True,
            generate_manifests=True,
            batch_size=10,
        )

        self.orchestrator = MDPipelineOrchestrator(pipeline_config)
        logger.info(f"PostCrawlerRAGProcessor initialized")
        logger.info(f"Input: {filtered_content_dir}")
        logger.info(f"Output: {rag_output_dir}")

    def process_filtered_pages(
        self,
        batch_size: int = 10,
        skip_existing: bool = False,
        enable_validation: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all filtered HTML pages and generate RAG JSON outputs.
        This is the main method to call after crawler engine completes.

        Args:
            batch_size: Number of files to process in each batch
            skip_existing: Skip files that already have RAG output (for resume capability)
            enable_validation: Validate RAG output after processing

        Returns:
            Dictionary containing processing results and statistics
        """
        logger.info("=" * 60)
        logger.info("Starting Post-Crawler RAG Processing")
        logger.info("=" * 60)

        # Check if filtered content directory exists
        filtered_path = Path(self.filtered_content_dir)
        if not filtered_path.exists():
            error_msg = f"Filtered content directory not found: {filtered_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "stats": None}

        # Count available HTML files
        html_files = list(filtered_path.glob("*.html"))
        if not html_files:
            warning_msg = f"No HTML files found in {filtered_path}"
            logger.warning(warning_msg)
            return {"success": False, "error": warning_msg, "stats": None}

        logger.info(f"Found {len(html_files)} HTML files to process")

        # Create RAG processing configuration
        rag_config = RAGProcessingConfig(
            filtered_content_dir=self.filtered_content_dir,
            rag_output_dir=self.rag_output_dir,
            process_images_dir="crawler/process-images",
            file_pattern="*.html",
            preserve_image_structure=True,
            generate_manifests=True,
            batch_size=batch_size,
            skip_existing=skip_existing,
            enable_validation=enable_validation,
        )

        # Process the filtered pages
        try:
            result = self.orchestrator.process_filtered_pages_batch(rag_config)

            if result["success"]:
                logger.info("=" * 60)
                logger.info("Post-Crawler RAG Processing COMPLETED Successfully!")
                logger.info("=" * 60)
                logger.info(
                    f"Processed: {result['stats'].processed_files}/{result['stats'].total_files} files"
                )
                logger.info(
                    f"Success rate: {(result['stats'].processed_files / result['stats'].total_files * 100):.1f}%"
                )
                logger.info(
                    f"Total chunks generated: {result['stats'].total_chunks_generated}"
                )
                logger.info(f"Output directory: {result['output_directory']}")
            else:
                logger.error("Post-Crawler RAG Processing FAILED")

            return result

        except Exception as e:
            error_msg = f"Error during RAG processing: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "stats": None}

    def process_single_filtered_page(self, html_file_path: str) -> Dict[str, Any]:
        """
        Process a single filtered HTML page and return its RAG JSON output.

        Args:
            html_file_path: Path to the HTML file to process

        Returns:
            Dictionary containing the RAG document data and processing info
        """
        html_path = Path(html_file_path)

        if not html_path.exists():
            return {
                "success": False,
                "error": f"File not found: {html_file_path}",
                "rag_data": None,
            }

        try:
            # Process the single file
            result = self.orchestrator.process_single_file(
                str(html_path), self.rag_output_dir
            )

            if result["success"]:
                # Load the generated RAG JSON
                rag_file_path = result["output_file"]
                with open(rag_file_path, "r", encoding="utf-8") as f:
                    rag_data = json.load(f)

                return {
                    "success": True,
                    "rag_data": rag_data,
                    "output_file": rag_file_path,
                    "processing_stats": result["stats"],
                }
            else:
                return {"success": False, "error": result["error"], "rag_data": None}

        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing file: {str(e)}",
                "rag_data": None,
            }

    def get_rag_output_for_page(self, page_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the RAG JSON output for a specific page by name.

        Args:
            page_name: Name of the page (without extension)

        Returns:
            RAG JSON data if found, None otherwise
        """
        rag_file_path = Path(self.rag_output_dir) / f"{page_name}_rag.json"

        if not rag_file_path.exists():
            logger.warning(f"RAG file not found: {rag_file_path}")
            return None

        try:
            with open(rag_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading RAG file {rag_file_path}: {e}")
            return None

    def list_available_rag_files(self) -> List[str]:
        """
        List all available RAG JSON files.

        Returns:
            List of RAG file names (without _rag.json suffix)
        """
        rag_dir = Path(self.rag_output_dir)
        if not rag_dir.exists():
            return []

        rag_files = list(rag_dir.glob("*_rag.json"))
        return [f.stem.replace("_rag", "") for f in rag_files]

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the RAG processing results.

        Returns:
            Dictionary containing processing summary statistics
        """
        rag_dir = Path(self.rag_output_dir)

        if not rag_dir.exists():
            return {
                "total_rag_files": 0,
                "total_chunks": 0,
                "total_images": 0,
                "processing_manifest_exists": False,
            }

        rag_files = list(rag_dir.glob("*_rag.json"))
        total_chunks = 0
        total_images = 0

        # Analyze RAG files
        for rag_file in rag_files:
            try:
                with open(rag_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total_chunks += data.get("total_chunks", 0)
                    total_images += len(data.get("images", []))
            except Exception as e:
                logger.warning(f"Error reading {rag_file}: {e}")

        # Check for processing manifest
        manifest_file = Path(self.debug_report_dir) / "rag_processing_manifest.json"
        manifest_exists = manifest_file.exists()

        summary = {
            "total_rag_files": len(rag_files),
            "total_chunks": total_chunks,
            "total_images": total_images,
            "processing_manifest_exists": manifest_exists,
            "output_directory": str(rag_dir),
        }

        # Add manifest data if available
        if manifest_exists:
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    summary["manifest_data"] = manifest_data
            except Exception as e:
                logger.warning(f"Error reading manifest: {e}")

        return summary


# Convenience functions for easy integration
def process_crawler_output(
    filtered_content_dir: str = "crawler/result_data/filtered_content",
    rag_output_dir: str = "crawler/result_data/rag_output",
    batch_size: int = 10,
    skip_existing: bool = False,
    enable_validation: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to process crawler output in one call.
    This is the main function to call after crawler engine completes.

    Args:
        filtered_content_dir: Directory where crawler saves filtered HTML pages
        rag_output_dir: Directory where RAG JSON files will be saved
        batch_size: Number of files to process in each batch
        skip_existing: Skip files that already have RAG output
        enable_validation: Validate RAG output after processing

    Returns:
        Dictionary containing processing results and statistics
    """
    processor = PostCrawlerRAGProcessor(filtered_content_dir, rag_output_dir)
    return processor.process_filtered_pages(
        batch_size, skip_existing, enable_validation
    )


def get_rag_json_for_page(
    page_name: str, rag_output_dir: str = "crawler/result_data/rag_output"
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get RAG JSON output for a specific page.

    Args:
        page_name: Name of the page (without extension)
        rag_output_dir: Directory where RAG JSON files are stored

    Returns:
        RAG JSON data if found, None otherwise
    """
    processor = PostCrawlerRAGProcessor(rag_output_dir=rag_output_dir)
    return processor.get_rag_output_for_page(page_name)


# Example usage
if __name__ == "__main__":
    # Example: Process all filtered pages after crawler completes
    result = process_crawler_output()

    if result["success"]:
        print(f"Successfully processed {result['stats'].processed_files} pages")
        print(f"Generated {result['stats'].total_chunks_generated} chunks")
        print(f"Output directory: {result['output_directory']}")
    else:
        print(f"Processing failed: {result['error']}")
