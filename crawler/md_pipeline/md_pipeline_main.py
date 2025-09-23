"""
MD Pipeline Main Orchestrator - Coordinates the complete MD pipeline process.
Efficiently processes HTML to RAG documents without reading complete long files.
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .html_to_md_converter import HTMLToMDConverter
from .md_rag_processor import MDRAGProcessor, RAGDocument, get_file_processing_stats
from .image_extractor_utility import ImageExtractorUtility


@dataclass
class PipelineConfig:
    """Configuration for the MD pipeline."""

    input_dir: str
    output_dir: str
    process_images_dir: str = "crawler/process-images"
    file_pattern: str = (
        "*.html"  # Glob pattern to match files (e.g., "*.html" matches all HTML files)
    )
    preserve_image_structure: bool = True
    generate_manifests: bool = True
    batch_size: int = 10  # Process files in batches to manage memory


@dataclass
class RAGProcessingConfig:
    """Configuration specifically for RAG processing of filtered pages from crawler."""

    filtered_content_dir: str = (
        "crawler/result_data/filtered_content"  # Where crawler saves filtered HTML
    )
    rag_output_dir: str = "crawler/result_data/rag_output"
    debug_report_dir: str = "crawler/debug/report"  # Where to save RAG JSON files
    process_images_dir: str = "crawler/process-images"  # Directory for image processing
    file_pattern: str = "*.html"  # Process all HTML files from crawler
    preserve_image_structure: bool = True
    generate_manifests: bool = True
    batch_size: int = 10  # Number of files to process in each batch
    skip_existing: bool = (
        False  # Skip files that already have RAG output (for resume capability)
    )
    enable_validation: bool = True  # Validate RAG output after processing


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""

    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_processing_time: float = 0.0
    total_chunks_generated: int = 0
    total_images_processed: int = 0
    average_file_size_reduction: float = 0.0
    failed_file_details: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.failed_file_details is None:
            self.failed_file_details = []


class MDPipelineOrchestrator:
    """
    Main orchestrator for the MD pipeline.
    Coordinates HTML->MD->RAG conversion with efficient processing.
    """

    def __init__(
        self,
        config: PipelineConfig,
        enable_url_mapping: bool = True,
        base_url: str = "",
    ):
        self.config = config
        self.rag_processor = MDRAGProcessor(
            enable_url_mapping=enable_url_mapping,
            base_url=base_url,
        )

        # Initialize HTML converter with URL mapper from RAG processor
        self.html_converter = HTMLToMDConverter(
            url_mapper=self.rag_processor.url_mapper,
            current_url=None,  # Will be set per file during processing
        )

        self.image_extractor = ImageExtractorUtility(
            config.process_images_dir,
            url_mapper=self.rag_processor.url_mapper,
            current_url=None,  # Will be set per file during processing
        )

        # Ensure output directories exist
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def process_single_file(
        self, html_file: str, output_base_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single HTML file through the complete pipeline.
        Returns processing results and statistics.
        """
        html_path = Path(html_file)
        output_dir = (
            Path(output_base_dir) if output_base_dir else Path(self.config.output_dir)
        )

        start_time = time.time()
        result = {
            "file": str(html_path),
            "success": False,
            "processing_time": 0.0,
            "error": None,
            "stats": {},
        }

        try:
            # Step 1: Check file size for efficient processing strategy
            file_size = html_path.stat().st_size

            # Step 1.5: Set current URL for HTML converter and image extractor (for relative path resolution)
            self.html_converter.current_url = (
                html_path.stem
            )  # Use filename for URL mapping
            self.image_extractor.current_url = (
                html_path.stem
            )  # Use filename for URL mapping

            # Step 2: Convert HTML to Markdown
            if file_size > 1024 * 1024:  # > 1MB
                # For large files, read efficiently
                with open(html_path, "r", encoding="utf-8") as f:
                    # Read in chunks to avoid memory issues
                    html_content = self._read_large_file_efficiently(f)
            else:
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

            markdown_result = self.html_converter.convert_html_to_md(
                html_content, str(html_file)
            )
            markdown_content = markdown_result["markdown_content"]
            html_metadata = markdown_result["metadata"]
            images_metadata = markdown_result.get("images_metadata", [])

            # Step 3: Process to RAG document
            rag_document = self.rag_processor.process_markdown_to_rag(
                markdown_content,
                str(html_file),
                metadata={
                    "original_file_size": file_size,
                    "html_metadata": html_metadata,
                },
                images_metadata=images_metadata,
            )

            # Step 4: Save RAG document
            output_file = output_dir / f"{html_path.stem}_rag.json"
            self.rag_processor.save_rag_document(rag_document, str(output_file))

            # Step 5: Process images if any
            images_processed = 0
            if rag_document.images:
                # Extract images using ImageExtractorUtility for proper format
                extracted_images = self.image_extractor.extract_images_from_markdown(
                    markdown_content
                )

                if extracted_images:
                    images_dir = output_dir / "images" / html_path.stem
                    copied_images = self.image_extractor.copy_images_to_output(
                        extracted_images,
                        str(images_dir),
                        self.config.preserve_image_structure,
                    )
                    images_processed = len(
                        [img for img in copied_images if "copied_path" in img]
                    )

            # Step 6: Generate processing statistics
            processing_stats = get_file_processing_stats(rag_document)
            processing_stats["images_processed"] = images_processed
            processing_stats["original_file_size"] = file_size
            processing_stats["markdown_size"] = len(markdown_content)
            processing_stats["size_reduction"] = (
                (file_size - len(markdown_content)) / file_size if file_size > 0 else 0
            )

            result.update(
                {
                    "success": True,
                    "output_file": str(output_file),
                    "rag_document_id": rag_document.document_id,
                    "stats": processing_stats,
                }
            )

        except Exception as e:
            result["error"] = str(e)

        finally:
            result["processing_time"] = time.time() - start_time

        return result

    def _read_large_file_efficiently(self, file_handle) -> str:
        """Read large files in chunks to manage memory efficiently."""
        content_parts = []
        chunk_size = 8192  # 8KB chunks

        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            content_parts.append(chunk)

        return "".join(content_parts)

    def process_directory(self, input_dir: Optional[str] = None) -> PipelineStats:
        """
        Process all HTML files in a directory through the pipeline.
        Uses batch processing for memory efficiency.
        """
        input_path = Path(input_dir) if input_dir else Path(self.config.input_dir)

        # Find all HTML files
        html_files = list(input_path.glob(self.config.file_pattern))

        stats = PipelineStats(total_files=len(html_files))
        start_time = time.time()

        # Process files in batches
        for i in range(0, len(html_files), self.config.batch_size):
            batch = html_files[i : i + self.config.batch_size]

            for html_file in batch:
                result = self.process_single_file(str(html_file))

                if result["success"]:
                    stats.processed_files += 1
                    stats.total_chunks_generated += result["stats"].get(
                        "total_chunks", 0
                    )
                    stats.total_images_processed += result["stats"].get(
                        "images_processed", 0
                    )

                    # Track size reduction
                    if "size_reduction" in result["stats"]:
                        current_avg = stats.average_file_size_reduction
                        new_reduction = result["stats"]["size_reduction"]
                        stats.average_file_size_reduction = (
                            current_avg * (stats.processed_files - 1) + new_reduction
                        ) / stats.processed_files
                else:
                    stats.failed_files += 1
                    stats.failed_file_details.append(
                        {"file": result["file"], "error": result["error"]}
                    )

                print(
                    f"Processed: {html_file.name} ({'SUCCESS' if result['success'] else 'FAILED'})"
                )

        stats.total_processing_time = time.time() - start_time

        # Generate overall pipeline manifest
        if self.config.generate_manifests:
            self._generate_pipeline_manifest(stats)

        return stats

    def _generate_pipeline_manifest(self, stats: PipelineStats) -> None:
        """Generate a manifest file for the entire pipeline run."""
        manifest = {
            "pipeline_run": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "input_dir": self.config.input_dir,
                    "output_dir": self.config.output_dir,
                    "file_pattern": self.config.file_pattern,
                    "batch_size": self.config.batch_size,
                },
                "statistics": {
                    "total_files": stats.total_files,
                    "processed_files": stats.processed_files,
                    "failed_files": stats.failed_files,
                    "success_rate": (
                        stats.processed_files / stats.total_files
                        if stats.total_files > 0
                        else 0
                    ),
                    "total_processing_time": stats.total_processing_time,
                    "average_processing_time": (
                        stats.total_processing_time / stats.total_files
                        if stats.total_files > 0
                        else 0
                    ),
                    "total_chunks_generated": stats.total_chunks_generated,
                    "total_images_processed": stats.total_images_processed,
                    "average_file_size_reduction": stats.average_file_size_reduction,
                },
                "failed_files": stats.failed_file_details,
            }
        }

        manifest_file = Path(self.config.output_dir) / "pipeline_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def compare_with_existing_pipeline(
        self, existing_rag_dir: str, comparison_output: str = None
    ) -> Dict[str, Any]:
        """
        Compare results with existing RAG pipeline.
        Analyzes file sizes, processing efficiency, and content quality.
        """
        existing_path = Path(existing_rag_dir)
        new_path = Path(self.config.output_dir)

        comparison = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "directories": {"existing": str(existing_path), "new": str(new_path)},
            "file_comparison": [],
            "summary": {
                "total_files_compared": 0,
                "size_reduction_average": 0.0,
                "processing_improvement": 0.0,
                "files_with_size_reduction": 0,
            },
        }

        # Find matching files
        existing_files = {f.stem: f for f in existing_path.glob("*.json")}
        new_files = {f.stem.replace("_rag", ""): f for f in new_path.glob("*_rag.json")}

        common_files = set(existing_files.keys()) & set(new_files.keys())

        total_size_reduction = 0.0
        files_with_reduction = 0

        for file_stem in common_files:
            existing_file = existing_files[file_stem]
            new_file = new_files[file_stem]

            # Compare file sizes
            existing_size = existing_file.stat().st_size
            new_size = new_file.stat().st_size
            size_reduction = (
                (existing_size - new_size) / existing_size if existing_size > 0 else 0
            )

            if size_reduction > 0:
                files_with_reduction += 1

            total_size_reduction += size_reduction

            # Load and compare content structure
            try:
                with open(existing_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                with open(new_file, "r", encoding="utf-8") as f:
                    new_data = json.load(f)

                file_comparison = {
                    "file": file_stem,
                    "existing_size": existing_size,
                    "new_size": new_size,
                    "size_reduction": size_reduction,
                    "existing_chunks": (
                        len(existing_data.get("chunks", []))
                        if isinstance(existing_data, dict)
                        else 0
                    ),
                    "new_chunks": new_data.get("total_chunks", 0),
                    "processing_strategy": new_data.get(
                        "processing_strategy", "unknown"
                    ),
                }

                comparison["file_comparison"].append(file_comparison)

            except Exception as e:
                comparison["file_comparison"].append(
                    {"file": file_stem, "error": f"Failed to compare content: {str(e)}"}
                )

        # Calculate summary statistics
        comparison["summary"]["total_files_compared"] = len(common_files)
        comparison["summary"]["size_reduction_average"] = (
            total_size_reduction / len(common_files) if common_files else 0
        )
        comparison["summary"]["files_with_size_reduction"] = files_with_reduction
        comparison["summary"]["size_reduction_percentage"] = (
            (files_with_reduction / len(common_files) * 100) if common_files else 0
        )

        # Save comparison report
        if comparison_output:
            output_file = Path(comparison_output)
        else:
            output_file = Path(self.config.output_dir) / "pipeline_comparison.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        return comparison

    def validate_pipeline_output(self) -> Dict[str, Any]:
        """
        Validate the pipeline output for consistency and quality.
        Checks for common issues and provides recommendations.
        """
        output_path = Path(self.config.output_dir)
        validation = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_directory": str(output_path),
            "validation_results": {
                "total_rag_files": 0,
                "valid_rag_files": 0,
                "files_with_images": 0,
                "files_with_chunks": 0,
                "average_chunks_per_file": 0.0,
                "issues": [],
            },
        }

        rag_files = list(output_path.glob("*_rag.json"))
        validation["validation_results"]["total_rag_files"] = len(rag_files)

        total_chunks = 0
        valid_files = 0
        files_with_images = 0
        files_with_chunks = 0

        for rag_file in rag_files:
            try:
                with open(rag_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Validate structure
                required_fields = [
                    "document_id",
                    "chunks",
                    "total_chunks",
                    "processing_strategy",
                ]
                missing_fields = [
                    field for field in required_fields if field not in data
                ]

                if missing_fields:
                    validation["validation_results"]["issues"].append(
                        {
                            "file": rag_file.name,
                            "type": "missing_fields",
                            "details": missing_fields,
                        }
                    )
                    continue

                valid_files += 1
                chunk_count = data.get("total_chunks", 0)
                total_chunks += chunk_count

                if chunk_count > 0:
                    files_with_chunks += 1

                if data.get("images"):
                    files_with_images += 1

            except Exception as e:
                validation["validation_results"]["issues"].append(
                    {
                        "file": rag_file.name,
                        "type": "validation_error",
                        "details": str(e),
                    }
                )

        # Calculate averages
        validation["validation_results"]["valid_rag_files"] = valid_files
        validation["validation_results"]["files_with_images"] = files_with_images
        validation["validation_results"]["files_with_chunks"] = files_with_chunks
        validation["validation_results"]["average_chunks_per_file"] = (
            total_chunks / valid_files if valid_files > 0 else 0
        )

        return validation

    def process_filtered_pages_batch(
        self, rag_config: RAGProcessingConfig
    ) -> Dict[str, Any]:
        """
        Process all filtered HTML pages from crawler and generate RAG JSON outputs.
        This method can be run independently multiple times for iterative refinement.

        Args:
            rag_config: Configuration for RAG processing

        Returns:
            Dictionary containing processing results and statistics
        """
        print(f"Starting RAG processing of filtered pages...")
        print(f"Input directory: {rag_config.filtered_content_dir}")
        print(f"Output directory: {rag_config.rag_output_dir}")

        # Ensure directories exist
        filtered_path = Path(rag_config.filtered_content_dir)
        output_path = Path(rag_config.rag_output_dir)

        if not filtered_path.exists():
            raise FileNotFoundError(
                f"Filtered content directory not found: {filtered_path}"
            )

        output_path.mkdir(parents=True, exist_ok=True)

        # Find all HTML files from crawler
        html_files = list(filtered_path.glob(rag_config.file_pattern))

        if not html_files:
            print(f"No HTML files found in {filtered_path}")
            return {
                "success": False,
                "message": "No HTML files found to process",
                "stats": PipelineStats(),
            }

        print(f"Found {len(html_files)} HTML files to process")

        # Initialize processing statistics
        stats = PipelineStats(total_files=len(html_files))
        start_time = time.time()

        # Track processed files for resume capability
        existing_rag_files = set()
        if rag_config.skip_existing:
            existing_rag_files = {
                f.stem.replace("_rag", "") for f in output_path.glob("*_rag.json")
            }
            print(
                f"Skip existing enabled: {len(existing_rag_files)} files already processed"
            )

        # Process files in batches
        for i in range(0, len(html_files), rag_config.batch_size):
            batch = html_files[i : i + rag_config.batch_size]
            batch_num = i // rag_config.batch_size + 1
            total_batches = (
                len(html_files) + rag_config.batch_size - 1
            ) // rag_config.batch_size

            print(
                f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} files)"
            )

            for html_file in batch:
                # Skip if already processed (resume capability)
                if rag_config.skip_existing and html_file.stem in existing_rag_files:
                    print(f"Skipping (already exists): {html_file.name}")
                    stats.processed_files += 1
                    continue

                # Process single file
                result = self.process_single_file(str(html_file), str(output_path))

                if result["success"]:
                    stats.processed_files += 1
                    stats.total_chunks_generated += result["stats"].get(
                        "total_chunks", 0
                    )
                    stats.total_images_processed += result["stats"].get(
                        "images_processed", 0
                    )

                    # Track size reduction
                    if "size_reduction" in result["stats"]:
                        current_avg = stats.average_file_size_reduction
                        new_reduction = result["stats"]["size_reduction"]
                        if stats.processed_files > 0:
                            stats.average_file_size_reduction = (
                                current_avg * (stats.processed_files - 1)
                                + new_reduction
                            ) / stats.processed_files
                        else:
                            stats.average_file_size_reduction = new_reduction

                    print(f"✓ {html_file.name} -> {Path(result['output_file']).name}")
                else:
                    stats.failed_files += 1
                    stats.failed_file_details.append(
                        {"file": result["file"], "error": result["error"]}
                    )
                    print(f"✗ {html_file.name} - ERROR: {result['error']}")

        stats.total_processing_time = time.time() - start_time

        # Generate processing manifest
        if rag_config.generate_manifests:
            self._generate_rag_processing_manifest(rag_config, stats)

        # Image enhancement is now handled during HTML-to-MD conversion
        # Enhanced metadata is embedded directly in the markdown content
        image_mapping_stats = None

        # Run validation if enabled
        validation_results = None
        if rag_config.enable_validation:
            print(f"\nValidating RAG output...")
            # Temporarily update config for validation
            original_output_dir = self.config.output_dir
            self.config.output_dir = rag_config.rag_output_dir
            validation_results = self.validate_pipeline_output()
            self.config.output_dir = original_output_dir

            print(
                f"Validation complete: {validation_results['validation_results']['valid_rag_files']} valid files"
            )

        # Print summary
        print(f"\n{'=' * 50}")
        print(f"RAG Processing Complete!")
        print(f"{'=' * 50}")
        print(f"Total files: {stats.total_files}")
        print(f"Processed: {stats.processed_files}")
        print(f"Failed: {stats.failed_files}")
        print(f"Success rate: {(stats.processed_files / stats.total_files * 100):.1f}%")
        print(f"Total chunks generated: {stats.total_chunks_generated}")
        print(f"Total images processed: {stats.total_images_processed}")
        print(f"Average size reduction: {stats.average_file_size_reduction:.2%}")
        print(f"Processing time: {stats.total_processing_time:.2f} seconds")
        print(f"Output directory: {output_path}")

        return {
            "success": True,
            "stats": stats,
            "validation": validation_results,
            "image_mapping_stats": image_mapping_stats,
            "output_directory": str(output_path),
            "config": rag_config,
        }

    def _generate_rag_processing_manifest(
        self, rag_config: RAGProcessingConfig, stats: PipelineStats
    ) -> None:
        """Generate a manifest file specifically for RAG processing run."""
        manifest = {
            "rag_processing_run": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "filtered_content_dir": rag_config.filtered_content_dir,
                    "rag_output_dir": rag_config.rag_output_dir,
                    "file_pattern": rag_config.file_pattern,
                    "batch_size": rag_config.batch_size,
                    "skip_existing": rag_config.skip_existing,
                    "enable_validation": rag_config.enable_validation,
                },
                "statistics": {
                    "total_files": stats.total_files,
                    "processed_files": stats.processed_files,
                    "failed_files": stats.failed_files,
                    "success_rate": (
                        stats.processed_files / stats.total_files
                        if stats.total_files > 0
                        else 0
                    ),
                    "total_processing_time": stats.total_processing_time,
                    "average_processing_time": (
                        stats.total_processing_time / stats.total_files
                        if stats.total_files > 0
                        else 0
                    ),
                    "total_chunks_generated": stats.total_chunks_generated,
                    "total_images_processed": stats.total_images_processed,
                    "average_file_size_reduction": stats.average_file_size_reduction,
                },
                "failed_files": stats.failed_file_details,
            }
        }

        manifest_file = (
            Path(rag_config.debug_report_dir) / "rag_processing_manifest.json"
        )
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Manifest saved: {manifest_file}")


# CLI interface functions
def create_pipeline_from_config(config_file: str) -> MDPipelineOrchestrator:
    """Create pipeline orchestrator from configuration file."""
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    config = PipelineConfig(**config_data)
    return MDPipelineOrchestrator(config)


def run_pipeline_cli():
    """Command-line interface for running the MD pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MD Pipeline - Convert HTML to RAG documents"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Input directory containing HTML files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for RAG documents"
    )
    parser.add_argument("--config", help="Configuration file (JSON)")
    parser.add_argument(
        "--compare-with", help="Directory with existing RAG files for comparison"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate pipeline output"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for processing"
    )

    args = parser.parse_args()

    if args.config:
        orchestrator = create_pipeline_from_config(args.config)
    else:
        config = PipelineConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        orchestrator = MDPipelineOrchestrator(config)

    # Run pipeline
    print(f"Starting MD pipeline processing...")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")

    stats = orchestrator.process_directory()

    print(f"\nPipeline completed!")
    print(f"Processed: {stats.processed_files}/{stats.total_files} files")
    print(f"Failed: {stats.failed_files} files")
    print(f"Total chunks generated: {stats.total_chunks_generated}")
    print(f"Total images processed: {stats.total_images_processed}")
    print(f"Average size reduction: {stats.average_file_size_reduction:.2%}")
    print(f"Total processing time: {stats.total_processing_time:.2f} seconds")

    # Run comparison if requested
    if args.compare_with:
        print(f"\nComparing with existing pipeline: {args.compare_with}")
        comparison = orchestrator.compare_with_existing_pipeline(args.compare_with)
        print(f"Files compared: {comparison['summary']['total_files_compared']}")
        print(
            f"Average size reduction: {comparison['summary']['size_reduction_average']:.2%}"
        )
        print(
            f"Files with size reduction: {comparison['summary']['files_with_size_reduction']}"
        )

    # Run validation if requested
    if args.validate:
        print(f"\nValidating pipeline output...")
        validation = orchestrator.validate_pipeline_output()
        print(f"Valid RAG files: {validation['validation_results']['valid_rag_files']}")
        print(
            f"Files with images: {validation['validation_results']['files_with_images']}"
        )
        print(
            f"Average chunks per file: {validation['validation_results']['average_chunks_per_file']:.1f}"
        )
        if validation["validation_results"]["issues"]:
            print(f"Issues found: {len(validation['validation_results']['issues'])}")


if __name__ == "__main__":
    run_pipeline_cli()
