"""
Independent RAG Processing Entry Point
Processes filtered HTML pages from crawler to generate RAG JSON outputs.
This can be run independently multiple times for iterative refinement.
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import Optional

from .md_pipeline.md_pipeline_main import (
    MDPipelineOrchestrator,
    PipelineConfig,
    RAGProcessingConfig,
)
from logger_config import get_logger, init_logging

# Initialize logging
init_logging(level="INFO", use_colors=True)
logger = get_logger(__name__)


def create_rag_config_from_args(args) -> RAGProcessingConfig:
    """Create RAG processing configuration from command line arguments."""
    return RAGProcessingConfig(
        filtered_content_dir=args.filtered_content_dir,
        rag_output_dir=args.rag_output_dir,
        process_images_dir=args.process_images_dir,
        file_pattern=args.file_pattern,
        preserve_image_structure=args.preserve_image_structure,
        generate_manifests=args.generate_manifests,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        enable_validation=args.enable_validation,
    )


def create_rag_config_from_file(config_file: str) -> RAGProcessingConfig:
    """Create RAG processing configuration from JSON file."""
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    return RAGProcessingConfig(**config_data)


def process_filtered_to_rag(
    rag_config: Optional[RAGProcessingConfig] = None, config_file: Optional[str] = None
) -> dict:
    """
    Main function to process filtered HTML pages to RAG JSON outputs.

    Args:
        rag_config: RAG processing configuration
        config_file: Path to configuration file (alternative to rag_config)

    Returns:
        Dictionary containing processing results
    """
    if config_file:
        rag_config = create_rag_config_from_file(config_file)
    elif rag_config is None:
        # Use default configuration
        rag_config = RAGProcessingConfig()

    logger.info("Starting RAG processing of filtered pages")
    logger.info(f"Configuration: {rag_config}")

    try:
        # Create pipeline orchestrator with minimal config (we'll override directories)
        pipeline_config = PipelineConfig(
            input_dir=rag_config.filtered_content_dir,
            output_dir=rag_config.rag_output_dir,
            process_images_dir=rag_config.process_images_dir,
            batch_size=rag_config.batch_size,
        )

        orchestrator = MDPipelineOrchestrator(pipeline_config)

        # Process filtered pages
        result = orchestrator.process_filtered_pages_batch(rag_config)

        if result["success"]:
            logger.info("RAG processing completed successfully")
            logger.info(f"Processed {result['stats'].processed_files} files")
            logger.info(f"Generated {result['stats'].total_chunks_generated} chunks")
            logger.info(f"Output saved to: {result['output_directory']}")
        else:
            logger.error(
                f"RAG processing failed: {result.get('message', 'Unknown error')}"
            )

        return result

    except Exception as e:
        logger.error(f"RAG processing failed with exception: {e}")
        raise


def main():
    """Command-line interface for RAG processing."""
    parser = argparse.ArgumentParser(
        description="Process filtered HTML pages to RAG JSON outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output directories
    parser.add_argument(
        "--filtered-content-dir",
        default="crawler/result_data/filtered_content",
        help="Directory containing filtered HTML files from crawler",
    )
    parser.add_argument(
        "--rag-output-dir",
        default="crawler/result_data/rag_output",
        help="Directory to save RAG JSON files",
    )
    parser.add_argument(
        "--process-images-dir",
        default="crawler/process-images",
        help="Directory for image processing",
    )

    # Processing options
    parser.add_argument(
        "--file-pattern",
        default="*.html",
        help='File pattern to match (e.g., "*.html", "page_*.html")',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files to process in each batch",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have RAG output (resume capability)",
    )

    # Feature toggles
    parser.add_argument(
        "--no-preserve-image-structure",
        action="store_false",
        dest="preserve_image_structure",
        help="Do not preserve original image directory structure",
    )
    parser.add_argument(
        "--no-manifests",
        action="store_false",
        dest="generate_manifests",
        help="Do not generate manifest files",
    )
    parser.add_argument(
        "--no-validation",
        action="store_false",
        dest="enable_validation",
        help="Do not validate RAG output after processing",
    )

    # Configuration file
    parser.add_argument(
        "--config", help="JSON configuration file (overrides command line arguments)"
    )

    # Utility options
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List files that would be processed and exit",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing RAG output without processing",
    )

    args = parser.parse_args()

    try:
        # Create configuration
        if args.config:
            rag_config = create_rag_config_from_file(args.config)
            print(f"Using configuration from: {args.config}")
        else:
            rag_config = create_rag_config_from_args(args)

        # List files mode
        if args.list_files:
            filtered_path = Path(rag_config.filtered_content_dir)
            if not filtered_path.exists():
                print(f"ERROR: Directory not found: {filtered_path}")
                return 1

            html_files = list(filtered_path.glob(rag_config.file_pattern))
            print(f"Found {len(html_files)} HTML files in {filtered_path}:")
            for html_file in html_files:
                print(f"  - {html_file.name}")
            return 0

        # Validate only mode
        if args.validate_only:
            output_path = Path(rag_config.rag_output_dir)
            if not output_path.exists():
                print(f"ERROR: RAG output directory not found: {output_path}")
                return 1

            # Create minimal pipeline for validation
            pipeline_config = PipelineConfig(
                input_dir=rag_config.filtered_content_dir,
                output_dir=rag_config.rag_output_dir,
            )
            orchestrator = MDPipelineOrchestrator(pipeline_config)
            validation = orchestrator.validate_pipeline_output()

            print(f"Validation Results:")
            print(
                f"  Total RAG files: {validation['validation_results']['total_rag_files']}"
            )
            print(
                f"  Valid files: {validation['validation_results']['valid_rag_files']}"
            )
            print(
                f"  Files with images: {validation['validation_results']['files_with_images']}"
            )
            print(
                f"  Average chunks per file: {validation['validation_results']['average_chunks_per_file']:.1f}"
            )

            if validation["validation_results"]["issues"]:
                print(
                    f"  Issues found: {len(validation['validation_results']['issues'])}"
                )
                for issue in validation["validation_results"]["issues"]:
                    print(
                        f"    - {issue['file']}: {issue['type']} - {issue['details']}"
                    )
            else:
                print("  No issues found!")

            return 0

        # Normal processing
        result = process_filtered_to_rag(rag_config=rag_config)

        if result["success"]:
            print(f"\n✅ RAG processing completed successfully!")
            print(f"📁 Output directory: {result['output_directory']}")
            return 0
        else:
            print(
                f"\n❌ RAG processing failed: {result.get('message', 'Unknown error')}"
            )
            return 1

    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Detailed error information:")
        return 1


# Convenience functions for programmatic use
def process_default_directories():
    """Process with default directory configuration."""
    return process_filtered_to_rag()


def process_with_custom_dirs(filtered_dir: str, output_dir: str):
    """Process with custom input and output directories."""
    config = RAGProcessingConfig(
        filtered_content_dir=filtered_dir, rag_output_dir=output_dir
    )
    return process_filtered_to_rag(rag_config=config)


def process_with_resume(filtered_dir: str, output_dir: str):
    """Process with resume capability (skip existing files)."""
    config = RAGProcessingConfig(
        filtered_content_dir=filtered_dir, rag_output_dir=output_dir, skip_existing=True
    )
    return process_filtered_to_rag(rag_config=config)


if __name__ == "__main__":
    exit(main())
