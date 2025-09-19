"""Refactored main module using modular components."""

import logging
from .crawler_engine.config import CrawlConfig
from .crawler_engine.main import EnhancedMadCapExtractor
from logger_config import init_logging, get_logger

# Initialize enhanced logging with traceback support and colors
init_logging(
    level=logging.INFO, use_colors=True, include_traceback=True, include_locals=True
)
logger = get_logger(__name__)


async def main():
    """Main execution function for testing with modular components"""
    config = CrawlConfig(
        base_url="crawler/result_data/raw_html/",
        # base_url="https://d3u2d4xznamk2r.cloudfront.net",
        max_pages=1000,
    )

    extractor = EnhancedMadCapExtractor(config)

    try:
        logger.info("Starting enhanced crawl with modular components...")
        await extractor.crawl_site()

        # Generate summary report
        summary = await extractor.generate_summary_report()

        logger.info(f"Processing completed successfully!")
        logger.info(f"Files saved in: {extractor.config.result_data__dir}")

    except Exception as e:
        logger.error(f"Enhanced crawl failed: {e}")
        raise


async def execute_crawler():
    await main()
