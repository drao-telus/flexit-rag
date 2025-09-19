"""Main crawler engine for orchestrating the crawling process."""

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime

import aiofiles
from playwright.async_api import async_playwright, Page
from bs4 import BeautifulSoup

from .config import CrawlConfig
from .page_processor import PageProcessor
from logger_config import get_logger, log_exception_with_context, exception_handler

logger = get_logger(__name__)


class EnhancedMadCapExtractor:
    """Enhanced MadCap Flare extractor with filtering and cleaning capabilities"""

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.visited_urls: Set[str] = set()
        self.content_store: Dict[str, Dict] = {}
        self.failed_urls: Set[str] = set()
        self.local_file_paths: List[str] = []
        self.processor = PageProcessor(config)

        # Create Result data directory structure
        self.setup_directories()

    def setup_directories(self):
        """Create the result data directory structure"""
        base_dir = Path(self.config.result_data__dir)

        # Create subdirectories
        self.raw_html_dir = base_dir / "raw_html"
        self.filtered_content_dir = base_dir / "filtered_content"
        self.processed_data_dir = base_dir / "processed_data"

        # Create directories
        for directory in [
            self.raw_html_dir,
            self.filtered_content_dir,
            self.processed_data_dir,
        ]:
            if directory.exists() and directory in [
                self.filtered_content_dir,
                self.processed_data_dir,
            ]:
                shutil.rmtree(directory)
                logger.info(f"Emptied existing directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created directory structure in {base_dir}")

    def load_predefined_page_links(self) -> List[str]:
        """Load predefined page links from the traverse file"""
        try:
            # Read the traverse file
            with open(self.config.page_links_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the JSON-like structure to extract page_links
            import ast

            parsed_data = ast.literal_eval(content)
            page_links = parsed_data.get("page_links", [])

            # Convert relative paths to full URLs
            full_urls = []
            for link in page_links:
                if link.startswith("/"):
                    # Remove leading slash and join with base URL
                    clean_link = link.lstrip("/")
                    full_url = urljoin(self.config.base_url + "/", clean_link)
                else:
                    full_url = urljoin(self.config.base_url, link)
                full_urls.append(full_url)

            logger.info(
                f"Loaded {len(full_urls)} predefined page links from {self.config.page_links_file}"
            )
            return full_urls

        except Exception as e:
            logger.error(f"Error loading predefined page links: {e}")
            logger.info("Falling back to dynamic link discovery")
            return []

    def get_safe_filename(self, url: str) -> str:
        """Generate a safe filename from URL"""
        if "http" not in self.config.base_url:
            return Path(url).stem

        parsed = urlparse(url)
        filename = parsed.path.replace("/", "_").replace("\\", "_")
        if filename.startswith("_"):
            filename = filename[1:]
        if not filename or filename == "_":
            filename = "index"

        # Remove file extension if present
        if filename.endswith(".htm") or filename.endswith(".html"):
            filename = filename.rsplit(".", 1)[0]

        # Ensure filename is safe
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
        return filename[:100]  # Limit length

    async def save_raw_html(self, url: str, original_soup: BeautifulSoup) -> None:
        """Save raw HTML"""
        safe_filename = self.get_safe_filename(url)

        # Save raw HTML
        if self.config.save_raw_html:
            raw_path = self.raw_html_dir / f"{safe_filename}.html"
            async with aiofiles.open(raw_path, "w", encoding="utf-8") as f:
                await f.write(str(original_soup))

    async def save_filtered_html(
        self, url: str, original_soup: BeautifulSoup
    ) -> BeautifulSoup:
        # Save filtered content (main content only) + cleaned HTML
        safe_filename = self.get_safe_filename(url)
        if self.config.save_filtered_content:
            filtered_soup = self.processor.extract_main_content_only(original_soup)
            cleaned_soup = self.processor.clean_content(filtered_soup)
            filtered_path = self.filtered_content_dir / f"{safe_filename}.html"
            async with aiofiles.open(filtered_path, "w", encoding="utf-8") as f:
                await f.write(str(cleaned_soup))
            return cleaned_soup

    async def crawl_site(self) -> Dict[str, Dict]:
        """Main crawling method"""
        if "http" in self.config.base_url:
            return await self._process_remote_site()
        else:
            return await self._process_local_files()

    async def _process_remote_site(self) -> Dict[str, Dict]:
        """Process remote website using predefined links with parallel processing"""
        logger.info(f"Starting crawl of {self.config.base_url}")

        # Load predefined page links if enabled
        if self.config.use_predefined_links:
            page_urls = self.load_predefined_page_links()
            if not page_urls:
                logger.warning(
                    "No predefined links found, falling back to single page processing"
                )
                page_urls = [self.config.base_url]
        else:
            page_urls = [self.config.base_url]

        # Limit pages based on max_pages configuration
        page_urls = page_urls[: self.config.max_pages]
        logger.info(
            f"Processing {len(page_urls)} pages with concurrent limit of {self.config.concurrent_limit}"
        )

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)

            try:
                # Process pages in concurrent batches
                await self._process_pages_in_batches(browser, page_urls)
            except Exception as e:
                logger.error(f"Error during batch processing: {e}")
            finally:
                await browser.close()

        logger.info(
            f"Crawl completed. Processed {len(self.content_store)} pages, Failed: {len(self.failed_urls)}"
        )

        # Automatically trigger RAG processing after crawling completes
        await self._trigger_post_crawler_rag_processing()

        return self.content_store

    async def _process_pages_in_batches(self, browser, page_urls: List[str]) -> None:
        """Process pages in concurrent batches"""
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrent_limit)

        # Process pages in batches
        for i in range(0, len(page_urls), self.config.batch_size):
            batch = page_urls[i : i + self.config.batch_size]
            logger.info(
                f"Processing batch {i // self.config.batch_size + 1}: {len(batch)} pages"
            )

            # Create tasks for concurrent processing
            tasks = []
            for url in batch:
                task = self._process_single_page_with_semaphore(browser, semaphore, url)
                tasks.append(task)

            # Execute batch concurrently
            await asyncio.gather(*tasks, return_exceptions=True)

            # Small delay between batches to be respectful to the server
            if i + self.config.batch_size < len(page_urls):
                await asyncio.sleep(self.config.delay_between_requests)

    async def _process_single_page_with_semaphore(
        self, browser, semaphore: asyncio.Semaphore, url: str
    ) -> None:
        """Process a single page with semaphore control"""
        async with semaphore:
            # Skip if already processed or should not be crawled
            if url in self.visited_urls or not self._should_crawl_url(url):
                return

            # Create a new page for this request
            page = await browser.new_page()

            try:
                await page.set_extra_http_headers(
                    {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                )

                await self._process_page_content(page, url)

            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                self.failed_urls.add(url)
            finally:
                await page.close()
                # Rate limiting delay
                await asyncio.sleep(self.config.delay_between_requests)

    async def _process_page_content(self, page: Page, url: str) -> None:
        """Process the content of a single page"""
        self.visited_urls.add(url)
        logger.info(f"Processing ({len(self.visited_urls)}): {url}")

        try:
            response = await page.goto(
                url, timeout=120000, wait_until="domcontentloaded"
            )
            if response.status != 200:
                logger.warning(f"HTTP {response.status} for {url}")
                self.failed_urls.add(url)
                return

            # Try to wait for selector, but continue if it fails
            try:
                await page.wait_for_selector(".MCBreadcrumbsLink", timeout=30000)
            except Exception as e:
                logger.warning(f"MCBreadcrumbsLink not found for {url}: {e}")

            html_content = await page.content()
            self._process_html(html_content, url)
            logger.debug(f"Successfully processed: {url}")

        except Exception as e:
            logger.error(f"Error processing content for {url}: {e}")
            self.failed_urls.add(url)
            raise

    async def _process_local_files(self) -> Dict[str, Dict]:
        """Process local HTML files"""
        logger.info(f"Processing local files from {self.config.base_url}")

        base_path = Path(self.config.base_url)
        if base_path.is_dir():
            html_files = list(base_path.glob("*.htm")) + list(base_path.glob("*.html"))
            self.local_file_paths = [str(f) for f in html_files]
        else:
            self.local_file_paths = [self.config.base_url]

        for file_path in self.local_file_paths[: self.config.max_pages]:
            try:
                await self._process_local_file(file_path)
            except Exception as e:
                logger.error(f"Error processing local file {file_path}: {e}")
                self.failed_urls.add(file_path)

        logger.info(
            f"Local file processing completed. Processed {len(self.content_store)} files"
        )

        # Automatically trigger RAG processing after local file processing completes
        await self._trigger_post_crawler_rag_processing()

        return self.content_store

    async def _process_local_file(self, file_path: str) -> None:
        """Process a single local HTML file"""

        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                html_content = await f.read()
                await self._process_html(html_content, file_path)

        except Exception as e:
            # Use enhanced logging with full traceback and context
            log_exception_with_context(
                logger, f"Error processing local file {file_path}", include_locals=True
            )
            self.failed_urls.add(file_path)

    async def _process_html(self, html_content, url):
        cleaned_html = self.clean_img_attributes(html_content)
        soup = BeautifulSoup(cleaned_html, "html.parser")

        # Save different variants of the page
        await self.save_raw_html(url, soup)
        await self.save_filtered_html(url, soup)

        # Extract semantic content
        # semantic_content = self.content_extractor.extract_semantic_content(soup, url)
        # self.content_store[url] = semantic_content

        logger.debug(f"Successfully processed: {url}")

    def _should_crawl_url(self, url: str) -> bool:
        """Determine if URL should be crawled"""
        parsed = urlparse(url)
        base_parsed = urlparse(self.config.base_url)

        if parsed.netloc != base_parsed.netloc:
            return False

        for pattern in self.config.exclude_patterns:
            if pattern in url:
                return False

        return url.endswith(".htm") or url.endswith(".html")

    async def generate_summary_report(self) -> Dict:
        """Generate a comprehensive summary report"""
        report = {
            "processing_summary": {
                "total_pages_processed": len(self.content_store),
                "failed_pages": len(self.failed_urls),
                "success_rate": len(self.content_store)
                / (len(self.content_store) + len(self.failed_urls))
                * 100
                if (len(self.content_store) + len(self.failed_urls)) > 0
                else 0,
                "timestamp": datetime.now().isoformat(),
            },
            "content_analysis": {
                "failed_page_url": list(self.failed_urls),
                "pages_with_images": sum(
                    1
                    for page in self.content_store.values()
                    if page["metadata"].get("has_images", False)
                ),
                "pages_with_tables": sum(
                    1
                    for page in self.content_store.values()
                    if page["metadata"].get("has_tables", False)
                ),
                "total_content_elements": sum(
                    len(page.get("content", [])) for page in self.content_store.values()
                ),
            },
            "directory_structure": {
                "raw_html": str(self.raw_html_dir),
                "filtered_content": str(self.filtered_content_dir),
                "processed_data": str(self.processed_data_dir),
            },
        }

        # Save summary report
        report_path = self.config.processing_summary_file
        async with aiofiles.open(report_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(report, indent=2, ensure_ascii=False))

        return report

    def clean_img_attributes(self, html):
        """Keep only src and class on img tags"""

        # Pattern to match img tags and capture src and class
        pattern = r'<img[^>]*?(?:src\s*=\s*["\']([^"\']*)["\'])?[^>]*?(?:class\s*=\s*["\']([^"\']*)["\'])?[^>]*/?>'

        def replace_img(match):
            full_match = match.group(0)
            # Extract src and class from the full match
            src = re.search(r'src\s*=\s*"([^"]*)"', full_match, re.I)
            if not src:
                src = re.search(r"src\s*=\s*'([^']*)'", full_match, re.I)
            class_attr = re.search(r'class\s*=\s*"([^"]*)"', full_match, re.I)
            if not class_attr:
                class_attr = re.search(r"class\s*=\s*'([^']*)'", full_match, re.I)

            src_val = src.group(1) if src else ""
            class_val = class_attr.group(1) if class_attr else ""

            # Build new tag
            attrs = []
            if class_val:
                attrs.append(f'class="{class_val}"')
            if src_val:
                attrs.append(f'src="{src_val}"')

            return f"<img {' '.join(attrs)}/>" if attrs else "<img/>"

        return re.sub(pattern, replace_img, html, flags=re.IGNORECASE)

    async def _trigger_post_crawler_rag_processing(self) -> None:
        """
        Trigger RAG processing after crawler completes.
        This method automatically processes filtered HTML pages to generate RAG JSON outputs.
        """
        try:
            logger.info("=" * 60)
            logger.info("TRIGGERING POST-CRAWLER RAG PROCESSING")
            logger.info("=" * 60)

            # Import the post-crawler processor
            from ..md_pipeline.post_crawler_processor import process_crawler_output

            # Set up directories for RAG processing
            filtered_content_dir = str(self.filtered_content_dir)
            rag_output_dir = str(self.filtered_content_dir.parent / "rag_output")

            logger.info(f"Filtered content directory: {filtered_content_dir}")
            logger.info(f"RAG output directory: {rag_output_dir}")

            # Check if there are filtered HTML files to process
            filtered_files = list(self.filtered_content_dir.glob("*.html"))
            if not filtered_files:
                logger.warning("No filtered HTML files found. Skipping RAG processing.")
                return

            logger.info(f"Found {len(filtered_files)} filtered HTML files to process")

            # Process the crawler output
            result = process_crawler_output(
                filtered_content_dir=filtered_content_dir,
                rag_output_dir=rag_output_dir,
                batch_size=10,
                skip_existing=False,
                enable_validation=True,
            )

            if result["success"]:
                logger.info("=" * 60)
                logger.info("POST-CRAWLER RAG PROCESSING COMPLETED SUCCESSFULLY!")
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
                logger.info(
                    f"Total images processed: {result['stats'].total_images_processed}"
                )
                logger.info(f"RAG output directory: {result['output_directory']}")

                # Log individual RAG files created
                rag_dir = Path(result["output_directory"])
                rag_files = list(rag_dir.glob("*_rag.json"))
                logger.info(f"Generated RAG files:")
                for rag_file in rag_files:
                    logger.info(f"  - {rag_file.name}")

            else:
                logger.error("POST-CRAWLER RAG PROCESSING FAILED!")
                logger.error(f"Error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Error during post-crawler RAG processing: {e}")
            logger.error("RAG processing failed, but crawler completed successfully")
