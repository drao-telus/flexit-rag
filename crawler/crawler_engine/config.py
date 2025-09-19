"""Configuration module for the MadCap crawler."""

from dataclasses import dataclass
from typing import List
from pathlib import Path

# Get project root (where your main script is)

project_root = Path(__file__).parent.absolute()


@dataclass
class CrawlConfig:
    """Configuration for the crawler"""

    base_url: str
    max_pages: int = 1000
    delay_between_requests: float = 2.0
    include_conditions: List[str] = None
    exclude_patterns: List[str] = None
    save_raw_html: bool = True
    save_filtered_content: bool = True
    concurrent_limit: int = 5
    batch_size: int = 20
    use_predefined_links: bool = True
    empty_learning_corner: bool = True
    result_data__dir = Path("crawler/result_data")
    processed_data_dir = Path("crawler/result_data/processed_data")
    page_links_file: str = "crawler/ulr/page_url.py"
    unmatched_links_file = "crawler/debug/report/unmatched_links_file.json"
    processing_summary_file = "crawler/debug/report/processing_summary.json"

    def __post_init__(self):
        if self.include_conditions is None:
            self.include_conditions = ["General.HTML5Only", None]
        if self.exclude_patterns is None:
            self.exclude_patterns = ["javascript:", "#", "mailto:"]
