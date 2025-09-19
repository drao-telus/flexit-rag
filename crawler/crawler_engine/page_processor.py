"""Page processing module for filtering and cleaning HTML content."""

import re
from bs4 import BeautifulSoup

from .config import CrawlConfig


class PageProcessor:
    """Handles page filtering and cleaning operations"""

    def __init__(self, config: CrawlConfig):
        self.config = config

    def filter_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Filter out unwanted content from the page"""
        # Create a copy to avoid modifying the original
        filtered_soup = BeautifulSoup(str(soup), "html.parser")

        # Remove HTML comments
        self._remove_comments(filtered_soup)

        # Remove navigation elements
        nav_selectors = [
            "nav",
            ".off-canvas",
            ".title-bar",
            ".navigation",
            ".sidenav",
            ".TopicMenu",
            '[data-mc-ignore="true"]',
            ".menu-icon-container",
            ".central-account-wrapper",
            ".nav-search-wrapper",
        ]

        for selector in nav_selectors:
            for element in filtered_soup.select(selector):
                element.decompose()

        # Remove footer content
        footer_selectors = [
            "footer",
            ".TableStyle-SimpleWithPadding",
            'table[summary="footer"]',
        ]

        for selector in footer_selectors:
            for element in filtered_soup.select(selector):
                element.decompose()

        # Remove scripts and styles
        for element in filtered_soup(["script", "style"]):
            element.decompose()

        # Remove elements with specific data attributes
        for element in filtered_soup.find_all(attrs={"data-mc-ignore": "true"}):
            element.decompose()

        # Remove empty divs and spans
        for element in filtered_soup.find_all(["div", "span"]):
            if not element.get_text(strip=True) and not element.find_all():
                element.decompose()

        return filtered_soup

    def clean_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Clean and optimize the content"""
        # Start with filtered content
        cleaned_soup = self.filter_content(soup)

        # Remove unnecessary attributes
        unwanted_attrs = [
            "data-mc-*",
            "aria-*",
            "tabindex",
            "style",
            "data-sticky*",
            "data-resize*",
            "data-events*",
            "data-toggle",
            "aria-controls",
            "aria-expanded",
            "data-off-canvas*",
            "data-accordion*",
        ]

        for element in cleaned_soup.find_all():
            attrs_to_remove = []
            for attr in element.attrs:
                for unwanted in unwanted_attrs:
                    if unwanted.endswith("*"):
                        if attr.startswith(unwanted[:-1]):
                            attrs_to_remove.append(attr)
                    elif attr == unwanted:
                        attrs_to_remove.append(attr)

            for attr in attrs_to_remove:
                del element.attrs[attr]

        # Clean up whitespace
        for element in cleaned_soup.find_all(text=True):
            if element.parent.name not in ["script", "style"]:
                cleaned_text = re.sub(r"\s+", " ", element.strip())
                if cleaned_text != element:
                    element.replace_with(cleaned_text)

        # Remove empty elements
        for element in cleaned_soup.find_all():
            if (
                not element.get_text(strip=True)
                and not element.find_all(["img"])
                and element.name not in ["img"]
            ):
                element.decompose()

        return cleaned_soup

    def extract_main_content_only(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract only the main content area"""
        main_content = soup.find("div", {"role": "main", "id": "mc-main-content"})
        breadcrumbs_div = soup.find("div", class_="MCBreadcrumbsBox_0")
        if main_content:
            # Create a new document with just the main content
            new_soup = BeautifulSoup(
                '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Extracted Content</title></head><body></body></html>',
                "html.parser",
            )
            new_soup.body.append(breadcrumbs_div.extract())
            new_soup.body.append(main_content.extract())
            return new_soup

        return soup

    def _remove_comments(self, soup: BeautifulSoup) -> None:
        """Remove HTML comments and JavaScript comments from the soup"""
        from bs4 import Comment

        # Remove HTML comments (<!-- comment -->)
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()

        # Remove JavaScript comments from script tags
        for script in soup.find_all("script"):
            if script.string:
                # Remove single-line JavaScript comments (// comment)
                cleaned_script = re.sub(
                    r"//.*?$", "", script.string, flags=re.MULTILINE
                )
                # Remove multi-line JavaScript comments (/* comment */)
                cleaned_script = re.sub(
                    r"/\*.*?\*/", "", cleaned_script, flags=re.DOTALL
                )
                script.string = cleaned_script

        # Remove CSS comments from style tags
        for style in soup.find_all("style"):
            if style.string:
                # Remove CSS comments (/* comment */)
                cleaned_style = re.sub(r"/\*.*?\*/", "", style.string, flags=re.DOTALL)
                style.string = cleaned_style

        # Remove inline JavaScript comments from event handlers and other attributes
        for element in soup.find_all():
            for attr_name, attr_value in list(element.attrs.items()):
                if isinstance(attr_value, str) and any(
                    js_indicator in attr_name.lower()
                    for js_indicator in [
                        "onclick",
                        "onload",
                        "onchange",
                        "onsubmit",
                        "href",
                    ]
                ):
                    if "javascript:" in attr_value.lower():
                        # Clean JavaScript comments from inline handlers
                        cleaned_attr = re.sub(r"//.*?(?=;|$)", "", attr_value)
                        cleaned_attr = re.sub(
                            r"/\*.*?\*/", "", cleaned_attr, flags=re.DOTALL
                        )
                        element.attrs[attr_name] = cleaned_attr
