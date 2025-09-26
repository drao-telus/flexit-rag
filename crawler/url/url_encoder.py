#!/usr/bin/env python3
"""
URL Encoder Utility for properly encoding URLs with spaces and special characters.
Specifically designed to handle GitHub raw URLs and CloudFront URLs.
"""

import re
from urllib.parse import quote, unquote, urlparse, urlunparse
from typing import Optional

import logging

logger = logging.getLogger(__name__)


def encode_url_path(url: str) -> str:
    """
    Properly encode URL path components, especially spaces and special characters.
    Converts spaces to %20 and other special characters to their encoded equivalents.

    Args:
        url: The URL to encode (can be full URL or relative path)

    Returns:
        URL with properly encoded path components

    Examples:
        >>> encode_url_path("https://example.com/Annual Enrollment Guide/image.png")
        "https://example.com/Annual%20Enrollment%20Guide/image.png"

        >>> encode_url_path("/images/Employee Admin Ref/screenshot.jpg")
        "/images/Employee%20Admin%20Ref/screenshot.jpg"
    """
    if not url:
        return url

    try:
        # Check if it's a full URL or just a path
        if url.startswith(("http://", "https://")):
            # Parse full URL
            parsed = urlparse(url)
            # Encode the path component, preserving forward slashes
            encoded_path = quote(parsed.path, safe="/")
            # Reconstruct the URL with encoded path
            return urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    encoded_path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )
        else:
            # Just encode the path directly
            return quote(url, safe="/")

    except Exception as e:
        logger.warning(f"Error encoding URL '{url}': {e}")
        return url


def encode_github_raw_url(github_url: str) -> str:
    """
    Specifically encode GitHub raw URLs to ensure web accessibility.
    Handles the common pattern of GitHub raw URLs with spaces in directory names.

    Args:
        github_url: GitHub raw URL to encode

    Returns:
        Properly encoded GitHub raw URL

    Examples:
        >>> encode_github_raw_url("https://raw.githubusercontent.com/user/repo/main/images/Annual Enrollment Guide/image.png")
        "https://raw.githubusercontent.com/user/repo/main/images/Annual%20Enrollment%20Guide/image.png"
    """
    if not github_url:
        return github_url

    # Check if it's a GitHub raw URL
    if "raw.githubusercontent.com" not in github_url:
        logger.debug(f"Not a GitHub raw URL, using standard encoding: {github_url}")
        return encode_url_path(github_url)

    return encode_url_path(github_url)


def is_url_encoded(url: str) -> bool:
    """
    Check if a URL is already properly encoded.

    Args:
        url: URL to check

    Returns:
        True if URL is already encoded, False otherwise

    Examples:
        >>> is_url_encoded("https://example.com/Annual%20Enrollment%20Guide/image.png")
        True

        >>> is_url_encoded("https://example.com/Annual Enrollment Guide/image.png")
        False
    """
    if not url:
        return True

    try:
        # If decoding the URL changes it, then it was encoded
        decoded = unquote(url)
        return decoded != url
    except Exception:
        return False


def encode_relative_path(relative_path: str) -> str:
    """
    Encode a relative path, commonly used for image paths.

    Args:
        relative_path: Relative path to encode

    Returns:
        Encoded relative path

    Examples:
        >>> encode_relative_path("Annual Enrollment Guide/Images/screenshot.png")
        "Annual%20Enrollment%20Guide/Images/screenshot.png"
    """
    if not relative_path:
        return relative_path

    # Remove leading slash if present, then encode, then add it back
    leading_slash = relative_path.startswith("/")
    clean_path = relative_path.lstrip("/")

    encoded = quote(clean_path, safe="/")

    return f"/{encoded}" if leading_slash else encoded


def safe_encode_url(url: str) -> str:
    """
    Safely encode a URL, checking if it's already encoded to prevent double-encoding.

    Args:
        url: URL to encode

    Returns:
        Safely encoded URL

    Examples:
        >>> safe_encode_url("https://example.com/Annual Enrollment Guide/image.png")
        "https://example.com/Annual%20Enrollment%20Guide/image.png"

        >>> safe_encode_url("https://example.com/Annual%20Enrollment%20Guide/image.png")
        "https://example.com/Annual%20Enrollment%20Guide/image.png"  # No double encoding
    """
    if not url:
        return url

    # If already encoded, return as-is
    if is_url_encoded(url):
        logger.debug(f"URL already encoded: {url}")
        return url

    # Otherwise, encode it
    encoded = encode_url_path(url)
    logger.debug(f"Encoded URL: {url} -> {encoded}")
    return encoded


def validate_encoded_url(url: str) -> bool:
    """
    Validate that an encoded URL is properly formatted and accessible.

    Args:
        url: Encoded URL to validate

    Returns:
        True if URL appears to be properly encoded, False otherwise
    """
    if not url:
        return False

    try:
        # Check for common encoding issues
        if " " in url:
            logger.warning(f"URL contains unencoded spaces: {url}")
            return False

        # Check if URL can be parsed
        parsed = urlparse(url)
        if not parsed.scheme and not parsed.path:
            logger.warning(f"URL cannot be parsed: {url}")
            return False

        # Check for double encoding (e.g., %2520 instead of %20)
        if "%25" in url:
            logger.warning(f"URL may be double-encoded: {url}")
            return False

        return True

    except Exception as e:
        logger.warning(f"Error validating URL '{url}': {e}")
        return False


def batch_encode_urls(urls: list) -> list:
    """
    Encode a batch of URLs safely.

    Args:
        urls: List of URLs to encode

    Returns:
        List of encoded URLs
    """
    encoded_urls = []

    for url in urls:
        try:
            encoded = safe_encode_url(url)
            encoded_urls.append(encoded)
        except Exception as e:
            logger.error(f"Error encoding URL '{url}': {e}")
            encoded_urls.append(url)  # Keep original if encoding fails

    return encoded_urls


# Convenience functions for common use cases
def encode_cloudfront_url(cloudfront_url: str) -> str:
    """
    Encode CloudFront URLs specifically.

    Args:
        cloudfront_url: CloudFront URL to encode

    Returns:
        Encoded CloudFront URL
    """
    return safe_encode_url(cloudfront_url)


def encode_image_path(image_path: str) -> str:
    """
    Encode image paths, handling common image directory structures.

    Args:
        image_path: Image path to encode

    Returns:
        Encoded image path
    """
    return encode_relative_path(image_path)


# Main functions for integration
def create_encoded_github_url(base_url: str, relative_path: str) -> str:
    """
    Create a properly encoded GitHub URL from base URL and relative path.

    Args:
        base_url: GitHub base URL (e.g., "https://raw.githubusercontent.com/user/repo/main")
        relative_path: Relative path to the file

    Returns:
        Properly encoded complete GitHub URL
    """
    if not base_url or not relative_path:
        return ""

    # Ensure base URL doesn't end with slash
    clean_base = base_url.rstrip("/")

    # Ensure relative path starts with slash
    clean_path = relative_path if relative_path.startswith("/") else f"/{relative_path}"

    # Combine and encode
    full_url = f"{clean_base}{clean_path}"
    return safe_encode_url(full_url)


if __name__ == "__main__":
    # Test the encoding functions
    test_urls = [
        "https://raw.githubusercontent.com/user/repo/main/images/Annual Enrollment Guide/image.png",
        "https://d3u2d4xznamk2r.cloudfront.net/Employee Admin Ref/Images/screenshot.jpg",
        "/images/Plan Setup Reference/diagram.png",
        "Template_Guide/Images/example.jpg",
    ]

    print("Testing URL encoding:")
    for url in test_urls:
        encoded = safe_encode_url(url)
        print(f"Original: {url}")
        print(f"Encoded:  {encoded}")
        print(f"Valid:    {validate_encoded_url(encoded)}")
        print("-" * 50)
