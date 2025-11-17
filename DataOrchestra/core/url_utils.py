from __future__ import annotations
from urllib.parse import urlparse, urljoin

def normalise_url(url: str) -> str:
    """
    Normalises a URL by converting to lowercase, removing fragment and query params.
    """
    url = url.lower().strip()
    if "#" in url:
        url = url.split("#")[0]
    if "?" in url:
        url = url.split("?")[0]
    return url.rstrip("/")

def same_netloc(a: str, b: str) -> bool:
    """Checks if two URLs share the same network location."""
    return urlparse(a).netloc == urlparse(b).netloc

def is_internal_link(href: str, base_url: str) -> bool:
    """
    Checks if a link is internal to the base URL's domain.
    Handles relative and absolute links.
    """
    if not href or href.startswith(("javascript:", "mailto:", "tel:")):
        return False
    
    # Join the link with the base URL to get an absolute URL
    absolute_url = urljoin(base_url, href)
    
    # Compare the network location (domain)
    return same_netloc(absolute_url, base_url)
