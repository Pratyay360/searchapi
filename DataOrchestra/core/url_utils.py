from __future__ import annotations

from urllib.parse import urljoin, urlparse


def normalize_url(url: str):
    url = url.lower().strip()
    if "#" in url:
        url = url.split("#")[0]
    if "?" in url:
        url = url.split("?")[0]
    return url.rstrip("/")


def same_netloc(a: str, b: str):
    return urlparse(a).netloc == urlparse(b).netloc


def is_internal_link(href: str, base_url: str):
    if not href or href.startswith(("javascript:", "mailto:", "tel:")):
        return False

    absolute_url = urljoin(base_url, href)
    return same_netloc(absolute_url, base_url)
