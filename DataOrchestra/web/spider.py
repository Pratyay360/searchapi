"""
Web spider for crawling websites.
"""

import os
import shutil
import subprocess
import requests
from pathlib import Path
from typing import Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from ..core import FileProcessingResult
from ..core.exceptions import CrawlError
from ..core.config import get_config
from fake_useragent import UserAgent

ua = UserAgent()


class WebSpider:
    """Web spider for crawling websites."""

    def __init__(self, base_url: str, output_dir: Optional[Path] = None):
        self.base_url = base_url
        self.config = get_config()
        self.output_dir = output_dir or self.config.output_dir
        self.visited_urls: Set[str] = set()
        self.domain = urlparse(base_url).netloc

    def crawl(self, max_pages: Optional[int] = None, max_depth: Optional[int] = None) -> list[FileProcessingResult]:
        max_pages = max_pages or self.config.web.max_pages
        max_depth = max_depth or self.config.web.max_depth
        results = []
        try:
            self._crawl_page(self.base_url, 0, max_pages, max_depth, results)
            return results
        except Exception as e:
            raise CrawlError(
                f"Failed to crawl website: {self.base_url}") from e

    def _crawl_page(self, url: str, depth: int, max_pages: int, max_depth: int, results: list[FileProcessingResult]) -> None:
        if len(self.visited_urls) >= max_pages or depth > max_depth:
            return

        if url in self.visited_urls:
            return

        self.visited_urls.add(url)

        try:
            # Fetch the page
            result = self._fetch_page(url)
            if result.success:
                results.append(result)

                # Parse the page and find links
                if depth < max_depth:
                    soup = BeautifulSoup(result.data, "html.parser")
                    for link in soup.find_all("a", href=True):
                        # Normalize and validate href to avoid bs4 typing issues (e.g. AttributeValueList)
                        raw_href = link.get("href")
                        if raw_href is None:
                            continue
                        href = str(raw_href).strip()
                        # Skip empty, fragments, mailto, javascript links
                        if not href or href.startswith(("#", "mailto:", "javascript:")):
                            continue

                        absolute_url = urljoin(url, href)
                        parsed = urlparse(absolute_url)
                        # Only follow http(s) links within the same domain
                        if parsed.scheme in ("http", "https") and parsed.netloc == self.domain:
                            self._crawl_page(
                                absolute_url, depth + 1, max_pages, max_depth, results)

        except Exception as e:
            results.append(FileProcessingResult(
                success=False,
                error=str(e),
                metadata={"url": url}
            ))

    def _fetch_page(self, url: str) -> FileProcessingResult:
        result = FileProcessingResult(
            success=False,
            metadata={"url": url}
        )

        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Parse URL to create a safe filename
            parsed_url = urlparse(url)
            path = parsed_url.path.replace("/", "_") or "index"
            filename = f"{self.domain}{path}.html"
            output_path = self.output_dir / filename

            headers = {
                "User-Agent": ua.random
            }

            response = requests.get(
                url,
                headers=headers,
                timeout=self.config.web.timeout
            )
            response.raise_for_status()

            # Save the content
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            result.success = True
            result.data = response.text
            result.output_path = output_path
            result.file_size_after = output_path.stat().st_size

            return result

        except Exception as e:
            result.error = str(e)
            return result


def crawl_website(base_url: str, output_dir: Optional[Path] = None) -> list[FileProcessingResult]:
    spider = WebSpider(base_url, output_dir)
    return spider.crawl()
