import secrets
import time
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fake_useragent import FakeUserAgentError, UserAgent

from ..core import FileProcessingResult
from ..core.config import get_config


class WebSpider:
    def __init__(self, base_url: str, output_dir: Optional[Path] = None):
        self.base_url = base_url
        self.config = get_config()
        self.output_dir = output_dir or self.config.output_dir
        self.visited_urls: Set[str] = set()
        self.domain = urlparse(base_url).netloc
        self._ua = None

    @property
    def ua(self):
        if self._ua is None:
            try:
                self._ua = UserAgent()
            except (FakeUserAgentError, Exception):
                self._ua = (
                    lambda: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
        return self._ua

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        sorted_query = urlencode({k: sorted(v) for k, v in query.items()}, doseq=True)
        return (
            f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{sorted_query}"
            if sorted_query
            else f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        )

    def _fetch_page(self, url: str):
        result = FileProcessingResult(success=False, metadata={"url": url})
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            timestamp_token = f"{int(time.time() * 1000)}_{secrets.token_hex(4)}"
            safe_domain = self.domain.replace(":", "_").replace(".", "_")
            filename = f"{safe_domain}_{timestamp_token}.html"
            output_path = self.output_dir / filename

            headers = {"User-Agent": self.ua.random}
            response = requests.get(
                url, headers=headers, timeout=self.config.web.timeout
            )
            response.raise_for_status()

            if response.encoding is None or response.encoding.lower() == "iso-8859-1":
                try:
                    import charset_normalizer

                    detected = charset_normalizer.detect(response.content)
                    if detected["encoding"]:
                        response.encoding = detected["encoding"]
                except ImportError:
                    pass

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            result.success = True
            result.data = response.text
            result.output_path = output_path
            result.file_size_after = output_path.stat().st_size
            result.metadata.update(
                {
                    "filename": filename,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                    "encoding": response.encoding,
                }
            )
            return result

        except Exception as e:
            result.error = str(e)
            return result

    def _crawl_page(
        self,
        url: str,
        depth: int,
        max_pages: int,
        max_depth: int,
        results: List[FileProcessingResult],
    ):
        if len(self.visited_urls) >= max_pages or depth > max_depth:
            return

        normalized_url = self._normalize_url(url)
        if normalized_url in self.visited_urls:
            return

        self.visited_urls.add(normalized_url)

        if hasattr(self.config.web, "delay") and depth > 0:
            time.sleep(self.config.web.delay)

        try:
            result = self._fetch_page(url)
            if result.success:
                results.append(result)

                if depth < max_depth:
                    soup = BeautifulSoup(result.data, "html.parser")
                    for link in soup.find_all("a", href=True):
                        raw_href = link.get("href")
                        if raw_href is None:
                            continue
                        href = str(raw_href).strip()
                        if not href or href.startswith(("#", "mailto:", "javascript:")):
                            continue
                        absolute_url = urljoin(url, href)
                        parsed = urlparse(absolute_url)
                        if (
                            parsed.scheme in ("http", "https")
                            and parsed.netloc == self.domain
                        ):
                            self._crawl_page(
                                absolute_url,
                                depth + 1,
                                max_pages,
                                max_depth,
                                results,
                            )

        except Exception as e:
            results.append(
                FileProcessingResult(success=False, error=str(e), metadata={"url": url})
            )

    def crawl(
        self, max_pages: Optional[int] = None, max_depth: Optional[int] = None
    ):
        max_pages = max_pages or self.config.web.max_pages
        max_depth = max_depth or self.config.web.max_depth
        results: List[FileProcessingResult] = []
        try:
            self._crawl_page(self.base_url, 0, max_pages, max_depth, results)
            return results
        except Exception as e:
            print(f"Failed to crawl website: {self.base_url} with error: {e}")
            return []


def crawl_website(
    base_url: str, output_dir: Optional[Path] = None
):
    spider = WebSpider(base_url, output_dir)
    return spider.crawl()
