from __future__ import annotations
import argparse
import json
import logging
import os
import random
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tqdm import tqdm
import logging

ua = UserAgent()
URLS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://www.bbc.com/news/technology"
]

# Common headers
BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
MAX_RETRIES = 4
BACKOFF_FACTOR = 0.6
TIMEOUT = 12

###############################################################################
# LOGGER
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(threadName)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crawler")


###############################################################################
# DATA MODEL
###############################################################################
@dataclass
class Page:
    idx: int
    url: str
    level: int = 0
    status_code: Optional[int] = None
    error: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    discovered_urls: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

def normalise_url(url: str) -> str:
    url = url.lower().strip()
    if "#" in url:
        url = url.split("#")[0]
    if "?" in url:
        url = url.split("?")[0]
    return url.rstrip("/")

def same_netloc(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc

def build_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(BASE_HEADERS)
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

def discover_links(page_url: str, html: str, seen: Set[str]) -> List[str]:
    """Return list of *new* internal absolute URLs."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href_val = a.get("href")
        if not href_val:
            continue
        if isinstance(href_val, (list, tuple)):
            href_str = href_val[0]
        else:
            href_str = href_val
        href = str(href_str).split("#")[0].strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:")):
            continue

        abs_url = urljoin(page_url, href)
        if not same_netloc(abs_url, page_url):
            continue

        abs_url = normalise_url(abs_url)
        if abs_url and abs_url not in seen:
            links.append(abs_url)

    return links


def extract_text(soup: BeautifulSoup) -> str:
    for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
        element.decompose()
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text


def fetch(
    page_url: str,
    idx: int,
    level: int,
    session: requests.Session,
    seen: Set[str],
    max_extra_depth: int
) -> List[Page]:
    pages: List[Page] = []
    main = Page(idx=idx, url=page_url, level=level)

    try:
        headers = {"User-Agent": f"{ua.random}"}
        log.debug("Fetching %s (level %d)", page_url, level)

        resp = session.get(page_url, headers=headers, timeout=TIMEOUT)
        main.status_code = resp.status_code

        if resp.status_code != 200:
            main.error = f"HTTP {resp.status_code}"
            pages.append(main)
            return pages

        content_type = resp.headers.get("content-type", "").lower()
        if "html" not in content_type:
            main.error = f"Non-HTML content: {content_type}"
            pages.append(main)
            return pages

        soup = BeautifulSoup(resp.content, "lxml")
        main.title = soup.title.string.strip() if soup.title else None
        main.text = extract_text(soup)
        if level < max_extra_depth:
            discovered = discover_links(page_url, resp.text, seen)
            main.discovered_urls = discovered
            log.debug("Discovered %d links from %s", len(discovered), page_url)
    except requests.exceptions.Timeout:
        main.error = "Timeout"
    except requests.exceptions.ConnectionError:
        main.error = "Connection error"
    except requests.exceptions.TooManyRedirects:
        main.error = "Too many redirects"
    except Exception as exc:
        main.error = f"{type(exc).__name__}: {str(exc)}"
        log.debug("Fetch error for %s: %s", page_url, exc)
    pages.append(main)
    return pages


def load_urls(path: Optional[str] = None) -> List[str]:
    if path and Path(path).is_file():
        with open(path, "r", encoding="utf-8") as fh:
            candidates = [
                line.strip() for line in fh if line.strip() and not line.startswith("#")
            ]
    else:
        candidates = URLS

    seen = set()
    out: List[str] = []
    for u in candidates:
        u_norm = normalise_url(u)
        if u_norm and u_norm not in seen:
            seen.add(u_norm)
            out.append(u)
    log.info("Loaded %d unique seed URLs", len(out))
    return out


def setup_output_dir() -> Path:
    tstamp = time.strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path("runs") / tstamp
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    return out_dir


def save_page(page: Page, out_dir: Path):
    meta_file = out_dir / "url_metadata.jsonl"
    txt_file = out_dir / "raw" / f"{page.idx:04d}_l{page.level}.txt"

    # Save metadata
    with meta_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")

    # Save text content
    if page.text:
        txt_file.write_text(page.text, encoding="utf-8")


def graceful_shutdown(sig, frame):
    log.warning("Interrupted – finishing pending tasks...")
    sys.exit(0)


def crawl(
    seed_urls: List[str],
    workers: int = 5,
    delay_range: tuple[float, float] = (1.0, 3.0),
    max_extra_depth: int = 2,
    timeout: int = 12,
    max_pages: int = 4000,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
) -> Path:
    global TIMEOUT
    TIMEOUT = timeout

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if output_dir is None:
        output_dir = setup_output_dir()
    else:
        (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    session = build_session()
    seen: Set[str] = set(normalise_url(u) for u in seed_urls)
    to_crawl: deque[tuple[str, int]] = deque((u, 0) for u in seed_urls)
    lock = threading.Lock()
    idx_counter = 0
    total_crawled = 0

    def next_idx() -> int:
        nonlocal idx_counter
        with lock:
            idx_counter += 1
            return idx_counter

    log.info(
        "Starting crawl with %d seed URLs into %s (depth=%d, workers=%d)",
        len(seed_urls), output_dir, max_extra_depth, workers
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        pbar = tqdm(total=len(to_crawl), unit="page", desc="Crawling")

        try:
            while to_crawl or futures:
                if total_crawled >= max_pages:
                    log.info("Reached maximum pages limit (%d)", max_pages)
                    break

                # Submit new tasks
                while to_crawl and len(futures) < workers and total_crawled < max_pages:
                    url, lvl = to_crawl.popleft()
                    future = pool.submit(
                        fetch, url, next_idx(), lvl, session, seen, max_extra_depth
                    )
                    futures[future] = (url, lvl)

                if not futures:
                    break

                # Process completed
                try:
                    done_futures = list(as_completed(futures, timeout=1.0))
                except TimeoutError:
                    continue

                for future in done_futures:
                    url, lvl = futures.pop(future)
                    try:
                        pages = future.result()
                        for page in pages:
                            save_page(page, output_dir)
                            total_crawled += 1

                            if page.discovered_urls and lvl < max_extra_depth:
                                with lock:
                                    for u in page.discovered_urls:
                                        u_norm = normalise_url(u)
                                        if u_norm not in seen:
                                            seen.add(u_norm)
                                            to_crawl.append((u, lvl + 1))
                                            pbar.total += 1
                    except Exception as e:
                        log.error("Unexpected error processing %s: %s", url, e)

                    pbar.update(1)
                    time.sleep(random.uniform(*delay_range))
        finally:
            pbar.close()
    log.info("Crawl finished - %d pages saved in %s",
             total_crawled, output_dir)
    return output_dir


def spider():
    parser = argparse.ArgumentParser(
        description="Crawl websites and internal links")
    parser.add_argument(
        "-u", "--urls", help="File with seed URLs (one per line)")
    parser.add_argument("-w", "--workers", type=int, default=5)
    parser.add_argument("-d", "--delay", type=float, nargs=2,
                        default=[1, 3], metavar=("MIN", "MAX"))
    parser.add_argument("-x", "--max-extra-depth", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--max-pages", type=int, default=4000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, graceful_shutdown)
    seeds = load_urls(args.urls)
    crawl(
        seed_urls=seeds,
        workers=args.workers,
        delay_range=tuple(args.delay),
        max_extra_depth=args.max_extra_depth,
        timeout=args.timeout,
        max_pages=args.max_pages,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    spider()
