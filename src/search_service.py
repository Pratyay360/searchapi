"""Search api
Provides a standalone search engine capabilities through REST and MCP interfaces.
"""

from __future__ import annotations

import asyncio
import logging
import time
import typing as t
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

import httpx
from lxml import html as _lh

logger = logging.getLogger("search_service")


@dataclass
class SearchResult:
    """Single normalized search result."""

    title: str
    url: str
    snippet: str | None = None
    engine: str | None = None
    rank: int | None = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "engine": self.engine,
            "rank": self.rank,
        }


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    total: int | None = None
    engines_used: list[str] = field(default_factory=list)
    engines_failed: dict[str, str] = field(default_factory=dict)
    engines_attempted: list[str] = field(default_factory=list)
    """Names of every engine the service queried for this request.

    Lets callers distinguish a total failure (every attempted engine is in
    ``engines_failed``) from a legitimate empty result (engines succeeded but
    the query matched nothing)."""

    def to_dict(self) -> dict:
        d: dict = {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total": self.total,
            "engines_used": self.engines_used,
        }
        if self.engines_failed:
            d["engines_failed"] = self.engines_failed
        if self.engines_attempted:
            d["engines_attempted"] = self.engines_attempted
        return d


_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "msclkid",
    "ref",
    "ref_src",
    "ref_url",
    "source",
    "si",
    "mc_cid",
    "mc_eid",
    "wickedid",
    "_ga",
    "_gl",
    "_hsenc",
    "_hsmi",
    "yclid",
    "igshid",
    "trk",
    "trkCampaign",
}

_SESSION_PARAMS = {
    "sessionid",
    "session_id",
    "sid",
    "phpsessid",
    "jsessionid",
    "aspsessionid",
}


def _date_offset(today_str: str, units: int = 1, unit_type: str = "days") -> str:
    """Subtract *units* of *unit_type* from *today_str* (YYYYMMDD) -> YYYYMMDD."""
    today = datetime.strptime(today_str, "%Y%m%d")
    unit_map = {
        "days": "days",
        "weeks": "weeks",
        "months": "months",
        "years": "years",
    }
    key = unit_map.get(unit_type, "days")
    if key in ("months", "years"):
        # Approximate: months as 30 days, years as 365 days
        approx_days = 30 if key == "months" else 365
        offset_date = today - timedelta(days=approx_days * units)
    else:
        kwargs = {key: units}
        offset_date = today - timedelta(**kwargs)  # type: ignore[arg-type]
    return offset_date.strftime("%Y%m%d")


def _strip_www(netloc: str) -> str:
    if netloc.startswith("www."):
        return netloc[4:]
    return netloc


def canonical_url(url: str) -> str:
    """Normalize a URL into a canonical form for deduplication.

    - Protocol-relative URLs (``//example.com``) → ``https://example.com``
    - Lowercase netloc, strip ``www.`` prefix
    - Strip fragments, trailing slash, tracking/session params
    - Sort remaining query params, remove empty values
    """
    if not url:
        return url
    try:
        if url.startswith("//"):
            url = "https:" + url
        parsed = urlparse(url)
        if not parsed.netloc:
            return url
        netloc = _strip_www(parsed.netloc.lower())
        path = parsed.path.rstrip("/") or "/"
        clean_params = sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=False)
            if key.lower() not in _TRACKING_PARAMS
            and key.lower() not in _SESSION_PARAMS
        )
        return urlunparse(
            (
                parsed.scheme.lower(),
                netloc,
                path,
                parsed.params,
                urlencode(clean_params),
                "",
            )
        )
    except Exception:
        return url


_ENGINE_PRIORITY: dict[str, int] = {
    "duckduckgo": 0,
    "google": 1,
    "mojeek": 2,
    "wikipedia": 3,
    "bing": 4,
}


def _result_quality(r: SearchResult) -> tuple:
    """Score a result for quality comparison. Higher tuple = better.

    Factors: has_snippet > snippet_len > title_len > engine_priority
    """
    has_snippet = 1 if (r.snippet and r.snippet.strip()) else 0
    snippet_len = len(r.snippet or "")
    title_len = len(r.title or "")
    eng_rank = _ENGINE_PRIORITY.get(r.engine or "", 99)
    return (has_snippet, snippet_len, title_len, -eng_rank)


def _best_result(a: SearchResult, b: SearchResult) -> SearchResult:
    """Pick the best result among two with the same canonical URL."""
    if _result_quality(a) >= _result_quality(b):
        return a
    return b


# ---------------------------------------------------------------------------
# Engine / provider base
# ---------------------------------------------------------------------------


class EngineBlockedError(Exception):
    """Engine returned a CAPTCHA / anti-bot challenge instead of results.

    This is *transient* — the caller could succeed with different headers,
    cookies, or network path — but the current request cannot be fulfilled.
    The service layer records it in ``engines_failed`` so it is visible to
    both REST and MCP consumers.
    """


class EngineProvider:
    """Base class for a search engine provider."""

    name: str
    timeout: float = 15.0
    retries: int = 2

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        language: str = "all",
        safesearch: int = 0,
        time_range: str | None = None,
        pageno: int = 1,
    ) -> list[SearchResult]:
        raise NotImplementedError

    async def search_with_retry(
        self,
        client: httpx.AsyncClient,
        query: str,
        language: str = "all",
        safesearch: int = 0,
        time_range: str | None = None,
        pageno: int = 1,
    ) -> list[SearchResult]:
        last_exc: Exception | None = None
        for attempt in range(1 + self.retries):
            try:
                return await self.search(
                    client=client,
                    query=query,
                    language=language,
                    safesearch=safesearch,
                    time_range=time_range,
                    pageno=pageno,
                )
            except EngineBlockedError:
                # Blocked/CAPTCHA — retrying the same way won't help, fail fast.
                raise
            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
            ) as exc:
                last_exc = exc
                logger.warning(
                    "Engine %s attempt %d/%d: %s",
                    self.name,
                    attempt + 1,
                    1 + self.retries,
                    exc,
                )
                if attempt < self.retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                continue
            except Exception as exc:
                logger.warning("Engine %s non-transient failure: %s", self.name, exc)
                raise
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DuckDuckGo HTML provider
# ---------------------------------------------------------------------------


class DuckDuckGoProvider(EngineProvider):
    """DuckDuckGo HTML search (no-JS variant via html.duckduckgo.com/html/)."""

    name = "duckduckgo"
    timeout = 20.0
    retries = 2

    _TIME_RANGE_MAP: dict[str, str] = {
        "day": "d",
        "week": "w",
        "month": "m",
        "year": "y",
    }

    def __init__(self) -> None:
        self._vqd: dict[tuple[str, str], str] = {}

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        language: str = "all",
        safesearch: int = 0,
        time_range: str | None = None,
        pageno: int = 1,
    ) -> list[SearchResult]:
        # DDG does not accept queries with more than 499 chars.
        if len(query) >= 500:
            return []

        # DDG form data.  On the first page ``b`` must be present (empty),
        # exactly as the SearXNG duckduckgo engine does it.
        region = "wt-wt" if language == "all" else language
        data: dict[str, str] = {"q": query}
        if pageno == 1:
            data["b"] = ""
        else:
            vqd = self._vqd.get((query, region))
            if not vqd:
                return []
            offset = 10 + (pageno - 2) * 15
            data.update(
                {
                    "vqd": vqd,
                    "nextParams": "",
                    "api": "d.js",
                    "o": "json",
                    "v": "l",
                    "dc": str(offset + 1),
                    "s": str(offset),
                }
            )
        if time_range:
            ddg_t = self._TIME_RANGE_MAP.get(time_range)
            if ddg_t:
                data["df"] = ddg_t
        data["kl"] = region

        # Cookies mirror what the SearXNG DDG engine sets.  These are part of
        # DDG's bot-blocker evasion: without ``kl``/``df``/``ah``/``l`` the
        # server is significantly more likely to serve a challenge page.
        cookies = {"kl": region}
        if "df" in data:
            cookies["df"] = data["df"]
        # ad/ah/l are language/region identifiers.  ``ah`` == the DDG region,
        # ``l`` == the DDG region used for the UI, ``ad`` == the DDG lang tag.
        cookies["ad"] = region
        cookies["ah"] = region
        cookies["l"] = region

        # Accept-Language derived from the requested language, mirroring
        # SearXNG's behaviour: e.g. "en-US" -> "en-US,en;q=0.7".
        ui_lang = region
        if ui_lang.startswith("wt"):
            accept_lang = "en-US,en;q=0.9"
        else:
            # Convert e.g. "en_US" -> "en-US" for the Accept-Language header.
            accept_lang = ui_lang.replace("_", "-") + ",en;q=0.9"

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": accept_lang,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://html.duckduckgo.com/",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Cookie": "; ".join(f"{key}={value}" for key, value in cookies.items()),
        }

        resp = await client.post(
            "https://html.duckduckgo.com/html/",
            data=data,
            headers=headers,
            follow_redirects=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        if self._is_captcha(resp.text):
            raise EngineBlockedError(
                "DuckDuckGo returned a CAPTCHA challenge instead of results"
            )

        doc = _lh.fromstring(resp.text)
        vqd = doc.xpath('//input[@name="vqd"]/@value')
        if vqd:
            self._vqd[(query, region)] = vqd[0]
        return self._parse_results(doc)

    @staticmethod
    def _is_captcha(html_text: str) -> bool:
        """Detect if DDG returned a CAPTCHA challenge page.

        Keyword heuristics only apply when the results container is absent,
        so ordinary result pages that merely mention "challenge"/"captcha"
        are not misclassified.
        """
        if 'id="challenge-form"' in html_text:
            return True
        if 'id="links"' in html_text:
            return False
        lower = html_text.lower()
        return "challenge" in lower and "captcha" in lower

    def _parse_results(self, html_text: str | _lh.HtmlElement) -> list[SearchResult]:
        results: list[SearchResult] = []
        if isinstance(html_text, str):
            try:
                doc = _lh.fromstring(html_text)
            except Exception:
                return results
        else:
            doc = html_text

        for div_result in doc.xpath(
            '//div[@id="links"]/div[contains(@class, "web-result")]'
        ):
            if "result--ad" in (div_result.get("class", "")):
                continue
            title_links = div_result.xpath(".//h2/a")
            if not title_links:
                continue
            link = title_links[0]
            title = "".join(link.itertext()).strip()
            url = link.get("href", "")
            if not title or not url:
                continue
            snippet_els = div_result.xpath('.//a[contains(@class, "result__snippet")]')
            snippet = "".join(snippet_els[0].itertext()).strip() if snippet_els else ""
            results.append(
                SearchResult(
                    title=title, url=url, snippet=snippet or None, engine=self.name
                )
            )
        return results


# ---------------------------------------------------------------------------
# Google HTML scraping provider
# ---------------------------------------------------------------------------


class GoogleProvider(EngineProvider):
    """Google HTML search via scraping the standard web results page.

    Uses the same approach as the SearXNG google engine: sends a GET to
    ``https://www.google.com/search`` with appropriate headers and parses
    the HTML response.

    .. warning::
       Google may block requests that don't have proper browser-like headers.
       This provider is best-effort and may occasionally get CAPTCHA'd.
    """

    name = "google"
    timeout = 20.0
    retries = 1

    _SAFE_MAP = {0: "off", 1: "medium", 2: "high"}

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        language: str = "all",
        safesearch: int = 0,
        time_range: str | None = None,
        pageno: int = 1,
    ) -> list[SearchResult]:
        params: dict[str, t.Any] = {
            "q": query,
            "hl": "en" if language == "all" else language.split("-")[0],
            "start": (pageno - 1) * 10,
        }
        if safesearch:
            params["safe"] = self._SAFE_MAP.get(safesearch, "medium")
        if time_range:
            tr_map = {"day": "d", "week": "w", "month": "m", "year": "y"}
            params["tbs"] = f"qdr:{tr_map.get(time_range, 'w')}"

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
        }

        url = f"https://www.google.com/search?{urlencode(params)}"
        try:
            resp = await client.get(
                url, headers=headers, follow_redirects=True, timeout=self.timeout
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Google search failed for %r: %s", query, exc)
            raise

        return self._parse_results(resp.text)

    def _parse_results(self, html_text: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        try:
            doc = _lh.fromstring(html_text)
        except Exception:
            return results

        # Remove scripts/styles for cleaner extraction
        for tag in doc.xpath("//script | //style"):
            tag.getparent().remove(tag)

        # Google search result containers
        # Only direct children of #search are organic results — this avoids
        # matching ads, knowledge panels, "people also ask" blocks, etc.
        for result_div in doc.xpath('//div[@id="search"]/div[contains(@class, "g")]'):
            link_els = result_div.xpath(
                './/a[not(contains(@href, "google")) and @href]'
            )
            if not link_els:
                continue
            link = link_els[0]
            href = link.get("href", "")

            # Clean Google redirect URLs
            if href.startswith("/url?q="):
                href = unquote(href[7:].split("&sa=U")[0])

            if not href.startswith("http"):
                continue

            title = "".join(link.itertext()).strip()
            if not title:
                continue

            # Extract snippet / description
            snippet_divs = result_div.xpath(
                './/div[contains(@class, "VwiC3b") or contains(@class, "st") or contains(@class, "BNeawe")]'
            )
            snippet = ""
            for sd in snippet_divs:
                text = "".join(sd.itertext()).strip()
                if text and len(text) > len(snippet):
                    snippet = text

            results.append(
                SearchResult(
                    title=title, url=href, snippet=snippet or None, engine=self.name
                )
            )

        return results


# ---------------------------------------------------------------------------
# Mojeek HTML scraping provider
# ---------------------------------------------------------------------------


class MojeekProvider(EngineProvider):
    """Mojeek HTML search (no API key required).

    Scrapes ``https://www.mojeek.com/search`` which accepts queries without
    any API key or registration.  Based on the SearXNG mojeek engine.

    .. warning::
       Mojeek may occasionally show a CAPTCHA for automated requests.
       This provider is best-effort.
    """

    name = "mojeek"
    timeout = 20.0
    retries = 2

    _BASE = "https://www.mojeek.com"
    _TIME_RANGE_MAP: dict[str, str] = {
        "day": "days",
        "week": "weeks",
        "month": "months",
        "year": "years",
    }

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        language: str = "all",
        safesearch: int = 0,
        time_range: str | None = None,
        pageno: int = 1,
    ) -> list[SearchResult]:
        params: dict[str, t.Any] = {
            "q": query,
            "safe": min(safesearch, 1),
        }

        # Setting s=0 on the first page triggers a rate-limit
        if pageno > 1:
            params["s"] = 10 * (pageno - 1)

        if time_range:
            unit = self._TIME_RANGE_MAP.get(time_range)
            if unit:
                today = time.strftime("%Y%m%d")
                ago = _date_offset(today, units=1, unit_type=unit)
                params["since"] = ago

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.mojeek.com/",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
        }
        # Same cookies the SearXNG Mojeek engine sends — empty lb = "all languages",
        # arc="none" = no region filter.  Without these Mojeek is more likely to
        # challenge automated clients.
        cookies = {"lb": "" if language == "all" else language, "arc": "none"}
        headers["Cookie"] = "; ".join(
            f"{key}={value}" for key, value in cookies.items()
        )

        url = f"{self._BASE}/search?{urlencode(params)}"
        resp = await client.get(
            url,
            headers=headers,
            follow_redirects=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        if self._is_captcha(resp.text):
            raise EngineBlockedError(
                "Mojeek returned a CAPTCHA challenge instead of results"
            )

        results = self._parse_results(resp.text)
        if not results and '<ul class="results-standard">' in resp.text:
            # Page structure exists but parser found nothing — layout may have changed.
            logger.warning(
                "Mojeek results container present but no results parsed for query: %s",
                query,
            )
        return results

    @staticmethod
    def _is_captcha(html_text: str) -> bool:
        """Detect if Mojeek returned a CAPTCHA / challenge page.

        Keyword heuristics only apply when the results container is absent,
        so ordinary result pages that merely mention "captcha"/"verify"
        are not misclassified.
        """
        if 'id="challenge-form"' in html_text:
            return True
        if '<ul class="results-standard">' in html_text:
            return False
        lower = html_text.lower()
        return (
            "captcha" in lower and "verify" in lower
        ) or "verification required" in lower

    def _parse_results(self, html_text: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        try:
            doc = _lh.fromstring(html_text)
        except Exception:
            return results

        # XPaths from the SearXNG Mojeek engine
        for result_elem in doc.xpath(
            '//ul[@class="results-standard"]/li/a[@class="ob"]'
        ):
            # URL: ./@href
            url = "".join(result_elem.xpath("./@href")).strip()
            if not url:
                continue
            if url.startswith("/"):
                url = self._BASE + url

            # Title: ../h2/a
            title_els = result_elem.xpath("../h2/a")
            title = "".join(title_els[0].itertext()).strip() if title_els else ""

            # Snippet: ..//p[@class="s"] — go up to <li> and find <p class="s">
            snippet_els = result_elem.xpath('..//p[@class="s"]')
            snippet = "".join(snippet_els[0].itertext()).strip() if snippet_els else ""

            if not title and not url:
                continue

            results.append(
                SearchResult(
                    title=title, url=url, snippet=snippet or None, engine=self.name
                )
            )

        return results


class WikipediaProvider(EngineProvider):
    """Wikipedia search via the MediaWiki Action API."""

    name = "wikipedia"
    timeout = 10.0
    retries = 1

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        language: str = "all",
        safesearch: int = 0,
        time_range: str | None = None,
        pageno: int = 1,
    ) -> list[SearchResult]:
        wiki_language = language.split("-")[0].lower()
        if language == "all" or not wiki_language.isalpha() or len(wiki_language) > 3:
            wiki_language = "en"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 10,
            "sroffset": (pageno - 1) * 10,
            "srnamespace": 0,
            "srprop": "snippet",
            "format": "json",
            "formatversion": 2,
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "searchapi/0.3.0 (SearXNG-style metasearch API)",
        }
        resp = await client.get(
            f"https://{wiki_language}.wikipedia.org/w/api.php",
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._parse_results(resp.json(), wiki_language)

    def _parse_results(
        self, data: t.Any, wiki_language: str = "en"
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        if not isinstance(data, dict):
            return results
        items = data.get("query", {}).get("search", [])
        if not isinstance(items, list):
            return results
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            if not isinstance(title, str) or not title:
                continue
            snippet = item.get("snippet")
            if isinstance(snippet, str):
                try:
                    snippet = _lh.fromstring(f"<div>{snippet}</div>").text_content()
                except Exception:
                    pass
            else:
                snippet = None
            results.append(
                SearchResult(
                    title=title,
                    url=f"https://{wiki_language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    snippet=snippet or None,
                    engine=self.name,
                )
            )
        return results


class TTLCache:
    """Simple in-memory cache with TTL expiry."""

    def __init__(self, ttl_seconds: float = 60) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, t.Any]] = {}

    def get(self, key: str) -> t.Any | None:
        if key in self._store:
            expires_at, value = self._store[key]
            if time.monotonic() < expires_at:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: t.Any) -> None:
        self._store[key] = (time.monotonic() + self._ttl, value)


class SearchService:
    """Aggregating search service that queries multiple engine providers."""

    def __init__(
        self,
        engines: list[EngineProvider] | None = None,
        max_concurrency: int = 3,
        default_limit: int = 10,
        cache_ttl: int = 60,
    ) -> None:
        self.engines = _default_providers() if engines is None else engines
        self.max_concurrency = max_concurrency
        self.default_limit = default_limit
        self._cache = TTLCache(ttl_seconds=cache_ttl) if cache_ttl > 0 else None

    @staticmethod
    def _cache_key(
        query: str,
        language: str,
        safesearch: int,
        time_range: str | None,
        offset: int,
        limit: int,
        engines: tuple[str, ...] | None,
    ) -> str:
        parts = [
            f"q={query}",
            f"lang={language}",
            f"ss={safesearch}",
            f"tr={time_range or ''}",
            f"off={offset}",
            f"lim={limit}",
        ]
        if engines:
            parts.append(f"eng={','.join(sorted(engines))}")
        return "|".join(parts)

    async def search(
        self,
        query: str,
        language: str = "all",
        safesearch: int = 0,
        time_range: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        engines: list[str] | None = None,
    ) -> SearchResponse:
        limit = self.default_limit if limit is None else limit
        ckey: str | None = None
        if self._cache:
            ckey = self._cache_key(
                query,
                language,
                safesearch,
                time_range,
                offset,
                limit,
                tuple(engines) if engines else None,
            )
            cached = self._cache.get(ckey)
            if cached is not None:
                return cached  # type: ignore

        selected = (
            [e for e in self.engines if e.name in set(engines)]
            if engines
            else self.engines
        )

        if not selected:
            resp = SearchResponse(
                query=query,
                results=[],
                total=0,
                engines_used=[],
                engines_attempted=[],
            )
            if self._cache and ckey:
                self._cache.set(ckey, resp)
            return resp

        semaphore = asyncio.Semaphore(self.max_concurrency)

        pages = min(5, max(1, (offset + limit + 9) // 10))

        async def run_one(
            provider: EngineProvider, client: httpx.AsyncClient
        ) -> tuple[str, list[SearchResult], str | None]:
            async with semaphore:
                try:
                    results: list[SearchResult] = []
                    seen_urls: set[str] = set()
                    for pageno in range(1, pages + 1):
                        try:
                            page_results = await provider.search_with_retry(
                                client=client,
                                query=query,
                                language=language,
                                safesearch=safesearch,
                                time_range=time_range,
                                pageno=pageno,
                            )
                        except Exception as exc:
                            if not results:
                                raise
                            message = f"{exc.__class__.__name__}: {exc}"
                            logger.warning(
                                "Engine %s stopped at page %d: %s",
                                provider.name,
                                pageno,
                                message,
                            )
                            return (provider.name, results, message)
                        new_results = [
                            result
                            for result in page_results
                            if canonical_url(result.url) not in seen_urls
                        ]
                        if not new_results:
                            break
                        results.extend(new_results)
                        seen_urls.update(
                            canonical_url(result.url) for result in new_results
                        )
                    return (provider.name, results, None)
                except EngineBlockedError as exc:
                    # Expected under automation — log at INFO, not ERROR.
                    logger.info("Engine %s blocked: %s", provider.name, exc)
                    return (provider.name, [], str(exc))
                except Exception as exc:
                    msg = f"{exc.__class__.__name__}: {exc}"
                    logger.error("Engine %s failed: %s", provider.name, msg)
                    return (provider.name, [], msg)

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.max_concurrency * 30.0), follow_redirects=True
        ) as client:
            outcomes = await asyncio.gather(
                *[run_one(e, client) for e in selected], return_exceptions=True
            )

        engine_results: list[tuple[str, list[SearchResult]]] = []
        engines_used: list[str] = []
        engines_failed: dict[str, str] = {}
        engines_attempted: list[str] = []
        for outcome in outcomes:
            if isinstance(outcome, tuple):
                name, results, err = outcome
                engines_attempted.append(name)
                if err:
                    engines_failed[name] = err
                if results or not err:
                    engines_used.append(name)
                if results:
                    engine_results.append((name, results))

        if not engine_results:
            resp = SearchResponse(
                query=query,
                results=[],
                total=0,
                engines_used=engines_used,
                engines_failed=engines_failed,
                engines_attempted=engines_attempted,
            )
            if self._cache and ckey and not engines_failed:
                self._cache.set(ckey, resp)
            return resp

        all_results: list[SearchResult] = []
        for name, results in engine_results:
            for r in results:
                r.engine = name
                all_results.append(r)

        best_by_canon: dict[str, SearchResult] = {}
        for r in all_results:
            canon = canonical_url(r.url)
            best_by_canon[canon] = (
                _best_result(best_by_canon[canon], r) if canon in best_by_canon else r
            )

        deduped = list(best_by_canon.values())
        for i, r in enumerate(deduped, 1):
            r.rank = i

        total = len(deduped)
        sliced = deduped[offset : offset + limit]
        resp = SearchResponse(
            query=query,
            results=sliced,
            total=total,
            engines_used=engines_used,
            engines_failed=engines_failed,
            engines_attempted=engines_attempted,
        )

        if self._cache and ckey and not engines_failed:
            self._cache.set(ckey, resp)
        return resp


def _default_providers() -> list[EngineProvider]:
    return [
        DuckDuckGoProvider(),
        WikipediaProvider(),
        MojeekProvider(),
        GoogleProvider(),
    ]


_service: SearchService | None = None


def configure_search_service(
    engines: list[EngineProvider] | None = None,
    max_concurrency: int = 3,
    default_limit: int = 10,
    cache_ttl: int = 60,
) -> SearchService:
    global _service
    if _service is not None:
        logger.warning("configure_search_service called more than once, ignoring")
        return _service
    _service = SearchService(
        engines=engines,
        max_concurrency=max_concurrency,
        default_limit=default_limit,
        cache_ttl=cache_ttl,
    )
    return _service


def get_search_service() -> SearchService:
    global _service
    if _service is None:
        _service = SearchService()
    return _service
