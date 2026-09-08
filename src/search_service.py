"""Search api
Provides a standalone search engine capabilities through REST and MCP interfaces.

Engine providers port the data-fetch logic (endpoints, form fields, headers,
cookies, pagination and parsing) of the corresponding upstream SearXNG engines.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import re
import time
import typing as t
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from urllib.parse import (
    parse_qs,
    parse_qsl,
    quote,
    unquote,
    urlencode,
    urlparse,
    urlunparse,
)

import httpx
from lxml import html as _lh

logger = logging.getLogger("search_service")

_USERAGENTS: dict[str, t.Any] = {
    "os": ["Windows NT 10.0; Win64; x64", "X11; Linux x86_64"],
    "ua": "Mozilla/5.0 ({os}; rv:{version}) Gecko/20100101 Firefox/{version}",
    "versions": ["154.0", "153.0"],
}
"""Mirror of searx/data/useragents.json."""


def gen_useragent() -> str:
    """Random desktop browser User-Agent (port of ``searx.utils.gen_useragent``)."""
    return _USERAGENTS["ua"].format(
        os=random.choice(_USERAGENTS["os"]),
        version=random.choice(_USERAGENTS["versions"]),
    )


def searxng_useragent() -> str:
    """Static identity User-Agent for API-type requests (port of
    ``searx.utils.searxng_useragent``, ``SearXNG/{VERSION_TAG} {suffix}``)."""
    try:
        pkg_version = _package_version("searchweb")
    except PackageNotFoundError:
        pkg_version = "unknown"
    return f"searchweb/{pkg_version} (SearXNG-style metasearch API)"


_HTTP_USER_AGENT = gen_useragent()
"""Static per-process desktop User-Agent; engines whose bot protection keys
state to the UA (e.g. DuckDuckGo's vqd) must never vary it between requests,
and mobile UAs change server layouts and break parsing."""


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


def _date_offset(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")


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
    "brave": 5,
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
        raise RuntimeError(
            f"Engine {self.name} exhausted {self.retries} retries: {last_exc}"
        ) from last_exc


# ---------------------------------------------------------------------------
# DuckDuckGo (port of searx/engines/duckduckgo.py, no-JS html engine)
# ---------------------------------------------------------------------------

_DDG_BANGS: frozenset[str] | None = None
_DDG_BANGS_URL = (
    "https://raw.githubusercontent.com/searxng/searxng/master/"
    "searx/data/external_bangs.json"
)


def _ddg_bangs() -> frozenset[str]:
    global _DDG_BANGS
    if _DDG_BANGS is None:
        try:
            resp = httpx.get(
                _DDG_BANGS_URL,
                headers={"User-Agent": searxng_useragent()},
                timeout=10,
            )
            resp.raise_for_status()
            _DDG_BANGS = frozenset(resp.json().keys())
        except Exception:
            _DDG_BANGS = frozenset()
    return _DDG_BANGS


def _quote_ddg_bangs(query: str) -> str:
    """Quote ``!bang`` directives so DDG does not redirect away from the
    results page (mirrors upstream ``quote_ddg_bangs``)."""
    quoted: list[str] = []
    bangs = _ddg_bangs()
    for val in re.split(r"(\s+)", query):
        if not val.strip():
            continue
        if val.startswith("!") and val[1:] in bangs:
            val = f"'{val}'"
        quoted.append(val)
    return " ".join(quoted)


def _ddg_region(language: str) -> str:
    """Map a locale tag to DuckDuckGo's region code (``wt-wt`` = all regions).

    Approximates upstream trait mapping: DDG orders its tags
    territory-language (``en-US`` -> ``us-en``); tags that already look like
    DDG codes pass through unchanged.
    """
    if language == "all":
        return "wt-wt"
    tag = language.replace("_", "-").lower()
    if re.fullmatch(r"[a-z]{2,3}-[a-z]{2,3}", tag):
        first, second = tag.split("-")
        return f"{second}-{first}"
    return tag


class DuckDuckGoProvider(EngineProvider):
    """DuckDuckGo WEB search via the no-JS html endpoint (POST form data)."""

    name = "duckduckgo"
    timeout = 20.0
    retries = 2

    _URL = "https://html.duckduckgo.com/html/"
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
        if len(query) >= 500:
            return []

        query = _quote_ddg_bangs(query)
        region = _ddg_region(language)

        data: dict[str, t.Any] = {"q": query}
        if pageno == 1:
            data["b"] = ""
        else:
            # vqd is required for follow-up pages; requesting them without one
            # is an immediate bot signal and lowers the IP reputation.
            vqd = self._vqd.get((query, region))
            if not vqd:
                raise EngineBlockedError(f"vqd missed (page: {pageno})")
            data["vqd"] = vqd
            if region.startswith("zh"):
                return []
            data.update(
                {
                    "nextParams": "",
                    "api": "d.js",
                    "o": "json",
                    "v": "l",
                }
            )
            offset = 10 + (pageno - 2) * 15
            data["dc"] = offset + 1
            data["s"] = offset

        # Upstream puts empty kl in the form data when region is "all".
        data["kl"] = "" if region == "wt-wt" else region

        cookies: dict[str, str] = {}
        if region != "wt-wt":
            cookies["kl"] = region
        t_range = self._TIME_RANGE_MAP.get(time_range or "", "")
        if t_range:
            data["df"] = t_range
            cookies["df"] = t_range

        accept_lang = (
            "en-US,en;q=0.9"
            if language == "all"
            else f"{language.replace('_', '-')},en;q=0.9"
        )
        headers = {
            "User-Agent": _HTTP_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": accept_lang,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": self._URL,
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
        }

        resp = await client.post(
            self._URL,
            data=data,
            headers=headers,
            cookies=cookies,
            follow_redirects=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        if resp.status_code == 303:
            return []

        doc = _lh.fromstring(resp.text)
        if doc.xpath("//form[@id='challenge-form']"):
            raise EngineBlockedError(f"CAPTCHA (kl: {data.get('kl')})")

        form_vqd = doc.xpath('//input[@name="vqd"]/@value')
        if form_vqd:
            self._vqd[(query, region)] = form_vqd[0]

        return self._parse_results(doc)

    def _parse_results(self, doc: _lh.HtmlElement) -> list[SearchResult]:
        results: list[SearchResult] = []
        for div_result in doc.xpath(
            '//div[@id="links"]/div[contains(@class, "web-result")]'
        ):
            if "result--ad" in (div_result.get("class") or ""):
                continue
            title_links = div_result.xpath(".//h2/a")
            hrefs = div_result.xpath(".//h2/a/@href")
            if not title_links or not hrefs:
                continue
            title = "".join(title_links[0].itertext()).strip()
            url = hrefs[0]
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
# Google (port of searx/engines/google.py, WML/XML layout)
# ---------------------------------------------------------------------------


class GoogleProvider(EngineProvider):
    """Google WEB search via the WML layout.

    The normal web layout requires JavaScript; upstream requests the legacy
    WML layout with Nokia user agents instead, which returns server-rendered
    results without a JS engine.
    """

    name = "google"
    timeout = 20.0
    retries = 1

    nokia_useragents = (
        "Nokia7610/2.0 (5.0509.0) SymbianOS/7.0s Series60/2.1 Profile/MIDP-2.0 Configuration/CLDC-1.0",
        "Nokia7610/2.0 (7.0642.0) SymbianOS/7.0s Series60/2.1 Profile/MIDP-2.0 Configuration/CLDC-1.0",
        "Nokia6230/2.0 (05.50) Profile/MIDP-2.0 Configuration/CLDC-1.1",
        "Nokia6230i/2.0 (03.80) Profile/MIDP-2.0 Configuration/CLDC-1.1",
        "Nokia6280/2.0 (03.60) Profile/MIDP-2.0 Configuration/CLDC-1.1",
        "NokiaN72/2.0617.1.0.3 Series60/2.8 Profile/MIDP-2.0 Configuration/CLDC-1.1",
    )

    _TIME_RANGE_MAP: dict[str, str] = {
        "day": "d",
        "week": "w",
        "month": "m",
        "year": "y",
    }
    _FILTER_MAP = {0: "off", 1: "medium", 2: "high"}
    _LANG_ALIASES = {"zh": "zh-CN", "no": "nb"}

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        language: str = "all",
        safesearch: int = 0,
        time_range: str | None = None,
        pageno: int = 1,
    ) -> list[SearchResult]:
        args: dict[str, t.Any] = {
            "q": query,
            "sca_esv": "1",
            "ie": "utf8",
            "oe": "utf8",
        }

        if language == "all":
            args["hl"] = "en"
            args["lr"] = ""
        else:
            lang = self._LANG_ALIASES.get(
                language.split("-")[0].lower(), language.split("-")[0].lower()
            )
            args["hl"] = lang
            args["lr"] = f"lang_{lang}"
            parts = language.split("-")
            if len(parts) > 1 and parts[1].isalpha() and len(parts[1]) == 2:
                args["cr"] = f"country{parts[1].upper()}"

        start = (pageno - 1) * 10
        if start:
            args["start"] = start
        if time_range and time_range in self._TIME_RANGE_MAP:
            args["tbs"] = "qdr:" + self._TIME_RANGE_MAP[time_range]
        if safesearch:
            args["safe"] = self._FILTER_MAP.get(safesearch, "medium")

        headers = {
            "User-Agent": random.choice(self.nokia_useragents),
            "Accept": "*/*",
        }
        cookies = {"CONSENT": "YES+"}

        url = f"https://www.google.com/wml/search?{urlencode(args)}"
        resp = await client.get(
            url,
            headers=headers,
            cookies=cookies,
            follow_redirects=True,
            timeout=self.timeout,
        )
        if resp.url.host == "sorry.google.com" or resp.url.path.startswith("/sorry"):
            raise EngineBlockedError("Google sorry page")
        resp.raise_for_status()
        if len(resp.text) < 2000 and "/sorry/" in resp.text:
            raise EngineBlockedError("Google sorry redirect page")

        return self._parse_results(resp.text)

    @staticmethod
    def _unwrap_url(raw_url: str) -> str:
        if raw_url.startswith("/url?q="):
            return unquote(raw_url[7:].split("&sa=U")[0])
        return raw_url

    def _parse_results(self, html_text: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        text = html_text
        if text.lstrip().startswith("<?xml"):
            text = text.split("?>", 1)[-1]
        try:
            doc = _lh.fromstring(text)
        except Exception:
            return results

        for result in doc.xpath('//div[contains(@class, "zMzFAb")]'):
            titles = result.xpath(
                './/a[contains(@class, "fuLhoc")]//span[contains(@class, "CVA68e")]'
            )
            if not titles:
                continue
            hrefs = result.xpath('.//a[contains(@class, "fuLhoc")]/@href')
            if not hrefs:
                continue
            title = "".join(titles[0].itertext()).strip()
            url = self._unwrap_url(hrefs[0])
            if not url.startswith("http"):
                continue
            contents = result.xpath(
                './/div[contains(@class, "taTFJ")]//span[contains(@class, "FrIlee")]'
            )
            snippet = "".join(contents[0].itertext()).strip() if contents else ""
            results.append(
                SearchResult(
                    title=title, url=url, snippet=snippet or None, engine=self.name
                )
            )
        return results


# ---------------------------------------------------------------------------
# Mojeek (port of searx/engines/mojeek.py, general search)
# ---------------------------------------------------------------------------


class MojeekProvider(EngineProvider):
    """Mojeek HTML search (general, no API key required)."""

    name = "mojeek"
    timeout = 20.0
    retries = 2

    _BASE = "https://www.mojeek.com"
    _TIME_RANGE_DELTA = {"day": 1, "week": 7, "month": 30, "year": 365}

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        language: str = "all",
        safesearch: int = 0,
        time_range: str | None = None,
        pageno: int = 1,
    ) -> list[SearchResult]:
        args: dict[str, t.Any] = {
            "q": query,
            "safe": min(safesearch, 1),
        }

        # Setting the page number on the first page (s=0) triggers a rate-limit.
        if pageno > 1:
            args["s"] = 10 * (pageno - 1)

        if time_range and time_range in self._TIME_RANGE_DELTA:
            args["since"] = _date_offset(self._TIME_RANGE_DELTA[time_range])

        # Cookie values mirror what upstream fills from Mojeek's preferences:
        # lb = language filter ("" = all), arc = region ("" = auto-detect,
        # "none" = no location bias).
        lang = "" if language == "all" else language.split("-")[0].lower()
        region = "none" if language == "all" else ""
        parts = language.split("-")
        if len(parts) > 1 and parts[1].isalpha() and len(parts[1]) == 2:
            region = parts[1].lower()
        cookies = {"lb": lang, "arc": region}

        headers = {
            "User-Agent": _HTTP_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{self._BASE}/",
        }

        url = f"{self._BASE}/search?{urlencode(args)}"
        resp = await client.get(
            url,
            headers=headers,
            cookies=cookies,
            follow_redirects=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        if self._is_captcha(resp.text):
            raise EngineBlockedError(
                "Mojeek returned a CAPTCHA challenge instead of results"
            )

        return self._parse_results(resp.text)

    @staticmethod
    def _is_captcha(html_text: str) -> bool:
        """Heuristics apply only when the results container is absent, so
        ordinary result pages mentioning "captcha" are not misclassified."""
        if 'id="challenge-form"' in html_text:
            return True
        if 'class="captcha-wrap"' in html_text:
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

        for result_elem in doc.xpath(
            '//ul[@class="results-standard"]/li/a[@class="ob"]'
        ):
            url = "".join(result_elem.xpath("./@href")).strip()
            if not url:
                continue
            if url.startswith("/"):
                url = self._BASE + url

            title_els = result_elem.xpath("../h2/a")
            title = "".join(title_els[0].itertext()).strip() if title_els else ""

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


# ---------------------------------------------------------------------------
# Bing (port of searx/engines/bing.py, web search)
# ---------------------------------------------------------------------------


class BingProvider(EngineProvider):
    """Bing WEB search.

    Upstream supports neither paging nor time ranges for the web category
    (both depend on JavaScript); the provider therefore only serves the
    first result page.
    """

    name = "bing"
    timeout = 15.0
    retries = 2

    _BASE = "https://www.bing.com"
    _SAFE_MAP = {0: "off", 1: "moderate", 2: "strict"}

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        language: str = "all",
        safesearch: int = 0,
        time_range: str | None = None,
        pageno: int = 1,
    ) -> list[SearchResult]:
        if pageno > 1:
            return []

        params: dict[str, t.Any] = {
            "q": query,
            "adlt": self._SAFE_MAP.get(safesearch, "off"),
        }

        headers = {
            "User-Agent": _HTTP_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        # Bing market codes are full <language>-<country> tags; the mkt
        # parameter is the recommended primary locale signal.
        norm = language.replace("_", "-")
        m = re.fullmatch(r"([a-z]{2,3})-([A-Za-z]{2})", norm)
        if m:
            market = f"{m.group(1)}-{m.group(2).upper()}"
            params["mkt"] = market
            headers["Accept-Language"] = f"{market},{market.split('-')[0]};q=0.9"

        url = f"{self._BASE}/search?{urlencode(params)}"
        resp = await client.get(
            url,
            headers=headers,
            follow_redirects=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        return self._parse_results(resp.text)

    @staticmethod
    def _unwrap_url(href: str) -> str:
        """Decode Bing's ``/ck/a?...&u=a1<base64url>`` tracking redirects."""
        qs = parse_qs(urlparse(href).query)
        u_values = qs.get("u")
        if u_values and u_values[0].startswith("a1"):
            encoded = u_values[0][2:]
            encoded += "=" * (-len(encoded) % 4)
            return base64.urlsafe_b64decode(encoded).decode("utf-8", errors="replace")
        return href

    def _parse_results(self, html_text: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        try:
            doc = _lh.fromstring(html_text)
        except Exception:
            return results

        for item in doc.xpath('//ol[@id="b_results"]/li[contains(@class, "b_algo")]'):
            links = item.xpath(".//h2/a")
            if not links:
                continue
            href = links[0].get("href", "")
            title = "".join(links[0].itertext()).strip()
            if not href or not title:
                continue
            if href.startswith("https://www.bing.com/ck/a?"):
                href = self._unwrap_url(href)

            # remove decorative icons Bing injects into <p> elements
            content_parts: list[str] = []
            for p in item.xpath(".//p"):
                for icon in p.xpath('.//span[@class="algoSlug_icon"]'):
                    icon.getparent().remove(icon)
                content_parts.append(" ".join(p.itertext()).strip())
            snippet = " ".join(part for part in content_parts if part)

            results.append(
                SearchResult(
                    title=title, url=href, snippet=snippet or None, engine=self.name
                )
            )

        return results


# ---------------------------------------------------------------------------
# Brave (port of searx/engines/brave.py, search category)
# ---------------------------------------------------------------------------


class BraveProvider(EngineProvider):
    """Brave search (web category, server-rendered HTML)."""

    name = "brave"
    timeout = 15.0
    retries = 2

    _BASE = "https://search.brave.com"
    _TIME_RANGE_MAP: dict[str, str] = {
        "day": "pd",
        "week": "pw",
        "month": "pm",
        "year": "py",
    }
    _SAFE_MAP = {0: "off", 1: "moderate", 2: "strict"}
    _UI_LANGS = {
        "ca",
        "de-de",
        "en-ca",
        "en-gb",
        "en-us",
        "es",
        "fr-ca",
        "fr-fr",
        "ja-jp",
        "pt-br",
        "sq-al",
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
        args: dict[str, t.Any] = {
            "q": query,
            "source": "web",
        }
        if pageno > 1:
            args["offset"] = pageno - 1
        if time_range and time_range in self._TIME_RANGE_MAP:
            args["tf"] = self._TIME_RANGE_MAP[time_range]

        # Region/language selection happens via cookies: country from the
        # region part of the locale, ui_lang from Brave's small UI language
        # set (defaults to en-us).
        parts = language.split("-")
        if (
            language != "all"
            and len(parts) > 1
            and parts[-1].isalpha()
            and len(parts[-1]) == 2
        ):
            country = parts[-1].lower()
        else:
            country = "all"
        ui_lang = language.replace("_", "-").lower()
        if ui_lang not in self._UI_LANGS:
            ui_lang = "en-us"

        headers = {
            "User-Agent": _HTTP_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        }
        cookies = {
            "safesearch": self._SAFE_MAP.get(safesearch, "off"),
            "useLocation": "0",
            "summarizer": "0",
            "country": country,
            "ui_lang": ui_lang,
        }

        url = f"{self._BASE}/search?{urlencode(args)}"
        resp = await client.get(
            url,
            headers=headers,
            cookies=cookies,
            follow_redirects=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        return self._parse_results(resp.text)

    def _parse_results(self, html_text: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        try:
            doc = _lh.fromstring(html_text)
        except Exception:
            return results

        for result in doc.xpath("//div[contains(@class, 'snippet ')]"):
            hrefs = result.xpath(".//a/@href")
            titles = result.xpath(".//div[contains(@class, 'title')]")
            if not hrefs or not titles:
                continue
            url = hrefs[0]
            # partial url likely means it's an ad
            if not urlparse(url).netloc:
                continue
            title = "".join(titles[0].itertext()).strip()
            if not title:
                continue

            snippet = ""
            contents = result.xpath(
                ".//div[contains(concat(' ', @class, ' '), ' content ')]"
            )
            if contents:
                snippet = " ".join(contents[0].itertext()).strip()
                pub_spans = contents[0].xpath(
                    ".//span[contains(@class, 't-secondary')]"
                )
                if pub_spans:
                    pub_date = " ".join(pub_spans[0].itertext()).strip()
                    if pub_date and snippet.startswith(pub_date):
                        snippet = snippet[len(pub_date) :].strip("- \n\t")

            results.append(
                SearchResult(
                    title=title, url=url, snippet=snippet or None, engine=self.name
                )
            )

        return results


# ---------------------------------------------------------------------------
# Wikipedia (port of searx/engines/mediawiki.py against wikipedia.org)
# ---------------------------------------------------------------------------


class WikipediaProvider(EngineProvider):
    """Wikipedia fulltext search via the MediaWiki Action API."""

    name = "wikipedia"
    timeout = 10.0
    retries = 1

    page_size = 5

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        language: str = "all",
        safesearch: int = 0,
        time_range: str | None = None,
        pageno: int = 1,
    ) -> list[SearchResult]:
        wiki_language = "en" if language == "all" else language.split("-")[0].lower()
        if not wiki_language.isalpha() or len(wiki_language) > 3:
            wiki_language = "en"

        args = {
            "action": "query",
            "list": "search",
            "format": "json",
            "srsearch": query,
            "sroffset": (pageno - 1) * self.page_size,
            "srlimit": self.page_size,
            "srwhat": "text",
            "srprop": "sectiontitle|snippet|timestamp|categorysnippet",
            "srsort": "relevance",
            "srenablerewrites": "1",
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": searxng_useragent(),
        }
        resp = await client.get(
            f"https://{wiki_language}.wikipedia.org/w/api.php",
            params=args,
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
            if (item.get("snippet") or "").startswith("#REDIRECT"):
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

            url = f"https://{wiki_language}.wikipedia.org/wiki/" + quote(
                title.replace(" ", "_")
            )
            sectiontitle = item.get("sectiontitle")
            if sectiontitle:
                url += "#" + quote(sectiontitle.replace(" ", "_"))
                title += f" / {sectiontitle}"

            results.append(
                SearchResult(
                    title=title, url=url, snippet=snippet or None, engine=self.name
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
                return cached

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
        BingProvider(),
        BraveProvider(),
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
