"""Tests for the SearXNG-style search service.

Covers URL canonicalization, date offsets, result-quality scoring, dedup,
aggregation (limit/offset), and partial-failure tolerance.
"""

from __future__ import annotations

import httpx
import pytest

from src.search_service import (
    DuckDuckGoProvider,
    EngineProvider,
    SearchResult,
    SearchService,
    WikipediaProvider,
    _best_result,
    _date_offset,
    _result_quality,
    canonical_url,
)


# ---------------------------------------------------------------------------
# URL canonicalization (dedup rules)
# ---------------------------------------------------------------------------


class TestCanonicalUrl:
    def test_strips_tracking_params(self):
        url = "https://example.com/page?utm_source=x&utm_medium=y&id=5"
        assert canonical_url(url) == "https://example.com/page?id=5"

    def test_strips_session_params(self):
        url = "https://example.com/page?sid=abc123&keep=1"
        assert canonical_url(url) == "https://example.com/page?keep=1"

    def test_sorts_remaining_params(self):
        url = "https://example.com/page?b=2&a=1"
        assert canonical_url(url) == "https://example.com/page?a=1&b=2"

    def test_strips_www_lowercases_netloc(self):
        url = "https://WWW.Example.COM/Path/"
        assert canonical_url(url) == "https://example.com/Path"

    def test_strips_fragment_and_trailing_slash(self):
        url = "https://example.com/a/b/#section"
        assert canonical_url(url) == "https://example.com/a/b"

    def test_protocol_relative_url(self):
        assert canonical_url("//example.com/x") == "https://example.com/x"

    def test_same_doc_different_tracking_dedupe_to_same_canonical(self):
        a = canonical_url("https://example.com/doc?utm_source=newsletter")
        b = canonical_url("https://www.example.com/doc")
        assert a == b


# ---------------------------------------------------------------------------
# Date offsets
# ---------------------------------------------------------------------------


class TestDateOffset:
    def test_days(self):
        assert _date_offset("20260801", units=1, unit_type="days") == "20260731"

    def test_weeks(self):
        assert _date_offset("20260801", units=2, unit_type="weeks") == "20260718"

    def test_months(self):
        # approximate: 1 month == 30 days
        assert _date_offset("20260801", units=1, unit_type="months") == "20260702"

    def test_years(self):
        # approximate: 1 year == 365 days
        assert _date_offset("20260801", units=1, unit_type="years") == "20250801"

    def test_unknown_unit_falls_back_to_days(self):
        assert _date_offset("20260801", units=1, unit_type="fortnights") == "20260731"


# ---------------------------------------------------------------------------
# Result quality scoring
# ---------------------------------------------------------------------------


def _res(
    title: str = "t",
    url: str = "https://example.com/x",
    snippet: str | None = None,
    engine: str = "duckduckgo",
) -> SearchResult:
    return SearchResult(title=title, url=url, snippet=snippet, engine=engine)


class TestResultQuality:
    def test_snippet_wins_over_no_snippet(self):
        with_snippet = _res(snippet="some text")
        no_snippet = _res(snippet=None)
        assert _result_quality(with_snippet) > _result_quality(no_snippet)

    def test_best_result_keeps_better_snippet(self):
        good = _res(title="a", snippet="a much longer, informative snippet here")
        bad = _res(title="b", snippet="tiny")
        best = _best_result(good, bad)
        assert best is good

    def test_best_result_prefers_engine_priority_on_tie(self):
        ddg = _res(engine="duckduckgo", snippet="same")
        google = _res(engine="google", snippet="same")
        assert _best_result(google, ddg) is ddg


# ---------------------------------------------------------------------------
# Aggregation service
# ---------------------------------------------------------------------------


class FakeProvider(EngineProvider):
    """Engine provider returning canned results (no network)."""

    retries = 0
    timeout = 1.0

    def __init__(
        self,
        name: str,
        results: list[SearchResult] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self.name = name
        self._results = results or []
        self._exc = exc

    async def search(
        self,
        client,
        query,
        language="all",
        safesearch=0,
        time_range=None,
        pageno=1,
    ):
        if self._exc:
            raise self._exc
        return list(self._results)


def _provider_results(prefix: str, n: int, engine: str) -> list[SearchResult]:
    return [
        SearchResult(
            title=f"{prefix} title {i}",
            url=f"https://{prefix}.example.com/page/{i}",
            snippet=f"{prefix} snippet {i}",
            engine=engine,
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_search_dedups_same_canonical_url_across_engines():
    dup_a = SearchResult(
        title="Duck", url="https://example.com/doc?utm_source=a", snippet="s1", engine="duckduckgo"
    )
    dup_b = SearchResult(
        title="Google", url="https://www.example.com/doc", snippet="s2 longer snippet", engine="google"
    )
    svc = SearchService(
        engines=[
            FakeProvider("duckduckgo", [dup_a, _res(url="https://unique.example.com/1")]),
            FakeProvider("google", [dup_b]),
        ],
        cache_ttl=0,
    )
    resp = await svc.search(query="test")
    urls = [r.url for r in resp.results]
    # The two providers returned the same canonical document -> collapsed to one.
    assert len(urls) == len({canonical_url(u) for u in urls})
    assert resp.total == 2


@pytest.mark.asyncio
async def test_search_limit_and_offset():
    svc = SearchService(
        engines=[FakeProvider("duckduckgo", _provider_results("a", 10, "duckduckgo"))],
        cache_ttl=0,
    )
    resp = await svc.search(query="test", limit=3, offset=2)
    assert len(resp.results) == 3
    assert resp.total == 10
    assert resp.results[0].url == "https://a.example.com/page/2"


@pytest.mark.asyncio
async def test_search_partial_failure_keeps_successful_results():
    svc = SearchService(
        engines=[
            FakeProvider("good", _provider_results("g", 3, "good")),
            FakeProvider("bad", exc=httpx.ConnectError("boom")),
        ],
        cache_ttl=0,
    )
    resp = await svc.search(query="test")
    assert resp.total == 3
    assert resp.engines_used == ["good"]
    assert "bad" in resp.engines_failed
    assert "bad" in resp.engines_attempted


@pytest.mark.asyncio
async def test_search_all_engines_fail():
    svc = SearchService(
        engines=[
            FakeProvider("one", exc=httpx.ConnectError("boom")),
            FakeProvider("two", exc=RuntimeError("nope")),
        ],
        cache_ttl=0,
    )
    resp = await svc.search(query="test")
    assert resp.results == []
    assert resp.total == 0
    assert set(resp.engines_failed) == {"one", "two"}


@pytest.mark.asyncio
async def test_search_engine_allowlist():
    svc = SearchService(
        engines=[
            FakeProvider("duckduckgo", _provider_results("d", 2, "duckduckgo")),
            FakeProvider("google", _provider_results("g", 2, "google")),
        ],
        cache_ttl=0,
    )
    resp = await svc.search(query="test", engines=["google"])
    assert resp.engines_used == ["google"]
    assert all(r.engine == "google" for r in resp.results)


@pytest.mark.asyncio
async def test_search_unknown_engine_returns_empty():
    svc = SearchService(
        engines=[FakeProvider("duckduckgo", _provider_results("d", 2, "duckduckgo"))],
        cache_ttl=0,
    )
    resp = await svc.search(query="test", engines=["does-not-exist"])
    assert resp.results == []
    assert resp.engines_attempted == []


def test_canonical_url_preserves_repeated_and_encoded_params():
    url = "HTTPS://Example.com/search?tag=a&tag=b&q=hello%20world"
    assert canonical_url(url) == "https://example.com/search?q=hello+world&tag=a&tag=b"


@pytest.mark.asyncio
async def test_empty_engine_list_does_not_enable_defaults():
    svc = SearchService(engines=[], cache_ttl=0)
    resp = await svc.search(query="test")
    assert resp.engines_attempted == []


class RecordingProvider(FakeProvider):
    def __init__(self):
        super().__init__("recording")
        self.calls = []

    async def search(
        self,
        client,
        query,
        language="all",
        safesearch=0,
        time_range=None,
        pageno=1,
    ):
        self.calls.append((language, pageno))
        return [
            SearchResult(
                title=f"result {pageno}",
                url=f"https://example.com/{pageno}",
            )
        ]


@pytest.mark.asyncio
async def test_search_propagates_language_and_fetches_required_pages():
    provider = RecordingProvider()
    svc = SearchService(engines=[provider], cache_ttl=0)
    resp = await svc.search(query="test", language="de-DE", limit=2, offset=10)
    assert provider.calls == [("de-DE", 1), ("de-DE", 2)]
    assert resp.total == 2


@pytest.mark.asyncio
async def test_search_caps_upstream_page_requests():
    provider = RecordingProvider()
    svc = SearchService(engines=[provider], cache_ttl=0)
    await svc.search(query="test", limit=50, offset=10_000)
    assert len(provider.calls) == 5


@pytest.mark.asyncio
async def test_successful_empty_engine_is_reported_as_used():
    svc = SearchService(engines=[FakeProvider("empty")], cache_ttl=0)
    resp = await svc.search(query="test")
    assert resp.engines_used == ["empty"]
    assert resp.engines_failed == {}


class FailingSecondPageProvider(FakeProvider):
    retries = 0

    async def search(
        self,
        client,
        query,
        language="all",
        safesearch=0,
        time_range=None,
        pageno=1,
    ):
        if pageno == 2:
            raise httpx.ConnectError("boom")
        return _provider_results("first", 10, self.name)


@pytest.mark.asyncio
async def test_search_keeps_results_when_later_page_fails():
    svc = SearchService(
        engines=[FailingSecondPageProvider("partial")],
        cache_ttl=0,
    )
    resp = await svc.search(query="test", limit=5, offset=10)
    assert resp.total == 10
    assert resp.engines_used == ["partial"]
    assert "partial" in resp.engines_failed


def test_wikipedia_parses_action_api_results():
    results = WikipediaProvider()._parse_results(
        {
            "query": {
                "search": [
                    {
                        "title": "Python (programming language)",
                        "snippet": "A <span class=\"searchmatch\">programming</span> language",
                    }
                ]
            }
        }
    )
    assert results == [
        SearchResult(
            title="Python (programming language)",
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            snippet="A programming language",
            engine="wikipedia",
        )
    ]


@pytest.mark.asyncio
async def test_duckduckgo_uses_first_page_token_for_next_page():
    requests = []

    def handler(request: httpx.Request):
        requests.append(request)
        return httpx.Response(
            200,
            text='''
                <html><body>
                    <form><input name="vqd" value="token"></form>
                    <div id="links"><div class="web-result">
                        <h2><a href="https://example.com/result">Result</a></h2>
                    </div></div>
                </body></html>
            ''',
        )

    provider = DuckDuckGoProvider()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await provider.search(client, "test", pageno=1)
        await provider.search(client, "test", pageno=2)

    assert requests[1].content
    assert b"vqd=token" in requests[1].content
    assert b"s=10" in requests[1].content
