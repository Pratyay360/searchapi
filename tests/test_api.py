from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main
from main import app
from src.search_service import SearchResult, SearchService
from tests.test_search_service import FakeProvider


@pytest.fixture
def mock_search_service(monkeypatch):
    fake_results = [
        SearchResult(
            title="Result 1",
            url="https://example.com/1",
            snippet="Snippet 1",
            engine="fake",
        )
    ]
    fake_svc = SearchService(
        engines=[FakeProvider("fake", results=fake_results)],
        cache_ttl=0,
    )
    monkeypatch.setattr(main, "search_svc", fake_svc)
    return fake_svc


def test_search_safesearch_coercion_string(mock_search_service):
    client = TestClient(app)
    response = client.get("/search/?query=python&safesearch=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["title"] == "Result 1"


def test_search_safesearch_invalid(mock_search_service):
    client = TestClient(app)
    response = client.get("/search/?query=python&safesearch=3")
    assert response.status_code == 422
    assert "safesearch" in response.text


def test_search_time_range_valid(mock_search_service):
    client = TestClient(app)
    response = client.get("/search/?query=python&time_range=week")
    assert response.status_code == 200


def test_search_time_range_invalid(mock_search_service):
    client = TestClient(app)
    response = client.get("/search/?query=python&time_range=invalid")
    assert response.status_code == 422
    assert "time_range" in response.text


def test_search_total_failure(monkeypatch):
    class AllFailedProvider(FakeProvider):
        async def search(self, *args, **kwargs):
            raise Exception("Service unavailable")

    fake_svc = SearchService(
        engines=[AllFailedProvider("bad_engine")],
        cache_ttl=0,
    )
    monkeypatch.setattr(main, "search_svc", fake_svc)

    client = TestClient(app)
    response = client.get("/search/?query=crash")
    assert response.status_code == 502
    assert response.json()["error"] == "All search engines failed to return results"
    assert "bad_engine" in response.json()["engines_failed"]


def test_search_unhandled_exception(monkeypatch):
    async def mock_search_raise(*args, **kwargs):
        raise RuntimeError("Catastrophic failure")

    fake_svc = SearchService(engines=[], cache_ttl=0)
    monkeypatch.setattr(fake_svc, "search", mock_search_raise)
    monkeypatch.setattr(main, "search_svc", fake_svc)

    client = TestClient(app)
    response = client.get("/search/?query=crash")
    assert response.status_code == 400
    assert "Catastrophic failure" in response.json()["error"]
