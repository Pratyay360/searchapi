"""SearXNG Search API – REST + MCP dual-use server."""

from __future__ import annotations

from typing import Any, Literal, cast

from fake_useragent import UserAgent
from fastapi import FastAPI, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi_mcp import FastApiMCP
from habanero import Crossref

from src.search_service import SearchService, configure_search_service

ua = UserAgent()

search_svc: SearchService = configure_search_service(
    max_concurrency=3,
    default_limit=10,
    cache_ttl=60,
)

app = FastAPI(title="Search API", version="0.3.0")


async def _search_paper(query: str, max_results: int = 5) -> list[str]:
    cr = Crossref()
    try:
        results = cast(dict[str, Any], cr.works(query=query, limit=max_results))
        return [item["DOI"] for item in results["message"]["items"] if item.get("DOI")]
    except Exception as e:
        raise RuntimeError(f"Paper search failed: {e}") from e


def _search_response(resp) -> JSONResponse:
    return JSONResponse(content=resp.to_dict(), status_code=status.HTTP_200_OK)


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return JSONResponse(
        content={
            "message": "Search API is online and ready to use!",
            "mcp server": "/mcp",
        }
    )


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return JSONResponse(content={"status": "ok"})


@app.get("/search/", status_code=status.HTTP_200_OK)
async def result_search(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0),
    safesearch: int = Query(0, ge=0, le=2),
    time_range: Literal["day", "week", "month", "year"] | None = Query(None),
    language: str = Query("all"),
):
    try:
        resp = await search_svc.search(
            query=query,
            limit=limit,
            offset=offset,
            safesearch=safesearch,
            time_range=time_range,
            language=language,
        )
        # Total failure: every engine that was queried errored or was blocked.
        # (An empty result set with healthy engines is a valid 200 response.)
        all_failed = bool(resp.engines_attempted) and all(
            name in resp.engines_failed for name in resp.engines_attempted
        )
        if all_failed:
            return JSONResponse(
                content={
                    "error": "All search engines failed to return results",
                    "engines_failed": resp.engines_failed,
                    "engines_attempted": resp.engines_attempted,
                    "query": query,
                    "results": [],
                    "total": 0,
                },
                status_code=status.HTTP_502_BAD_GATEWAY,
            )
        return _search_response(resp)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "query": query, "results": [], "total": 0},
            status_code=status.HTTP_400_BAD_REQUEST,
        )


@app.get("/search/engine", status_code=status.HTTP_200_OK)
async def search_engine(
    query: str = Query(...),
    engine: str = Query(
        ..., description="Engine name (e.g. duckduckgo, google, wikipedia, brave)"
    ),
    limit: int = Query(10, ge=1, le=50),
):
    try:
        resp = await search_svc.search(query=query, limit=limit, engines=[engine])
        return _search_response(resp)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_400_BAD_REQUEST
        )


@app.get("/search/paper", status_code=status.HTTP_200_OK)
async def search_paper_route(
    query: str = Query(...),
    limit: int = Query(5, ge=1, le=20),
):
    try:
        results = await _search_paper(query, limit)
        return JSONResponse(
            content={"results": results}, status_code=status.HTTP_200_OK
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_400_BAD_REQUEST
        )


@app.get("/searchpdfs/", status_code=status.HTTP_200_OK)
async def search_pdfs(query: str = Query(...), limits: int = Query(10, ge=1, le=50)):
    try:
        resp = await search_svc.search(query=f"filetype:pdf {query}", limit=limits)
        return JSONResponse(
            content=[r.url for r in resp.results], status_code=status.HTTP_200_OK
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/search/specific/", status_code=status.HTTP_200_OK)
async def search_specific(
    query: str = Query(...),
    filetype: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
):
    try:
        resp = await search_svc.search(
            query=f"filetype:{filetype} {query}", limit=limit
        )
        return JSONResponse(
            content=[r.url for r in resp.results], status_code=status.HTTP_200_OK
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/repositories/", status_code=status.HTTP_200_OK)
async def search_repositories(
    query: str = Query(...), limit: int = Query(10, ge=1, le=50)
):
    try:
        urls: list[str] = []
        for site in ("github.com", "gitlab.com"):
            resp = await search_svc.search(query=f"{query} site:{site}", limit=limit)
            urls.extend(r.url for r in resp.results)
        return JSONResponse(content=urls, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/wiki/", status_code=status.HTTP_200_OK)
async def search_wikipedia(
    query: str = Query(...), limit: int = Query(10, ge=1, le=50)
):
    try:
        urls: list[str] = []
        for site in (
            "wikipedia.org",
            "wikibooks.org",
            "wiktionary.org",
            "grokipedia.com",
            "wikiquote.org",
            "wikisource.org",
        ):
            resp = await search_svc.search(query=f"{query} site:{site}", limit=limit)
            urls.extend(r.url for r in resp.results)
        return JSONResponse(content=urls, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/books/", status_code=status.HTTP_200_OK)
async def search_books(query: str = Query(...), limit: int = Query(10, ge=1, le=50)):
    try:
        resp = await search_svc.search(query=query, limit=limit)
        return JSONResponse(
            content=[r.to_dict() for r in resp.results], status_code=status.HTTP_200_OK
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/news/", status_code=status.HTTP_200_OK)
async def search_news(query: str = Query(...), limit: int = Query(10, ge=1, le=50)):
    try:
        resp = await search_svc.search(query=query, limit=limit)
        return JSONResponse(
            content=[r.to_dict() for r in resp.results], status_code=status.HTTP_200_OK
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/useragent/", status_code=status.HTTP_200_OK)
async def return_useragent():
    try:
        return JSONResponse(content=ua.random, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(path="favicon.ico")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

