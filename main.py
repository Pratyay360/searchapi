from typing import List
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from searchweb import search, searchBooks, searchNews, searchEngine, searchPaper
from fake_useragent import UserAgent

ua = UserAgent()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return JSONResponse(
        content={"message": "DataOrchestra API is online and ready to use!"}
    )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return JSONResponse(content={"status": "ok"})

@app.get("/search/", status_code=status.HTTP_200_OK)
async def result_search(query: str, limit: int):
    try:
        results = await search(query, lim)
        return JSONResponse(content=results, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_400_BAD_REQUEST
        )

@app.get("/search/engine", status_code=status.HTTP_200_OK)
async def search_engine(query: str, limit: int, engine: Literal["bing", "brave", "duckduckgo", "google", "mojeek", "yandex", "yahoo", "wikipedia"]):
    try:
        results = await searchEngine(query, limit, engine)
        return JSONResponse(content=results, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_400_BAD_REQUEST
        )

@app.get("/search/paper", status_code=status.HTTP_200_OK)
async def search_paper(query: str, limit: int):
    try:
        results = await searchPaper(query, limit)
        return JSONResponse(content=results, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_400_BAD_REQUEST
        )


@app.get("/searchpdfs/", status_code=status.HTTP_200_OK)
async def search_pdfs(query: str, limits: int):
    try:
        raw_results = await search(f"filetype:pdf {query}", limits)
        urls = []
        for result in raw_results:
            urls.append(result)
        return JSONResponse(content=urls, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/search/specific/", status_code=status.HTTP_200_OK)
async def search_specific(query: str, filetype: str, limit: int):
    try:
        raw_results = await search(f"filetype:{filetype} {query}", limit)
        urls = []
        for result in raw_results:
            urls.append(result)
        return JSONResponse(content=urls, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@app.get("/repositories/", status_code=status.HTTP_200_OK)
async def search_repositories(query: str, limit: int):
    try:
        res = []
        s = await search(f"{query} site:github.com", limit)
        res.extend(s)
        s = await search(f"{query} site:gitlab.com", limit)
        res.extend(s)
        return JSONResponse(content=res, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.get("/wiki/", status_code=status.HTTP_200_OK)
async def search_wikipedia(query: str, limit: int):
    try:
        merged = []
        res = await search(f"{query} site:wikipedia.org", limit)
        merged.extend(res)
        res = await search(f"{query} site:wikibooks.org", limit)
        merged.extend(res)
        res = await search(f"{query} site:wiktionary.org", limit)
        merged.extend(res)
        res = await search(f"{query} site:grokipedia.com", limit)
        merged.extend(res)
        res = await search(f"{query} site:wikiquote.org", limit)
        merged.extend(res)
        res = await search(f"{query} site:wikisource.org", limit)
        merged.extend(res)
        return JSONResponse(content=merged, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )



@app.get("/books/", status_code=status.HTTP_200_OK)
async def search_books(query: str, limit: int):
    try:
        raw_results = await searchBooks(f"{query}", limits)
        urls = []
        for result in raw_results:
            urls.append(result)
        return JSONResponse(content=urls, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )



@app.get("/news/", status_code=status.HTTP_200_OK)
async def search_news(query: str, limit: int):
    try:
        raw_results = await searchNews(query, limit)
        urls = []
        for result in raw_results:
            urls.append(result)
        return JSONResponse(content=urls, status_code=status.HTTP_200_OK)
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
    return FileResponse(path='favicon.ico')

   