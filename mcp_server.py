from mcp.server.fastmcp import FastMCP
from searchweb import search, searchBooks, searchNews, searchEngine, searchPaper

mcp = FastMCP("SearchAPI")


@mcp.tool()
async def web_search(query: str, limit: int = 10) -> list[str]:
    """Search the web for the given query and return a list of URLs."""
    # pyrefly: ignore [bad-return]
    return await search(query, limit)


@mcp.tool()
async def search_by_engine(query: str, engine: str = "duckduckgo", limit: int = 10) -> list[str]:
    """Search the web using a specific engine and return a list of URLs.

    Available engines: bing, brave, duckduckgo, google, mojeek, yandex, yahoo, wikipedia.
    """
    valid_engines = ["bing", "brave", "duckduckgo", "google", "mojeek", "yandex", "yahoo", "wikipedia"]
    if engine not in valid_engines:
        raise ValueError(f"Invalid engine: {engine}. Choose from {valid_engines}")
    return await searchEngine(query, engine, limit)


@mcp.tool()
async def search_papers(query: str, limit: int = 5) -> list[str]:
    """Search for academic papers and return a list of DOIs."""
    return await searchPaper(query, limit)


@mcp.tool()
async def search_books(query: str, limit: int = 10) -> list[str]:
    """Search for books and return a list of URLs."""
    # pyrefly: ignore [bad-return]
    return await searchBooks(query, limit)


@mcp.tool()
async def search_news(query: str, limit: int = 10) -> list[str]:
    """Search for news articles and return a list of URLs."""
    # pyrefly: ignore [bad-return]
    return await searchNews(query, limit)


@mcp.tool()
async def search_pdfs(query: str, limit: int = 10) -> list[str]:
    """Search for PDF documents and return a list of URLs."""
    # pyrefly: ignore [bad-return]
    return await search(f"filetype:pdf {query}", limit)


@mcp.tool()
async def search_filetype(query: str, filetype: str, limit: int = 10) -> list[str]:
    """Search for files of a specific type and return a list of URLs.

    filetype examples: pdf, doc, docx, ppt, pptx, xls, xlsx.
    """
    # pyrefly: ignore [bad-return]
    return await search(f"filetype:{filetype} {query}", limit)


@mcp.tool()
async def search_repositories(query: str, limit: int = 10) -> list[str]:
    """Search for code repositories on GitHub and GitLab and return a list of URLs.

    The limit applies per site, so up to 2*limit URLs may be returned in total.
    """
    res = []
    s = await search(f"{query} site:github.com", limit)
    # pyrefly: ignore [bad-argument-type]
    res.extend(s)
    s = await search(f"{query} site:gitlab.com", limit)
    # pyrefly: ignore [bad-argument-type]
    res.extend(s)
    return res


@mcp.tool()
async def search_wiki(query: str, limit: int = 10) -> list[str]:
    """Search Wikipedia and Wikimedia sites and return a list of URLs.

    The limit applies per site, so up to 5*limit URLs may be returned in total.
    """
    merged = []
    for site in ["wikipedia.org", "wikibooks.org", "wiktionary.org", "wikiquote.org", "wikisource.org"]:
        res = await search(f"{query} site:{site}", limit)
        # pyrefly: ignore [bad-argument-type]
        merged.extend(res)
    return merged


if __name__ == "__main__":
    mcp.run()
