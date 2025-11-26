import asyncio
import time
from random import shuffle
from typing import List, Optional
import requests
import urllib3
from crossref.restful import Works
from ddgs import DDGS
from fake_useragent import UserAgent

urllib3.disable_warnings()
ua = UserAgent()


async def search(query: str, max_results: int = 10):
    max_retries = 7
    ddgs = DDGS(timeout=100)
    for attempt in range(max_retries):
        try:
            results = ddgs.text(query, max_results=max_results)
            res = []
            for result in results:
                res.append(result.get("href", "No URL"))
            return res
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                continue
            else:
                raise RuntimeError(f"Search failed after {max_retries} attempts")


async def searchBooks(query: str, max_results: int = 10):
    max_retries = 7
    ddgs = DDGS(timeout=100)
    for attempt in range(max_retries):
        try:
            results = ddgs.books(query, max_results=max_results)
            res = []
            for result in results:
                res.append(result.get("href", "No URL"))
            return res
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                continue
            else:
                raise RuntimeError(f"Search failed after {max_retries} attempts")


async def searchNews(query: str, max_results: int = 10):
    max_retries = 7
    ddgs = DDGS(timeout=100)
    for attempt in range(max_retries):
        try:
            results = ddgs.news(query, max_results=max_results)
            res = []
            for result in results:
                res.append(result.get("href", "No URL"))
            return res
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                continue
            else:
                raise RuntimeError(f"Search failed after {max_retries} attempts")


async def searchPaper(query: str, max_results: int = 5):
    works = Works()
    try:
        a = requests.get(works.query(query).url, headers={"User-Agent": f"{ua.random}"})
        res = []
        x = a.json().get("message").get("items")
        for i in x:
            res.append(i.get("DOI"))
        return res
    except Exception as e:
        raise RuntimeError(f"Search failed {e}")


async def searchEngine(query: str, engine: str = "auto", max_results: int = 10):
    ddgs = DDGS(timeout=100)
    try:
        results = ddgs.text(query, max_results=max_results, backend=engine)
        res = []
        for result in results:
            res.append(result.get("href", "No URL"))
        return res
    except Exception as e:
        raise RuntimeError(f"Search failed {e}")
