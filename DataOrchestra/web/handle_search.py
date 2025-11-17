from ddgs import DDGS
import os
import shutil
import subprocess


def clone_repo(repo_url: str, clone_dir: str = "repo_clone"):
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    subprocess.run(["git", "clone", "--depth", "1",
                    repo_url, clone_dir], check=True)
    return clone_dir


def web_search(param: str):
    results = DDGS().text(f"{param}", backend="auto")
    data = list(results)
    urls = [item['href'] for item in data]
    return urls


def search_pdf(param: str):
    results = DDGS().text(f"filetype:pdf {param}", backend="auto")
    data = list(results)
    urls = [item['href'] for item in data]
    return urls


def search_gitlab(param: str):
    results = DDGS().text(f"site:gitlab.com {param}", backend="auto")
    data = list(results)
    urls = [item['href'] for item in data]
    return urls


def search_github(param: str):
    results = DDGS().text(f"site:github.com {param}", backend="auto")
    data = list(results)
    urls = [item['href'] for item in data]
    return urls


def search_repos(param: str):
    github_results = search_github(param)
    gitlab_results = search_gitlab(param)
    res = github_results + gitlab_results
    return res

