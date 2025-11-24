# Search API

This project provides a lightweight, deployable web interface for performing internet searches using the [DDGS](https://github.com/deedy5/ddgs) library. It exposes multiple structured endpoints through a FastAPI backend, enabling flexible and programmatic access to general, PDF, repository, and Wikipedia searches.

# Deployment

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FPratyay360%2Fsearch-api)

# Installation guide

```bash
podman-compose up -d             #to start the service
podman-compose logs -f           #to view logs
podman-compose down              #to stop the service

```

---

## Features

For generating dataset for training/finetuning llms a huge number of dataset is required for that you can use this api to fetch relevent urls on your domain.

### Framework

- Built with **FastAPI**, offering a clean and performant API layer.

### Why?

so, I was trying to fine tune a llm on a particular domain and i needed a lot of relevent information to build a dataset. but I found no free search engine api options (except duckduckgo ) Kudos to them. I had tried 1st with [Instant Data Scraper](https://chromewebstore.google.com/detail/instant-data-scraper/ofaokhiedipichpaobibbnahnkdoiiah) It was a manual thing. why should I be copying and pasting urls from an web extension no way.

So I was looking for some options and found DDGS a cli based meta search engine. It's good but having an api is always sweet. So I built this api.

I have tried following the KISS(Keep it simple, stupid) principle while building this project. Also given option for simple deployment because of this principle.

Have built this just to satisfy my need. If you find it useful please star ⭐ this repo.

### Middleware

- Includes **CORSMiddleware** with permissive configuration:
  - All origins allowed
  - All methods allowed
  - Credentials supported
  - Designed for easy cross-origin integration

### Endpoints

#### `GET /`

Returns a simple message confirming that the API is operational.

#### `GET /health`

Returns a simple message confirming that the API is operational.

#### `GET /useragent`

Returns a random user agent.

---

#### `GET /paper/`

searches for research papers related to the `query` .

```json

{
  query: "topics ..", # string data type ...
 }

```

#### `GET /search/engine`

Performs a general web search on a particulatr search engine `engine` ,
`query` and returns up to `limit` results as a list of URLs.

```json

{
  engine: "Enter Your Engine", # string data type  eg ["bing", "brave", "duckduckgo", "google", "mojeek", "yandex", "yahoo", "wikipedia"]

  query: "Enter Your Query", # string data type ...
  limit: "Enter Your Limit" # int data type
 }

```

** this can raise some flags for some search engines btw so use the search endpoint instead (it just randomly selects a search engine and searches on it)**

#### `GET /search/specific/{filetype}`

Searches specifically for a particular type of resource related to the query.

```json

{
  query: "Enter Your Query", # string data type
  filetype: "Enter Your File Type", # pdf, doc, docx, ppt, pptx, xls, xlsx, etc
  limit: "Enter Your Limit" # int data type
 }

```

## All the routes bellow will be taking `query` and `limit` as parameters

---

#### `GET /search/`

Performs a general web search for the specified `query` and returns up to `limit` results as a list of URLs.

#### `GET /searchpdf/`

Searches specifically for PDF resources related to the query.

#### `GET /books/`

Searches specifically for books related to the query searches annas archive.

#### `GET /repositories/`

Queries GitHub and GitLab for repositories matching the given search term.

#### `GET /wiki/`

Aggregates results from Wikipedia and related Wikimedia platforms for more comprehensive content retrieval.

#### `GET /news/`

Returns a list of news articles related to the specified query.

### Error Handling

- All endpoints return structured JSON responses on failure.
- Errors use standard HTTP status codes such as:
  - `400 Bad Request`
  - `500 Internal Server Error`
- Responses include clear diagnostic messages.

## Rate Limiting & Usage Notes

To avoid getting rate limited:

- Use proxies, VPNs, or Tor as a routing layer (not the browser one).
- When invoking the API repeatedly, apply a **politeness delay** to avoid overloading upstream engines.

# Disclaimer

This project is intended for **educational use only**.
You are fully responsible for complying with all applicable laws of your country (Be a law abiding citizen).

## Donate me if you liked this project

You can find my donation link on my [github profile](https://github.com/pratyay360#-want-to-support-my-work-) and [personal website](https://pratyay.vercel.app/#donate)
