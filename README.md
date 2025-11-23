# Search API

This project provides a lightweight, deployable web interface for performing internet searches using the [DDGS](https://github.com/deedy5/ddgs) library. It exposes multiple structured endpoints through a FastAPI backend, enabling flexible and programmatic access to general, PDF, repository, and Wikipedia searches.

# Deployment

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FPratyay360%2Fsearch-api)

# Installation guide

```

podman-compose up -d             #to start the service
podman-compose logs -f           #to view logs
podman-compose down              #to stop the service

```

### Don't Forget to Star this repo :)

---

## Features

For generating dataset for training/finetuning llms a huge number of dataset is required for that you can use this api to fetch relevent urls on your domain.

### Framework

- Built with **FastAPI**, offering a clean and performant API layer.

### Middleware

- Includes **CORSMiddleware** with permissive configuration:
  - All origins allowed
  - All methods allowed
  - Credentials supported
  - Designed for easy cross-origin integration

### Endpoints

#### `GET /`

Returns a simple message confirming that the API is operational.

#### `POST /search/{query}/{lim}`

Performs a general web search for the specified query and returns up to `lim` results as a list of URLs.

#### `POST /search/pdfs/{query}/{lim}`

Searches specifically for PDF resources related to the query.

#### `POST /search/repositories/{query}/{lim}`

Queries GitHub and GitLab for repositories matching the given search term.

#### `POST /search/wikipedia/{query}/{lim}`

Aggregates results from Wikipedia and related Wikimedia platforms for more comprehensive content retrieval.

### Error Handling

- All endpoints return structured JSON responses on failure.
- Errors use standard HTTP status codes such as:
  - `400 Bad Request`
  - `500 Internal Server Error`
- Responses include clear diagnostic messages.

## Rate Limiting & Usage Notes

To avoid getting rate limited:

- Use proxies, VPNs, or Tor as a routing layer.
- When invoking the API repeatedly, apply a **politeness delay** to avoid overloading upstream engines.

# Disclaimer

This project is intended for **educational use only**.
You are fully responsible for complying with all applicable laws in your jurisdiction.

## Donate me if you liked this project

You can find my donation link on my [github profile](https://github.com/pratyay360#-want-to-support-my-work-) and [personal website](https://pratyay.vercel.app/#donate)
