# Search API

A lightweight, easy to use and deploy API for web searches. It uses SearXNG at core, It gives you flexible, programmatic access to multiple search engines + can be used as mcp

Best part? It’s 100% free. No premium tiers, no hidden fees, no paid API keys. you just pay for your infrastructure usage...

***

# Deploy and Host

Getting this up and running is super straightforward. Here’s the quick rundown on hosting it.

## About Hosting
Because it’s a lightweight app and api 1st by default, it runs almost anywhere. You can throw it on your local machine, a Raspberry Pi, or a cheap cloud VPS. even you can deploy fastapi apps on Vercel too or use Docker which is highly recommended to keep things simple.

## Why Deploy
Why deploying. The link of the example instance is just for demo, this app is intended to be used with and on your own hardware full control over everything. Plus, your app don't just depends on my free trial use your own server.. :) 

## Common Use Cases
* **AI Agents:** Giving LLMs live web access for up-to-date info 
* **Research Tools:** Aggregating news, articles, or data from multiple sources at once.
* **Automation:** Feeding clean, structured search data into rag, llm fine tuning workloads.

## Dependencies for

### Deployment Dependencies
For local testing, you really just uv. For a solid production setup, you’ll 

* ** docker/nerdctl or podman :** Honestly the easiest way to bundle everything and keep things isolated.
* **A basic hosting: you can host on vercel, render , railway and fastapicloud because of their so much generous free limits
