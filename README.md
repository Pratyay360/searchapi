# Search API

This project provides a lightweight, deployable api interface for performing web searches using searxng as the base. It exposes multiple structured endpoints through a FastAPI backend, enabling flexible and programmatic access to different search engines,, and btw everything is free. no need for paid.. stuff ..

last i have checked brave costs about $$5$ for 1000 searches ...

# special mention

duckduckgo has a free api endpoint 
```
curl -X GET \
  'https://api.duckduckgo.com/?q=duckduckgo&format=json&pretty=1&no_redirect=1' \
  -H 'User-Agent: Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/555.96 (KHTML, like Gecko) Chrome/107.3.11.12 Safari/542.70 Edg/113.6.8.0' \
  -H 'Accept: application/json'
```

<div>
<a href="https://imgflip.com/i/axyo2u"><img src="https://i.imgflip.com/axyo2u.jpg" title="made at imgflip.com"/></a><div><a href="https://imgflip.com/memegenerator"></a></div>
</div>

# Deployment

[render](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FPratyay360%2Fsearchapi)

[railway](https://railway.com/deploy/6EC5D8?referralCode=MojJNc&utm_medium=integration&utm_source=template&utm_campaign=generic)


```bash
ghcr.io/pratyay360/searchapi:latest

docker.io/pratyay360/searchapi:latest

quay.io/pratyay360/searchapi:latest
```

## Installation guide

```bash
buildah build 
podman images 
```

### vs code config example

```json
{
  "mcpServers": {
    "searchapi": {
      "type": "remote",
      "url": ["https://searchapi.fastapicloud.dev/mcp"]
    }
  }
}
```

### Why?

so, I was trying to fine tune a llm on a particular domain and i needed a lot of relevent information to build a dataset. but I found no free search engine api options (except duckduckgo ) Kudos to them.

### btw have put that on hold :) cz it's not exciting any more.

<div>
<a href="https://imgflip.com/i/axyoxn"><img src="https://i.imgflip.com/axyoxn.jpg" title="made at imgflip.com"/></a><div><a href="https://imgflip.com/memegenerator">            </a></div>
</div> 

###

## Rate Limiting & Usage Notes

To avoid getting rate limited:

- Use proxies, VPNs, or Tor as a routing layer (not the browser). [for proxying with tor you can use use](https://flathub.org/en/apps/io.frama.tractor.carburetor)
- When invoking the API repeatedly, apply a **politeness delay** to avoid overloading upstream engines and getting captcha issues.

This project is intended for **educational use only**.


## Donate me if you liked this project

<div>
<a href="https://github.com/sponsors/Pratyay360" target="_blank">
  <img src="https://github.com/sponsors/Pratyay360/button" alt="Sponsor Pratyay360" />
</a>
</div>
