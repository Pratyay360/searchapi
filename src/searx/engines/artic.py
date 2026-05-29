# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Art Institute of Chicago

Explore thousands of artworks from The Art Institute of Chicago.

* https://artic.edu

"""

from json import loads
from urllib.parse import urlencode

about = {
    "website": "https://www.artic.edu",
    "wikidata_id": "Q239303",
    "official_api_documentation": "http://api.artic.edu/docs/",
    "use_official_api": True,
    "require_api_key": False,
    "results": "JSON",
}

categories = ["images"]
paging = True
page_size = 20

search_api = "https://api.artic.edu/api/v1/artworks/search?"
image_api = "https://www.artic.edu/iiif/2/"


def request(query, params):

    args = urlencode(
        {
            "q": query,
            "page": params["pageno"],
            "fields": "id,title,artist_display,medium_display,image_id,date_display,dimensions,artist_titles",
            "limit": page_size,
        }
    )
    params["url"] = search_api + args

    logger.debug("query_url --> %s", params["url"])
    return params


def response(resp):

    results = []
    json_data = loads(resp.text)

    for result in json_data["data"]:
        if not result["image_id"]:
            continue

        results.append(
            {
                "url": "https://artic.edu/artworks/{id}".format(**result),
                "title": result["title"]
                + " ({date_display}) // {artist_display}".format(**result),
                "content": "{medium_display} // {dimensions}".format(**result),
                "author": ", ".join(result["artist_titles"]),
                "img_src": image_api + "/{image_id}/full/843,/0/default.jpg".format(**result),
                "template": "images.html",
            }
        )

    return results
