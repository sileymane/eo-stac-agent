"""External services: geocoding (Nominatim) and STAC catalog search (Earth Search)."""
import os
from datetime import date

import httpx
from pystac_client import Client

from .models import OPTICAL_COLLECTIONS

STAC_URL = os.getenv("STAC_URL", "https://earth-search.aws.element84.com/v1")
NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
# Nominatim's usage policy requires an identifying User-Agent.
USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "eo-stac-agent/0.1")


def geocode(place: str) -> list[float]:
    """Return the bounding box [west, south, east, north] of a place name."""
    resp = httpx.get(
        NOMINATIM_URL,
        params={"q": place, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Could not geocode '{place}'")
    south, north, west, east = map(float, results[0]["boundingbox"])
    return [west, south, east, north]


def search_items(
    collection: str,
    bbox: list[float],
    start: date,
    end: date,
    max_cloud_cover: float | None,
    max_items: int = 100,
) -> list[dict]:
    """Search the STAC catalog and return a compact summary of each item."""
    # Earth Search supports the STAC "query" extension but not CQL2 "filter".
    query = None
    if max_cloud_cover is not None and collection in OPTICAL_COLLECTIONS:
        query = {"eo:cloud_cover": {"lte": max_cloud_cover}}

    search = Client.open(STAC_URL).search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query=query,
        max_items=max_items,
    )
    return [_summarize_item(item) for item in search.items()]


def rank_items(items: list[dict], limit: int) -> list[dict]:
    """Clearest first (unknown cloud cover last), then most recent."""
    by_date = sorted(items, key=lambda i: i["datetime"] or "", reverse=True)
    by_cloud = sorted(by_date, key=lambda i: i["cloud_cover"] if i["cloud_cover"] is not None else 101)
    return by_cloud[:limit]


def _summarize_item(item) -> dict:
    props = item.properties
    thumbnail = item.assets.get("thumbnail")
    visual = item.assets.get("visual")
    cloud = props.get("eo:cloud_cover")
    return {
        "id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "platform": props.get("platform"),
        "tile": props.get("grid:code") or props.get("s2:mgrs_tile"),
        "cloud_cover": round(cloud, 1) if cloud is not None else None,
        "thumbnail": thumbnail.href if thumbnail else None,
        "visual": visual.href if visual else None,
        "stac_url": item.get_self_href(),
    }
