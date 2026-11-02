"""Data models shared by the graph nodes."""
from datetime import date
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

Collection = Literal["sentinel-2-l2a", "landsat-c2-l2", "sentinel-1-grd"]

# Collections where a cloud-cover filter makes sense (SAR sees through clouds).
OPTICAL_COLLECTIONS = {"sentinel-2-l2a", "landsat-c2-l2"}


class SearchRequest(BaseModel):
    """Structured search parameters extracted from the user's sentence by the LLM."""

    place: str | None = Field(
        None, description="Place name to geocode, e.g. 'Toulouse, France'. Null if the user gave coordinates."
    )
    bbox: list[float] | None = Field(
        None, description="[west, south, east, north] in WGS84 degrees, only if the user gave coordinates explicitly."
    )
    collection: Collection = Field(
        "sentinel-2-l2a",
        description="sentinel-2-l2a (optical 10m), landsat-c2-l2 (optical 30m) or sentinel-1-grd (SAR radar).",
    )
    start_date: date = Field(description="Start of the acquisition window (inclusive).")
    end_date: date = Field(description="End of the acquisition window (inclusive).")
    max_cloud_cover: float | None = Field(
        None, ge=0, le=100, description="Maximum cloud cover in percent. Null if not mentioned or for SAR."
    )
    limit: int = Field(5, ge=1, le=50, description="How many products the user wants back.")


class AgentState(TypedDict, total=False):
    query: str                # the original user sentence
    request: SearchRequest    # current search parameters (may be relaxed over time)
    bbox: list[float]         # resolved area of interest
    items: list[dict]         # ranked search results
    attempts: int             # number of STAC searches performed
    relaxations: list[str]    # human-readable log of what was relaxed
    error: str                # set when a step fails unrecoverably
    answer: str               # final response for the user
