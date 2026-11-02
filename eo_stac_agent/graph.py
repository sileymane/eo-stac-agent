r"""LangGraph workflow:

    parse_request -> geocode -> search --(results)--------------> summarize -> END
                        |          ^  \--(none, retries left)--> relax --+
                        |          |_________________________________|
                        \--(error)--------------------------------------> summarize
"""
import json
import os
from datetime import date, timedelta
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

from . import tools
from .models import OPTICAL_COLLECTIONS, AgentState, SearchRequest

MAX_ATTEMPTS = 4          # 1 initial search + up to 3 relaxations
CLOUD_STEP = 20           # percentage points added per relaxation
DATE_STEP = timedelta(days=15)
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))  # seconds per LLM request
LLM_RETRIES = 1                                       # retries after the first request fails


@lru_cache
def get_llm():
    # "provider:model", e.g. "openai:gpt-4.1-mini" or "google_genai:gemini-3.8-flash"
    model = os.getenv("LLM_MODEL", "openai:gpt-4.1-mini")
    kwargs = {"temperature": 0, "timeout": LLM_TIMEOUT, "max_retries": LLM_RETRIES}
    if model.startswith("google_genai:"):
        # google-genai counts max_retries as total attempts (original included); OpenAI counts retries.
        kwargs["max_retries"] = LLM_RETRIES + 1
        # Parsing and formatting need no deep reasoning; Gemini's default thinking makes calls very slow.
        kwargs["reasoning_effort"] = os.getenv("LLM_REASONING_EFFORT", "low")
    return init_chat_model(model, **kwargs)


PARSE_PROMPT = """You convert Earth observation data requests into catalog search parameters.
Today's date is {today}. Resolve relative dates ("last summer", "past 2 weeks") against it.
If no date is given, use the last 30 days. If no collection is implied, use sentinel-2-l2a.
Use sentinel-1-grd for radar/SAR requests, landsat-c2-l2 for Landsat.

Request: {query}"""

SUMMARY_PROMPT = """You are an Earth observation data assistant. Answer the user's request using ONLY the
search results below. Start with one sentence on what was found. If the search had to be relaxed,
say clearly how (the user asked for something stricter). Then give a markdown table with columns:
date, platform, tile, cloud %, id. After the table, list the STAC item URL of the best product.
Be concise.

User request: {query}
Final search parameters: {request}
Relaxations applied: {relaxations}
Results (best first): {items}"""


# --- nodes -------------------------------------------------------------------

def parse_request(state: AgentState) -> AgentState:
    llm = get_llm().with_structured_output(SearchRequest, method="function_calling")
    request = llm.invoke(PARSE_PROMPT.format(today=date.today().isoformat(), query=state["query"]))
    return {"request": request, "attempts": 0, "relaxations": []}


def geocode(state: AgentState) -> AgentState:
    request = state["request"]
    if request.bbox:
        return {"bbox": request.bbox}
    if not request.place:
        return {"error": "No area of interest found in the request (give a place name or a bbox)."}
    try:
        return {"bbox": tools.geocode(request.place)}
    except Exception as exc:
        return {"error": f"Geocoding failed: {exc}"}


def search(state: AgentState) -> AgentState:
    request = state["request"]
    try:
        items = tools.search_items(
            request.collection, state["bbox"], request.start_date, request.end_date, request.max_cloud_cover
        )
    except Exception as exc:
        return {"error": f"STAC search failed: {exc}", "attempts": state["attempts"] + 1}
    return {"items": tools.rank_items(items, request.limit), "attempts": state["attempts"] + 1}


def relax(state: AgentState) -> AgentState:
    """Loosen the search a bit: cloud cover first (optical only), then the date window."""
    request = state["request"]
    cloud = request.max_cloud_cover
    if request.collection in OPTICAL_COLLECTIONS and cloud is not None and cloud < 100:
        new_cloud = min(100.0, cloud + CLOUD_STEP)
        update = {"max_cloud_cover": new_cloud}
        note = f"max cloud cover {cloud:g}% -> {new_cloud:g}%"
    else:
        start = request.start_date - DATE_STEP
        end = min(request.end_date + DATE_STEP, date.today())
        update = {"start_date": start, "end_date": end}
        note = f"date window widened to {start} / {end}"
    return {"request": request.model_copy(update=update), "relaxations": state["relaxations"] + [note]}


def summarize(state: AgentState) -> AgentState:
    if state.get("error"):
        return {"answer": f"Sorry, I couldn't complete the search. {state['error']}"}
    if not state.get("items"):
        tried = "; ".join(state["relaxations"]) or "no relaxation possible"
        return {"answer": f"No products found, even after relaxing the search ({tried})."}
    prompt = SUMMARY_PROMPT.format(
        query=state["query"],
        request=state["request"].model_dump_json(),
        relaxations=state["relaxations"] or "none",
        items=json.dumps(state["items"], indent=1),
    )
    try:
        return {"answer": get_llm().invoke(prompt).content}
    except Exception as exc:
        # The search already succeeded: still give the user the products, just without the LLM prose.
        return {"answer": _fallback_answer(state), "error": f"Summary LLM call failed: {exc}"}


def _fallback_answer(state: AgentState) -> str:
    items = state["items"]
    lines = [f"Found {len(items)} product(s) (summary unavailable, LLM call failed)."]
    if state["relaxations"]:
        lines.append("Search relaxed: " + "; ".join(state["relaxations"]) + ".")
    lines += ["", "| date | platform | tile | cloud % | id |", "|---|---|---|---|---|"]
    for i in items:
        cloud = "n/a" if i["cloud_cover"] is None else f"{i['cloud_cover']:g}"
        lines.append(f"| {(i['datetime'] or '')[:10]} | {i['platform']} | {i['tile']} | {cloud} | {i['id']} |")
    lines += ["", f"Best product: {items[0]['stac_url']}"]
    return "\n".join(lines)


# --- routing -----------------------------------------------------------------

def route_after_geocode(state: AgentState) -> str:
    return "summarize" if state.get("error") else "search"


def route_after_search(state: AgentState) -> str:
    if state.get("error") or state.get("items") or state["attempts"] >= MAX_ATTEMPTS:
        return "summarize"
    return "relax"


# --- graph -------------------------------------------------------------------

def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("parse_request", parse_request)
    builder.add_node("geocode", geocode)
    builder.add_node("search", search)
    builder.add_node("relax", relax)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "parse_request")
    builder.add_edge("parse_request", "geocode")
    builder.add_conditional_edges("geocode", route_after_geocode, ["search", "summarize"])
    builder.add_conditional_edges("search", route_after_search, ["relax", "summarize"])
    builder.add_edge("relax", "search")
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=checkpointer)
