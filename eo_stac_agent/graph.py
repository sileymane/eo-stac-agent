r"""LangGraph workflow:

    parse_request -> geocode -> search --(results, index asked)--> analyze --> summarize -> END
          |             |        ^  |  \--(results, plain search)-----------> summarize
          |             |        |  \--(none, retries left)--> relax --+
          |             |        |_________________________________|
          \-------------\--(error)----------------------------------------> summarize

A plain data request stops at the product list. An analysis question ("how did vegetation around
Dakar change between July and August?") also searches a baseline period, computes the index on the
pixels, and writes a PNG map plus a provenance record (see analysis.py).

Compiled with a checkpointer, each thread_id is a conversation: follow-ups ("same but August")
start from the previous turn's parsed request.
"""
import json
import os
from datetime import date, timedelta
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from . import analysis, tools
from .models import ANALYSIS_COLLECTIONS, OPTICAL_COLLECTIONS, AgentState, SearchRequest

MAX_ATTEMPTS = 4          # 1 initial search + up to 3 relaxations
CLOUD_STEP = 20           # percentage points added per relaxation
DATE_STEP = timedelta(days=15)
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))  # seconds per LLM request
LLM_RETRIES = 1                                       # retries after the first request fails


def llm_model() -> str:
    # "provider:model", e.g. "openai:gpt-4.1-mini" or "google_genai:gemini-3.5-flash"
    return os.getenv("LLM_MODEL", "openai:gpt-4.1-mini")


@lru_cache
def get_llm():
    model = llm_model()
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
Set index only for analysis questions about vegetation (ndvi) or water (ndwi); index analysis needs
sentinel-2-l2a. For a change question, the later period is start_date/end_date and the earlier one
is baseline_start_date/baseline_end_date. If the user names whole months, use the whole months.
{followup}
Request: {query}"""

FOLLOWUP_CONTEXT = """
This conversation already has earlier requests (oldest first):
{history}
Search parameters of the last request: {previous}
If the new request refers back to it ("same but in August", "what about Landsat?", "only under 5% cloud"),
start from those parameters and change only what the user asks for. If it is a new, self-contained
request, ignore them.
"""
FOLLOWUP_HISTORY = 3      # earlier user requests shown to the parser

SUMMARY_PROMPT = """You are an Earth observation data assistant. Answer the user's request using ONLY the
search results below. Start with one sentence on what was found. If the search had to be relaxed,
say clearly how (the user asked for something stricter). Then give a markdown table with columns:
date, platform, tile, cloud %, id. After the table, list the STAC item URL of the best product.
If the analysis note is not "none", say in one sentence that the requested analysis could not be done and why.
Be concise.

User request: {query}
Final search parameters: {request}
Relaxations applied: {relaxations}
Analysis note: {analysis_note}
Results (best first): {items}"""

ANALYSIS_PROMPT = """You are an Earth observation analyst. Answer the user's question using ONLY the
numbers below. Never invent values. Index values range from -1 to 1; for NDVI, higher means more
or healthier vegetation; for NDWI, positive values usually indicate open water.

Format:
1. Two or three sentences answering the question directly, citing the key numbers (means, share of
   pixels that increased/decreased).
2. **Confidence**: one or two sentences grounded in the uncertainty notes. You may name plausible
   causes (season, rainfall, harvest) only as hypotheses to check, never as findings.
3. **Data**: scene id and acquisition date used for each period; mention any search relaxation.
4. **Outputs**: the PNG and provenance file paths.

User question: {query}
Search relaxations: {relaxations}
Analysis: {analysis}"""


# --- nodes -------------------------------------------------------------------

def parse_request(state: AgentState) -> AgentState:
    # Reset everything left over from the previous turn of this thread (no-op on a fresh thread).
    reset = {"error": None, "bbox": None, "items": [], "baseline_items": [], "attempts": 0,
             "relaxations": [], "analysis": None, "analysis_error": None,
             "messages": [HumanMessage(state["query"])]}
    prompt = PARSE_PROMPT.format(
        today=date.today().isoformat(), query=state["query"], followup=_followup_context(state)
    )
    llm = get_llm().with_structured_output(SearchRequest, method="function_calling")
    try:
        request = llm.invoke(prompt)
    except Exception as exc:
        return {**reset, "error": f"Could not understand the request (LLM call failed): {exc}"}
    return {**reset, "request": request, "parsed_request": request}


def _followup_context(state: AgentState) -> str:
    if not state.get("parsed_request"):
        return ""
    earlier = [m.content for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    history = "\n".join(f"- {q}" for q in earlier[-FOLLOWUP_HISTORY:])
    return FOLLOWUP_CONTEXT.format(
        history=history, previous=state["parsed_request"].model_dump_json(exclude_none=True)
    )


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
    request, bbox = state["request"], state["bbox"]
    attempts = state["attempts"] + 1
    try:
        update = {"items": _search_window(request, bbox, request.start_date, request.end_date)}
        if request.has_baseline:
            update["baseline_items"] = _search_window(
                request, bbox, request.baseline_start_date, request.baseline_end_date
            )
    except Exception as exc:
        return {"error": f"STAC search failed: {exc}", "attempts": attempts}
    return {**update, "attempts": attempts}


def _search_window(request: SearchRequest, bbox: list[float], start: date, end: date) -> list[dict]:
    items = tools.search_items(request.collection, bbox, start, end, request.max_cloud_cover)
    return tools.rank_items(items, request.limit)


def relax(state: AgentState) -> AgentState:
    """Loosen the search a bit: cloud cover first (optical only), then the date window(s)."""
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
        if request.has_baseline:
            update["baseline_start_date"] = request.baseline_start_date - DATE_STEP
            update["baseline_end_date"] = request.baseline_end_date + DATE_STEP
            note += (f"; baseline widened to {update['baseline_start_date']} / "
                     f"{update['baseline_end_date']}")
    return {"request": request.model_copy(update=update), "relaxations": state["relaxations"] + [note]}


def analyze(state: AgentState) -> AgentState:
    request = state["request"]
    if request.collection not in ANALYSIS_COLLECTIONS:
        return {"analysis_error": f"{request.index.upper()} analysis is only available for "
                                  f"{', '.join(sorted(ANALYSIS_COLLECTIONS))}, not {request.collection}."}
    try:
        return {"analysis": analysis.run(state, llm_model())}
    except Exception as exc:
        # Keep going: the search results are still worth reporting.
        return {"analysis_error": f"Analysis failed: {exc}"}


def summarize(state: AgentState) -> AgentState:
    update = _compose_answer(state)
    return {**update, "messages": [AIMessage(update["answer"])]}


def _compose_answer(state: AgentState) -> AgentState:
    if state.get("error"):
        return {"answer": f"Sorry, I couldn't complete the search. {state['error']}"}
    if not state.get("items"):
        tried = "; ".join(state["relaxations"]) or "no relaxation possible"
        return {"answer": f"No products found, even after relaxing the search ({tried})."}
    if state.get("analysis"):
        prompt = ANALYSIS_PROMPT.format(
            query=state["query"],
            relaxations=state["relaxations"] or "none",
            analysis=json.dumps(state["analysis"], indent=1),
        )
    else:
        prompt = SUMMARY_PROMPT.format(
            query=state["query"],
            request=state["request"].model_dump_json(),
            relaxations=state["relaxations"] or "none",
            analysis_note=state.get("analysis_error") or "none",
            # Band hrefs are for analysis.py, not the reader: keep them out of the prompt.
            items=json.dumps([{k: v for k, v in i.items() if k != "bands"} for i in state["items"]], indent=1),
        )
    try:
        # .text, not .content: Gemini 3.x returns content as a list of blocks.
        return {"answer": get_llm().invoke(prompt).text}
    except Exception as exc:
        # The search already succeeded: still give the user the products, just without the LLM prose.
        return {"answer": _fallback_answer(state), "error": f"Summary LLM call failed: {exc}"}


def _fallback_answer(state: AgentState) -> str:
    items = state["items"]
    lines = [f"Found {len(items)} product(s) (summary unavailable, LLM call failed)."]
    if state["relaxations"]:
        lines.append("Search relaxed: " + "; ".join(state["relaxations"]) + ".")
    if state.get("analysis_error"):
        lines.append(state["analysis_error"])
    if result := state.get("analysis"):
        lines += ["", f"{result['index'].upper()} analysis:"]
        for name, period in result["periods"].items():
            lines.append(f"- {name} ({period['scene']['id']}): {period['stats']}")
        if result["change"]:
            lines.append(f"- change: {result['change']}")
        lines += [f"- caveat: {note}" for note in result["uncertainty"]]
        lines.append(f"- outputs: {result['png']}, {result['provenance']}")
    lines += ["", "| date | platform | tile | cloud % | id |", "|---|---|---|---|---|"]
    for i in items:
        cloud = "n/a" if i["cloud_cover"] is None else f"{i['cloud_cover']:g}"
        lines.append(f"| {(i['datetime'] or '')[:10]} | {i['platform']} | {i['tile']} | {cloud} | {i['id']} |")
    lines += ["", f"Best product: {items[0]['stac_url']}"]
    return "\n".join(lines)


# --- routing -----------------------------------------------------------------

def route_after_parse(state: AgentState) -> str:
    return "summarize" if state.get("error") else "geocode"


def route_after_geocode(state: AgentState) -> str:
    return "summarize" if state.get("error") else "search"


def route_after_search(state: AgentState) -> str:
    if state.get("error"):
        return "summarize"
    request = state["request"]
    complete = bool(state.get("items")) and (bool(state.get("baseline_items")) or not request.has_baseline)
    if not complete and state["attempts"] < MAX_ATTEMPTS:
        return "relax"
    # Out of retries with only the target period found: analyze still reports that period alone.
    return "analyze" if request.index and state.get("items") else "summarize"


# --- graph -------------------------------------------------------------------

def make_checkpointer() -> InMemorySaver:
    # Our own state types must be allow-listed, or LangGraph refuses to restore them from a checkpoint.
    serde = JsonPlusSerializer(allowed_msgpack_modules=[(SearchRequest.__module__, SearchRequest.__name__)])
    return InMemorySaver(serde=serde)


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("parse_request", parse_request)
    builder.add_node("geocode", geocode)
    builder.add_node("search", search)
    builder.add_node("relax", relax)
    builder.add_node("analyze", analyze)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "parse_request")
    builder.add_conditional_edges("parse_request", route_after_parse, ["geocode", "summarize"])
    builder.add_conditional_edges("geocode", route_after_geocode, ["search", "summarize"])
    builder.add_conditional_edges("search", route_after_search, ["relax", "analyze", "summarize"])
    builder.add_edge("relax", "search")
    builder.add_edge("analyze", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=checkpointer)
