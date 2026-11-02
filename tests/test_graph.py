"""Offline tests: the LLM and external services are replaced by fakes."""
from datetime import date

import pytest
from langchain_core.messages import AIMessage

from eo_stac_agent import graph, tools
from eo_stac_agent.models import SearchRequest


class FakeLLM:
    def __init__(self, request: SearchRequest):
        self.request = request
        self.prompts = []

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:  # first call is parse_request
            return self.request
        return AIMessage(content="SUMMARY")


def item(cloud, day="2025-07-10"):
    return {"id": f"S2_{cloud}", "datetime": f"{day}T10:00:00", "platform": "sentinel-2a",
            "tile": "31TCJ", "cloud_cover": cloud, "thumbnail": None, "visual": None, "stac_url": "u"}


@pytest.fixture
def setup(monkeypatch):
    """Install a fake LLM and fake tools; returns a list recording each search's parameters."""
    searches = []

    def install(request, catalog):
        llm = FakeLLM(request)
        monkeypatch.setattr(graph, "get_llm", lambda: llm)
        monkeypatch.setattr(tools, "geocode", lambda place: [1.35, 43.53, 1.52, 43.67])

        def fake_search(collection, bbox, start, end, max_cloud):
            searches.append({"cloud": max_cloud, "start": start, "end": end})
            return [i for i in catalog if max_cloud is None or i["cloud_cover"] <= max_cloud]

        monkeypatch.setattr(tools, "search_items", fake_search)
        return searches

    return install


def july(**kw):
    return SearchRequest(place="Toulouse", start_date=date(2025, 7, 1), end_date=date(2025, 7, 31), **kw)


def test_direct_hit_no_relaxation(setup):
    searches = setup(july(max_cloud_cover=10), [item(5), item(30)])
    result = graph.build_graph().invoke({"query": "q"})
    assert len(searches) == 1
    assert [i["cloud_cover"] for i in result["items"]] == [5]
    assert result["relaxations"] == []
    assert result["answer"] == "SUMMARY"


def test_relaxes_cloud_cover_until_results(setup):
    # Like Toulouse in July 2025: clearest scene is 32.6% cloud.
    searches = setup(july(max_cloud_cover=10), [item(32.6), item(47.9), item(88.5)])
    result = graph.build_graph().invoke({"query": "q"})
    assert [s["cloud"] for s in searches] == [10, 30, 50]
    assert [i["cloud_cover"] for i in result["items"]] == [32.6, 47.9]
    assert result["relaxations"] == ["max cloud cover 10% -> 30%", "max cloud cover 30% -> 50%"]


def test_gives_up_after_max_attempts(setup):
    searches = setup(july(max_cloud_cover=0), [item(99)])
    result = graph.build_graph().invoke({"query": "q"})
    assert len(searches) == graph.MAX_ATTEMPTS
    assert result["items"] == []
    assert result["answer"].startswith("No products found")


def test_sar_widens_dates_instead_of_cloud(setup):
    request = july(collection="sentinel-1-grd", max_cloud_cover=None)
    searches = setup(request, [])
    graph.build_graph().invoke({"query": "q"})
    assert all(s["cloud"] is None for s in searches)
    assert searches[1]["start"] == date(2025, 6, 16)
    assert searches[1]["end"] == date(2025, 8, 15)


def test_missing_area_of_interest_skips_search(setup):
    searches = setup(SearchRequest(start_date=date(2025, 7, 1), end_date=date(2025, 7, 31)), [item(5)])
    result = graph.build_graph().invoke({"query": "q"})
    assert searches == []
    assert "No area of interest" in result["answer"]


def test_summary_failure_falls_back_to_table(setup, monkeypatch):
    setup(july(max_cloud_cover=10), [item(32.6), item(47.9)])
    llm = graph.get_llm()

    def invoke(prompt):
        llm.prompts.append(prompt)
        if len(llm.prompts) == 1:
            return llm.request
        raise TimeoutError("deadline exceeded")

    monkeypatch.setattr(llm, "invoke", invoke)
    result = graph.build_graph().invoke({"query": "q"})
    assert "Summary LLM call failed" in result["error"]
    assert "Found 2 product(s)" in result["answer"]
    assert "| 2025-07-10 | sentinel-2a | 31TCJ | 32.6 | S2_32.6 |" in result["answer"]
    assert "max cloud cover 10% -> 30%" in result["answer"]


def test_rank_items_clearest_then_newest():
    items = [item(None, "2025-07-20"), item(10, "2025-07-01"), item(10, "2025-07-15"), item(3)]
    ranked = tools.rank_items(items, limit=3)
    assert [(i["cloud_cover"], i["datetime"][:10]) for i in ranked] == [
        (3, "2025-07-10"), (10, "2025-07-15"), (10, "2025-07-01")]
