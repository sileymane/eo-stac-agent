"""Offline tests: the LLM and external services are replaced by fakes."""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from langchain_core.messages import AIMessage

from eo_stac_agent import analysis, graph, tools
from eo_stac_agent.models import SearchRequest


class FakeLLM:
    """Structured calls (parse_request) return the queued requests in order; plain calls return SUMMARY."""

    def __init__(self, requests: list[SearchRequest]):
        self.requests = list(requests)
        self.parse_prompts = []

    def with_structured_output(self, schema, **kwargs):
        return FakeParser(self)

    def invoke(self, prompt):
        return AIMessage(content="SUMMARY")


class FakeParser:
    def __init__(self, llm: FakeLLM):
        self.llm = llm

    def invoke(self, prompt):
        self.llm.parse_prompts.append(prompt)
        return self.llm.requests.pop(0)


def item(cloud, day="2025-07-10"):
    return {"id": f"S2_{cloud}", "datetime": f"{day}T10:00:00", "platform": "sentinel-2a",
            "tile": "31TCJ", "cloud_cover": cloud, "thumbnail": None, "visual": None, "stac_url": "u"}


@pytest.fixture
def setup(monkeypatch):
    """Install a fake LLM and fake tools; returns a list recording each search's parameters.

    `request` is one SearchRequest, or a list of them (one per conversation turn).
    `catalog` is a list of items, or a function start_date -> items (to give each period its own scenes)."""
    searches = []

    def install(request, catalog):
        llm = FakeLLM(request if isinstance(request, list) else [request])
        monkeypatch.setattr(graph, "get_llm", lambda: llm)
        monkeypatch.setattr(tools, "geocode", lambda place: [1.35, 43.53, 1.52, 43.67])

        def fake_search(collection, bbox, start, end, max_cloud):
            searches.append({"cloud": max_cloud, "start": start, "end": end})
            pool = catalog(start) if callable(catalog) else catalog
            return [i for i in pool if max_cloud is None or i["cloud_cover"] <= max_cloud]

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
        raise TimeoutError("deadline exceeded")

    monkeypatch.setattr(llm, "invoke", invoke)  # only the summary call; parsing goes through FakeParser
    result = graph.build_graph().invoke({"query": "q"})
    assert "Summary LLM call failed" in result["error"]
    assert "Found 2 product(s)" in result["answer"]
    assert "| 2025-07-10 | sentinel-2a | 31TCJ | 32.6 | S2_32.6 |" in result["answer"]
    assert "max cloud cover 10% -> 30%" in result["answer"]


def test_parse_failure_skips_to_summary(setup, monkeypatch):
    searches = setup(july(), [item(5)])
    def invoke(self, prompt):
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(FakeParser, "invoke", invoke)
    result = graph.build_graph().invoke({"query": "q"})
    assert searches == []
    assert result["answer"].startswith("Sorry, I couldn't complete the search.")
    assert "503 UNAVAILABLE" in result["answer"]


def converse(*queries):
    """Run several turns in one checkpointed thread; returns the final state after each turn."""
    app = graph.build_graph(checkpointer=graph.make_checkpointer())
    config = {"configurable": {"thread_id": "t1"}}
    return [app.invoke({"query": q}, config) for q in queries]


def august(**kw):
    return SearchRequest(place="Toulouse", start_date=date(2025, 8, 1), end_date=date(2025, 8, 31), **kw)


def test_followup_starts_from_original_not_relaxed_request(setup):
    # Turn 1 needs relaxing to 30%; the follow-up must still be offered the user's 10%.
    searches = setup([july(max_cloud_cover=10), august(max_cloud_cover=10)], [item(25)])
    first, second = converse("S2 over Toulouse in July, <10% cloud", "same but August")
    llm = graph.get_llm()

    assert first["relaxations"] == ["max cloud cover 10% -> 30%"]
    assert "earlier requests" not in llm.parse_prompts[0]
    assert "- S2 over Toulouse in July, <10% cloud" in llm.parse_prompts[1]
    assert '"max_cloud_cover":10.0' in llm.parse_prompts[1]
    assert searches[2]["cloud"] == 10 and searches[2]["start"] == date(2025, 8, 1)
    assert [type(m).__name__ for m in second["messages"]] == ["HumanMessage", "AIMessage"] * 2


def test_previous_turn_error_does_not_leak(setup):
    no_place = SearchRequest(start_date=date(2025, 7, 1), end_date=date(2025, 7, 31))
    searches = setup([no_place, july(max_cloud_cover=10)], [item(5)])
    first, second = converse("July imagery", "over Toulouse")

    assert "No area of interest" in first["answer"]
    assert second["error"] is None
    assert second["answer"] == "SUMMARY"
    assert len(searches) == 1


def test_rank_items_clearest_then_newest():
    items = [item(None, "2025-07-20"), item(10, "2025-07-01"), item(10, "2025-07-15"), item(3)]
    ranked = tools.rank_items(items, limit=3)
    assert [(i["cloud_cover"], i["datetime"][:10]) for i in ranked] == [
        (3, "2025-07-10"), (10, "2025-07-15"), (10, "2025-07-01")]


# --- analysis (pixel reads replaced by uniform synthetic scenes) ---------------

S2_BANDS = {n: {"href": f"https://cog/{n}.tif", "scale": 0.0001, "offset": -0.1} for n in ("red", "nir", "green", "scl")}


def s2(item_id, day, cloud=5):
    return {**item(cloud, day), "id": item_id, "bands": S2_BANDS}


def fake_pixels(scenes):
    """load_bands replacement: scenes maps item id -> (nir DN, red DN, SCL class), uniform over the grid."""
    def load(item, names, bbox, shape):
        nir, red, scl = scenes[item["id"]]
        values = {"nir": nir, "red": red, "green": 1000, "scl": scl}
        return {n: np.full(shape, values[n], dtype=np.uint16) for n in names}
    return load


def change_question(**kw):
    return july(index="ndvi", baseline_start_date=date(2025, 6, 1), baseline_end_date=date(2025, 6, 30), **kw)


@pytest.fixture
def outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    return tmp_path


def test_change_question_writes_png_and_provenance(setup, monkeypatch, outputs):
    by_period = {date(2025, 6, 1): [s2("JUNE", "2025-06-15")], date(2025, 7, 1): [s2("JULY", "2025-07-15")]}
    searches = setup(change_question(max_cloud_cover=10), lambda start: by_period[start])
    # June: nir 0.2, red 0.1 -> NDVI 0.333. July: nir 0.3, red 0.05 -> NDVI 0.714.
    monkeypatch.setattr(analysis, "load_bands", fake_pixels({"JUNE": (3000, 2000, 4), "JULY": (4000, 1500, 4)}))
    result = graph.build_graph().invoke({"query": "did vegetation change?"})

    a = result["analysis"]
    assert len(searches) == 2  # target and baseline searched in the same attempt
    assert a["periods"]["baseline"]["stats"]["mean"] == pytest.approx(0.333, abs=1e-3)
    assert a["periods"]["target"]["stats"]["mean"] == pytest.approx(0.714, abs=1e-3)
    assert a["change"]["share_increase"] == 1.0
    assert any("30 days apart" in note for note in a["uncertainty"])
    assert Path(a["png"]).stat().st_size > 0
    prov = json.loads(Path(a["provenance"]).read_text(encoding="utf-8"))
    assert prov["query"] == "did vegetation change?"
    assert prov["periods"]["baseline"]["scene"]["id"] == "JUNE"
    assert prov["periods"]["target"]["band_hrefs"]["nir"] == "https://cog/nir.tif"
    assert prov["final_request"]["baseline_start_date"] == "2025-06-01"
    assert prov["software"]["rasterio"]
    assert result["answer"] == "SUMMARY"


def test_cloudy_scene_over_area_falls_back_to_next(setup, monkeypatch, outputs):
    # Low tile-level cloud cover, but the area itself is under cloud (SCL 9) in the first scene.
    setup(july(index="ndvi"), [s2("CLOUDY", "2025-07-05", cloud=1), s2("CLEAR", "2025-07-20", cloud=4)])
    monkeypatch.setattr(analysis, "load_bands", fake_pixels({"CLOUDY": (3000, 2000, 9), "CLEAR": (3000, 2000, 4)}))
    a = graph.build_graph().invoke({"query": "q"})["analysis"]

    assert a["periods"]["target"]["scene"]["id"] == "CLEAR"
    assert a["periods"]["target"]["scenes_tried"] == [
        {"id": "CLOUDY", "valid_fraction": 0.0}, {"id": "CLEAR", "valid_fraction": 1.0}]
    assert a["change"] is None
    assert any("after trying 2" in note for note in a["uncertainty"])


def test_missing_baseline_still_analyzes_target(setup, monkeypatch, outputs):
    searches = setup(change_question(max_cloud_cover=10),
                     lambda start: [] if start.month <= 6 else [s2("JULY", "2025-07-15")])
    monkeypatch.setattr(analysis, "load_bands", fake_pixels({"JULY": (4000, 1500, 4)}))
    result = graph.build_graph().invoke({"query": "q"})

    assert len(searches) == 2 * graph.MAX_ATTEMPTS  # kept relaxing for the baseline, then gave up
    assert result["analysis"]["change"] is None
    assert any("no scene found for the baseline" in n for n in result["analysis"]["uncertainty"])


def test_index_on_unsupported_collection_reports_search_only(setup, outputs):
    setup(july(index="ndvi", collection="landsat-c2-l2"), [item(5)])
    result = graph.build_graph().invoke({"query": "q"})

    assert result["analysis"] is None
    assert "only available for sentinel-2-l2a" in result["analysis_error"]
    assert result["answer"] == "SUMMARY"
    assert list(outputs.iterdir()) == []
