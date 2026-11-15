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
        self.summary_prompts = []

    def with_structured_output(self, schema, **kwargs):
        return FakeParser(self)

    def invoke(self, prompt):
        self.summary_prompts.append(prompt)
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
    """load_bands replacement: scenes maps item id -> (nir DN, red DN, SCL class[, green DN[, half]]),
    uniform over the grid. half = "west"/"east" makes the tile cover only that half (SCL 0 elsewhere)."""
    def load(item, names, bbox, shape):
        nir, red, scl, *extra = scenes[item["id"]]
        green = extra[0] if extra else 1000
        half = extra[1] if len(extra) > 1 else None
        values = {"nir": nir, "red": red, "green": green, "scl": scl}
        out = {n: np.full(shape, values[n], dtype=np.uint16) for n in names}
        if half:
            outside = np.s_[:, shape[1] // 2:] if half == "west" else np.s_[:, :shape[1] // 2]
            for n in names:
                out[n][outside] = 0
        return out
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
    assert a["comparability"]["same_season"]  # mid-June vs mid-July: within tolerance, no seasonal warning
    assert not any("not seasonally comparable" in note for note in a["uncertainty"])
    assert Path(a["png"]).stat().st_size > 0
    prov = json.loads(Path(a["provenance"]).read_text(encoding="utf-8"))
    assert prov["query"] == "did vegetation change?"
    assert prov["periods"]["baseline"]["scenes"][0]["id"] == "JUNE"
    assert prov["periods"]["target"]["band_hrefs"]["JULY"]["nir"] == "https://cog/nir.tif"
    assert prov["final_request"]["baseline_start_date"] == "2025-06-01"
    assert prov["software"]["rasterio"]
    assert prov["method"]["definition"]["sentinel2"] == "(B08 - B04) / (B08 + B04)"
    assert prov["comparability"]["days_apart"] == 30
    assert result["answer"] == "SUMMARY"


def test_cloudy_scene_over_area_falls_back_to_next(setup, monkeypatch, outputs):
    # Low tile-level cloud cover, but the area itself is under cloud (SCL 9) in the first scene.
    setup(july(index="ndvi"), [s2("CLOUDY", "2025-07-05", cloud=1), s2("CLEAR", "2025-07-20", cloud=4)])
    monkeypatch.setattr(analysis, "load_bands", fake_pixels({"CLOUDY": (3000, 2000, 9), "CLEAR": (3000, 2000, 4)}))
    a = graph.build_graph().invoke({"query": "q"})["analysis"]

    assert a["periods"]["target"]["scenes"][0]["id"] == "CLEAR"
    assert [(t["tiles"], t["valid_fraction"]) for t in a["periods"]["target"]["overpasses_tried"]] == [
        (["CLOUDY"], 0.0), (["CLEAR"], 1.0)]
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


# --- provider settings --------------------------------------------------------

@pytest.mark.parametrize("model, effort, expected", [
    ("openai:gpt-4.1-mini", "", {"max_retries": 1, "timeout": 60.0}),
    # google-genai counts the first request as an attempt, and thinks slowly unless told otherwise.
    ("google_genai:gemini-3.5-flash", "", {"max_retries": 2, "timeout": 60.0, "reasoning_effort": "low"}),
    # langchain-openrouter takes milliseconds: 60 would be a 60 ms timeout.
    ("openrouter:google/gemini-3.5-flash", "", {"max_retries": 1, "timeout": 60_000}),
    ("openrouter:google/gemini-3.5-flash", "low", {"max_retries": 1, "timeout": 60_000, "reasoning": {"effort": "low"}}),
])
def test_get_llm_provider_settings(monkeypatch, model, effort, expected):
    calls = []
    monkeypatch.setattr(graph, "init_chat_model", lambda m, **kw: calls.append((m, kw)))
    monkeypatch.setenv("LLM_MODEL", model)
    monkeypatch.setenv("LLM_REASONING_EFFORT", effort)
    graph.get_llm.cache_clear()
    try:
        graph.get_llm()
    finally:
        graph.get_llm.cache_clear()
    (called_model, kwargs), = calls
    assert called_model == model
    assert {k: v for k, v in kwargs.items() if k != "temperature"} == expected


def test_two_period_search_without_index_shows_both_periods_to_summary(setup):
    # Seen live: a model parsed both periods but no index; the summary, shown only the target
    # results, then claimed the baseline period had no images.
    by_period = {date(2025, 6, 1): [s2("JUNE", "2025-06-15")], date(2025, 7, 1): [s2("JULY", "2025-07-15")]}
    setup(july(baseline_start_date=date(2025, 6, 1), baseline_end_date=date(2025, 6, 30)),
          lambda start: by_period[start])
    result = graph.build_graph().invoke({"query": "q"})

    assert result["analysis"] is None
    prompt = graph.get_llm().summary_prompts[-1]
    assert '"id": "JUNE"' in prompt and '"id": "JULY"' in prompt
    assert "https://cog/" not in prompt  # band hrefs stay out of the prompt


def test_single_period_summary_says_no_baseline(setup):
    setup(july(), [item(5)])
    graph.build_graph().invoke({"query": "q"})
    assert "Baseline period results (best first): none" in graph.get_llm().summary_prompts[-1]


def test_change_across_seasons_is_flagged(setup, monkeypatch, outputs):
    # Dakar-like: July baseline vs September target is rainy-season greening, not necessarily change.
    request = SearchRequest(place="Dakar", start_date=date(2025, 9, 1), end_date=date(2025, 9, 30), index="ndvi",
                            baseline_start_date=date(2025, 7, 1), baseline_end_date=date(2025, 7, 31))
    by_period = {7: [s2("JUL", "2025-07-05")], 9: [s2("SEP", "2025-09-03")]}
    setup(request, lambda start: by_period[start.month])
    monkeypatch.setattr(analysis, "load_bands", fake_pixels({"JUL": (3000, 2000, 4), "SEP": (4000, 1500, 4)}))
    result = graph.build_graph().invoke({"query": "q"})

    a = result["analysis"]
    assert a["comparability"] == {"days_apart": 60, "seasonal_offset_days": 60, "same_season": False,
                                  "same_year": True, "tolerance_days": analysis.SEASON_TOLERANCE_DAYS}
    assert any("not seasonally comparable" in note and "2025-07-05" in note for note in a["uncertainty"])


def test_same_season_different_years_notes_weather(setup, monkeypatch, outputs):
    request = SearchRequest(place="Dakar", start_date=date(2025, 7, 1), end_date=date(2025, 7, 31), index="ndvi",
                            baseline_start_date=date(2024, 7, 1), baseline_end_date=date(2024, 7, 31))
    by_year = {2024: [s2("Y24", "2024-07-10")], 2025: [s2("Y25", "2025-07-15")]}
    setup(request, lambda start: by_year[start.year])
    monkeypatch.setattr(analysis, "load_bands", fake_pixels({"Y24": (3000, 2000, 4), "Y25": (3000, 2000, 4)}))
    a = graph.build_graph().invoke({"query": "q"})["analysis"]

    assert a["comparability"]["same_season"] and not a["comparability"]["same_year"]
    assert any("each year's weather" in note for note in a["uncertainty"])


def test_ndwi_answer_names_the_mcfeeters_variant(setup, monkeypatch, outputs):
    setup(july(index="ndwi"), [s2("W", "2025-07-15")])
    monkeypatch.setattr(analysis, "load_bands", fake_pixels({"W": (1500, 1, 6)}))

    def invoke(prompt):
        raise TimeoutError("deadline exceeded")

    monkeypatch.setattr(graph.get_llm(), "invoke", invoke)  # summary fails -> deterministic fallback answer
    result = graph.build_graph().invoke({"query": "q"})

    assert result["analysis"]["index_definition"]["name"] == "NDWI (McFeeters)"
    assert "NDWI (McFeeters) analysis, (B03 - B08) / (B03 + B08)" in result["answer"]  # fallback answer


def test_overpass_tiles_are_mosaicked_even_beyond_the_display_limit(setup, monkeypatch, outputs):
    # The area straddles two tiles of the same overpass; the cloudier one is outside the top `limit`=1.
    west = {**s2("WEST", "2025-07-15", cloud=1), "tile": "28PCC"}
    east = {**s2("EAST", "2025-07-15", cloud=3), "tile": "28PDC"}
    other_day = s2("OTHER", "2025-07-20", cloud=2)
    setup(july(index="ndvi", limit=1), [west, east, other_day])
    monkeypatch.setattr(analysis, "load_bands", fake_pixels({
        "WEST": (3000, 2000, 4, 1000, "west"), "EAST": (3000, 2000, 4, 1000, "east"), "OTHER": (3000, 2000, 9)}))
    result = graph.build_graph().invoke({"query": "q"})

    target = result["analysis"]["periods"]["target"]
    assert [i["id"] for i in result["items"]] == ["WEST"]          # what the user is shown
    assert [s["id"] for s in target["scenes"]] == ["WEST", "EAST"]  # what the analysis used
    assert target["stats"]["valid_fraction"] == 1.0 and target["stats"]["no_data_fraction"] == 0.0


def test_partial_tile_coverage_is_reported_as_no_data_not_cloud(setup, monkeypatch, outputs):
    setup(july(index="ndvi"), [s2("HALF", "2025-07-15")])
    monkeypatch.setattr(analysis, "load_bands", fake_pixels({"HALF": (3000, 2000, 4, 1000, "west")}))
    a = graph.build_graph().invoke({"query": "q"})["analysis"]

    assert a["periods"]["target"]["stats"]["no_data_fraction"] == pytest.approx(0.5, abs=0.01)
    assert any("50% not covered by that date's tiles" in note for note in a["uncertainty"])


def test_ndwi_change_reports_water_area_gained_and_lost(setup, monkeypatch, outputs):
    # The lake fills the west half in May and the whole box in October; the land does not change.
    request = SearchRequest(place="Lac de Guiers", start_date=date(2025, 10, 1), end_date=date(2025, 10, 31),
                            index="ndwi", baseline_start_date=date(2025, 5, 1), baseline_end_date=date(2025, 5, 31))
    by_month = {5: [s2("MAY_LAKE", "2025-05-26"), s2("MAY_LAND", "2025-05-26", cloud=6)],
                10: [s2("OCT", "2025-10-28")]}
    setup(request, lambda start: by_month[start.month])
    # Water: green 0.2, nir 0.05 -> NDWI +0.6. Land: green 0.05, nir 0.3 -> NDWI -0.71.
    monkeypatch.setattr(analysis, "load_bands", fake_pixels({
        "MAY_LAKE": (1500, 1, 6, 3000, "west"), "MAY_LAND": (4000, 1, 4, 1500, "east"), "OCT": (1500, 1, 6, 3000)}))
    result = graph.build_graph().invoke({"query": "q"})

    a = result["analysis"]
    may, october = a["periods"]["baseline"]["water"], a["periods"]["target"]["water"]
    assert may["water_share_of_observed"] == pytest.approx(0.5, abs=0.01) and october["water_share_of_observed"] == 1.0
    assert october["water_km2"] == pytest.approx(2 * may["water_km2"], rel=0.01)
    water = a["change"]["water"]
    assert water["lost_km2"] == 0 and water["gained_km2"] == pytest.approx(may["water_km2"], rel=0.01)
    assert water["net_km2"] == pytest.approx(water["gained_km2"], abs=0.01)
    assert any("bright or turbid water as cloud" in note for note in a["uncertainty"])
    prov = json.loads(Path(a["provenance"]).read_text(encoding="utf-8"))
    assert prov["method"]["water_threshold"] == "NDWI > 0.0"
    assert set(prov["periods"]["baseline"]["band_hrefs"]) == {"MAY_LAKE", "MAY_LAND"}


def test_water_totals_over_different_ground_are_flagged(setup, monkeypatch, outputs):
    # May sees only the west half (e.g. the rest masked as cloud), October sees everything.
    request = SearchRequest(place="Lac de Guiers", start_date=date(2025, 10, 1), end_date=date(2025, 10, 31),
                            index="ndwi", baseline_start_date=date(2025, 5, 1), baseline_end_date=date(2025, 5, 31))
    by_month = {5: [s2("MAY", "2025-05-26")], 10: [s2("OCT", "2025-10-28")]}
    setup(request, lambda start: by_month[start.month])
    monkeypatch.setattr(analysis, "load_bands", fake_pixels({
        "MAY": (1500, 1, 6, 3000, "west"), "OCT": (1500, 1, 6, 3000)}))
    a = graph.build_graph().invoke({"query": "q"})["analysis"]

    assert any("cover different ground" in note for note in a["uncertainty"])
    assert a["change"]["water"]["gained_km2"] == 0  # the unseen half is not counted as new water
