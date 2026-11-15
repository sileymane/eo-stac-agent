"""Pixel analysis: a spectral index over the area of interest, its change between two periods,
a PNG map and a provenance record that lets anyone audit or reproduce the result.

Only the area's window is read from each Cloud-Optimized GeoTIFF, warped onto one shared
lon/lat grid so scenes from different tiles or UTM zones can be compared pixel by pixel.
"""
import json
import math
import os
import platform
import re
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from . import tools

GRID_MAX_PX = 1024        # longest side of the analysis grid; never finer than the 10 m native pixels
MIN_VALID = 0.5           # a scene needs this share of clear pixels over the area to be used
MAX_SCENES_TRIED = 3      # acquisition dates tried per period, clearest first
MAX_TILES = 4             # tiles of one overpass mosaicked together (an area can straddle tile edges)
CHANGE_THRESHOLD = 0.1    # index change counted as a real increase/decrease
WELL_COVERED = 0.8        # below this share of clear pixels, flag the result as less reliable
CLOUD_BUFFER_M = 60       # SCL misses thin cloud edges: also mask this far around clouds/shadows
WATER_THRESHOLD = 0.0     # McFeeters NDWI above this counts as open water (McFeeters 1996)

# Sentinel-2 Scene Classification (SCL) classes kept as clear observations:
# 4 vegetation, 5 bare soil, 6 water, 7 unclassified, 11 snow/ice.
# Dropped: 0 no data, 1 saturated, 2 dark/shadow, 3 cloud shadow, 8-9 cloud, 10 cirrus.
SCL_CLEAR = (4, 5, 6, 7, 11)
SCL_CLOUDY = (3, 8, 9, 10)
# Clear but irrelevant to the question: water (sea, lakes) would swamp vegetation statistics.
# Reported as a separate share instead of counting as "stable" NDVI ~ 0.
INDEX_EXCLUDED = {"ndvi": (6,)}

# index -> (a, b) for the normalized difference (a - b) / (a + b)
INDEX_BANDS = {"ndvi": ("nir", "red"), "ndwi": ("green", "nir")}
INDEX_CMAP = {"ndvi": "RdYlGn", "ndwi": "RdYlBu"}

# What each label means. "NDWI" names two different indices in the literature, so the label alone is
# ambiguous: this agent computes McFeeters' green-NIR open-water NDWI, never Gao's NIR-SWIR one.
# Sentinel-2 band numbers match the Earth Search asset keys in INDEX_BANDS (green B03, red B04, nir B08).
INDEX_DEFINITIONS = {
    "ndvi": {"name": "NDVI", "reference": "Rouse et al. (1974)", "measures": "vegetation greenness",
             "sentinel2": "(B08 - B04) / (B08 + B04)"},
    "ndwi": {"name": "NDWI (McFeeters)", "reference": "McFeeters (1996)", "measures": "open surface water",
             "sentinel2": "(B03 - B08) / (B03 + B08)",
             "not": "Gao (1996) NDWI, (NIR - SWIR) / (NIR + SWIR), vegetation water content"},
}

# Dates further apart than this in the annual cycle (ignoring the year) are not seasonally comparable:
# their difference is likely to include normal phenology (rainy season, crop growth, senescence).
SEASON_TOLERANCE_DAYS = 30


# --- I/O (the only network access; tests replace load_bands) ------------------

def load_bands(item: dict, names: tuple[str, ...], bbox: list[float], shape: tuple[int, int]) -> dict:
    """Read raw DN values of the given bands over bbox, resampled onto the shared EPSG:4326 grid."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT

    height, width = shape
    out = {}
    for name in names:
        # SCL is categorical: never average class codes.
        resampling = Resampling.nearest if name == "scl" else Resampling.average
        with rasterio.open(item["bands"][name]["href"]) as src, WarpedVRT(
            src, crs="EPSG:4326", transform=from_bounds(*bbox, width, height),
            width=width, height=height, resampling=resampling, nodata=0,
        ) as vrt:
            out[name] = vrt.read(1)
    return out


# --- pure computation ---------------------------------------------------------

def _extent_m(bbox: list[float]) -> tuple[float, float]:
    """(width, height) of the bbox in metres, at its mid latitude."""
    west, south, east, north = bbox
    return (east - west) * 111_320 * math.cos(math.radians((south + north) / 2)), (north - south) * 110_574


def grid_shape(bbox: list[float]) -> tuple[int, int]:
    """(height, width) of the analysis grid: square-ish pixels, at most GRID_MAX_PX, at least 10 m."""
    width_m, height_m = _extent_m(bbox)
    pixels_long_side = min(GRID_MAX_PX, max(width_m, height_m) / 10)
    metres_per_px = max(width_m, height_m) / pixels_long_side
    return max(1, round(height_m / metres_per_px)), max(1, round(width_m / metres_per_px))


def grid_resolution_m(bbox: list[float], shape: tuple[int, int]) -> float:
    return _extent_m(bbox)[0] / shape[1]


def pixel_area_km2(bbox: list[float], shape: tuple[int, int]) -> float:
    width_m, height_m = _extent_m(bbox)
    return width_m / shape[1] * height_m / shape[0] / 1e6


def compute_index(index: str, dn: dict, bands: dict, buffer_px: int = 0) -> np.ndarray:
    """Normalized-difference index; NaN where the pixel is cloudy (plus buffer_px around clouds),
    shadowed, has no data, or is excluded for this index (water for NDVI)."""
    a_name, b_name = INDEX_BANDS[index]
    a = dn[a_name] * bands[a_name]["scale"] + bands[a_name]["offset"]
    b = dn[b_name] * bands[b_name]["scale"] + bands[b_name]["offset"]
    clear = (np.isin(dn["scl"], SCL_CLEAR) & ~np.isin(dn["scl"], INDEX_EXCLUDED.get(index, ()))
             & ~_dilate(np.isin(dn["scl"], SCL_CLOUDY), buffer_px)
             & (dn[a_name] > 0) & (dn[b_name] > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = (a - b) / (a + b)
    return np.where(clear & np.isfinite(values), np.clip(values, -1, 1), np.nan).astype(np.float32)


def excluded_fraction(index: str, scl: np.ndarray) -> float:
    """Share of the area deliberately left out for this index (e.g. water for NDVI)."""
    return float(np.isin(scl, INDEX_EXCLUDED.get(index, ())).mean())


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(px):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def valid_fraction(values: np.ndarray, excluded: float = 0.0) -> float:
    """Share of the area that could be observed, ignoring deliberately excluded pixels."""
    relevant = 1.0 - excluded
    return float(np.isfinite(values).mean() / relevant) if relevant > 0 else 0.0


def index_stats(values: np.ndarray, excluded: float = 0.0) -> dict:
    clear = values[np.isfinite(values)]
    base = {"valid_fraction": _r(valid_fraction(values, excluded)), "excluded_fraction": _r(excluded)}
    if clear.size == 0:
        return base
    p10, median, p90 = np.percentile(clear, [10, 50, 90])
    return {"mean": _r(clear.mean()), "median": _r(median), "p10": _r(p10), "p90": _r(p90), **base}


def change_stats(before: np.ndarray, after: np.ndarray, excluded: float = 0.0) -> dict:
    """Per-pixel change, computed only where both dates are clear."""
    both = np.isfinite(before) & np.isfinite(after)
    relevant = 1.0 - excluded
    base = {"valid_fraction_both": _r(both.mean() / relevant) if relevant > 0 else 0.0}
    if not both.any():
        return base
    diff = (after - before)[both]
    return {
        "mean_change": _r(diff.mean()),
        "share_increase": _r((diff > CHANGE_THRESHOLD).mean()),
        "share_decrease": _r((diff < -CHANGE_THRESHOLD).mean()),
        "share_stable": _r((np.abs(diff) <= CHANGE_THRESHOLD).mean()),
        **base,
    }


def mosaic(layers: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Combine the tiles of one overpass, given as (index values, SCL) on the shared grid.

    Each pixel comes from the first tile that has data there (SCL != 0), values and SCL together,
    so a pixel that tile flags as cloud stays masked rather than being filled from a neighbour."""
    values, scl = layers[0][0].copy(), layers[0][1].copy()
    for tile_values, tile_scl in layers[1:]:
        fill = (scl == 0) & (tile_scl != 0)
        values[fill], scl[fill] = tile_values[fill], tile_scl[fill]
    return values, scl


def water_extent(values: np.ndarray, px_km2: float) -> dict:
    """Open-water area on one date, from NDWI (McFeeters); only clear pixels can be counted."""
    observed = np.isfinite(values)
    water = observed & (values > WATER_THRESHOLD)
    return {"water_km2": _km2(water, px_km2), "observed_km2": _km2(observed, px_km2),
            "water_share_of_observed": _r(water.sum() / observed.sum()) if observed.any() else None}


def water_change(before: np.ndarray, after: np.ndarray, px_km2: float) -> dict:
    """Water gained and lost, on pixels clear on both dates only (so both totals cover the same ground)."""
    both = np.isfinite(before) & np.isfinite(after)
    was, now = both & (before > WATER_THRESHOLD), both & (after > WATER_THRESHOLD)
    return {"compared_km2": _km2(both, px_km2), "water_before_km2": _km2(was, px_km2),
            "water_after_km2": _km2(now, px_km2), "gained_km2": _km2(now & ~was, px_km2),
            "lost_km2": _km2(was & ~now, px_km2), "net_km2": round(_km2(now, px_km2) - _km2(was, px_km2), 2)}


def _km2(mask: np.ndarray, px_km2: float) -> float:
    return round(float(mask.sum() * px_km2), 2)


def _r(x) -> float:
    return round(float(x), 3)


# --- orchestration --------------------------------------------------------------

def acquisitions(items: list[dict]) -> list[list[dict]]:
    """Group scenes into overpasses (same day, same satellite): the tiles that together cover an area
    straddling tile edges. Clearest overpass first, then newest; clearest tiles first within each."""
    def cloud(item):
        return item["cloud_cover"] if item["cloud_cover"] is not None else 101

    groups: dict[tuple, list[dict]] = {}
    for item in items:
        groups.setdefault(((item["datetime"] or "")[:10], item["platform"]), []).append(item)
    newest_first = sorted(groups.values(), key=lambda g: g[0]["datetime"] or "", reverse=True)
    clearest_first = sorted(newest_first, key=lambda g: sum(map(cloud, g)) / len(g))
    return [sorted(g, key=cloud)[:MAX_TILES] for g in clearest_first]


def pick_acquisition(items: list[dict], index: str, bbox: list[float], shape: tuple[int, int]) -> dict:
    """Try the clearest overpasses in turn, each mosaicked from its tiles; keep the first that covers
    the area well enough.

    Catalog cloud cover is per tile, not per area: a 5% tile can still be cloudy over the city, and
    one tile may cover only part of it. Returns the chosen scenes, index values, SCL mosaic,
    excluded fraction and the overpasses tried."""
    a_name, b_name = INDEX_BANDS[index]
    names = (a_name, b_name, "scl")
    buffer_px = math.ceil(CLOUD_BUFFER_M / grid_resolution_m(bbox, shape))
    tried, best = [], None
    for overpass in acquisitions(items)[:MAX_SCENES_TRIED]:
        tiles = [i for i in overpass if set(names) <= i.get("bands", {}).keys()]
        if not tiles:
            continue
        layers = []
        for item in tiles:
            dn = load_bands(item, names, bbox, shape)
            layers.append((compute_index(index, dn, item["bands"], buffer_px), dn["scl"]))
        values, scl = mosaic(layers)
        excluded = excluded_fraction(index, scl)
        share = valid_fraction(values, excluded)
        tried.append({"date": (tiles[0]["datetime"] or "")[:10], "tiles": [i["id"] for i in tiles],
                      "valid_fraction": _r(share), "no_data_fraction": _r((scl == 0).mean())})
        if best is None or share > best["share"]:
            best = {"scenes": tiles, "values": values, "scl": scl, "excluded": excluded, "share": share}
        if share >= MIN_VALID:
            break
    if best is None:
        raise ValueError("no scene with the bands needed for the analysis")
    return {**best, "tried": tried}


def run(state: dict, llm_model: str) -> dict:
    """Analyse the target period (and the baseline, if any); write analysis.png and provenance.json."""
    request, bbox, index = state["request"], state["bbox"], state["request"].index
    shape = grid_shape(bbox)
    out_dir = _output_dir(request.place or "bbox", index)

    px_km2 = pixel_area_km2(bbox, shape)
    # All search results, not only the top `limit` shown to the user: an overpass's other tiles are needed.
    candidates = state.get("candidates") or {}

    target = pick_acquisition(candidates.get("target") or state["items"], index, bbox, shape)
    periods = {"target": _period(request.start_date, request.end_date, target, index, px_km2)}
    target_values, baseline_values, change = target["values"], None, None
    if request.has_baseline and state.get("baseline_items"):
        base = pick_acquisition(candidates.get("baseline") or state["baseline_items"], index, bbox, shape)
        periods["baseline"] = _period(request.baseline_start_date, request.baseline_end_date, base, index, px_km2)
        baseline_values = base["values"]
        change = change_stats(baseline_values, target_values, max(target["excluded"], base["excluded"]))
        if index == "ndwi":
            change["water"] = water_change(baseline_values, target_values, px_km2)
    comparability = date_comparability(periods)

    uncertainty = _uncertainty(request, periods, change, comparability)
    png = out_dir / "analysis.png"
    render_png(png, index, bbox, periods, target_values, baseline_values)
    result = {
        "index": index, "index_definition": INDEX_DEFINITIONS[index], "periods": periods, "change": change,
        "comparability": comparability, "uncertainty": uncertainty,
        "png": str(png), "provenance": str(out_dir / "provenance.json"),
    }
    write_provenance(out_dir / "provenance.json", state, result, shape, llm_model)
    return result


def _period(start: date, end: date, picked: dict, index: str, px_km2: float) -> dict:
    first = picked["scenes"][0]
    period = {
        "window": f"{start} / {end}",
        "date": (first["datetime"] or "")[:10],
        "platform": first["platform"],
        "scenes": [{k: item.get(k) for k in ("id", "datetime", "tile", "cloud_cover", "stac_url")}
                   for item in picked["scenes"]],
        "overpasses_tried": picked["tried"],
        "stats": {**index_stats(picked["values"], picked["excluded"]),
                  "no_data_fraction": _r((picked["scl"] == 0).mean())},
    }
    if index == "ndwi":
        period["water"] = water_extent(picked["values"], px_km2)
    return period


def seasonal_offset_days(a: date, b: date) -> int:
    """Distance between two dates' positions in the annual cycle, ignoring the year (0..182 days)."""
    # Day-of-year shifts by one after February in leap years, so compare in one non-leap year.
    a, b = (date(2001, d.month, min(d.day, 28) if d.month == 2 else d.day) for d in (a, b))
    gap = abs((a - b).days)
    return min(gap, 365 - gap)


def date_comparability(periods: dict) -> dict | None:
    """How comparable the baseline and target acquisition dates are; None without a dated baseline."""
    if "baseline" not in periods:
        return None
    t, b = (_day(periods[p]["date"]) for p in ("target", "baseline"))
    if date.min in (t, b):
        return None
    offset = seasonal_offset_days(t, b)
    return {"days_apart": abs((t - b).days), "seasonal_offset_days": offset,
            "same_season": offset <= SEASON_TOLERANCE_DAYS, "same_year": t.year == b.year,
            "tolerance_days": SEASON_TOLERANCE_DAYS}


def _uncertainty(request, periods: dict, change: dict | None, comparability: dict | None) -> list[str]:
    notes = []
    label = INDEX_DEFINITIONS[request.index]["name"]
    excluded = max(p["stats"]["excluded_fraction"] for p in periods.values())
    if excluded > 0:
        notes.append(f"about {excluded:.0%} of the area is water (sea, lakes) and is excluded from "
                     f"{label} statistics; percentages refer to the remaining land")
    for name, period in periods.items():
        share, no_data = period["stats"]["valid_fraction"], period["stats"]["no_data_fraction"]
        if share < WELL_COVERED:
            outside = f"{no_data:.0%} not covered by that date's tiles, " if no_data > 0.01 else ""
            notes.append(f"{name}: only {share:.0%} of the area is clear ({outside}the rest cloud or shadow)")
        if len(period["overpasses_tried"]) > 1:
            notes.append(f"{name}: the clearest overpasses in the catalog were cloudy over the area or only "
                         f"partly covered it; used {period['date']} after trying {len(period['overpasses_tried'])}")
    if request.has_baseline and "baseline" not in periods:
        notes.append("no scene found for the baseline period: change could not be computed")
    if change is not None:
        if change["valid_fraction_both"] < WELL_COVERED:
            notes.append(f"change is computed on the {change['valid_fraction_both']:.0%} of the area clear on both dates")
        t, b = periods["target"], periods["baseline"]
        if change.get("water"):
            observed = sorted(periods[p]["water"]["observed_km2"] for p in ("baseline", "target"))
            if observed[0] < 0.9 * observed[1]:
                notes.append(f"the per-date water totals cover different ground ({observed[0]} vs {observed[1]} km2 "
                             f"observed), so they must not be compared with each other; only the gained/lost "
                             f"figures, on ground clear on both dates, measure change")
            notes.append(f"water area is compared on the {change['water']['compared_km2']} km2 clear on both dates; "
                         f"per-date totals count only clear pixels, and Sentinel-2's scene classification "
                         f"sometimes labels bright or turbid water as cloud, which removes it from that date")
        if comparability:
            notes.append(f"single-scene snapshots {comparability['days_apart']} days apart: short-term effects "
                         f"(rain, irrigation, harvest, tide) can dominate the difference")
            if not comparability["same_season"]:
                notes.append(f"not seasonally comparable: the dates ({_date(periods['baseline'])} and "
                             f"{_date(periods['target'])}) are {comparability['seasonal_offset_days']} days apart "
                             f"in the annual cycle, so normal seasonal change (rainy season, crop growth, "
                             f"senescence) is probably part of the difference; compare the same months of "
                             f"different years to separate it")
            elif not comparability["same_year"]:
                notes.append("same time of year in different years: the seasonal cycle is roughly controlled for, "
                             "but each year's weather (e.g. a wetter or drier season) is not")
        if t["platform"] != b["platform"]:
            notes.append(f"different satellites ({b['platform']} vs {t['platform']}); "
                         f"L2A processing makes them comparable but not identical")
    else:
        notes.append("single-scene snapshot: values describe one acquisition date, not the whole period")
    return notes


def _day(iso: str | None) -> date:
    return date.fromisoformat(iso[:10]) if iso else date.min


def _output_dir(place: str, index: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", place.lower()).strip("-")[:40]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(os.getenv("OUTPUT_DIR", "outputs")) / f"{stamp}_{slug}_{index}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- outputs --------------------------------------------------------------------

def render_png(path: Path, index: str, bbox: list[float], periods: dict,
               target: np.ndarray, baseline: np.ndarray | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = INDEX_DEFINITIONS[index]["name"]
    panels = []
    if baseline is not None:
        panels.append((f"{name} {_date(periods['baseline'])}", baseline, INDEX_CMAP[index], -1, 1))
    panels.append((f"{name} {_date(periods['target'])}", target, INDEX_CMAP[index], -1, 1))
    if baseline is not None:
        panels.append((f"{name} change", target - baseline, "RdBu", -0.5, 0.5))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5), squeeze=False)
    extent = (bbox[0], bbox[2], bbox[1], bbox[3])
    for ax, (title, values, cmap, vmin, vmax) in zip(axes[0], panels):
        cmap = plt.get_cmap(cmap).with_extremes(bad="lightgrey")  # masked: cloud, shadow, no data
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, shrink=0.8)
    masked = ["cloud", "shadow"] + (["water"] if INDEX_EXCLUDED.get(index) else []) + ["no data"]
    fig.suptitle(f"grey = masked ({', '.join(masked)})", fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _date(period: dict) -> str:
    return period["date"]


def write_provenance(path: Path, state: dict, result: dict, shape: tuple[int, int], llm_model: str) -> None:
    a_name, b_name = INDEX_BANDS[result["index"]]
    record = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": state["query"],
        "parsed_request": state["parsed_request"].model_dump(mode="json"),
        "final_request": state["request"].model_dump(mode="json"),
        "relaxations": state.get("relaxations", []),
        "bbox_wgs84": state["bbox"],
        "stac_api": tools.STAC_URL,
        "llm_model": llm_model,
        "method": {
            "index": result["index"],
            "definition": result["index_definition"],
            "formula": f"({a_name} - {b_name}) / ({a_name} + {b_name})",
            "reflectance": "DN * scale + offset, from the STAC raster:bands extension",
            "cloud_mask": f"Sentinel-2 SCL, clear classes kept: {list(SCL_CLEAR)}; "
                          f"{CLOUD_BUFFER_M} m buffer around classes {list(SCL_CLOUDY)}",
            "excluded_classes": list(INDEX_EXCLUDED.get(result["index"], ())),
            "grid": {"crs": "EPSG:4326", "height": shape[0], "width": shape[1],
                     "resampling": "average (reflectance), nearest (SCL)"},
            "scene_selection": f"overpasses (same day and satellite) clearest first, each mosaicked from up "
                               f"to {MAX_TILES} tiles (first tile with data wins per pixel); first with >= "
                               f"{MIN_VALID:.0%} clear pixels, at most {MAX_SCENES_TRIED} tried per period",
            "pixel_area_km2": pixel_area_km2(state["bbox"], shape),
            **({"water_threshold": f"NDWI > {WATER_THRESHOLD}"} if result["index"] == "ndwi" else {}),
            "change_threshold": CHANGE_THRESHOLD,
        },
        "periods": {
            name: {**period, "band_hrefs": {scene["id"]: _band_hrefs(state, name, scene["id"], (a_name, b_name, "scl"))
                                            for scene in period["scenes"]}}
            for name, period in result["periods"].items()
        },
        "change": result["change"],
        "comparability": result["comparability"],
        "uncertainty": result["uncertainty"],
        "outputs": {"png": Path(result["png"]).name},
        "software": _versions(),
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _band_hrefs(state: dict, period: str, item_id: str, names: tuple[str, ...]) -> dict:
    fallback = state["items"] if period == "target" else state.get("baseline_items", [])
    items = (state.get("candidates") or {}).get(period) or fallback
    item = next(i for i in items if i["id"] == item_id)
    return {n: item["bands"][n]["href"] for n in names}


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "rasterio", "pystac-client", "langgraph", "langchain"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    try:
        import rasterio
        out["gdal"] = rasterio.__gdal_version__
    except ImportError:
        pass
    return out
