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
MAX_SCENES_TRIED = 3      # per period, clearest first
CHANGE_THRESHOLD = 0.1    # index change counted as a real increase/decrease
WELL_COVERED = 0.8        # below this share of clear pixels, flag the result as less reliable
CLOUD_BUFFER_M = 60       # SCL misses thin cloud edges: also mask this far around clouds/shadows

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

def grid_shape(bbox: list[float]) -> tuple[int, int]:
    """(height, width) of the analysis grid: square-ish pixels, at most GRID_MAX_PX, at least 10 m."""
    west, south, east, north = bbox
    width_m = (east - west) * 111_320 * math.cos(math.radians((south + north) / 2))
    height_m = (north - south) * 110_574
    pixels_long_side = min(GRID_MAX_PX, max(width_m, height_m) / 10)
    metres_per_px = max(width_m, height_m) / pixels_long_side
    return max(1, round(height_m / metres_per_px)), max(1, round(width_m / metres_per_px))


def grid_resolution_m(bbox: list[float], shape: tuple[int, int]) -> float:
    width_m = (bbox[2] - bbox[0]) * 111_320 * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
    return width_m / shape[1]


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


def _r(x) -> float:
    return round(float(x), 3)


# --- orchestration --------------------------------------------------------------

def pick_scene(items: list[dict], index: str, bbox: list[float], shape: tuple[int, int]):
    """Try the clearest scenes in turn; keep the first one that covers the area well enough.

    Catalog cloud cover is per tile, not per area: a 5% tile can still be cloudy over the city.
    Returns (item, index values, excluded fraction, scenes tried)."""
    a_name, b_name = INDEX_BANDS[index]
    buffer_px = math.ceil(CLOUD_BUFFER_M / grid_resolution_m(bbox, shape))
    tried, best = [], None
    for item in items[:MAX_SCENES_TRIED]:
        if not {a_name, b_name, "scl"} <= item.get("bands", {}).keys():
            continue
        dn = load_bands(item, (a_name, b_name, "scl"), bbox, shape)
        values = compute_index(index, dn, item["bands"], buffer_px)
        excluded = excluded_fraction(index, dn["scl"])
        share = valid_fraction(values, excluded)
        tried.append({"id": item["id"], "valid_fraction": _r(share)})
        if best is None or share > best[3]:
            best = (item, values, excluded, share)
        if share >= MIN_VALID:
            break
    if best is None:
        raise ValueError("no scene with the bands needed for the analysis")
    return best[0], best[1], best[2], tried


def run(state: dict, llm_model: str) -> dict:
    """Analyse the target period (and the baseline, if any); write analysis.png and provenance.json."""
    request, bbox, index = state["request"], state["bbox"], state["request"].index
    shape = grid_shape(bbox)
    out_dir = _output_dir(request.place or "bbox", index)

    target_item, target_values, target_excl, target_tried = pick_scene(state["items"], index, bbox, shape)
    periods = {"target": _period(request.start_date, request.end_date, target_item, target_values,
                                 target_excl, target_tried)}
    baseline_values, change = None, None
    if request.has_baseline and state.get("baseline_items"):
        base_item, baseline_values, base_excl, base_tried = pick_scene(state["baseline_items"], index, bbox, shape)
        periods["baseline"] = _period(request.baseline_start_date, request.baseline_end_date, base_item,
                                      baseline_values, base_excl, base_tried)
        change = change_stats(baseline_values, target_values, max(target_excl, base_excl))

    uncertainty = _uncertainty(request, periods, change)
    png = out_dir / "analysis.png"
    render_png(png, index, bbox, periods, target_values, baseline_values)
    result = {
        "index": index, "periods": periods, "change": change, "uncertainty": uncertainty,
        "png": str(png), "provenance": str(out_dir / "provenance.json"),
    }
    write_provenance(out_dir / "provenance.json", state, result, shape, llm_model)
    return result


def _period(start: date, end: date, item: dict, values: np.ndarray, excluded: float, tried: list[dict]) -> dict:
    return {
        "window": f"{start} / {end}",
        "scene": {k: item.get(k) for k in ("id", "datetime", "platform", "tile", "cloud_cover", "stac_url")},
        "scenes_tried": tried,
        "stats": index_stats(values, excluded),
    }


def _uncertainty(request, periods: dict, change: dict | None) -> list[str]:
    notes = []
    excluded = max(p["stats"]["excluded_fraction"] for p in periods.values())
    if excluded > 0:
        notes.append(f"about {excluded:.0%} of the area is water (sea, lakes) and is excluded from "
                     f"{request.index.upper()} statistics; percentages refer to the remaining land")
    for name, period in periods.items():
        share = period["stats"]["valid_fraction"]
        if share < WELL_COVERED:
            notes.append(f"{name}: only {share:.0%} of the area is clear (rest is cloud, shadow or outside the tile)")
        if len(period["scenes_tried"]) > 1:
            notes.append(f"{name}: the clearest scenes in the catalog were cloudy over the area; "
                         f"used {period['scene']['id']} after trying {len(period['scenes_tried'])}")
    if request.has_baseline and "baseline" not in periods:
        notes.append("no scene found for the baseline period: change could not be computed")
    if change is not None:
        if change["valid_fraction_both"] < WELL_COVERED:
            notes.append(f"change is computed on the {change['valid_fraction_both']:.0%} of the area clear on both dates")
        t, b = periods["target"]["scene"], periods["baseline"]["scene"]
        gap = abs((_day(t["datetime"]) - _day(b["datetime"])).days)
        notes.append(f"single-scene snapshots {gap} days apart: short-term effects (rain, irrigation, harvest, "
                     f"tide) can dominate the difference")
        if t["platform"] != b["platform"]:
            notes.append(f"different satellites ({b['platform']} vs {t['platform']}); "
                         f"L2A processing makes them comparable but not identical")
    else:
        notes.append("single-scene snapshot: values describe one acquisition date, not the whole period")
    return notes


def _day(iso: str | None) -> date:
    return datetime.fromisoformat(iso).date() if iso else date.min


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

    name = index.upper()
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
    return (period["scene"]["datetime"] or "")[:10]


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
            "formula": f"({a_name} - {b_name}) / ({a_name} + {b_name})",
            "reflectance": "DN * scale + offset, from the STAC raster:bands extension",
            "cloud_mask": f"Sentinel-2 SCL, clear classes kept: {list(SCL_CLEAR)}; "
                          f"{CLOUD_BUFFER_M} m buffer around classes {list(SCL_CLOUDY)}",
            "excluded_classes": list(INDEX_EXCLUDED.get(result["index"], ())),
            "grid": {"crs": "EPSG:4326", "height": shape[0], "width": shape[1],
                     "resampling": "average (reflectance), nearest (SCL)"},
            "scene_selection": f"clearest first; first with >= {MIN_VALID:.0%} clear pixels, "
                               f"at most {MAX_SCENES_TRIED} tried per period",
            "change_threshold": CHANGE_THRESHOLD,
        },
        "periods": {
            name: {**period, "band_hrefs": _band_hrefs(state, name, period["scene"]["id"], (a_name, b_name, "scl"))}
            for name, period in result["periods"].items()
        },
        "change": result["change"],
        "uncertainty": result["uncertainty"],
        "outputs": {"png": Path(result["png"]).name},
        "software": _versions(),
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _band_hrefs(state: dict, period: str, item_id: str, names: tuple[str, ...]) -> dict:
    items = state["items"] if period == "target" else state.get("baseline_items", [])
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
