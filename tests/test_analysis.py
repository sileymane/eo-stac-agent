"""Pure pixel computations: no network, no LLM."""
from datetime import date

import numpy as np
import pytest

from eo_stac_agent import analysis

BANDS = {n: {"scale": 0.0001, "offset": -0.1} for n in ("red", "nir", "green")}


def dn(nir, red, scl):
    return {"nir": np.array(nir, dtype=np.uint16), "red": np.array(red, dtype=np.uint16),
            "scl": np.array(scl, dtype=np.uint8)}


def test_ndvi_applies_offset_and_masks_clouds_shadows_nodata():
    values = analysis.compute_index("ndvi", dn(
        nir=[3000, 3000, 3000, 0],
        red=[2000, 2000, 2000, 2000],
        scl=[4, 9, 3, 4],  # clear, cloud, cloud shadow, clear-but-no-data
    ), BANDS)
    # With the -0.1 offset: nir 0.2, red 0.1 -> 0.333 (ignoring the offset would give 0.2).
    assert values[0] == pytest.approx(1 / 3, abs=1e-6)
    assert np.isnan(values[1:]).all()


def test_zero_reflectance_sum_is_masked_not_infinite():
    # DN 1000 -> reflectance 0.0 for both bands: 0/0 must become NaN, not crash or inf.
    values = analysis.compute_index("ndvi", dn(nir=[1000], red=[1000], scl=[4]), BANDS)
    assert np.isnan(values).all()


def test_ndwi_uses_green_and_nir():
    bands = dn(nir=[1500], red=[1], scl=[6])
    bands["green"] = np.array([3000], dtype=np.uint16)
    values = analysis.compute_index("ndwi", bands, BANDS)
    assert values[0] == pytest.approx((0.2 - 0.05) / (0.2 + 0.05), abs=1e-6)  # water is positive


def test_change_stats_only_counts_pixels_clear_on_both_dates():
    before = np.array([0.2, 0.2, 0.5, np.nan], dtype=np.float32)
    after = np.array([0.5, 0.25, 0.2, 0.9], dtype=np.float32)
    stats = analysis.change_stats(before, after)
    assert stats["valid_fraction_both"] == 0.75
    assert stats["share_increase"] == pytest.approx(1 / 3, abs=1e-3)
    assert stats["share_decrease"] == pytest.approx(1 / 3, abs=1e-3)
    assert stats["share_stable"] == pytest.approx(1 / 3, abs=1e-3)


def test_fully_cloudy_scene_gives_empty_stats():
    assert analysis.index_stats(np.full((2, 2), np.nan)) == {"valid_fraction": 0.0, "excluded_fraction": 0.0}


def test_water_is_excluded_from_ndvi_but_kept_for_ndwi():
    scene = dn(nir=[3000, 3000], red=[2000, 2000], scl=[4, 6])  # land, water
    scene["green"] = np.array([2000, 2000], dtype=np.uint16)
    ndvi = analysis.compute_index("ndvi", scene, BANDS)
    assert np.isfinite(ndvi[0]) and np.isnan(ndvi[1])
    assert np.isfinite(analysis.compute_index("ndwi", scene, BANDS)).all()
    # Coverage is measured on land only: half the area is sea, all the land is clear -> 100%.
    excluded = analysis.excluded_fraction("ndvi", scene["scl"])
    assert excluded == 0.5
    assert analysis.valid_fraction(ndvi, excluded) == 1.0


def test_cloud_buffer_masks_pixels_next_to_clouds():
    scl = np.full((1, 7), 4, dtype=np.uint8)
    scl[0, 3] = 9  # one cloudy pixel in the middle
    scene = {"nir": np.full((1, 7), 3000, np.uint16), "red": np.full((1, 7), 2000, np.uint16), "scl": scl}
    values = analysis.compute_index("ndvi", scene, BANDS, buffer_px=2)
    assert np.isnan(values[0]).tolist() == [False, True, True, True, True, True, False]


def test_grid_is_capped_and_never_finer_than_10m():
    big = analysis.grid_shape([1.0, 43.0, 2.0, 44.0])           # ~80 x 110 km
    assert max(big) == analysis.GRID_MAX_PX
    small = analysis.grid_shape([1.0, 43.0, 1.01, 43.01])       # ~0.8 x 1.1 km
    assert small == pytest.approx((111, 81), abs=1)             # ~10 m pixels


@pytest.mark.parametrize("a, b, expected", [
    ("2025-07-05", "2025-09-03", 60),    # same year, two months apart in the cycle
    ("2024-07-10", "2025-07-15", 5),     # one year apart, same season
    ("2024-12-20", "2025-01-10", 21),    # wraps around the new year
    ("2025-01-01", "2025-07-02", 182),   # opposite seasons
])
def test_seasonal_offset_ignores_the_year(a, b, expected):
    assert analysis.seasonal_offset_days(date.fromisoformat(a), date.fromisoformat(b)) == expected


def test_ndwi_is_always_named_mcfeeters_not_gao():
    ndwi = analysis.INDEX_DEFINITIONS["ndwi"]
    assert "McFeeters" in ndwi["name"] and "Gao" in ndwi["not"]
    assert analysis.INDEX_BANDS["ndwi"] == ("green", "nir")  # the definition and the computation agree


def test_mosaic_fills_only_where_the_first_tile_has_no_data():
    nan = np.nan
    first = (np.array([0.5, nan, nan], dtype=np.float32), np.array([4, 9, 0], dtype=np.uint8))  # clear, cloud, no data
    second = (np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([4, 4, 4], dtype=np.uint8))
    values, scl = analysis.mosaic([first, second])
    # Pixel 0 keeps the first tile, pixel 1 stays cloud-masked (same overpass, same cloud), pixel 2 is filled.
    assert values[0] == pytest.approx(0.5) and np.isnan(values[1]) and values[2] == pytest.approx(0.3)
    assert scl.tolist() == [4, 9, 4]


def test_acquisitions_group_tiles_of_one_overpass_clearest_first():
    def item(i, day, platform, cloud):
        return {"id": i, "datetime": f"{day}T11:00:00", "platform": platform, "cloud_cover": cloud}
    groups = analysis.acquisitions([
        item("A1", "2025-07-05", "sentinel-2a", 20), item("B1", "2025-07-10", "sentinel-2b", 2),
        item("A2", "2025-07-05", "sentinel-2a", 10), item("B2", "2025-07-10", "sentinel-2b", 4)])
    assert [[i["id"] for i in g] for g in groups] == [["B1", "B2"], ["A2", "A1"]]


def test_water_change_counts_only_pixels_clear_on_both_dates():
    before = np.array([0.5, -0.5, 0.5, np.nan], dtype=np.float32)  # water, land, water, cloud
    after = np.array([0.5, 0.4, -0.3, 0.6], dtype=np.float32)      # water, water, land, water
    w = analysis.water_change(before, after, px_km2=0.01)
    assert w == {"compared_km2": 0.03, "water_before_km2": 0.02, "water_after_km2": 0.02,
                 "gained_km2": 0.01, "lost_km2": 0.01, "net_km2": 0.0}


def test_pixel_area_matches_the_bbox_area():
    bbox = [-16.0, 15.8, -15.7, 16.4]
    shape = analysis.grid_shape(bbox)
    width_m, height_m = analysis._extent_m(bbox)
    assert analysis.pixel_area_km2(bbox, shape) * shape[0] * shape[1] == pytest.approx(width_m * height_m / 1e6)
