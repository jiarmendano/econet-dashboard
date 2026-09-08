"""Regression guard for the station reach raster's ring-winding fix.

Plain assertions, run directly (`python test_cell_rectangles_winding.py`)
-- same convention as test_zip_radius.py, this repo's other standalone
regression script (see CLAUDE.md: no pytest, no runner, no test suite
in the usual sense).

Background, so this is never rediagnosed from scratch: a single-cell
render test found Plotly's geo choropleth filling the whole map with a
tiny hole where the cell should be. RFC7946=YES on st_write() had no
effect -- sf/GDAL already emit RFC 7946-correct (counter-clockwise)
exterior rings, so there was nothing for GDAL to normalise. Reversing
the ring to CLOCKWISE fixed it. Plotly's geo traces render on a bundled
d3-geo that wants the opposite of RFC 7946's own winding rule for a
plain small polygon here -- verified empirically against this
project's pinned Plotly, not a spec violation in the data. See
CLAUDE.md, "Plotly geo winding (settled)", and
Build_cell_rectangles_geojson.R's winding-reversal step, for the full
diagnosis.

This guard exists because the failure mode is silent: a future Plotly
release that aligns with RFC 7946 -- or a rebuild of
cell_rectangles.geojson that drops the reversal step -- would flip
every cell's fill to its complement with no error, only a map that
looks wrong. Catch it here instead.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
GEOJSON_PATH = REPO_ROOT / "cell_rectangles.geojson"


def signed_area(ring):
    """Shoelace signed area of a closed [lon, lat] ring. Negative means
    clockwise (map with north up, east right) -- what this project's
    pinned Plotly needs here. Positive (counter-clockwise) is RFC 7946's
    own rule, and is exactly the winding that fills the COMPLEMENT
    instead, per the diagnosis above."""
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        a += x1 * y2 - x2 * y1
    return a / 2


def exterior_rings(geometry):
    """The exterior ring of a Polygon, or of each part of a
    MultiPolygon -- only the rings this fix's winding governs (these
    cells have no holes, so there are no interior rings to check)."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [poly[0] for poly in geometry["coordinates"]]
    raise ValueError(f"unexpected geometry type {geometry['type']!r}")


with open(GEOJSON_PATH, encoding="utf-8") as f:
    geojson = json.load(f)

bad = []
for feature in geojson["features"]:
    cell_id = feature["properties"]["cell_id"]
    for ring in exterior_rings(feature["geometry"]):
        area = signed_area(ring)
        if area >= 0:
            bad.append((cell_id, area))

if bad:
    raise AssertionError(
        f"{len(bad)} of {len(geojson['features'])} exterior ring(s) in "
        "cell_rectangles.geojson wind counter-clockwise (RFC 7946's own "
        "rule) instead of clockwise. This project's pinned Plotly fills "
        "the COMPLEMENT of a counter-clockwise ring here -- the opposite "
        "of the RFC 7946 spec, verified empirically, not a bug in this "
        "test. Do NOT re-diagnose this from scratch: see "
        "Build_cell_rectangles_geojson.R's winding-reversal step and "
        "CLAUDE.md, 'Plotly geo winding (settled)'. Likely cause: the "
        "reversal step was skipped or removed on a rebuild, or the "
        "pinned Plotly version changed (see CLAUDE.md for which version "
        "this guard was verified against) and Plotly's own convention "
        "flipped to match RFC 7946. First few offending cells (cell_id, "
        f"signed_area): {bad[:5]}"
    )

print(f"OK: all {len(geojson['features'])} features' exterior rings wind "
     "clockwise (this project's Plotly convention, opposite RFC 7946).")
