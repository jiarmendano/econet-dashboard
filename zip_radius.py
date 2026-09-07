"""Point-level (ZIP + radius) cell resolution.

Standalone helper module -- NOT wired into biomet_app.py yet. See
CLAUDE.md, "Point-level selection: data prepared, app side not built
yet." Only reads conus_grid.parquet, zip_to_cell.parquet and
cell_pentad/*.parquet; writes nothing and has no Streamlit dependency.

The correction this module exists for: a naive "read the ZIP's own
state, load that one cell_pentad file, then filter by radius" flow
silently clips the selection to whichever side of a state line the ZIP
happens to be matched to. Rhode Island is the extreme case -- its own
file has 2 cells, so a real 100 km radius from Providence has to reach
into Massachusetts and Connecticut, and a flow that only ever opens
cell_pentad_Rhode_Island.parquet cannot produce that regardless of the
radius typed in.

The fix is ordering: resolve the radius against the grid geographically
FIRST (conus_grid.parquet, one row per cell, already small enough to
hold in memory), THEN load only the cell_pentad state files that
selection actually touches, THEN filter each loaded file down to
exactly the selected cells.

zip_to_cell.parquet carries two different state fields; this module
uses each for its own job, not one field doing both:
  - admin_state: the ZCTA's real administrative state (from the 2020
    Census ZCTA-to-county relationship file, area-weighted majority --
    see Build_zip_to_cell.R). Used here for DISPLAY, i.e. home_state in
    select_cells_for_zip_radius()'s report.
  - cell_state: the state of the ZCTA's single NEAREST grid cell. Not
    used by this module's radius flow at all (cells_within_radius()
    gets each selected cell's state straight from conus_grid.parquet),
    kept in zip_to_cell.parquet only for a plain nearest-cell lookup
    that isn't a radius search.
Conflating them was the original bug: Providence, RI (ZIP 02903) is
close enough to the Massachusetts border that its nearest single grid
cell is a Massachusetts cell, which is a defensible answer for "which
file" and a wrong, trust-costing one for "what state is this" on any
screen a person reads.

When this becomes a UI control, the radius input/slider should not
default below ~50 km: grid cells are ~55-60 km apart, so a 25 km
radius from a real, ordinary ZIP (verified on Waco, TX, far from any
border) already returns zero cells -- not a bug, just a radius smaller
than the grid can resolve, but confusing as a default that silently
finds nothing.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
GRID_PATH = REPO_ROOT / "conus_grid.parquet"
ZIP_PATH = REPO_ROOT / "zip_to_cell.parquet"
CELL_PENTAD_DIR = REPO_ROOT / "cell_pentad"

EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points -- a real
    geodesic distance, not the cos(lat)-scaled planar approximation
    Aggregate_station_pentad.R and Build_zip_to_cell.R use for
    NEAREST-cell matching. That approximation is fine there because
    only relative ordering matters (which cell is closest); here the
    radius is a number the user typed in kilometres, so it has to mean
    that literally. Spherical haversine is accurate to within ~0.5% of
    the WGS84 ellipsoidal distance at CONUS latitudes and the tens-of-km
    distances this module deals with -- far smaller than the ~55 km
    cell spacing the radius is being compared against, and not worth a
    new dependency (pyproj) over.
    """
    lat1r, lon1r, lat2r, lon2r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def haversine_km_vec(lat1, lon1, lat2_arr, lon2_arr):
    """haversine_km, vectorised: one point against an array of points."""
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r = np.radians(lat2_arr)
    lon2r = np.radians(lon2_arr)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def load_grid():
    return pd.read_parquet(GRID_PATH)


def load_zip_lookup():
    return pd.read_parquet(ZIP_PATH)


def zip_centroid(zip_code, zip_df=None):
    """The ZIP's own centroid (lat, lon), from the Census ZCTA gazetteer
    via zip_to_cell.parquet -- NOT the matched nearest cell's centre.
    A radius search has to start from where the ZIP actually is, not
    from whatever grid cell it happened to snap to.
    """
    if zip_df is None:
        zip_df = load_zip_lookup()
    zip_code = str(zip_code).zfill(5)
    row = zip_df.loc[zip_df["zip"] == zip_code]
    if row.empty:
        raise ValueError(f"ZIP {zip_code} not found in zip_to_cell.parquet "
                        "(not on the CONUS grid, or not a valid ZCTA)")
    r = row.iloc[0]
    return float(r["lat"]), float(r["lon"])


def cells_within_radius(lat, lon, radius_km, grid_df=None):
    """Step 1: every conus_grid.parquet cell whose CENTRE is within
    radius_km of (lat, lon), by real geodesic distance, with no
    reference to any state at all yet.
    """
    if grid_df is None:
        grid_df = load_grid()
    d = haversine_km_vec(lat, lon, grid_df["lat"].to_numpy(), grid_df["lon"].to_numpy())
    out = grid_df.assign(distance_km=d)
    return out.loc[out["distance_km"] <= radius_km].sort_values("distance_km").reset_index(drop=True)


def load_cell_pentad_for_states(states):
    """Step 2: load only the cell_pentad_<STATE>.parquet files the
    selected cells actually touch -- never the ZIP's own single "home"
    state file alone, which is exactly the bug this module exists to
    avoid.
    """
    frames = []
    for state in states:
        path = CELL_PENTAD_DIR / f"cell_pentad_{state.replace(' ', '_')}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No cell_pentad file for state {state!r}: {path}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def select_cells_for_zip_radius(zip_code, radius_km, grid_df=None, zip_df=None):
    """The full corrected flow, in order:
      1. cells_within_radius() -- resolve the cell set geographically,
         against the whole grid, before any state is chosen.
      2. load_cell_pentad_for_states() -- load only the states that
         selection spans.
      3. an inner merge back onto the selected (lon, lat) keys -- each
         loaded state file carries every cell in that state, not only
         the ones inside the radius.

    Returns (matched_cells, cell_pentad_df, report):
      matched_cells -- one row per selected grid cell (lon, lat, state,
        region, area_weight, n_cells_state, distance_km), nearest first
      cell_pentad_df -- the selected cells' full pentad x year rows
      report -- dict with n_cells, states, home_state (the ZIP's REAL
        administrative state, from zip_to_cell.parquet's admin_state --
        not cell_state, which is a single nearest-cell match and can
        name the wrong state for a ZIP right at a border, e.g.
        Providence, RI), cross_state (bool), nearest_km, farthest_km
        and a human-readable message
    """
    if zip_df is None:
        zip_df = load_zip_lookup()
    if grid_df is None:
        grid_df = load_grid()

    zip_code = str(zip_code).zfill(5)
    lat, lon = zip_centroid(zip_code, zip_df)
    home_state = zip_df.loc[zip_df["zip"] == zip_code, "admin_state"].iloc[0]

    matched_cells = cells_within_radius(lat, lon, radius_km, grid_df)
    states = sorted(matched_cells["state"].unique().tolist())

    if states:
        cell_pentad_df = load_cell_pentad_for_states(states)
        selected_keys = matched_cells[["lon", "lat"]].drop_duplicates()
        cell_pentad_df = cell_pentad_df.merge(selected_keys, on=["lon", "lat"], how="inner")
    else:
        cell_pentad_df = pd.DataFrame()

    other_states = [s for s in states if s != home_state]
    cross_state = len(other_states) > 0
    if len(matched_cells) == 0:
        message = (f"No grid cell centre falls within {radius_km:.0f} km of ZIP "
                  f"{zip_code} -- the radius is smaller than half the ~55 km "
                  f"cell spacing and missed every cell centre.")
    elif cross_state:
        message = (f"This {radius_km:.0f} km selection around ZIP {zip_code} "
                  f"({home_state}) spans {len(states)} states: also includes "
                  f"cells in {', '.join(other_states)}.")
    else:
        message = (f"This {radius_km:.0f} km selection around ZIP {zip_code} "
                  f"stays within {home_state}.")

    report = {
        "zip": zip_code,
        "radius_km": radius_km,
        "lat": lat,
        "lon": lon,
        "n_cells": len(matched_cells),
        "states": states,
        "home_state": home_state,
        "cross_state": cross_state,
        "message": message,
        "nearest_km": float(matched_cells["distance_km"].min()) if len(matched_cells) else None,
        "farthest_km": float(matched_cells["distance_km"].max()) if len(matched_cells) else None,
    }
    return matched_cells, cell_pentad_df, report
