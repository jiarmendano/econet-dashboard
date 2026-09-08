"""Offline precompute for Region Matching's "station reach" tab (Task N).

Builds station_reach.parquet: for every (grid cell, station, region
window), the COMPOSITE sigma dissimilarity of that single cell's own
climate against the station -- "how far does this station's climate
reach", the inverse of Advanced search's own question (which station
matches this region). Composite only: single-variable views are a
later, separate addition in native units, not sigma (see the report
this script's own output feeds into).

Run from the repo root: `python precompute_station_reach.py`. Reads
station_pentad.parquet and cell_pentad/*.parquet; writes
station_reach.parquet next to them. Takes a couple of minutes; nothing
in biomet_app.py imports or runs this -- it is a standalone batch step,
the same role r-codes/*.R scripts play for the data one level up.

Method: the exact algorithm documented on sigma_dissimilarity() in
biomet_app.py (Mahony et al. 2017; Fitzpatrick & Dunn 2019) --
per-variable z-score against the station's own interannual SD, PCA on
the station's SD-scaled year-to-year anomalies kept to
PCA_VARIANCE_THRESHOLD = 95% cumulative variance, Mahalanobis distance
in that reduced space, converted to sigma via the chi distribution
(df = retained components) and norm.isf, chi.sf/norm.isf rather than
1-cdf/ppf to avoid float64 saturation near tail probability 0.
Reimplemented standalone here rather than imported: biomet_app.py is a
Streamlit script that renders its whole UI on import (st.set_page_config,
sidebar, data loads all run at module scope), so it cannot safely be
imported as a library from a batch job. Kept in sync by hand -- the
same relationship the R pipeline's own formula ports already have to
this file's documentation in CLAUDE.md, not a new kind of risk.

The one-cell difference from sigma_dissimilarity()'s own state-average
case: there the "region" side is itself a distribution (one value per
reference year) and sigma_dissimilarity() pools it down to a single
mean before ever touching it (region_mean = mean(region_py[v])) --
so a single grid cell's own pooled window mean, one number per
variable, plugs into exactly the same formula unchanged. There is no
extra pooling step to skip; sigma_dissimilarity() already only ever
wanted a mean.

Per CLAUDE.md's window-search precedent (search_best_matches()), the
station side is not fixed to one window: for each region window, the
station side is searched over the SAME-LENGTH candidate windows on the
quarter-start grid (region_windows.CANDIDATES_BY_LENGTH) and the best
(lowest) sigma is kept -- computed independently per CELL, since which
candidate window fits best can genuinely differ across the country for
the same station. The winning candidate is stored as `station_window`
(as a display-ready label, e.g. "Apr-Jun", not the internal key) even
though the tab itself never displays it: a later per-variable view
needs the six departures computed from the one alignment that produced
this sigma, and re-deriving that argmin from scratch would mean
rerunning this entire precompute a second time.

`sigma_same_window` is the sigma for the candidate whose window equals
the CELL's own region window (region_key) -- already one row of the
same sigma matrix the argmin is taken over, so a store, not a second
computation. Coincides with `sigma` exactly for the Annual window,
where "Annual" is its own and only candidate. Never lower than `sigma`
elsewhere, since `sigma` is a min over the same candidates.
"""
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import chi, norm

import region_windows as rw

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(REPO, "station_reach.parquet")

# The six default comparison variables (CLAUDE.md, "Comparison variables
# (default set)") -- the only quantity this file computes is their
# composite; single-variable views are a later addition, in native
# units, not sigma (see this script's own report).
VARS = ["T2M", "DTR", "interdiurnal_T2M", "T2MDEW", "PRECTOTCORR", "THI_ge_79"]

PCA_VARIANCE_THRESHOLD = 0.95   # matches biomet_app.py's own constant exactly
SD_EPS = 1e-9                   # matches sigma_dissimilarity()'s own usable-variable floor


def composite_sigma_matrix(station_mean, station_std, usable, kept_vecs, kept_vals, df,
                           cell_means):
    """sigma_dissimilarity()'s formula, vectorised over every cell at
    once for one (station, candidate window): departure -> PCA
    projection -> Mahalanobis distance -> chi.sf/norm.isf. Returns one
    sigma per cell (n_cells,), in the same row order as cell_means."""
    dep = (cell_means[usable].to_numpy() - station_mean[usable].to_numpy()) / station_std[usable].to_numpy()
    scores = dep @ kept_vecs                      # (n_cells, n_components)
    mahal2 = np.sum(scores ** 2 / kept_vals, axis=1)
    distance = np.sqrt(np.clip(mahal2, 0.0, None))
    tail_p = chi.sf(distance, df)
    return norm.isf(tail_p / 2)


def station_window_stats(station_pentad, station, pentads):
    """One (station, candidate window)'s mean/SD/PCA -- everything
    sigma_dissimilarity() needs from the station side alone, independent
    of any region or cell. Mirrors sigma_dissimilarity() exactly: usable
    variables are those with interannual SD > SD_EPS over the reference
    period; PCA keeps the leading components explaining
    PCA_VARIANCE_THRESHOLD of the station's own SD-scaled year-to-year
    variance."""
    sub = station_pentad[(station_pentad["station"] == station)
                         & (station_pentad["pentad"].isin(pentads))]
    annual = sub.groupby("year")[VARS].mean()

    mean = annual.mean()
    std = annual.std(ddof=1)
    usable = [v for v in VARS if std.get(v, 0.0) > SD_EPS]

    if not usable:
        return None

    z_year = (annual[usable] - mean[usable]) / std[usable]
    cov = z_year.cov().to_numpy()   # pandas .cov(): ddof=1, matches sigma_dissimilarity()

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    cumvar = np.cumsum(eigvals) / eigvals.sum()
    n_components = int(np.searchsorted(cumvar, PCA_VARIANCE_THRESHOLD, side="left")) + 1
    n_components = min(n_components, len(eigvals))

    return dict(mean=mean, std=std, usable=usable,
               kept_vecs=eigvecs[:, :n_components], kept_vals=eigvals[:n_components],
               df=n_components)


def main():
    t_total0 = time.perf_counter()

    print(f"windows: {len(rw.WINDOWS)} (expect 13)")
    for key, lk, p_start in rw.WINDOWS:
        print(f"  {key:10s} length_key={lk:5s} n_pentads={len(rw.WINDOW_PENTADS[key])}")

    t0 = time.perf_counter()
    station_pentad = pd.read_parquet(
        os.path.join(REPO, "station_pentad.parquet"),
        columns=["station", "pentad", "year"] + VARS)
    stations = sorted(station_pentad["station"].unique())
    print(f"\nstations: {stations}")
    print(f"station_pentad: {len(station_pentad)} rows, {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    cell_files = sorted(glob.glob(os.path.join(REPO, "cell_pentad", "cell_pentad_*.parquet")))
    frames = [pd.read_parquet(f, columns=["lon", "lat", "pentad", "year"] + VARS)
             for f in cell_files]
    cell_pentad = pd.concat(frames, ignore_index=True)
    del frames
    print(f"cell_pentad: {len(cell_pentad)} rows from {len(cell_files)} files, "
         f"{time.perf_counter()-t0:.2f}s")

    cells = cell_pentad[["lon", "lat"]].drop_duplicates().reset_index(drop=True)
    n_cells = len(cells)
    print(f"distinct cells: {n_cells}")

    # Mean of per-(cell, pentad) means, restricted to a window's pentad
    # list, equals the pooled mean over every (pentad, year) row in that
    # window -- valid because every pentad carries the exact same set of
    # reference years (cell_pentad has no gaps: CLAUDE.md, "There is no
    # imputation for the grid... POWER has no gaps"), so the groups being
    # averaged are balanced. Collapsing years first, once, cuts the
    # per-window groupby from 7.3M rows down to 2863 x 73 = 209k, run 13
    # times instead of the full frame 13 times.
    t0 = time.perf_counter()
    cell_pentad_mean = cell_pentad.groupby(["lon", "lat", "pentad"])[VARS].mean().reset_index()
    del cell_pentad
    print(f"per-cell-pentad means: {len(cell_pentad_mean)} rows, {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    cell_window_mean = {}
    for key in rw.WINDOW_KEYS:
        sub = cell_pentad_mean[cell_pentad_mean["pentad"].isin(rw.WINDOW_PENTADS[key])]
        g = sub.groupby(["lon", "lat"])[VARS].mean()
        g = g.reindex(pd.MultiIndex.from_frame(cells[["lon", "lat"]]))
        cell_window_mean[key] = g
    print(f"cell-side window means for {len(rw.WINDOW_KEYS)} windows: "
         f"{time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    station_stats = {}
    for station in stations:
        for key in rw.WINDOW_KEYS:
            station_stats[(station, key)] = station_window_stats(
                station_pentad, station, rw.WINDOW_PENTADS[key])
    print(f"station-side stats for {len(stations)} stations x {len(rw.WINDOW_KEYS)} "
         f"windows: {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    lon_arr = cells["lon"].to_numpy()
    lat_arr = cells["lat"].to_numpy()

    (out_lon, out_lat, out_station, out_window, out_station_window,
    out_sigma, out_sigma_same_window) = ([], [], [], [], [], [], [])

    for region_key, lk, _ in rw.WINDOWS:
        candidates = rw.CANDIDATES_BY_LENGTH[lk]
        same_idx = candidates.index(region_key)   # region_key is always its own length's own candidate
        cell_means = cell_window_mean[region_key]

        for station in stations:
            # (n_candidates, n_cells) sigma matrix, then argmin per cell --
            # the winning candidate can differ cell to cell, so this can't
            # be reduced with a running np.minimum alone if the winning
            # window identity is also needed (it is: `station_window`).
            sigmas = np.full((len(candidates), n_cells), np.inf)
            for ci, cand in enumerate(candidates):
                st = station_stats[(station, cand)]
                if st is None:
                    continue
                sigmas[ci] = composite_sigma_matrix(
                    st["mean"], st["std"], st["usable"],
                    st["kept_vecs"], st["kept_vals"], st["df"], cell_means)

            best_idx = np.argmin(sigmas, axis=0)
            best_sigma = sigmas[best_idx, np.arange(n_cells)]
            # Same candidate the region window itself sits at -- already
            # one of the rows in `sigmas`, so this is a store, not a new
            # computation. For Annual there is only one candidate
            # ("Annual" itself), so same_idx == best_idx always and the
            # two sigma columns coincide by construction.
            same_window_sigma = sigmas[same_idx]

            out_lon.append(lon_arr)
            out_lat.append(lat_arr)
            out_station.append(np.full(n_cells, station, dtype=object))
            out_window.append(np.full(n_cells, region_key, dtype=object))
            out_station_window.append(
                np.array([rw.DISPLAY_LABEL[candidates[i]] for i in best_idx], dtype=object))
            out_sigma.append(best_sigma)
            out_sigma_same_window.append(same_window_sigma)

    out = pd.DataFrame({
        "lon": np.concatenate(out_lon),
        "lat": np.concatenate(out_lat),
        "station": np.concatenate(out_station),
        "window": np.concatenate(out_window),
        "station_window": np.concatenate(out_station_window),
        "sigma": np.concatenate(out_sigma),
        "sigma_same_window": np.concatenate(out_sigma_same_window),
    })
    print(f"final table assembled: {len(out)} rows, {time.perf_counter()-t0:.2f}s")

    n_viol = int((out["sigma_same_window"] < out["sigma"] - 1e-6).sum())
    print(f"sigma_same_window < sigma (stored best) violations: {n_viol} (expect 0)")

    for c in ["station", "window", "station_window"]:
        out[c] = out[c].astype("category")
    out["lon"] = out["lon"].astype("float32")
    out["lat"] = out["lat"].astype("float32")
    out["sigma"] = out["sigma"].astype("float32")
    out["sigma_same_window"] = out["sigma_same_window"].astype("float32")

    t0 = time.perf_counter()
    out.to_parquet(OUT_PATH, index=False)
    print(f"written in {time.perf_counter()-t0:.2f}s")

    expected = n_cells * len(stations) * len(rw.WINDOWS)
    print(f"\nrow count: {len(out)}  (expected {n_cells}*{len(stations)}*{len(rw.WINDOWS)} = {expected})")
    print(f"file size: {os.path.getsize(OUT_PATH)} bytes = {os.path.getsize(OUT_PATH)/1024**2:.2f} MB")
    finite = np.isfinite(out["sigma"])
    print(f"sigma stats: min={out['sigma'][finite].min():.3f} max={out['sigma'][finite].max():.3f} "
         f"median={out['sigma'][finite].median():.3f} n_inf_or_nan={(~finite).sum()}")
    print(f"\nTOTAL WALL TIME: {time.perf_counter()-t_total0:.1f}s")


if __name__ == "__main__":
    main()
