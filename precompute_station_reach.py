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
where "Annual" is its own and only candidate and the trend filter does
not run at all (see below). Elsewhere, `sigma_same_window` was never
lower than `sigma` before the trend filter (since `sigma` was a plain
min over the same candidates); now it CAN be, exactly when the
same-window candidate itself is the one the filter excludes -- see the
"sigma_same_window < sigma" report line in main() for the current
count, re-measured after the filter's own rewrite below.

Task E adds six columns, dep_<VAR> for each of VARS: the native-unit
departure (cell mean minus station mean, signed) that the composite's
own z-score is built from before it is divided by the station's SD and
passed through PCA -- so this is a decomposition of the SAME comparison
`sigma` already reports, not a second, independently-optimised one.
Both sides are read off exactly what composite_sigma_matrix() already
uses for the WINNING candidate (best_idx): the cell's mean at the
region's own fixed window (cell_means, unchanged across candidates)
minus the station's mean at that winning candidate window
(station_stats[(station, candidate)]["mean"], which covers all of VARS
-- unlike "usable", which composite_sigma_matrix() applies only for the
PCA step, a variable with ~0 interannual SD still has a perfectly good
mean and so still gets a departure here). No variable is allowed to
pick its own best candidate window independently -- all six read off
the one candidate index the composite already minimised on.

The argmin over candidates also applies the same trend filter Advanced
search's own automatic search uses (CLAUDE.md, "Trend is a filter, not
a weight"; trajectory.py's trajectory_correlation(), shared rather than
a second copy that could drift): a candidate whose pentad-trajectory
correlation with the CELL is negative is excluded before the argmin,
same mean opposite seasonal trend being the case a level-only sigma
comparison cannot see for itself. NaN correlation (no usable variable,
fewer than 2 window positions, or a flat trajectory on either side) is
kept, not excluded, matching search_best_matches()'s own rule exactly.

The filter does not run at all for the Annual window: with only one
candidate ("Annual" itself), there is no alternative alignment to
prefer over an excluded one, so filtering would only ever have one
outcome to offer -- keep the one candidate, or discard it with nothing
to replace it. Annual's `sigma` is always the plain, unfiltered value.

For every other window length, a cell where EVERY same-length candidate
is excluded has no survivor to fall back to -- there is no rejected
candidate to fall back on, per CLAUDE.md's own filter rule, so that
(cell, station, region window) row is dropped from the output entirely
rather than filled with a value the filter just ruled out. Counts are
reported by main() at the end of the run, broken down by window
length."""
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import chi, norm

import region_windows as rw
from trajectory import trajectory_correlation

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
    out_dep = {v: [] for v in VARS}

    n_dropped = 0
    n_dropped_by_length = {}

    for region_key, lk, _ in rw.WINDOWS:
        candidates = rw.CANDIDATES_BY_LENGTH[lk]
        same_idx = candidates.index(region_key)   # region_key is always its own length's own candidate
        cell_means = cell_window_mean[region_key]
        cell_means_np = cell_means[VARS].to_numpy()   # (n_cells, 6), fixed across candidates
        has_choice = len(candidates) > 1   # False only for Annual

        # Per-cell, per-position (not pooled) trajectory, one array per
        # variable, (n_cells, n_positions), position-ordered by this
        # region window's own WINDOW_PENTADS (wrap-aware) -- the region
        # side trajectory_correlation() needs; cell_means above (the
        # pooled mean sigma is built from) can't stand in for it, a
        # single number per variable has no shape to correlate. Skipped
        # entirely for Annual: with one candidate there is no rotation
        # to prefer one alignment over another, so the trend filter has
        # nothing to filter -- computing it would cost real time (a
        # pivot over all 73 pentads x 2863 cells) for a result that can
        # only ever keep its one candidate anyway.
        if has_choice:
            region_pentads = rw.WINDOW_PENTADS[region_key]
            cell_pos_sub = cell_pentad_mean[cell_pentad_mean["pentad"].isin(region_pentads)]
            cell_pos_wide = cell_pos_sub.pivot(index=["lon", "lat"], columns="pentad", values=VARS)
            cell_pos_wide = cell_pos_wide.reindex(
                columns=pd.MultiIndex.from_product([VARS, region_pentads]))
            cell_pos_wide = cell_pos_wide.reindex(pd.MultiIndex.from_frame(cells[["lon", "lat"]]))
            cell_pos = {v: cell_pos_wide[v].to_numpy() for v in VARS}   # each (n_cells, n_positions)

        for station in stations:
            # (n_candidates, n_cells) sigma matrix, then argmin per cell --
            # the winning candidate can differ cell to cell, so this can't
            # be reduced with a running np.minimum alone if the winning
            # window identity is also needed (it is: `station_window`).
            sigmas = np.full((len(candidates), n_cells), np.inf)
            # (n_candidates, 6) station means, one row per candidate window --
            # the OTHER thing composite_sigma_matrix() reads per candidate,
            # kept here too so the six native departures can be read off the
            # exact winning candidate afterwards instead of recomputed.
            means = np.full((len(candidates), len(VARS)), np.nan)
            if has_choice:
                # (n_candidates, n_cells) trajectory correlation, one row
                # per candidate -- NaN (kept, per the filter's own rule)
                # wherever composite_sigma_matrix() itself has nothing to
                # compare (st is None) or trajectory_correlation() finds
                # a flat trajectory on either side.
                corrs = np.full((len(candidates), n_cells), np.nan)
            for ci, cand in enumerate(candidates):
                st = station_stats[(station, cand)]
                if st is None:
                    continue
                sigmas[ci] = composite_sigma_matrix(
                    st["mean"], st["std"], st["usable"],
                    st["kept_vecs"], st["kept_vals"], st["df"], cell_means)
                means[ci] = st["mean"][VARS].to_numpy()

                if not has_choice:
                    continue
                usable = st["usable"]
                cand_pentads = rw.WINDOW_PENTADS[cand]
                sta_sub = station_pentad[(station_pentad["station"] == station)
                                         & (station_pentad["pentad"].isin(cand_pentads))]
                sta_by_pentad = sta_sub.groupby("pentad")[VARS].mean().reindex(cand_pentads)
                region_batch = np.stack([cell_pos[v] for v in usable], axis=-1)
                corrs[ci] = trajectory_correlation(
                    region_batch, sta_by_pentad[usable].to_numpy(), usable,
                    st["mean"][usable].to_numpy(), st["std"][usable].to_numpy())

            if has_choice:
                # Trend filter (CLAUDE.md, "Trend is a filter, not a
                # weight"): a candidate with a real negative correlation
                # is excluded from the argmin entirely, same rule
                # search_best_matches() applies -- NaN correlation is
                # kept (not evidence of a mismatch), only corr < 0
                # excludes. No fallback to a rejected candidate: a cell
                # where every candidate is excluded has no survivor and
                # is dropped from the output entirely (keep_mask below),
                # not filled in with a candidate the filter just ruled
                # out.
                valid = (corrs >= 0) | np.isnan(corrs)
                sigmas = np.where(valid, sigmas, np.inf)
                # Dropped means "the filter excluded every candidate",
                # not "the surviving candidate's own sigma happens to be
                # non-finite" -- those are different failures.
                # composite_sigma_matrix() can legitimately saturate to
                # inf on an extreme mismatch (chi.sf underflowing to
                # exactly 0.0, then norm.isf(0.0) = inf) with nothing to
                # do with the trend filter at all; keep_mask used to be
                # np.isfinite(best_sigma), which conflated the two and
                # silently dropped a genuinely off-scale row right along
                # with a genuinely filter-excluded one. station_reach_map()
                # already renders non-finite sigma as "extremely novel
                # (off scale)", so that row has somewhere correct to go
                # -- it just needs to survive to the parquet.
                keep_mask = valid.any(axis=0)
            else:
                # Annual: no filter runs at all (see above), so nothing
                # is ever dropped here regardless of how the one
                # candidate's own sigma comes out.
                keep_mask = np.ones(n_cells, dtype=bool)

            best_idx = np.argmin(sigmas, axis=0)
            best_sigma = sigmas[best_idx, np.arange(n_cells)]
            n_drop = int((~keep_mask).sum())
            n_dropped += n_drop
            n_dropped_by_length[lk] = n_dropped_by_length.get(lk, 0) + n_drop

            # Same candidate the region window itself sits at -- already
            # one of the rows in `sigmas`, so this is a store, not a new
            # computation. For Annual there is only one candidate
            # ("Annual" itself), so same_idx == best_idx always and the
            # two sigma columns coincide by construction. Unaffected by
            # the trend filter above: this column answers "how novel is
            # the cell at the region's OWN window", not "what's the
            # best-aligned station window" -- but sigmas itself has now
            # had excluded candidates set to inf in-place, so a cell
            # whose SAME-WINDOW candidate was the one excluded reads inf
            # here too, same as it would for any other excluded
            # candidate -- not filtered a second, different way.
            same_window_sigma = sigmas[same_idx]

            # Native departure at the WINNING candidate only (Task E):
            # cell mean (region's own fixed window) minus station mean
            # (best_idx's candidate window), same two quantities
            # composite_sigma_matrix() already differenced before scaling
            # by SD and projecting through PCA -- no separate per-variable
            # argmin.
            dep = cell_means_np - means[best_idx]   # (n_cells, 6)

            out_lon.append(lon_arr[keep_mask])
            out_lat.append(lat_arr[keep_mask])
            out_station.append(np.full(keep_mask.sum(), station, dtype=object))
            out_window.append(np.full(keep_mask.sum(), region_key, dtype=object))
            out_station_window.append(
                np.array([rw.DISPLAY_LABEL[candidates[i]] for i in best_idx[keep_mask]],
                        dtype=object))
            out_sigma.append(best_sigma[keep_mask])
            out_sigma_same_window.append(same_window_sigma[keep_mask])
            for vi, v in enumerate(VARS):
                out_dep[v].append(dep[keep_mask, vi])

    print(f"\nDropped (every candidate excluded by the trend filter, no survivor -- "
         f"Annual never contributes here, the filter doesn't run on it): "
         f"{n_dropped} of {n_cells * len(stations) * len(rw.WINDOWS)}")
    for lk in ("3mo", "6mo", "9mo"):
        n_region_windows_here = sum(1 for _, l, _ in rw.WINDOWS if l == lk)
        n_combos = n_cells * len(stations) * n_region_windows_here
        print(f"  {lk:5s}: {n_dropped_by_length.get(lk, 0):5d} / {n_combos}")

    out = pd.DataFrame({
        "lon": np.concatenate(out_lon),
        "lat": np.concatenate(out_lat),
        "station": np.concatenate(out_station),
        "window": np.concatenate(out_window),
        "station_window": np.concatenate(out_station_window),
        "sigma": np.concatenate(out_sigma),
        "sigma_same_window": np.concatenate(out_sigma_same_window),
        **{f"dep_{v}": np.concatenate(out_dep[v]) for v in VARS},
    })
    print(f"final table assembled: {len(out)} rows, {time.perf_counter()-t0:.2f}s")

    # No longer "expect 0": that held only while `sigma` was an
    # unfiltered min over the exact same candidates sigma_same_window
    # is one of. Now that the trend filter can exclude the same-window
    # candidate itself, `sigma` (the filtered argmin) can legitimately
    # land on a WORSE candidate than the excluded same-window one --
    # sigma_same_window stays unfiltered (see its own comment above),
    # by design, so it isn't one of the trend filter's own inputs.
    n_gt = int((out["sigma_same_window"] < out["sigma"] - 1e-6).sum())
    print(f"sigma_same_window < sigma (same-window candidate itself excluded "
         f"by the trend filter): {n_gt}")

    for c in ["station", "window", "station_window"]:
        out[c] = out[c].astype("category")
    out["lon"] = out["lon"].astype("float32")
    out["lat"] = out["lat"].astype("float32")
    out["sigma"] = out["sigma"].astype("float32")
    out["sigma_same_window"] = out["sigma_same_window"].astype("float32")
    for v in VARS:
        out[f"dep_{v}"] = out[f"dep_{v}"].astype("float32")

    t0 = time.perf_counter()
    out.to_parquet(OUT_PATH, index=False)
    print(f"written in {time.perf_counter()-t0:.2f}s")

    max_rows = n_cells * len(stations) * len(rw.WINDOWS)
    print(f"\nrow count: {len(out)}  (max possible {n_cells}*{len(stations)}*{len(rw.WINDOWS)} = "
         f"{max_rows}, {max_rows - len(out)} dropped for having no surviving candidate)")
    print(f"file size: {os.path.getsize(OUT_PATH)} bytes = {os.path.getsize(OUT_PATH)/1024**2:.2f} MB")
    finite = np.isfinite(out["sigma"])
    print(f"sigma stats: min={out['sigma'][finite].min():.3f} max={out['sigma'][finite].max():.3f} "
         f"median={out['sigma'][finite].median():.3f} n_inf_or_nan={(~finite).sum()}")
    for v in VARS:
        col = out[f"dep_{v}"]
        n_nan = int(col.isna().sum())
        print(f"dep_{v}: min={col.min():.3f} max={col.max():.3f} n_nan={n_nan}")
    print(f"\nTOTAL WALL TIME: {time.perf_counter()-t_total0:.1f}s")


if __name__ == "__main__":
    main()
