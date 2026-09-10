"""Trajectory correlation shared between biomet_app.py (Advanced search's
automatic window search, search_best_matches()) and
precompute_station_reach.py (Station reach's own offline window search).
Moved here rather than duplicated so the two search a station's window
candidates with the exact same trend filter -- CLAUDE.md, "Trend is a
filter, not a weight": both are the same underlying question (find the
best-aligned station window for a fixed comparison window), and a
diverging copy in each place would eventually disagree on which
candidates get excluded, the same failure mode CLAUDE.md flags for
`coverage()`'s band and the multiselect session_state keys.

No Streamlit import here and nothing else heavy, the same reasoning
region_windows.py already documents for itself: precompute_station_reach.py
runs standalone from the command line and must not have to import
biomet_app.py (a Streamlit script that renders its whole UI on import).
"""
import numpy as np


def trajectory_correlation(region_values, station_values, usable,
                           station_mean, station_std):
    """Pearson correlation of the region's and the station's mean
    seasonal trajectory across the window, position by position, rather
    than by calendar pentad, since the two windows generally sit in
    different parts of the year. CLAUDE.md, "Trend is a filter, not a
    weight": same mean with opposite seasonal trend is the case a
    level-only score gets wrong, so a negative correlation here excludes
    the candidate from the search entirely (see the caller), regardless
    of how good its sigma dissimilarity is.

    Plain ndarrays, not pandas -- the two callers select `usable`
    (sigma_dissimilarity()'s usable list for this candidate: variables
    the station has nonzero interannual SD on) and z-score themselves
    before calling, so this function stays one shared implementation of
    the actual maths for both callers' own shapes, not a formula
    reimplemented differently in each:

    - biomet_app.py's search_best_matches(): one region trajectory
      against one station trajectory (single candidate, single window
      search). `region_values`/`station_values` shape (n_positions,
      n_usable); returns a scalar.
    - precompute_station_reach.py's own per-cell argmin: MANY region
      (cell) trajectories against the SAME station trajectory (one
      candidate, every grid cell at once) -- `region_values` shape
      (n_cells, n_positions, n_usable), `station_values` still
      (n_positions, n_usable); returns an (n_cells,) array. Computing
      one candidate's correlation for 2863 cells this way, instead of
      2863 separate scalar calls, is what keeps the precompute's own
      runtime in the same couple-of-minutes ballpark it already
      documents.

    `station_mean`/`station_std` are the station's own interannual
    mean/SD, restricted to `usable`, shape (n_usable,) -- both sides are
    z-scored against the STATION's own scale, the same one
    sigma_dissimilarity() puts every variable on, so variables with very
    different native units (mm vs degC vs a day count) contribute
    comparably to the single composite trajectory each side is averaged
    down to, rather than one dominating by magnitude alone; this
    composite trajectory is a design choice for combining variables, not
    a quantity either cited paper defines.

    Returns nan (never excluded -- a correlation against nothing to
    compare is not evidence of a mismatch) if there is no usable
    variable, fewer than 2 window positions, or either side's composite
    trajectory is flat. In the batch shape, that nan applies per cell,
    not to the whole batch."""
    batch = region_values.ndim == 3
    n_positions = region_values.shape[-2]
    if not usable or n_positions < 2:
        return np.full(region_values.shape[0], np.nan) if batch else np.nan

    reg_z = (region_values - station_mean) / station_std
    sta_z = (station_values - station_mean) / station_std
    reg_traj = reg_z.mean(axis=-1)   # (n_positions,) or (n_cells, n_positions)
    sta_traj = sta_z.mean(axis=-1)   # (n_positions,)

    if not batch:
        if np.isnan(reg_traj).any() or np.isnan(sta_traj).any():
            return np.nan
        if np.std(reg_traj) < 1e-12 or np.std(sta_traj) < 1e-12:
            return np.nan
        return float(np.corrcoef(reg_traj, sta_traj)[0, 1])

    if np.isnan(sta_traj).any() or np.std(sta_traj) < 1e-12:
        return np.full(region_values.shape[0], np.nan)
    reg_c = reg_traj - reg_traj.mean(axis=1, keepdims=True)
    sta_c = sta_traj - sta_traj.mean()
    reg_std = reg_traj.std(axis=1)
    sta_std = sta_traj.std()
    cov = (reg_c * sta_c).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / (reg_std * sta_std)
    flat_or_nan = (reg_std < 1e-12) | np.isnan(reg_traj).any(axis=1)
    return np.where(flat_or_nan, np.nan, corr)
