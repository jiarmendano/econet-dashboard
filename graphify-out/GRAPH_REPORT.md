# Graph Report - econet-dashboard  (2026-09-08)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 134 nodes · 188 edges · 23 communities (11 shown, 12 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c32ae8c6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- zip_radius.py
- station_pentad_year
- cache_data
- search_best_matches
- precompute_station_reach.py
- _profile_stats
- biomet_app.py
- _avg_conus_percentile
- window_date_range
- sigma_stepped_colorscale
- aggregate
- chart_or_table
- convert
- convert_delta
- month_scale
- pentad_of_doy
- region_map
- sigma_band_color
- sigma_bar_chart
- _toggle_all_months
- trend_rate_control
- _set_rm_window_annual
- _set_state_selection

## God Nodes (most connected - your core abstractions)
1. `select_cells_for_zip_radius()` - 9 edges
2. `station_pentad_year()` - 9 edges
3. `search_best_matches()` - 8 edges
4. `window_date_range()` - 7 edges
5. `region_station_boxplots()` - 6 edges
6. `cells_within_radius()` - 5 edges
7. `_coverage_band_sample()` - 5 edges
8. `region_radar()` - 5 edges
9. `sigma_dissimilarity()` - 5 edges
10. `load_grid()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `check()` --calls--> `select_cells_for_zip_radius()`  [EXTRACTED]
  test_zip_radius.py → zip_radius.py

## Import Cycles
- None detected.

## Communities (23 total, 12 thin omitted)

### Community 0 - "zip_radius.py"
Cohesion: 0.18
Nodes (17): check(), Verification cases for zip_radius.py's ZIP + radius cell resolution. Plain…, cells_within_radius(), haversine_km(), haversine_km_vec(), load_cell_pentad_for_states(), load_grid(), load_zip_lookup() (+9 more)

### Community 1 - "station_pentad_year"
Cohesion: 0.13
Nodes (17): _annual_mean_sample(), coverage(), _coverage_band_sample(), _pct_to_r(), A station is already a single cell's pentad x year series in…, The sample a p5-p95 band is built from for one variable, shared verbatim by…, One value per reference year -- the window mean that year, for every variable…, Genuine MESS (Elith, Kearney & Phillips 2010): "does the station's range… (+9 more)

### Community 2 - "cache_data"
Cohesion: 0.14
Nodes (15): conus_percentile_scale(), load_conus_grid(), load_counties(), load_data(), load_state_pentad(), load_station_pentad(), load_station_reach(), nc_map() (+7 more)

### Community 3 - "search_best_matches"
Cohesion: 0.14
Nodes (14): _hex_to_rgba(), Sigma dissimilarity (Mahony et al. 2017; applied to climate analogues in…, (p_lo, p_hi) for a win_pentads-long window starting at p_start, wrapping past…, The exact ordered pentad sequence of a win_pentads-long window starting at…, Pearson correlation of the region's and the station's mean seasonal trajectory…, Automatic best-match search (Task 14). The region's window (`p_lo`-`p_hi`,…, #RRGGBB" -> "rgba(r,g,b,alpha)", for a fillcolor that needs to be more…, The "Departure" display mode (renamed from "Distance" -- the term this… (+6 more)

### Community 4 - "precompute_station_reach.py"
Cohesion: 0.22
Nodes (9): composite_sigma_matrix(), main(), Offline precompute for Region Matching's "station reach" tab (Task N). Builds…, sigma_dissimilarity()'s formula, vectorised over every cell at once for one…, One (station, candidate window)'s mean/SD/PCA -- everything…, station_window_stats(), Shared window definitions for Region Matching's "station reach" tab (Task N):…, The `length`-long list of pentads starting at p_start, wrapping past… (+1 more)

### Community 5 - "_profile_stats"
Cohesion: 0.25
Nodes (8): _profile_stats(), pentad_year: one row per pentad x year already, one column per raw variable…, Area-weighted mean across `states`' state_pentad.parquet rows, weighted by each…, Pooled p5/p25/p50/p75/p95/mean over the whole window -- see…, Pooled p5/p25/p50/p75/p95/mean over the whole window, the exact same shape…, region_pentad_year(), region_profile(), station_profile()

### Community 7 - "_avg_conus_percentile"
Cohesion: 0.33
Nodes (6): _avg_conus_percentile(), _pentad_range(), _percentile_of(), Empirical percentile (0-100): share of a variable x pentad's CONUS distribution…, Inclusive pentad list from p_lo to p_hi -- wraps past n_pentads back to 1 when…, A single raw value's CONUS percentile rank, averaged across every pentad in the…

### Community 8 - "window_date_range"
Cohesion: 0.33
Nodes (6): _fmt_station_start(), _pentad_end_date(), _pentad_start_date(), Human-readable (start_date, end_date, wrapped) for a win_pentads- long window…, _station_col_header(), window_date_range()

### Community 9 - "sigma_stepped_colorscale"
Cohesion: 0.50
Nodes (4): Every grid cell, coloured by its own composite sigma dissimilarity against one…, A Plotly colorscale (list of [0-1 position, colour] stops) that reproduces…, sigma_stepped_colorscale(), station_reach_map()

### Community 10 - "aggregate"
Cohesion: 0.67
Nodes (3): aggregate(), Aggregate honouring each variable's rule: totals sum, the rest average., yearly_frame()

### Community 11 - "chart_or_table"
Cohesion: 0.67
Nodes (3): chart_or_table(), Switch between the figure and the numbers behind it., style_fig()

## Knowledge Gaps
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `station_pentad_year()` connect `station_pentad_year` to `search_best_matches`, `_profile_stats`, `biomet_app.py`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Why does `search_best_matches()` connect `search_best_matches` to `window_date_range`, `station_pentad_year`, `biomet_app.py`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Why does `conus_percentile_scale()` connect `cache_data` to `biomet_app.py`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Should `station_pentad_year` be split into smaller, more focused modules?**
  _Cohesion score 0.1323529411764706 - nodes in this community are weakly interconnected._
- **Should `cache_data` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._
- **Should `search_best_matches` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._