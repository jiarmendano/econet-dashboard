"""Regression test for the trend filter added to
precompute_station_reach.py's own window search (same rule as
biomet_app.py's search_best_matches(), shared via trajectory.py).

Run directly, no pytest -- same convention as test_zip_radius.py and
test_cell_rectangles_winding.py.

The case: BAHA and GOLD, region window 9mo_Q3, at three cells the
pre-filter file (station_reach.parquet.bak, kept alongside this file's
own commit) picked candidate 9mo_Q1 ("Jan-Sep") as the winning station
window -- sigma near 0 (an almost-perfect level match) but a
trajectory correlation around -0.96 (the two seasons run in opposite
phase). That is exactly the failure mode the trend filter exists to
catch: a level-only comparison cannot see it, but "same mean, opposite
seasonal trend" is a real mismatch (CLAUDE.md, "Trend is a filter, not
a weight").
"""
import pandas as pd

REPO_CELLS = [
    (-83.750, 32.5, "BAHA"),
    (-83.125, 32.5, "BAHA"),
    (-80.625, 33.0, "GOLD"),
]
REGION_WINDOW = "9mo_Q3"
ANTI_PHASE_LABEL = "Jan-Sep"   # rw.DISPLAY_LABEL["9mo_Q1"]


def main():
    reach = pd.read_parquet("station_reach.parquet")
    failures = []

    for lon, lat, station in REPO_CELLS:
        row = reach[(reach["lon"] == lon) & (reach["lat"] == lat)
                    & (reach["station"] == station) & (reach["window"] == REGION_WINDOW)]
        if row.empty:
            failures.append(f"{station} ({lon}, {lat}): no row found for {REGION_WINDOW}")
            continue
        label = row["station_window"].iloc[0]
        sigma = row["sigma"].iloc[0]
        if label == ANTI_PHASE_LABEL:
            failures.append(
                f"{station} ({lon}, {lat}): still won by anti-phase "
                f"{ANTI_PHASE_LABEL} (sigma={sigma:.4f}) -- trend filter did not apply")
        else:
            print(f"OK  {station} ({lon}, {lat}): winner is now {label} "
                 f"(sigma={sigma:.4f}), not {ANTI_PHASE_LABEL}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" ", f)
        raise SystemExit(1)
    print("\nAll BAHA/GOLD 9mo_Q3 regression cells pass.")


if __name__ == "__main__":
    main()
