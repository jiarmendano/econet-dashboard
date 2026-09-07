"""Verification cases for zip_radius.py's ZIP + radius cell resolution.

Plain assertions, run directly (`python test_zip_radius.py`) -- this
repo has no test suite or runner (see CLAUDE.md), so this mirrors the
R build scripts' own style: asserts plus a printed report, not pytest.

Five cases, chosen to exercise the fix this module exists for (a naive
"load the ZIP's own state file, then filter by radius" flow silently
clips the selection at a state line) and the two adjacent things that
needed checking once the fix was in: that the radius is a real
geodesic distance in km, and that a small radius can legitimately find
nothing.
"""
from zip_radius import select_cells_for_zip_radius, load_grid, load_zip_lookup

grid_df = load_grid()
zip_df = load_zip_lookup()


def check(zip_code, radius_km, label, **expect):
    matched, cell_pentad_df, report = select_cells_for_zip_radius(zip_code, radius_km, grid_df, zip_df)
    print(f"=== {label} -- ZIP {zip_code}, radius {radius_km} km ===")
    print(f"  home_state: {report['home_state']}   cells: {report['n_cells']}   "
         f"states: {report['states']}")
    if report["n_cells"]:
        print(f"  nearest/farthest: {report['nearest_km']:.2f} / {report['farthest_km']:.2f} km")
    print(f"  {report['message']}")

    if "home_state" in expect:
        assert report["home_state"] == expect["home_state"], \
            f"{label}: expected home_state {expect['home_state']!r}, got {report['home_state']!r}"
    if "n_cells" in expect:
        assert report["n_cells"] == expect["n_cells"], \
            f"{label}: expected {expect['n_cells']} cells, got {report['n_cells']}"
    if "states" in expect:
        assert set(report["states"]) == set(expect["states"]), \
            f"{label}: expected states {expect['states']}, got {report['states']}"
    if "cross_state" in expect:
        assert report["cross_state"] == expect["cross_state"], \
            f"{label}: expected cross_state={expect['cross_state']}, got {report['cross_state']}"

    # every loaded row must be one of the matched cells, and every
    # matched cell must have its full 73 x 35 pentad-year history --
    # step 3's filter must neither leak extra cells nor drop rows.
    if report["n_cells"]:
        n_distinct_loaded = cell_pentad_df[["lon", "lat"]].drop_duplicates().shape[0]
        assert n_distinct_loaded == report["n_cells"], \
            f"{label}: loaded {n_distinct_loaded} distinct cells, expected {report['n_cells']}"
        assert len(cell_pentad_df) == report["n_cells"] * 2555, \
            f"{label}: loaded {len(cell_pentad_df)} rows, expected {report['n_cells']} x 2555"
    else:
        assert len(cell_pentad_df) == 0

    print("  OK\n")
    return report


# Rhode Island: the case that forced the fix. Its own cell_pentad file
# has 2 cells; a real 100 km radius from Providence must reach past
# them into neighbouring states, not stop at Rhode Island's own file.
# Also confirms admin_state, not the ZIP's nearest-cell state (which is
# actually Massachusetts here -- Providence sits that close to the
# border), is what gets reported as "home".
check("02903", 100, "Providence, RI",
     home_state="Rhode Island", cross_state=True,
     states=["Rhode Island", "Massachusetts", "Connecticut", "New York"])

# A real state-line case away from any small-state edge effect.
check("64108", 100, "Kansas City, MO side",
     home_state="Missouri", cross_state=True,
     states=["Missouri", "Kansas"])

# Far from any border: should stay in one state, and give a sense of
# the normal cell count at 100 km (9, here).
check("76701", 100, "Waco, central TX",
     home_state="Texas", cross_state=False, states=["Texas"], n_cells=9)

# Same ZIP, radius shrunk to 25 km -- smaller than the ~55-60 km grid
# spacing, so finding zero cells is the CORRECT answer, not a bug. See
# zip_radius.py's own module docstring for why a UI radius control
# should not default this low.
check("76701", 25, "Waco, central TX (small radius)",
     n_cells=0, cross_state=False)

# Gulf coast: much of the 100 km circle is open water, so the cell
# count should come in below an equivalent inland location (Waco's 9)
# for a real geographic reason, not by chance.
report = check("77550", 100, "Galveston, TX (Gulf coast)",
               home_state="Texas", cross_state=False, states=["Texas"])
assert report["n_cells"] < 9, "Gulf coast cell count should be below the inland (Waco) baseline"

print("ALL CHECKS PASSED")
