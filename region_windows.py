"""Shared window definitions for Region Matching's "station reach" tab
(Task N): the fixed menu of 13 windows -- annual, plus 3/6/9-month
windows starting only at a quarter boundary on the 73-pentad grid --
that the offline precompute script (precompute_station_reach.py) and
the app's own window selector must agree on exactly.

One module, not duplicated into both places: CLAUDE.md is explicit
that two independent computations of "the same" thing always end up
disagreeing eventually (coverage()/region_radar()'s band, the
multiselect session_state keys), and a window definition is exactly
that kind of thing -- the precompute script has to search the same 13
positions the app's selector lists, or the two silently drift apart.

No Streamlit import here and nothing else heavy: precompute_station_reach.py
runs standalone from the command line and must not have to import
biomet_app.py (a Streamlit script that renders its whole UI on import,
not something a batch job can safely import as a library).

Quarter starts on the pentad grid (pentad_of_doy(doy) = (doy-1)//5 + 1,
biomet_app.py, non-leap reference calendar): Jan 1 -> pentad 1, Apr 1
(day 91) -> pentad 19, Jul 1 (day 182) -> pentad 37, Oct 1 (day 274) ->
pentad 55. A calendar quarter's exact length (90, 91, 92 or 92 days)
never divides evenly into 5-day pentads, so a fixed pentad-length per
window kind (18/37/55/73) is already an approximation of "3/6/9/12
months" no matter which day it starts on -- accepted as such, not
worth chasing tighter, since the display labels below are deliberately
coarse ("Jan-Mar") rather than claiming a precision the grid can't
back up. One consequence worth knowing: the 3-month windows starting
at pentad 55 (Oct) run pentads 55-72 and stop 5 days short of Dec 31
(pentad 73) -- the same wraparound-length arithmetic as every other
window here, not a separate bug.
"""

N_PENTADS_PER_YEAR = 73

QUARTER_STARTS = [1, 19, 37, 55]   # Jan 1, ~Apr 1, ~Jul 1, ~Oct 1
LENGTHS = {"3mo": 18, "6mo": 37, "9mo": 55, "12mo": 73}


def window_pentads(p_start, length, n_pentads=N_PENTADS_PER_YEAR):
    """The `length`-long list of pentads starting at p_start, wrapping
    past N_PENTADS_PER_YEAR back to 1 -- the same wraparound rule
    biomet_app.py's own _wrapped_window() uses for the automatic search."""
    return [((p_start - 1 + k) % n_pentads) + 1 for k in range(length)]


# (key, length_key, p_start) -- Annual first (it's the default/landing
# choice), then each length's four quarter starts in calendar order.
WINDOWS = [("Annual", "12mo", 1)]
for _lk in ("3mo", "6mo", "9mo"):
    for _qi, _start in enumerate(QUARTER_STARTS, start=1):
        WINDOWS.append((f"{_lk}_Q{_qi}", _lk, _start))

WINDOW_KEYS = [w[0] for w in WINDOWS]                      # menu/display order
WINDOW_LENGTH_KEY = {key: lk for key, lk, _ in WINDOWS}    # key -> "3mo"/"6mo"/"9mo"/"12mo"
WINDOW_PENTADS = {key: window_pentads(p_start, LENGTHS[lk])
                 for key, lk, p_start in WINDOWS}          # key -> list of pentads

# For a given length, the candidate windows the "best sigma over station
# windows restricted to quarter starts" search picks among -- e.g. a
# 6-month region window only ever compares against the four 6-month
# station windows, never a 3- or 9-month one.
CANDIDATES_BY_LENGTH = {}
for _key, _lk, _ in WINDOWS:
    CANDIDATES_BY_LENGTH.setdefault(_lk, []).append(_key)

# Deliberately coarse display labels -- see the module docstring for why
# these don't claim day-level precision the pentad grid doesn't have.
DISPLAY_LABEL = {
    "Annual": "Annual",
    "3mo_Q1": "Jan-Mar", "3mo_Q2": "Apr-Jun", "3mo_Q3": "Jul-Sep", "3mo_Q4": "Oct-Dec",
    "6mo_Q1": "Jan-Jun", "6mo_Q2": "Apr-Sep", "6mo_Q3": "Jul-Dec", "6mo_Q4": "Oct-Mar",
    "9mo_Q1": "Jan-Sep", "9mo_Q2": "Apr-Dec", "9mo_Q3": "Jul-Mar", "9mo_Q4": "Oct-Jun",
}
