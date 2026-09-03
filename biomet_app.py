"""
Biometeorological Data Explorer
Streamlit dashboard over the gap-filled daily station dataset.

Run with:
    streamlit run biomet_app.py

Expects ECONET_HSdata_d.csv next to this file, or set DATA_PATH below.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------- config ----

DATA_PATH = Path(__file__).parent / "ECONET_HSdata_d.csv"
COUNTIES_PATH = Path(__file__).parent / "nc_counties.geojson"
STATE_PENTAD_PATH = Path(__file__).parent / "state_pentad.parquet"
CONUS_GRID_PATH = Path(__file__).parent / "conus_grid.parquet"
STATION_PENTAD_PATH = Path(__file__).parent / "station_pentad.parquet"

APP_TITLE = "Biometeorological Data Explorer"
BASE_YEARS = (2006, 2025)      # fixed anomaly reference, never follows the filter
IMPUTED_FLAG = 50              # a year above this % imputed is drawn as synthetic

URL_NETWORK = "https://econet.climate.ncsu.edu"
URL_OFFICE = "https://climate.ncsu.edu"
URL_MERRA = "https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/"

st.set_page_config(page_title=APP_TITLE, page_icon="\u25e6",
                   layout="wide", initial_sidebar_state="expanded")

# Streamlit >=1.49 uses width="stretch"; on older versions swap for
# use_container_width=True.
W = "stretch"

# ----------------------------------------------------------------- theme ----

# Okabe-Ito, colourblind safe, one per station.
STATION_COLORS = ["#E69F00", "#56B4E9", "#009E73",
                  "#CC79A7", "#0072B2", "#D55E00"]

# USDA-ARS's own map colours (sampled from
# https://www.ars.usda.gov/ARSUserFiles/incme/images//usmap.jpg, one
# interior point per region, away from borders/labels/station markers).
# Kept as one constant, reused by both themes: these identify the
# agency's regions specifically, so they don't get a light/dark variant
# the way UI chrome does. Northeast and Midwest are hard to tell apart
# under simulated deuteranopia/protanopia — a property of the source
# map's own pastel palette, not adjusted here; see the region_map()
# choropleth if that turns out to matter in practice.
USDA_REGION_COLORS = ["#D5C9E1", "#B3FFFF", "#E0DAC2", "#C5FFE4", "#FFB1A7"]

THEMES = {
    "dark": dict(
        bg="#0E1418", panel="#161F26", line="#243139",
        text="#E3E9ED", muted="#94A5B0", accent="#D9542B",
        accent_soft="#3A2119", grid="#1E2A32", template="plotly_dark",
        region_colors=USDA_REGION_COLORS,
    ),
    "light": dict(
        bg="#FBFAF7", panel="#FFFFFF", line="#E2E0DA",
        text="#1B2429", muted="#535E67", accent="#C24A20",
        accent_soft="#F7E4DC", grid="#EDEBE5", template="plotly_white",
        region_colors=USDA_REGION_COLORS,
    ),
}

# Livestock Weather Safety Index thresholds: alert, danger, emergency
THRESHOLD_COLORS = {"THI75": "#E7CA45", "THI79": "#F68B00", "THI84": "#9A3C41"}

# Diverging pairs: warm/cool for temperature, dry/wet for precipitation.
# Precipitation gets its own pair so that red never reads as "wetter".
ANOM_TEMP = ("#C44536", "#3E7CB1")
ANOM_PRECIP = ("#2A9D8F", "#A6771C")   # (positive = wetter, negative = drier)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# USDA-ARS regions, canonical order (matches conus_grid.parquet / CLAUDE.md).
REGIONS = ["Northeast", "Southeast", "Midwest", "Plains", "Pacific West"]

# Standard postal codes, needed for Plotly's locationmode="USA-states" —
# conus_grid.parquet only carries full state names.
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

# "Data" is hidden from the nav (Region Matching took its slot) but its
# elif branch below is untouched and still reachable — restore the name
# here to bring it back.
NAV = ["Overview", "Time series", "Anomalies", "Region Matching"]


def inject_css(t):
    st.markdown(f"""
    <style>
      .stApp {{ background:{t['bg']}; color:{t['text']}; }}
      section[data-testid="stSidebar"] {{
          background:{t['panel']}; border-right:1px solid {t['line']}; }}
      section[data-testid="stSidebar"] * {{ color:{t['text']}; }}
      h1,h2,h3,h4 {{ color:{t['text']}; font-weight:600; letter-spacing:-.01em; }}
      .block-container {{ padding-top:2rem; max-width:1560px; }}

      /* menu, station and download buttons: stDownloadButton is a separate
         widget from stButton, so it needs to be listed explicitly or it
         falls back to Streamlit's own unthemed button style. */
      div[data-testid="stButton"] > button,
      div[data-testid="stDownloadButton"] > button {{
          border-radius:6px; border:1px solid {t['line']};
          background:transparent; color:{t['muted']};
          font-weight:500; text-align:left; padding:.45rem .8rem; }}
      div[data-testid="stButton"] > button:hover,
      div[data-testid="stDownloadButton"] > button:hover {{
          border-color:{t['accent']}; color:{t['text']}; }}
      div[data-testid="stButton"] > button[kind="primary"],
      div[data-testid="stDownloadButton"] > button[kind="primary"] {{
          background:{t['accent_soft']}; border-left:3px solid {t['accent']};
          color:{t['text']}; }}

      /* Select all / Unselect all: a plain stButton, tinted a bit so it
         doesn't read as just another month pill. Specificity has to match
         the button rule above (div[data-testid] > button) or it loses. */
      .st-key-months_toggle div[data-testid="stButton"] > button {{
          background:{t['line']}; text-align:center; }}
      .st-key-months_toggle div[data-testid="stButton"] > button:hover {{
          background:{t['accent_soft']}; }}

      /* pills (Months) and segmented control (Chart/Table/Download) share
         this widget; style selected vs unselected explicitly so the state
         is unambiguous and readable in both themes. */
      div[data-testid="stButtonGroup"] button {{
          background:transparent; border:1px solid {t['line']};
          color:{t['muted']}; }}
      div[data-testid="stButtonGroup"] button:hover {{
          border-color:{t['accent']}; color:{t['text']}; }}
      div[data-testid="stButtonGroup"] button[data-selected="true"] {{
          background:{t['accent_soft']}; border-color:{t['accent']};
          color:{t['text']}; font-weight:600; }}

      div[data-testid="stMetric"] {{
          background:{t['panel']}; border:1px solid {t['line']};
          border-radius:8px; padding:12px 14px; }}
      label[data-testid="stMetricLabel"] p {{
          color:{t['muted']}; font-size:.72rem; text-transform:uppercase;
          letter-spacing:.05em; line-height:1.25; }}
      div[data-testid="stMetricValue"] {{ color:{t['text']}; font-size:1.45rem; }}

      /* Streamlit truncates the label and delta lines with an ellipsis by
         default; narrow KPI columns need them to wrap instead. */
      label[data-testid="stMetricLabel"],
      label[data-testid="stMetricLabel"] p,
      div[data-testid="stMetricDelta"],
      div[data-testid="stMetricDelta"] p {{
          white-space:normal !important; overflow:visible !important;
          text-overflow:clip !important; }}

      /* Overview KPI grid: labels have room to spell words out, so they
         read as normal sentence case rather than the small-caps treatment
         used elsewhere for tight metric labels. */
      .st-key-ov_kpis label[data-testid="stMetricLabel"] p {{
          text-transform:none; letter-spacing:normal; }}

      /* Overview KPI grid: "per year"/"per season" sits next to the number
         instead of stacked below it. */
      .st-key-ov_kpis div[data-testid="stMetricValue"] {{
          display:inline-flex; vertical-align:baseline; }}
      .st-key-ov_kpis div[data-testid="stMetricValue"] + div {{
          display:inline-flex; vertical-align:baseline; margin-left:.5em; }}

      /* threshold KPIs: a colour bar on the card's left edge, not a border
         change on all sides. Card size/shape stays the same. */
      .st-key-ov_thi75 div[data-testid="stMetric"] {{
          border-left:4px solid {THRESHOLD_COLORS['THI75']}; }}
      .st-key-ov_thi79 div[data-testid="stMetric"] {{
          border-left:4px solid {THRESHOLD_COLORS['THI79']}; }}
      .st-key-ov_thi84 div[data-testid="stMetric"] {{
          border-left:4px solid {THRESHOLD_COLORS['THI84']}; }}

      /* selectbox / multiselect: closed field and the open dropdown panel
         (the panel renders in a portal at the document root, not inside
         .stApp, but plain testid selectors still reach it). */
      div[data-testid="stSelectbox"] div[role="group"],
      div[data-testid="stMultiSelect"] div[role="group"] {{
          background:{t['panel']}; border-color:{t['line']}; }}
      div[data-testid="stSelectbox"] input,
      div[data-testid="stMultiSelect"] input {{
          background:transparent; color:{t['text']}; }}
      div[data-testid="stSelectboxVirtualDropdown"],
      div[data-testid="stMultiSelectDropdown"] {{
          background:{t['panel']}; border:1px solid {t['line']};
          color:{t['text']}; }}
      div[data-testid="stSelectboxVirtualDropdown"] [role="option"],
      div[data-testid="stMultiSelectDropdown"] [role="option"] {{
          color:{t['text']}; }}
      div[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
      div[data-testid="stMultiSelectDropdown"] [role="option"]:hover,
      div[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
      div[data-testid="stMultiSelectDropdown"] [role="option"][aria-selected="true"] {{
          background:{t['accent_soft']}; }}

      /* expanders: themed chrome instead of Streamlit's default white card,
         everywhere one appears. */
      div[data-testid="stExpander"] {{
          background:{t['panel']}; border:1px solid {t['line']};
          border-radius:8px; }}
      div[data-testid="stExpander"] summary {{ color:{t['text']}; }}
      div[data-testid="stExpander"] summary:hover {{ color:{t['accent']}; }}
      div[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {{
          color:{t['muted']}; }}
      div[data-testid="stExpander"] details[open] summary {{
          border-bottom:1px solid {t['line']}; }}

      /* Overview's two section expanders: header reads at the same weight
         as an h3 ("### Summary table"), not the expander's default 16px/400.
         Scoped to each expander's own summary so the nested "Copy as
         tab-separated" expander inside the table section is unaffected. */
      .st-key-ov_section_station > div[data-testid="stLayoutWrapper"]
          > div[data-testid="stExpander"] > details > summary p,
      .st-key-ov_section_table > div[data-testid="stLayoutWrapper"]
          > div[data-testid="stExpander"] > details > summary p {{
          font-size:1.125rem; font-weight:600; letter-spacing:-.01em; }}

      /* map zoom controls: Plotly's modebar defaults to the top-right,
         translucent black, sized for a full toolbar. Move it to the
         bottom-right and theme it down. */
      /* stretched to full height so the toolbar can anchor to the bottom
         edge, but transparent everywhere except its own buttons — without
         pointer-events:none here, this invisible box sits over the whole
         map and swallows the scroll/drag meant for the chart underneath. */
      .modebar-container {{ height:100% !important; pointer-events:none; }}
      .js-plotly-plot .plotly .modebar {{
          top:auto !important; bottom:6px !important; right:6px !important; }}
      .modebar-container .modebar-group {{
          background:{t['panel']} !important; border:1px solid {t['line']};
          border-radius:6px; pointer-events:auto; }}
      .modebar-container .modebar-btn path {{ fill:{t['muted']} !important; }}
      .modebar-container .modebar-btn:hover path {{ fill:{t['accent']} !important; }}

      /* Plotly's own hover tooltip for these buttons defaults to opening
         below the button (top:110%), which is fine when the toolbar sits
         at the chart's top edge but pushes the tooltip outside the map
         entirely now that the toolbar sits at the bottom. Flip it to open
         upward instead, back over the map where there's room. */
      .js-plotly-plot .plotly .modebar [data-title]::after,
      .js-plotly-plot .plotly .modebar [data-title]::before {{
          top:auto !important; bottom:110% !important; }}

      /* Anomalies: the "averaged across stations" note sits in a column
         with no label of its own, so it lines up with the widget's label
         line by default; push it down to line up with the multiselect
         box instead. */
      .st-key-anom_stations_note {{ margin-top:28px; }}

      /* Region Matching's 7 blocks get the same expander-header weight
         Overview's two section expanders use, but as one attribute-prefix
         selector instead of Overview's enumerated pair — 7 keys in a
         comma list would be unwieldy where 2 isn't. */
      [class*="st-key-rm_block_"] > div[data-testid="stLayoutWrapper"]
          > div[data-testid="stExpander"] > details > summary p {{
          font-size:1.125rem; font-weight:600; letter-spacing:-.01em; }}

      /* Units and Theme, pinned top-left just outside the sidebar (300px
         is Streamlit's default sidebar width; if the sidebar is dragged
         to a different width, or collapsed, this stops lining up with its
         edge) so the pair stays on screen regardless of scroll position.
         Radios stack vertically by default; row them side by side to fit
         the header's height. */
      .st-key-topbar_controls {{
          position:fixed; top:14px; left:316px; z-index:1000000;
          width:fit-content !important;
          display:flex; flex-direction:row; align-items:center;
          background:{t['line']}; border-radius:8px; padding:6px 16px; }}
      .st-key-topbar_controls div[data-testid="stElementContainer"] {{
          width:auto; }}
      /* a rule between Units and Theme, so the four options don't read as
         one run of buttons. */
      .st-key-topbar_controls div[data-testid="stElementContainer"]:nth-child(2) {{
          margin-left:18px; padding-left:18px; border-left:1px solid {t['muted']}; }}

      .nav-caption {{ color:{t['muted']}; font-size:.72rem;
          text-transform:uppercase; letter-spacing:.08em; margin:.4rem 0 .2rem; }}
      .infobox {{ background:{t['panel']}; border:1px solid {t['line']};
          border-left:3px solid {t['accent']}; border-radius:6px;
          padding:12px 14px; font-size:.84rem; color:{t['muted']}; }}
      .infobox b {{ color:{t['text']}; }}
      .credits {{ color:{t['muted']}; font-size:.75rem; line-height:1.6; }}
      .credits a {{ color:{t['muted']}; text-decoration:underline; }}
      hr {{ border-color:{t['line']}; }}
    </style>""", unsafe_allow_html=True)


def style_fig(fig, t, height=430):
    fig.update_layout(
        template=t["template"], height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, title=None),
        hovermode="x unified")
    fig.update_xaxes(gridcolor=t["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=t["grid"], zeroline=False)
    return fig


# ------------------------------------------------------------- variables ----

VARS = {
    "Tmax_C":    dict(label="Max temperature",      agg="mean", kind="temp"),
    "Tmed_C":    dict(label="Average temperature",  agg="mean", kind="temp"),
    "Tmin_C":    dict(label="Min temperature",      agg="mean", kind="temp"),
    "Tdew_C":    dict(label="Dew point",            agg="mean", kind="temp"),
    "RHmax":     dict(label="Max relative humidity",     agg="mean", kind="pct"),
    "RHmed":     dict(label="Average relative humidity", agg="mean", kind="pct"),
    "RHmin":     dict(label="Min relative humidity",     agg="mean", kind="pct"),
    "THImax":    dict(label="THI max",              agg="mean", kind="index"),
    "THImed":    dict(label="THI average",          agg="mean", kind="index"),
    "THImin":    dict(label="THI min",              agg="mean", kind="index"),
    "Precip_mm": dict(label="Rainfall",              agg="sum",  kind="precip"),
    "THI75":     dict(label="Days THImax \u2265 75 (Alert)", agg="sum", kind="days"),
    "THI79":     dict(label="Days THImax \u2265 79 (Danger)", agg="sum", kind="days"),
    "THI84":     dict(label="Days THImax \u2265 84 (Emergency)", agg="sum", kind="days"),
}

# Region Matching's own variable registry. Not folded into VARS: the two
# describe different datasets entirely (VARS is the ECONet daily table;
# this is state_pentad.parquet's already pentad-aggregated columns), and
# state_pentad.parquet doesn't have most of what VARS lists at all. Same
# shape and philosophy as VARS otherwise: one entry per variable, its
# display label, and its kind for convert()/unit_label(). No "agg" key
# here, unlike VARS: every grid variable is a plain mean across whichever
# pentads/years get selected, because state_pentad.parquet was built
# specifically so that would always be true (see CLAUDE.md, "Store
# rates, not totals").
#
# PRECTOTCORR and THI_ge_79 are stored as rates (mean mm/day, fraction of
# days), not totals, for exactly that reason: a pentad total/count would
# carry pentad 12's extra leap day as a length artefact. Decision for the
# app side: convert to a per-window total/count at display time, the
# same "per period" convention VARS' own Precip_mm/THI75/THI79 already
# use for the ECONet side, rather than labelling these as rates here.
# kind below names that eventual display form, not the stored one; no
# block does the conversion yet.
#
# DTR and interdiurnal_T2M are temperature *differences*, not absolute
# readings, even though kind="temp" like T2M/T2MDEW: whichever later
# block displays them must scale with convert_delta() (no +32 offset),
# the same distinction Anomalies already draws for its own delta values
# using VARS' kind.
GRID_VARS = {
    "T2M":              dict(label="Mean temperature",              kind="temp",   default=True),
    "DTR":              dict(label="Diurnal temperature range",     kind="temp",   default=True),
    "interdiurnal_T2M": dict(label="Day to day temperature change", kind="temp",   default=True),
    "T2MDEW":           dict(label="Dew point",                     kind="temp",   default=True),
    "PRECTOTCORR":      dict(label="Precipitation",                 kind="precip", default=True),
    "THI_ge_79":        dict(label="Days THImax ≥ 79",         kind="days",   default=True),
    "THImax":           dict(label="Maximum THI",                   kind="index",  default=False),
    "RHmed":            dict(label="Mean relative humidity",        kind="pct",    default=False),
}

# Columns from the source file that this app does not use
DROP_COLS = ["THImax_rh", "THImin_rh", "THImed_rh"]


def unit_label(kind, metric):
    if kind == "temp":
        return "\u00b0C" if metric else "\u00b0F"
    if kind == "precip":
        return "mm" if metric else "in"
    if kind == "pct":
        return "%"
    if kind == "days":
        return "days"
    return ""          # THI is a unitless index


def unit_suffix(kind, metric):
    u = unit_label(kind, metric)
    return f" ({u})" if u else ""


def convert(series, kind, metric):
    """THI is a unitless index and is never converted."""
    if metric:
        return series
    if kind == "temp":
        return series * 9 / 5 + 32
    if kind == "precip":
        return series / 25.4
    return series


def convert_delta(series, kind, metric):
    """Scale a *difference* (an anomaly, a trend) with no offset: a 1 degC
    gap is 1.8 degF, not 33.8 degF. THI is unitless and never converted."""
    if metric:
        return series
    if kind == "temp":
        return series * 9 / 5
    if kind == "precip":
        return series / 25.4
    return series


# ------------------------------------------------------------------ data ----

@st.cache_data(show_spinner="Loading dataset")
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df["Imputed"] = df["Imputed"].astype(str).str.lower().isin(["true", "1"])
    return df


@st.cache_data
def load_counties(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading state/pentad aggregates")
def load_state_pentad(path):
    """Region Matching's only input besides the grid — one row per
    state x pentad x year, already collapsed from the per-cell grid in R.
    See CLAUDE.md, 'Region matching section', for why the collapse
    happens there and not here."""
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading grid cells")
def load_conus_grid(path):
    """Grid point coordinates plus each cell's state/region/area weight,
    used only for the map — Region Matching's actual numbers all come
    from load_state_pentad(), never from an individual cell."""
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading station pentad aggregates")
def load_station_pentad(path):
    """The six ECONet stations' nearest-cell pentad x year rates —
    aggregate_cell_to_pentad() in R, the same function state_pentad.parquet's
    own per-cell step uses, so the station and region sides of the
    comparison are never apples to oranges. See
    r-codes/Aggregate_station_pentad.R and pentad_lib.R."""
    return pd.read_parquet(path)


def aggregate(frame, var, by):
    """Aggregate honouring each variable's rule: totals sum, the rest average."""
    grouped = frame.groupby(by, observed=True)[var]
    rule = VARS[var]["agg"]
    return (grouped.sum() if rule == "sum" else grouped.mean()).reset_index()


# --------------------------------------------------------------- sidebar ----

def _toggle_all_months():
    """Runs as an on_click callback, before the pills widget redraws, since
    st.session_state can't be written to a widget's key after it's already
    been instantiated in the same script run."""
    all_selected = len(st.session_state.months_sel or []) == 12
    st.session_state.months_sel = [] if all_selected else list(range(1, 13))


# Theme toggle is disabled for now: st.dataframe and the top toolbar are
# native Streamlit widgets that don't follow the CSS-only "dark" theme, so
# the app looks broken with it on. Pin to "light" instead of exposing the
# toggle; THEMES["dark"] and every T[...] reference below are left in
# place so this can be flipped back on later.
st.session_state.theme = "light"
if "section" not in st.session_state:
    st.session_state.section = "Overview"

with st.sidebar:
    st.markdown(f"### {APP_TITLE}")
    st.markdown('<p class="nav-caption">Menu</p>', unsafe_allow_html=True)

    for item in NAV:
        if st.button(item, key=f"nav_{item}", width=W,
                     type="primary" if st.session_state.section == item
                     else "secondary"):
            st.session_state.section = item
            st.rerun()

section = st.session_state.section

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}. "
             "Put ECONET_HSdata_d.csv next to this script, "
             "or edit DATA_PATH at the top of the file.")
    st.stop()

data = load_data(DATA_PATH)
all_stations = sorted(data["Station"].unique())
color_map = dict(zip(all_stations, STATION_COLORS))
yr_lo, yr_hi = int(data["Year"].min()), int(data["Year"].max())

st.session_state.setdefault("kpi_station", all_stations[0])

with st.sidebar:
    st.divider()

    # Wrapped in its own keyed container, always built regardless of
    # section, so it can be hidden with CSS on Region Matching (below,
    # near inject_css) without ever skipping the widgets themselves.
    # Skipping them would drop their session_state and reset the user's
    # selection on the way back to Time series/Anomalies — the same class
    # of desync already documented for the multiselect and trend-rate keys.
    with st.container(key="sidebar_filters"):
        st.markdown('<p class="nav-caption">Filters</p>', unsafe_allow_html=True)

        # Every section picks its stations on the page itself now, so `df`
        # below always starts from the full set.
        sel_stations = all_stations

        years = st.slider("Years", yr_lo, yr_hi, (yr_lo, yr_hi))

        st.session_state.setdefault("months_sel", list(range(1, 13)))
        months = st.pills("Months", list(range(1, 13)), selection_mode="multi",
                          key="months_sel",
                          format_func=lambda m: MONTH_NAMES[m - 1])

        all_selected = len(st.session_state.months_sel or []) == 12
        st.button("Unselect all" if all_selected else "Select all",
                 key="months_toggle", width="content",
                 on_click=_toggle_all_months)

    if not months:
        months = list(range(1, 13))
    months = sorted(months)

# Units lives top-left, pinned just to the right of the sidebar via CSS
# (.st-key-topbar_controls), rather than inside the sidebar itself: the one
# control worth having on screen no matter how far the page is scrolled.
# The Theme radio that used to sit beside it is disabled for now (see the
# session_state.theme pin above) rather than removed, so it can be dropped
# back in here once dark mode's native-widget gaps are fixed.
with st.container(key="topbar_controls"):
    units = st.radio("Units", ["Metric (\u00b0C, mm)", "Imperial (\u00b0F, in)"],
                     horizontal=True, label_visibility="collapsed")
metric = units.startswith("Metric")

T = THEMES[st.session_state.theme]
inject_css(T)

# Sidebar year/month filters are meaningless on Region Matching (it loads
# state_pentad.parquet on its own, not the sidebar-filtered df) but stay
# mounted regardless of section — see the comment at sidebar_filters above
# for why — so they're hidden the same way .st-key-topbar_controls is
# positioned: CSS on the container's key, not a conditional around the
# widgets.
if section == "Region Matching":
    st.markdown('<style>.st-key-sidebar_filters { display:none !important; }'
               '</style>', unsafe_allow_html=True)

# NOTE: there used to be a native-theme sync here (st.html script comparing
# Streamlit's native theme against session_state.theme, syncing via
# localStorage, and forcing a navigation with theme/section carried across
# it in query params). It caused an infinite reload loop once the theme
# was pinned above, so it's removed entirely rather than patched. Native,
# canvas-rendered widgets (e.g. st.dataframe) now just follow whatever
# .streamlit/config.toml sets as the native theme, which is pinned to
# light there too.

full_year = len(months) == 12
PERIOD = "year" if full_year else ("month" if len(months) == 1 else "season")
months_label = ("all months" if full_year
                else ", ".join(MONTH_NAMES[m - 1] for m in months))

df = data[data["Station"].isin(sel_stations)
          & data["Year"].between(*years)
          & data["Month"].isin(months)].copy()

with st.sidebar:
    st.divider()
    st.markdown(
        f'<div class="credits">'
        f'{len(df):,} records selected<br><br>'
        f'Source: NC State Climate Office, ECONet '
        f'(<a href="{URL_NETWORK}" target="_blank">econet.climate.ncsu.edu</a>)'
        f'<br>Gaps filled with calibrated '
        f'<a href="{URL_MERRA}" target="_blank">MERRA-2</a> reanalysis'
        f'</div>', unsafe_allow_html=True)


# --------------------------------------------------------------- helpers ----

def chart_or_table(fig, table, key, filename, height=430):
    """Switch between the figure and the numbers behind it."""
    mode = st.segmented_control("View", ["Chart", "Table", "Download"],
                                default="Chart", key=key,
                                label_visibility="collapsed")
    if mode == "Table":
        st.dataframe(table, width=W, hide_index=True)
    elif mode == "Download":
        st.download_button("Download CSV", table.to_csv(index=False),
                           file_name=filename, mime="text/csv", key=key + "_dl")
        st.caption("Or copy the tab-separated block below straight into a sheet.")
        st.code(table.to_csv(index=False, sep="\t"), language=None)
    else:
        st.plotly_chart(style_fig(fig, T, height), width=W, key=key + "_fig")


def trend_rate_control(key):
    """Per year / per decade toggle. Each section keeps its own state:
    sharing one key across two call sites left the widget's on-screen
    selection and its session_state value out of sync in this Streamlit
    version, so each page owns its own control instead."""
    st.session_state.setdefault(key, "Per decade")
    st.segmented_control("Trend rate", ["Per year", "Per decade"], key=key)
    unit = st.session_state[key] or "Per decade"
    suffix = "year" if unit == "Per year" else "decade"
    mult = 1 if unit == "Per year" else 10
    return mult, suffix


def imputation_panel(frame, note=True):
    rows = (frame.groupby("Station", observed=True)["Imputed"]
            .agg(["size", "sum"]).reset_index())
    rows["pct"] = (100 * rows["sum"] / rows["size"]).round(1)
    body = " &nbsp;\u00b7&nbsp; ".join(
        f"<b>{r.Station}</b> {r.pct}%"
        for r in rows.sort_values("pct", ascending=False).itertuples())
    overall = round(100 * frame["Imputed"].mean(), 1) if len(frame) else 0
    tail = ("<br>Series above 50% are drawn in figures with open markers and "
            "should not be read as measurements." if note and overall >= 5
            else "")
    st.markdown(f'<div class="infobox">Imputed values in this selection: '
                f'<b>{overall}%</b><br>{body}{tail}</div>',
                unsafe_allow_html=True)


def yearly_frame(frame, var):
    agg = aggregate(frame, var, ["Station", "Year"])
    imp = (frame.groupby(["Station", "Year"], observed=True)["Imputed"]
           .mean().mul(100).round(1).reset_index(name="pct_imputed"))
    out = agg.merge(imp, on=["Station", "Year"])
    out["synthetic"] = out["pct_imputed"] > IMPUTED_FLAG
    return out


def line_by_station(tbl, var, ylab, stations, show_trend=False, show_mean=False):
    fig = go.Figure()
    for stn in stations:
        d = tbl[tbl["Station"] == stn].sort_values("Year")
        if d.empty:
            continue
        c = color_map[stn]
        fig.add_trace(go.Scatter(
            x=d["Year"], y=d[var], name=stn, mode="lines+markers",
            line=dict(color=c, width=2), marker=dict(size=5, color=c)))

        syn = d[d["synthetic"]]
        if not syn.empty:
            fig.add_trace(go.Scatter(
                x=syn["Year"], y=syn[var], mode="markers",
                marker=dict(size=9, color=T["bg"],
                            line=dict(color=c, width=2)),
                name=f"{stn} \u00b7 mostly imputed",
                hovertemplate="mostly imputed<extra></extra>"))

        if show_trend and len(d) > 2:
            k = np.polyfit(d["Year"], d[var], 1)
            fig.add_trace(go.Scatter(
                x=d["Year"], y=np.polyval(k, d["Year"]), mode="lines",
                line=dict(color=c, width=1.4, dash="dot"),
                name=f"{stn} \u00b7 trend", hoverinfo="skip"))

        if show_mean:
            fig.add_hline(y=d[var].mean(),
                          line=dict(color=c, width=1, dash="dash"), opacity=.5)

    fig.update_yaxes(title=ylab)
    fig.update_xaxes(title=None, dtick=2)
    return fig


def nc_map(active, height=330):
    """North Carolina with the six sites; the active one is highlighted."""
    sites = (data[["Station", "StationName", "Lat", "Lon"]]
             .drop_duplicates("Station").sort_values("Station"))
    counties = load_counties(COUNTIES_PATH)
    county_ids = [f["id"] for f in counties["features"]]

    fig = go.Figure()

    # county lines: subtle, filled to match the panel so they read as
    # texture rather than as boundaries competing with the state outline
    fig.add_trace(go.Choropleth(
        geojson=counties, locations=county_ids, z=[0] * len(county_ids),
        colorscale=[[0, T["panel"]], [1, T["panel"]]], showscale=False,
        marker_line_color=T["grid"], marker_line_width=.6,
        hoverinfo="skip"))

    # state outline on top, bolder, so it stays the dominant boundary
    fig.add_trace(go.Choropleth(
        locations=["NC"], locationmode="USA-states", z=[0],
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
        marker_line_color=T["line"], marker_line_width=1.4,
        hoverinfo="skip"))

    rest = sites[sites["Station"] != active]
    fig.add_trace(go.Scattergeo(
        lon=rest["Lon"], lat=rest["Lat"], text=rest["Station"],
        mode="markers", marker=dict(size=9, color=T["muted"],
                                    line=dict(color=T["bg"], width=1)),
        hovertemplate="%{text}<extra></extra>", showlegend=False))

    on = sites[sites["Station"] == active]
    fig.add_trace(go.Scattergeo(
        lon=on["Lon"], lat=on["Lat"], text=on["Station"],
        mode="markers+text", textposition="top center",
        textfont=dict(color=T["text"], size=11),
        marker=dict(size=15, color=T["accent"],
                    line=dict(color=T["bg"], width=1.5)),
        hovertemplate="%{text}<extra></extra>", showlegend=False))

    # no scope="usa": that locks in the Albers USA projection, which tilts
    # NC noticeably at this latitude. Mercator keeps the state level.
    fig.update_geos(projection_type="mercator", fitbounds="locations",
                    visible=False, bgcolor="rgba(0,0,0,0)")
    # dragmode left at its default (rather than False) so the geo subplot's
    # native pan-on-drag stays available; scroll-to-zoom is enabled where
    # the chart is rendered, via the plotly_chart config.
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ----------------------------------------------------- region matching ----

# Every Region Matching block title gets a short, always-visible line
# underneath it (st.caption(HINTS[...])) rather than a tooltip: there is
# too little to say about most of them to make hiding it behind a hover
# worthwhile. Non-obvious controls that take Streamlit's own help= (the
# station-source radio, the window slider, the states multiselect) use
# that directly instead. One dict either way, in the same spirit as
# THEMES and VARS. No em dashes: this is UI text, not code comments.
HINTS = {
    "rm_reference_period":
        "Select the period over which the biometeorological statistics "
        "are calculated.",
    "rm_station_source_block":
        "Choose which product supplies the North Carolina station side "
        "of the comparison.",
    "rm_station_source_control":
        "MERRA-2 is the same reanalysis product used to characterise the "
        "regions, which keeps any station-vs-region gap attributable to "
        "climate rather than to comparing two different data sources. "
        "ECONet is the station's own measurements, but only goes back to 2006.",
    "rm_window_block":
        "Choose the part of the year to compare, as a single window or "
        "the full year.",
    "rm_window_control":
        "Example: Jun 1 to Jul 30 is a 60-day summer window. Allowed from "
        "about a month up to the full year.",
    "rm_variables":
        "Six variables make up the default comparison set. THImax and "
        "RHmed are included but unchecked.",
    "rm_map_block":
        "Pick a region, then optionally narrow it to specific states "
        "within it.",
    "rm_map_states":
        "Narrows the region to just these states.",
    "rm_radar":
        "Region average conditions against up to three stations, on a "
        "percentile radar. Automatic best-match search is coming soon.",
    "rm_boxplots":
        "Coming soon: one boxplot per comparison variable, region against "
        "the selected stations.",
}


def _set_rm_window_annual():
    """Runs as an on_change callback, before the slider widget redraws
    (same timing constraint _toggle_all_months documents for months_sel):
    switching Coverage to Annual locks the slider to the full year rather
    than just disabling it at whatever range it last had."""
    if st.session_state.rm_window_mode == "Annual":
        st.session_state.rm_window = (date(2001, 1, 1), date(2001, 12, 31))


def month_scale():
    """A |--Jan--|--Feb--|--Mar--| ruler under the window slider so the
    selected range's position in the year is readable at a glance. Column
    widths are proportional to each month's day count on the same 2001,
    non-leap calendar the slider itself uses, so the tick marks line up
    with where its handles can actually land."""
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cols = " ".join(f"{d}fr" for d in days_in_month)
    cells = "".join(
        f'<div style="border-left:1px solid {T["line"]}; text-align:center; '
        f'color:{T["muted"]}; font-size:.72rem; padding:2px 0;">{m}</div>'
        for m in MONTH_NAMES)
    st.markdown(
        f'<div style="display:grid; grid-template-columns:{cols}; '
        f'border-right:1px solid {T["line"]};">{cells}</div>',
        unsafe_allow_html=True)


def region_map(state_regions, cells, active_region, active_states, height=460):
    """CONUS states filled by USDA-ARS region, with every grid cell drawn
    on top as a fixed backdrop and the current selection highlighted.
    scope="usa": nc_map()'s Mercator choice is to avoid tilting North
    Carolina at that latitude; at the full-country scale Albers (what
    scope="usa" gives Plotly) is the projection that's actually correct.
    """
    region_colors = dict(zip(REGIONS, T["region_colors"]))

    fig = go.Figure()

    for reg in REGIONS:
        sub = state_regions[state_regions["region"] == reg]
        c = region_colors[reg]
        fig.add_trace(go.Choropleth(
            locations=[STATE_ABBR[s] for s in sub["state"]],
            z=[1] * len(sub), locationmode="USA-states",
            colorscale=[[0, c], [1, c]], showscale=False,
            marker_line_color=T["bg"], marker_line_width=.5,
            marker_opacity=1.0 if reg == active_region else .25,
            text=sub["state"], hovertemplate="%{text}<extra></extra>",
            name=reg, showlegend=False))

    # every cell, muted, so the map reads the same regardless of selection
    fig.add_trace(go.Scattergeo(
        lon=cells["lon"], lat=cells["lat"], mode="markers",
        marker=dict(size=3, color=T["muted"], opacity=.5),
        hoverinfo="skip", showlegend=False))

    # the current selection highlighted on top: selected states if any,
    # else the whole active region
    if active_states:
        hi = cells[cells["state"].isin(active_states)]
    else:
        hi = cells[cells["region"] == active_region]
    fig.add_trace(go.Scattergeo(
        lon=hi["lon"], lat=hi["lat"], mode="markers",
        marker=dict(size=4.5, color=T["accent"]),
        hoverinfo="skip", showlegend=False))

    fig.update_geos(scope="usa", bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


def pentad_of_doy(doy):
    """Pentad index (1-73) for a day-of-year on the Window block's fixed
    non-leap reference calendar (2001). Mirrors pentad_of() in
    r-codes/pentad_lib.R exactly (same non-leap case; the Window slider
    can never land on Feb 29, so the leap case never applies here) —
    needed to turn the block's date range into the pentad bounds the
    profile queries below filter on."""
    return (doy - 1) // 5 + 1


# state_pentad.parquet stores PRECTOTCORR and THI_ge_79 as rates (mean
# mm/day, fraction of days), not totals — see GRID_VARS and CLAUDE.md,
# "Store rates, not totals". Decision from block 4: scale these by the
# window length in days at display time, here, rather than showing a
# rate. DTR and interdiurnal_T2M are temperature *differences*; they need
# convert_delta() rather than convert() wherever a profile gets displayed.
RATE_VARS = {"PRECTOTCORR", "THI_ge_79"}
DELTA_VARS = {"DTR", "interdiurnal_T2M"}

_PROFILE_STATS = {
    "mean": np.mean,
    "p5":  lambda x: np.percentile(x, 5),
    "p25": lambda x: np.percentile(x, 25),
    "p50": lambda x: np.percentile(x, 50),
    "p75": lambda x: np.percentile(x, 75),
    "p95": lambda x: np.percentile(x, 95),
}


def _profile_stats(pentad_year, variables, win_days):
    """pentad_year: one row per pentad x year already, one column per
    raw variable name — the "one value per pentad per year" both
    region_profile() and station_profile() build before calling this.
    Returns {variable: {stat: value}}, native rate units still, before
    convert()/convert_delta(): RATE_VARS are scaled by win_days here so
    every caller gets a per-window quantity, not a per-day rate."""
    out = {}
    for v in variables:
        x = pentad_year[v].to_numpy()
        scale = win_days if v in RATE_VARS else 1
        out[v] = {stat: fn(x) * scale for stat, fn in _PROFILE_STATS.items()}
    return out


def region_pentad_year(state_pentad, state_weights, states, p_lo, p_hi, y_lo, y_hi,
                       variables):
    """Area-weighted mean across `states`' state_pentad.parquet rows,
    weighted by each state's summed area_weight (conus_grid.parquet) —
    a bigger, more-cells state pulls the combined series more, the same
    way a multi-state selection should behave. Collapses to one row per
    pentad x year (exactly what Task 9 asks for) and stops there: the
    shared building block behind both region_profile() (pooled stats
    over the whole window) and climate_dissimilarity()/coverage()
    (per-pentad stats)."""
    sub = state_pentad[
        state_pentad["state"].isin(states)
        & state_pentad["pentad"].between(p_lo, p_hi)
        & state_pentad["year"].between(y_lo, y_hi)
    ].copy()
    sub["_w"] = sub["state"].map(state_weights)

    grouped = sub.groupby(["pentad", "year"])
    return pd.DataFrame({
        v: grouped.apply(lambda g, v=v: np.average(g[v], weights=g["_w"]))
        for v in variables
    }).reset_index()   # -> plain "pentad", "year" columns, not a MultiIndex


def region_profile(state_pentad, state_weights, states, p_lo, p_hi, y_lo, y_hi,
                   variables, win_days):
    """Pooled p5/p25/p50/p75/p95/mean over the whole window -- see
    region_pentad_year() for the per-pentad series this collapses."""
    pentad_year = region_pentad_year(state_pentad, state_weights, states,
                                     p_lo, p_hi, y_lo, y_hi, variables)
    return _profile_stats(pentad_year, variables, win_days)


def station_pentad_year(station_pentad, station, p_lo, p_hi, y_lo, y_hi):
    """A station is already a single cell's pentad x year series in
    station_pentad.parquet -- no collapsing step, unlike
    region_pentad_year(), just the same window/reference-period filter."""
    return station_pentad[
        (station_pentad["station"] == station)
        & station_pentad["pentad"].between(p_lo, p_hi)
        & station_pentad["year"].between(y_lo, y_hi)
    ]


def station_profile(station_pentad, station, p_lo, p_hi, y_lo, y_hi,
                    variables, win_days):
    """Pooled p5/p25/p50/p75/p95/mean over the whole window, the exact
    same shape region_profile() returns so the two sides are comparable
    at all."""
    sub = station_pentad_year(station_pentad, station, p_lo, p_hi, y_lo, y_hi)
    return _profile_stats(sub, variables, win_days)


@st.cache_data(show_spinner="Building the national percentile scale")
def conus_percentile_scale(y_lo, y_hi):
    """The fixed yardstick climate_dissimilarity() converts every raw
    p5/p95 onto: for each variable and each pentad, the sorted array of
    all 48 states' state_pentad.parquet values in that pentad, over the
    given reference period. A gap expressed in these units is comparable
    across variables with entirely different native magnitudes (a THI
    point is not a mm). Computed once per reference period and cached
    (Task 10's own instruction), not rebuilt on every score call; reloads
    state_pentad.parquet via load_state_pentad() rather than taking it as
    an argument so Streamlit hashes two ints, not a 122k-row frame."""
    sp = load_state_pentad(STATE_PENTAD_PATH)
    sub = sp[sp["year"].between(y_lo, y_hi)]
    return {
        v: sub.groupby("pentad")[v].apply(lambda s: np.sort(s.to_numpy())).to_dict()
        for v in GRID_VARS
    }


def _percentile_of(value, sorted_arr):
    """Empirical percentile (0-100): share of a variable x pentad's
    CONUS distribution at or below `value`."""
    if len(sorted_arr) == 0:
        return np.nan
    return 100 * np.searchsorted(sorted_arr, value, side="right") / len(sorted_arr)


def climate_dissimilarity(region_py, station_pentad, station, conus_scale,
                          p_lo, p_hi, y_lo, y_hi, variables):
    """Climate analogue dissimilarity: "how alike are these two
    climates" -- the score that ranks the six stations. Replaces what
    this file used to call mess_score(). Per variable:

        |region_p5 - station_p5| + |region_p95 - station_p95|

    both pooled over the whole window (the same single band coverage()
    and the eventual radar use), each side run through every pentad's
    own CONUS percentile scale in turn (a raw value means a different
    thing in January than in July, so the seasonal yardstick still has
    to be per pentad even though the band being measured against it does
    not) and averaged -- not summed -- over the window's pentads, then
    summed across `variables` for the station's total. Cite: Grenier et
    al. 2013, J. Appl. Meteor. Climatol. 52:4, for this choice of metric.

    Why this replaced the old score, for the record: the old one summed
    only the region's uncovered tails (asymmetric, MESS-derived), which
    saturates. Once the region fell fully outside the station's range on
    every pentad, the region's own value cancelled out of the
    *between-station* difference and the score stopped depending on the
    region at all -- interdiurnal_T2M landed on exactly 9.5238 in 29 of
    48 states, driven entirely by which two stations were being
    compared, not by the region. It also ignored over-coverage
    entirely (a station far *wider* than the region scored the same as
    a perfect match), which is why PRECTOTCORR was blind in about half
    the country in every window tested. The symmetric absolute-difference
    form here penalises both a station too narrow to cover the region
    and one much wider than it, and an all-|.| sum of two non-negative,
    unbounded terms does not saturate the same way.

    Coverage -- genuine MESS, Elith, Kearney & Phillips (2010), the
    asymmetric "does the station's range contain the region's" -- is a
    separate function now, coverage(): a different question, not a
    component of this one."""
    sta_py = station_pentad_year(station_pentad, station, p_lo, p_hi, y_lo, y_hi)

    out = {"total": 0.0}
    for v in variables:
        sta_p5, sta_p95 = np.percentile(sta_py[v], [5, 95])
        region_p5, region_p95 = np.percentile(region_py[v], [5, 95])

        pentad_dissim = []
        for p in range(p_lo, p_hi + 1):
            scale = conus_scale[v].get(p)
            if scale is None or len(scale) == 0:
                continue
            r5_pct  = _percentile_of(region_p5, scale)
            r95_pct = _percentile_of(region_p95, scale)
            s5_pct  = _percentile_of(sta_p5, scale)
            s95_pct = _percentile_of(sta_p95, scale)
            pentad_dissim.append(abs(r5_pct - s5_pct) + abs(r95_pct - s95_pct))

        dissim = float(np.mean(pentad_dissim)) if pentad_dissim else 0.0
        out[v] = {"dissimilarity": dissim}
        out["total"] += dissim

    return out


def coverage(region_py, station_pentad, station, p_lo, p_hi, y_lo, y_hi, variables):
    """Genuine MESS (Elith, Kearney & Phillips 2010): "does the station's
    range contain the region's?" Reference = station, projection =
    region (CLAUDE.md), asymmetric by design -- unlike
    climate_dissimilarity(), over-coverage (station wider than the
    region) scores the same as a perfect match. Per variable, the pooled
    fraction of the region's pentad x year values that fall inside the
    station's own pooled p5-p95 for the whole window. Feeds the radar's
    red segments (region_radar()); a separate question from
    climate_dissimilarity(), not a component of it, so it is not summed
    into a total here."""
    sta_py = station_pentad_year(station_pentad, station, p_lo, p_hi, y_lo, y_hi)

    out = {}
    for v in variables:
        sta_p5, sta_p95 = np.percentile(sta_py[v], [5, 95])
        region_vals = region_py[v].to_numpy()
        out[v] = float(np.mean((region_vals >= sta_p5) & (region_vals <= sta_p95)))

    return out


def _avg_conus_percentile(value, variable, conus_scale, p_lo, p_hi):
    """A single raw value's CONUS percentile rank, averaged across every
    pentad in the window. Positions a pooled p5/p95 on the radar's radial
    axis the same way climate_dissimilarity() runs a pooled value through
    every pentad's own scale in turn -- a raw value means a different
    thing in January than in July -- but returns the value's own average
    position rather than an already-differenced gap, since the radar
    needs to place two separate rings, not one number."""
    pcts = []
    for p in range(p_lo, p_hi + 1):
        scale = conus_scale[variable].get(p)
        if scale is not None and len(scale):
            pcts.append(_percentile_of(value, scale))
    return float(np.mean(pcts)) if pcts else np.nan


# Percentile 0 is deliberately not the radar's true centre: a hole big
# enough that an uncovered inner tail near the national 0th percentile
# stays readable instead of collapsing into an unreadable point.
RADAR_HOLE = 0.25
RADAR_MAX = 1.0


def _pct_to_r(pct):
    return RADAR_HOLE + (RADAR_MAX - RADAR_HOLE) * pct / 100


# Off for now: a real but tiny gap (well under 1-2 percentile points --
# e.g. GOLD vs North Carolina on T2M/DTR/T2MDEW) draws as two overlapping
# end-cap markers with no visible line between them, which reads as a
# stray dot/error rather than "basically covered," on exactly the axes
# where the station is a good match. Needs a minimum-gap threshold or a
# different visual language (e.g. a thin tick instead of a thick capped
# segment) before turning back on -- the geometry itself is correct
# (verified against the raw coverage()/percentile numbers).
RADAR_SHOW_ALERTS = False


def region_radar(variables, region_py, station_pentad, stations, focus,
                 conus_scale, p_lo, p_hi, y_lo, y_hi, height=520):
    """The percentile radar. One axis per variable, fixed order (the
    caller's `variables` order, not selection order, so it doesn't
    reshuffle between reruns). Radial scale is the CONUS percentile,
    0-100, with RADAR_HOLE's hole at the centre -- see there. Region
    average conditions is a filled, low-opacity, thin-outlined band from
    its pooled p5 to its pooled p95; each of `stations` (up to three,
    STATION_COLORS) is the same band unfilled, dashed on both edges, so
    it reads as an envelope meant to enclose the region. Alert segments —
    thick, round-capped (faked with matching end markers; Plotly line
    traces don't expose a cap style), in the accent colour, both tails,
    driven by coverage()'s own asymmetric direction (region beyond
    station) rather than climate_dissimilarity()'s symmetric gap, so a
    station much wider than the region draws no segment — are built but
    gated off by RADAR_SHOW_ALERTS (see there for why: a real but tiny
    gap reads as a stray dot, not an alert, on exactly the axes where a
    station is a good match). An axis where neither the region nor any
    shown station varies at all is greyed (CLAUDE.md, "no meaningful
    variation ... on either side"). All colours from THEMES/STATION_COLORS;
    nothing hard-coded."""
    n = len(variables)
    angles = [i * 360 / n for i in range(n)]

    def pooled_pct(py, v):
        lo, hi = np.percentile(py[v], [5, 95])
        return (_avg_conus_percentile(lo, v, conus_scale, p_lo, p_hi),
                _avg_conus_percentile(hi, v, conus_scale, p_lo, p_hi))

    region_pct = {v: pooled_pct(region_py, v) for v in variables}

    station_py = {stn: station_pentad_year(station_pentad, stn, p_lo, p_hi, y_lo, y_hi)
                 for stn in stations}
    station_pct = {stn: {v: pooled_pct(station_py[stn], v) for v in variables}
                  for stn in stations}

    greyed = set()
    for v in variables:
        region_flat = np.isclose(*np.percentile(region_py[v], [5, 95]))
        stations_flat = all(np.isclose(*np.percentile(station_py[stn][v], [5, 95]))
                            for stn in stations) if stations else True
        if region_flat and stations_flat:
            greyed.add(v)

    fig = go.Figure()

    # Region average conditions: two independently CLOSED polygons (same
    # construction as each station's outer/inner loops below, which is
    # why those close cleanly on every axis), with Plotly filling the gap
    # between consecutive traces via fill="tonext" rather than one path
    # threading both boundaries with a seam. Two earlier attempts built a
    # single outer+inner path instead (closing the outer loop before
    # jumping to inner, then removing that closing point) and each moved
    # a rendering seam to a different vertex rather than removing it --
    # tonext between two separately closed loops has no seam to place.
    # Order matters: the inner (no-fill) loop must be added first, so the
    # outer trace right after it has something to fill "toward".
    outer = [_pct_to_r(region_pct[v][1]) for v in variables]
    inner = [_pct_to_r(region_pct[v][0]) for v in variables]
    theta_closed = angles + [angles[0]]
    fig.add_trace(go.Scatterpolar(
        r=inner + [inner[0]], theta=theta_closed, mode="lines",
        line=dict(color=T["accent"], width=1.5),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatterpolar(
        r=outer + [outer[0]], theta=theta_closed, mode="lines", fill="tonext",
        fillcolor=T["accent_soft"], opacity=0.6,
        line=dict(color=T["accent"], width=1.5),
        name="Region average conditions", hoverinfo="skip"))

    for i, stn in enumerate(stations):
        c = STATION_COLORS[i % len(STATION_COLORS)]
        outer_s = [_pct_to_r(station_pct[stn][v][1]) for v in variables]
        inner_s = [_pct_to_r(station_pct[stn][v][0]) for v in variables]
        theta_closed = angles + [angles[0]]
        fig.add_trace(go.Scatterpolar(
            r=outer_s + [outer_s[0]], theta=theta_closed, mode="lines",
            line=dict(color=c, width=2, dash="dash"),
            name=stn, legendgroup=stn, hoverinfo="skip"))
        fig.add_trace(go.Scatterpolar(
            r=inner_s + [inner_s[0]], theta=theta_closed, mode="lines",
            line=dict(color=c, width=2, dash="dash"),
            name=stn, legendgroup=stn, showlegend=False, hoverinfo="skip"))

    if RADAR_SHOW_ALERTS and focus in station_pct:
        for i, v in enumerate(variables):
            r5_pct, r95_pct = region_pct[v]
            s5_pct, s95_pct = station_pct[focus][v]
            lo_gap = s5_pct - r5_pct     # region reaches below the station's floor
            hi_gap = r95_pct - s95_pct   # region reaches above the station's ceiling
            for gap, a, b in ((lo_gap, r5_pct, s5_pct), (hi_gap, s95_pct, r95_pct)):
                if gap > 0:
                    r_ab = [_pct_to_r(a), _pct_to_r(b)]
                    fig.add_trace(go.Scatterpolar(
                        r=r_ab, theta=[angles[i], angles[i]], mode="lines+markers",
                        line=dict(color=T["accent"], width=7),
                        marker=dict(color=T["accent"], size=8, symbol="circle"),
                        showlegend=False, hoverinfo="skip"))

    ticktext = [
        f'<span style="color:{T["muted"]}">{GRID_VARS[v]["label"]}</span>'
        if v in greyed else GRID_VARS[v]["label"]
        for v in variables
    ]
    fig.update_layout(
        height=height, showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"]),
        legend=dict(orientation="h", yanchor="top", y=-0.1, x=0),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, RADAR_MAX], showline=True, linecolor=T["line"],
                tickvals=[RADAR_HOLE, _pct_to_r(50), RADAR_MAX],
                ticktext=["0", "50", "100"],
                gridcolor=T["line"], tickfont=dict(color=T["muted"])),
            angularaxis=dict(
                tickvals=angles, ticktext=ticktext,
                direction="clockwise", rotation=90,
                gridcolor=T["line"], linecolor=T["line"],
                tickfont=dict(color=T["text"])),
        ),
        margin=dict(l=40, r=40, t=30, b=10))
    return fig


# ================================================================ sections ==

if section == "Overview":
    st.title("Overview")
    st.caption(f"{years[0]}\u2013{years[1]} \u00b7 {months_label}")

    with st.container(key="ov_section_station"), st.expander("Selected station", expanded=True):
        # Station picker for the KPIs only
        cols = st.columns(len(all_stations))
        for i, stn in enumerate(all_stations):
            if cols[i].button(stn, key=f"kpi_{stn}", width=W,
                              type="primary"
                              if st.session_state.kpi_station == stn
                              else "secondary"):
                st.session_state.kpi_station = stn
                st.rerun()

        active = st.session_state.kpi_station
        one = df[df["Station"] == active]
        n_years = max(one["Year"].nunique(), 1)

        left, right = st.columns([3, 2], gap="large")

        with left, st.container(key="ov_kpis"):
            r1 = st.columns(2)
            r1[0].metric(f"Mean max temperature{unit_suffix('temp', metric)}",
                         f"{convert(one['Tmax_C'].mean(), 'temp', metric):.1f}")
            r1[1].metric(f"Mean min temperature{unit_suffix('temp', metric)}",
                         f"{convert(one['Tmin_C'].mean(), 'temp', metric):.1f}")

            r2 = st.columns(2)
            r2[0].metric(f"Rainfall{unit_suffix('precip', metric)}",
                         f"{convert(one['Precip_mm'].sum() / n_years, 'precip', metric):,.0f}",
                         delta=f"per {PERIOD}", delta_color="off", delta_arrow="off")
            r2[1].metric(f"Mean relative humidity{unit_suffix('pct', metric)}",
                         f"{one['RHmed'].mean():.1f}")

            r3 = st.columns(2)
            r3[0].metric("Mean THImax", f"{one['THImax'].mean():.1f}")
            r3[1].metric("Mean THImin", f"{one['THImin'].mean():.1f}")

            r4 = st.columns(3)
            with r4[0], st.container(key="ov_thi75"):
                st.metric("Days THImax \u2265 75 (Alert)",
                         f"{one['THI75'].sum() / n_years:.0f}",
                         delta=f"per {PERIOD}", delta_color="off", delta_arrow="off")
            with r4[1], st.container(key="ov_thi79"):
                st.metric("Days THImax \u2265 79 (Danger)",
                         f"{one['THI79'].sum() / n_years:.0f}",
                         delta=f"per {PERIOD}", delta_color="off", delta_arrow="off")
            with r4[2], st.container(key="ov_thi84"):
                st.metric("Days THImax \u2265 84 (Emergency)",
                         f"{one['THI84'].sum() / n_years:.0f}",
                         delta=f"per {PERIOD}", delta_color="off", delta_arrow="off")

        with right:
            st.plotly_chart(nc_map(active), width=W,
                            config={"displayModeBar": True, "scrollZoom": True,
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": [
                                        "toImage", "pan2d", "select2d", "lasso2d",
                                        "resetViewMapbox", "hoverClosestGeo",
                                    ]})
            # The zoom in/out buttons compute their new scale from
            # geo.projection.scale, but a fitbounds-only view never sets
            # that until the chart handles its first scroll or drag — so a
            # button click as the very first interaction is a silent no-op.
            # A single imperceptibly small synthetic scroll, right after
            # mount, is enough for Plotly to populate it correctly (a
            # manual relayout guess is the wrong unit space here — this
            # goes through Plotly's own zoom math instead), so the buttons
            # work on the first real click too.
            st.html("""<script>
            (function() {
                var tries = 0;
                // wait for Plotly to finish attaching its own interaction
                // handlers, not just for the geo layout object to exist —
                // dispatching the nudge too early is a silent no-op.
                setTimeout(function() {
                    var iv = setInterval(function() {
                        tries++;
                        var gd = document.querySelector(
                            'div[data-testid="stPlotlyChart"] .js-plotly-plot');
                        var geo = gd && gd._fullLayout && gd._fullLayout.geo;
                        if (geo && geo.projection.scale !== undefined) {
                            clearInterval(iv);
                        } else if (geo) {
                            var r = gd.getBoundingClientRect();
                            var cx = r.left + r.width / 2, cy = r.top + r.height / 2;
                            var target = document.elementFromPoint(cx, cy) || gd;
                            target.dispatchEvent(new WheelEvent("wheel", {
                                bubbles: true, cancelable: true,
                                clientX: cx, clientY: cy, deltaY: -1, deltaMode: 0,
                            }));
                        }
                        if (tries > 20) clearInterval(iv);
                    }, 200);
                }, 900);
            })();
            </script>""", unsafe_allow_javascript=True)

    with st.container(key="ov_section_table"), st.expander("Comparison across stations", expanded=True):
        tbl_stations = st.session_state.get("ov_stations") or all_stations
        tbl_vars = st.session_state.get("ov_vars") or list(VARS)

        rows = []
        for v in tbl_vars:
            meta = VARS[v]
            rec = {"Variable": f"{meta['label']}{unit_suffix(meta['kind'], metric)}"}
            for stn in tbl_stations:
                sub = df[df["Station"] == stn]
                ny = max(sub["Year"].nunique(), 1)
                x = sub[v].sum() / ny if meta["agg"] == "sum" else sub[v].mean()
                rec[stn] = round(convert(x, meta["kind"], metric), 1)
            rows.append(rec)
        summary = pd.DataFrame(rows)

        st.dataframe(summary, width=W, hide_index=True)

        d1, d2 = st.columns([1, 3])
        d1.download_button("Download CSV", summary.to_csv(index=False),
                           file_name="summary_table.csv", mime="text/csv")
        with d2.expander("Copy as tab-separated"):
            st.code(summary.to_csv(index=False, sep="\t"), language=None)

        st.multiselect("Variables in table", list(VARS), default=list(VARS),
                       format_func=lambda v: VARS[v]["label"], key="ov_vars")
        st.multiselect("Stations in table", all_stations, default=all_stations,
                       key="ov_stations")

    imputation_panel(df)


elif section == "Time series":
    st.title("Time series")

    # variable, overlay, stations and trend rate live on the page, not the
    # sidebar: a symmetric two-row grid, a dropdown and a control per row.
    c1, c2 = st.columns([2, 1])
    var = c1.selectbox("Variable", list(VARS), index=0,
                       format_func=lambda v: VARS[v]["label"])
    with c2:
        overlay = st.segmented_control("Overlay", ["Trend", "Average"],
                                       key="ts_overlay")
    show_trend = overlay == "Trend"
    show_mean = overlay == "Average"

    c3, c4 = st.columns([2, 1])
    with c3:
        st.multiselect("Stations in chart", all_stations, default=all_stations,
                       key="ts_stations")
    with c4:
        trend_mult, trend_suffix = trend_rate_control("ts_trend_unit")

    ts_stations = st.session_state.get("ts_stations") or all_stations

    meta = VARS[var]
    tbl = yearly_frame(df, var)
    tbl[var] = convert(tbl[var], meta["kind"], metric).round(2)
    ylab = (f"{meta['label']} per {PERIOD}" if meta["kind"] == "days"
           else f"{meta['label']}{unit_suffix(meta['kind'], metric)}")

    st.caption(f"{ylab} \u00b7 {years[0]}\u2013{years[1]} \u00b7 {months_label}")

    fig = line_by_station(tbl, var, ylab, ts_stations, show_trend, show_mean)
    chart_or_table(fig, tbl.rename(columns={var: meta["label"]}),
                   key="ts", filename=f"{var}_by_year.csv", height=470)

    k = st.columns(max(len(ts_stations), 1))
    for i, stn in enumerate(ts_stations):
        d = tbl[tbl["Station"] == stn].sort_values("Year")
        if len(d) < 3:
            continue
        slope = np.polyfit(d["Year"], d[var], 1)[0] * trend_mult
        k[i].metric(f"{stn} \u00b7 average", f"{d[var].mean():.1f}",
                    f"Trend {slope:+.2f}/{trend_suffix}", delta_color="off",
                    delta_arrow="up" if slope >= 0 else "down")

    st.markdown("### Distribution across years")
    box_tbl = tbl[tbl["Station"].isin(ts_stations)]
    box = px.box(box_tbl, x="Station", y=var, color="Station",
                 color_discrete_map=color_map, points="all")
    box.update_layout(showlegend=False)
    box.update_yaxes(title=ylab)
    st.plotly_chart(style_fig(box, T, 380), width=W)

    imputation_panel(df)
    if var in ("THI75", "THI79", "THI84"):
        st.caption("Thresholds follow the Livestock Weather Safety Index: 75 "
                   "alert, 79 danger, 84 emergency. THI was calculated using "
                   "dew point temperature as the source of humidity "
                   "(THI = Air temperature + 0.36 · Dew point + 41.2).")


elif section == "Anomalies":
    st.title("Anomalies")

    # variable and trend-rate controls live on the page, not the sidebar,
    # in the same two-row grid Time series uses: a dropdown and a control
    # per row, both rows the same width.
    c1, c2 = st.columns([2, 1])
    var = c1.selectbox("Variable", list(VARS), index=1,
                       format_func=lambda v: VARS[v]["label"])
    with c2:
        trend_mult, trend_suffix = trend_rate_control("anom_trend_unit")

    c3, c4 = st.columns([2, 1])
    with c3:
        st.multiselect("Stations", all_stations, default=[all_stations[0]],
                       key="anom_stations")
    stns = st.session_state.get("anom_stations") or [all_stations[0]]
    with c4:
        # There's no second button row here the way Time series has trend
        # rate; the space instead explains what changes once more than one
        # station is in the mix. Nudged down to line up with the multiselect
        # box itself, not the "Stations" label above it.
        if len(stns) > 1:
            with st.container(key="anom_stations_note"):
                st.caption("Anomalies are averaged across the selected stations.")

    meta = VARS[var]
    as_pct = (st.toggle("Show as percent of baseline", value=False)
              if meta["kind"] == "precip" else False)

    base_src = data[data["Station"].isin(stns)
                    & data["Year"].between(*BASE_YEARS)
                    & data["Month"].isin(months)]
    # per station-year first, honouring each variable's own sum/average
    # rule, then averaged across the selected stations for each year, so
    # a total like rainfall doesn't get accumulated across stations.
    baseline = aggregate(base_src, var, ["Station", "Year"])[var].mean()

    per_station = aggregate(df[df["Station"].isin(stns)], var, ["Station", "Year"])
    cur = per_station.groupby("Year", as_index=False, observed=True)[var].mean()
    cur["anom"] = cur[var] - baseline
    if as_pct:
        cur["anom"] = 100 * cur["anom"] / baseline
        unit = "%"
    else:
        cur["anom"] = convert_delta(cur["anom"], meta["kind"], metric)
        unit = unit_label(meta["kind"], metric)

    pos, neg = ANOM_PRECIP if meta["kind"] == "precip" else ANOM_TEMP
    stn_label = stns[0] if len(stns) == 1 else f"{len(stns)} stations"
    st.caption(f"{stn_label} \u00b7 {meta['label']} \u00b7 baseline "
               f"{BASE_YEARS[0]}\u2013{BASE_YEARS[1]} over {months_label}")

    fig = go.Figure(go.Bar(
        x=cur["Year"], y=cur["anom"],
        marker_color=[pos if v >= 0 else neg for v in cur["anom"]],
        hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=T["muted"], width=1))
    fig.update_yaxes(title=f"Anomaly ({unit})" if unit else "Anomaly")
    fig.update_xaxes(title=None, dtick=2)

    left, right = st.columns([3, 1])
    with left:
        st.plotly_chart(style_fig(fig, T, 430), width=W)
    with right:
        st.metric("Window average anomaly",
                  f"{cur['anom'].mean():+.2f} {unit}".strip())
        if len(cur) > 2:
            slope = np.polyfit(cur["Year"], cur["anom"], 1)[0] * trend_mult
            st.metric("Trend", f"{slope:+.2f} {unit}/{trend_suffix}".strip())
        st.metric(f"Reference{unit_suffix(meta['kind'], metric)}",
                  f"{convert(baseline, meta['kind'], metric):.1f}",
                  delta=f"Reference period: {BASE_YEARS[0]}–{BASE_YEARS[1]}",
                  delta_color="off", delta_arrow="off")

    imputation_panel(df[df["Station"].isin(stns)])
    st.markdown(
        f'<div class="infobox">Anomalies here are computed against this '
        f"dataset's own {BASE_YEARS[0]}–{BASE_YEARS[1]} average (20 years), "
        f'not the standard 1991–2020 climatological reference period.</div>',
        unsafe_allow_html=True)


elif section == "Region Matching":
    st.title("Region Matching")

    # Independent of the sidebar-filtered `df` (and of `data`) entirely:
    # this section answers a national question the six-station ECONet
    # frame can't, over its own pre-aggregated input. See CLAUDE.md, "The
    # Region matching section", for the full block-by-block design.
    if (not STATE_PENTAD_PATH.exists() or not CONUS_GRID_PATH.exists()
            or not STATION_PENTAD_PATH.exists()):
        st.error("state_pentad.parquet, conus_grid.parquet and/or "
                 "station_pentad.parquet not found next to this script. "
                 "See CLAUDE.md's R preprocessing section for how they're "
                 "built.")
        st.stop()

    state_pentad = load_state_pentad(STATE_PENTAD_PATH)
    grid = load_conus_grid(CONUS_GRID_PATH)
    station_pentad = load_station_pentad(STATION_PENTAD_PATH)
    state_regions = grid.drop_duplicates("state")[["state", "region"]]
    state_weights = grid.groupby("state")["area_weight"].sum()

    with st.container(key="rm_block_reference"), \
         st.expander("1. Reference period", expanded=True):
        st.caption(HINTS["rm_reference_period"])
        rm_ref_years = st.slider("Years", 1991, 2025, (1991, 2025),
                                 key="rm_ref_years")
        if rm_ref_years[1] - rm_ref_years[0] + 1 < 20:
            st.warning("Under 20 years, the interannual variability "
                      "diagnostic (block 6) stops meaning much.")

    with st.container(key="rm_block_source"), \
         st.expander("2. NC stations data source", expanded=True):
        st.caption(HINTS["rm_station_source_block"])
        rm_source = st.radio(
            "NC stations data source", ["MERRA-2 (recommended)", "ECONet"],
            key="rm_source", horizontal=True, label_visibility="collapsed",
            help=HINTS["rm_station_source_control"])
        if rm_source == "ECONet" and rm_ref_years[0] < 2006:
            st.warning("ECONet station records only go back to 2006. Move "
                      "the reference period's start year to 2006 or later, "
                      "or switch back to MERRA-2.")

    with st.container(key="rm_block_window"), \
         st.expander("3. Time window exploration", expanded=True):
        st.caption(HINTS["rm_window_block"])

        st.session_state.setdefault("rm_window_mode", "Subannual")
        st.radio("Coverage", ["Annual", "Subannual"], key="rm_window_mode",
                 horizontal=True, on_change=_set_rm_window_annual)
        is_annual = st.session_state.rm_window_mode == "Annual"

        # No value= here: _set_rm_window_annual() writes rm_window's
        # session_state directly on toggle, and Streamlit warns (a real
        # policy check, not just style) if a keyed widget gets both an
        # explicit value and a pre-existing session_state entry. setdefault
        # supplies the one-time initial value instead, same as months_sel
        # and trend_rate_control's key do it elsewhere in this file.
        st.session_state.setdefault(
            "rm_window", (date(2001, 6, 1), date(2001, 7, 30)))
        rm_window = st.slider(
            "Window", min_value=date(2001, 1, 1), max_value=date(2001, 12, 31),
            step=timedelta(days=10), format="MMM D",
            key="rm_window", label_visibility="collapsed",
            disabled=is_annual, help=HINTS["rm_window_control"])
        if is_annual:
            rm_window = (date(2001, 1, 1), date(2001, 12, 31))
        month_scale()

        win_days = (rm_window[1] - rm_window[0]).days + 1
        if win_days < 28:
            st.warning("Window is under a month. About 30 days is the "
                      "shortest supported.")

    with st.container(key="rm_block_variables"), \
         st.expander("4. Variables", expanded=True):
        st.caption(HINTS["rm_variables"])

        # A multiselect, not a checkbox row: consistent with how every
        # other section in this app (ts_stations, anom_stations,
        # data_stations) already lets the user pick a subset from a
        # named list, and 8 options read tighter as one dropdown than as
        # 8 boxes at this column width.
        default_vars = [v for v, m in GRID_VARS.items() if m["default"]]
        rm_vars = st.multiselect(
            "Variables", list(GRID_VARS), default=default_vars,
            format_func=lambda v: GRID_VARS[v]["label"], key="rm_vars")
        if len(rm_vars) < 3:
            st.warning("Fewer than three variables selected. The "
                      "comparison degenerates below that.")

    with st.container(key="rm_block_map"), \
         st.expander("5. Region and states", expanded=True):
        st.caption(HINTS["rm_map_block"])

        st.session_state.setdefault("rm_region", REGIONS[0])
        rm_region = st.segmented_control("Region", REGIONS, key="rm_region")

        region_states = sorted(
            state_regions.loc[state_regions["region"] == rm_region, "state"])
        st.multiselect("States", region_states, default=region_states,
                       key=f"rm_states_{rm_region}",
                       help=HINTS["rm_map_states"])
        rm_states = st.session_state.get(f"rm_states_{rm_region}") or []

        if rm_states:
            n_cells = len(grid[grid["state"].isin(rm_states)])
            sel_label = f"{len(rm_states)} state{'s' if len(rm_states) != 1 else ''} selected"
        else:
            n_cells = len(grid[grid["region"] == rm_region])
            sel_label = f"{rm_region} (all {len(region_states)} states)"
        st.caption(f"{sel_label} · {n_cells:,} grid cells")

        st.plotly_chart(region_map(state_regions, grid, rm_region, rm_states),
                        width=W, config={"displayModeBar": False})

    with st.container(key="rm_block_radar"), \
         st.expander("6. Radar and station selection", expanded=True):
        st.caption(HINTS["rm_radar"])

        if rm_source == "ECONet":
            st.warning("The station profile below always uses the grid "
                      "cell (MERRA-2), regardless of this setting. An "
                      "ECONet-measurements pentad aggregation does not "
                      "exist yet.")

        if len(rm_vars) == 0:
            st.info("Select at least one variable in block 4 to see the radar.")
        else:
            p_lo = pentad_of_doy(rm_window[0].timetuple().tm_yday)
            p_hi = pentad_of_doy(rm_window[1].timetuple().tm_yday)
            y_lo, y_hi = rm_ref_years
            selected_states = rm_states if rm_states else region_states
            all_stations = sorted(station_pentad["station"].unique())
            # Fixed order every render: the caller's declared GRID_VARS order,
            # not multiselect selection order, which can reshuffle on rerun.
            ordered_vars = [v for v in GRID_VARS if v in rm_vars]

            region_py = region_pentad_year(state_pentad, state_weights,
                                           selected_states, p_lo, p_hi,
                                           y_lo, y_hi, ordered_vars)
            scale = conus_percentile_scale(y_lo, y_hi)

            radar_col, control_col = st.columns([3, 2])

            with control_col:
                st.session_state.setdefault("rm_focus_stations", [all_stations[0]])
                st.multiselect("Stations (up to 3)", all_stations, max_selections=3,
                               key="rm_focus_stations")
                focus_stations = st.session_state.get("rm_focus_stations") or []

                # The focus picker only matters for the alert segments,
                # currently off (RADAR_SHOW_ALERTS) -- showing a control
                # with no visible effect would be worse than not showing
                # it, so it's gated the same way.
                if RADAR_SHOW_ALERTS and len(focus_stations) > 1:
                    if st.session_state.get("rm_focus") not in focus_stations:
                        st.session_state.rm_focus = focus_stations[0]
                    st.radio("Focus (alert segments)", focus_stations, key="rm_focus")
                    focused = st.session_state.rm_focus
                elif focus_stations:
                    focused = focus_stations[0]
                else:
                    focused = None

            with radar_col:
                fig = region_radar(ordered_vars, region_py, station_pentad,
                                   focus_stations, focused, scale,
                                   p_lo, p_hi, y_lo, y_hi)
                st.plotly_chart(fig, width=W, config={"displayModeBar": False})
                st.caption(f"{p_lo}–{p_hi} pentads ({win_days} days) · "
                          f"{y_lo}–{y_hi} · {len(selected_states)} state"
                          f"{'s' if len(selected_states) != 1 else ''} in region "
                          f"average conditions")

            with control_col:
                if focus_stations:
                    cov_by_stn = {
                        stn: coverage(region_py, station_pentad, stn,
                                     p_lo, p_hi, y_lo, y_hi, ordered_vars)
                        for stn in focus_stations
                    }
                    dis_by_stn = {
                        stn: climate_dissimilarity(region_py, station_pentad, stn, scale,
                                                  p_lo, p_hi, y_lo, y_hi, ordered_vars)
                        for stn in focus_stations
                    }
                    rows = []
                    for v in ordered_vars:
                        rec = {"Variable": GRID_VARS[v]["label"]}
                        for stn in focus_stations:
                            rec[f"{stn} coverage"] = round(cov_by_stn[stn][v], 2)
                            rec[f"{stn} contribution"] = round(
                                dis_by_stn[stn][v]["dissimilarity"], 1)
                        rows.append(rec)
                    st.dataframe(pd.DataFrame(rows), width=W, hide_index=True)
                    st.caption("Coverage: share of region average conditions inside "
                              "the station's band (MESS). Contribution: that "
                              "variable's share of climate_dissimilarity's total.")

    with st.container(key="rm_block_boxplots"), \
         st.expander("7. Boxplots", expanded=False):
        st.caption(HINTS["rm_boxplots"])


# "Data" is hidden from NAV (Region Matching took its slot) but this branch
# and everything in it is untouched and still reachable — put "Data" back
# in NAV to restore it.
elif section == "Data":
    st.title("Data")

    # Station filter lives on the page, not the sidebar, matching the
    # other sections.
    st.multiselect("Stations", all_stations, default=all_stations,
                   key="data_stations")
    stns = st.session_state.get("data_stations") or all_stations

    show = df[df["Station"].isin(stns)].copy()
    st.caption(f"{len(show):,} records \u00b7 {years[0]}\u2013{years[1]} "
               f"\u00b7 {months_label}")

    for v, meta in VARS.items():
        if meta["kind"] in ("temp", "precip") and v in show:
            show[v] = convert(show[v], meta["kind"], metric).round(2)

    st.dataframe(show, width=W, hide_index=True, height=560)
    st.download_button("Download CSV", show.to_csv(index=False),
                       file_name="daily_records_filtered.csv", mime="text/csv")


else:  # Info — dropped from NAV for now, kept here in case it comes back
    st.title("Info")

    st.markdown("#### Stations")
    stations = (data[["Station", "StationName", "City", "County",
                      "Lat", "Lon", "Elev_m"]]
                .drop_duplicates("Station").sort_values("Station"))
    first_obs = (data[~data["Imputed"]].groupby("Station", observed=True)["Date"]
                 .min().dt.date.reset_index(name="First measured day"))
    pct = (data.groupby("Station", observed=True)["Imputed"].mean()
           .mul(100).round(1).reset_index(name="% imputed"))
    st.dataframe(stations.merge(first_obs, on="Station").merge(pct, on="Station"),
                 width=W, hide_index=True)

    st.markdown("#### Where the numbers come from")
    st.markdown(f"""
Daily records come from the North Carolina Environment and Climate Observing
Network ([ECONet]({URL_NETWORK})), run by the
[State Climate Office of North Carolina]({URL_OFFICE}). The window is
{BASE_YEARS[0]}\u2013{BASE_YEARS[1]}, complete calendar years only.

Gaps were filled rather than dropped. For temperature, dew point and humidity,
[MERRA-2]({URL_MERRA}) reanalysis was calibrated against the stations themselves
using hierarchical additive models: one shared correction curve for the network
plus a penalised per-station deviation, so a site with a short record borrows
strength from the others. Rainfall used empirical quantile mapping by
station and month, which corrects the reanalysis habit of drizzling on far too
many days while missing the heavy tail.

THI was calculated using dew point temperature as the source of humidity
(THI = Air temperature + 0.36\u00b7Dew point + 41.2). Thresholds of 75, 79
and 84 follow the Livestock Weather Safety Index.

One station has a much shorter record than the rest, and its early years are
modelled rather than measured. Those years carry the interannual signal of the
reanalysis, not of the site. Every view reports how much of what is on screen
is imputed, and heavily imputed years are drawn with open markers.
""")