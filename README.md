# Biometeorological Data Explorer

Dashboard over daily meteorological records from six weather stations in North
Carolina, 2006-2025. Temperature, rainfall and heat-stress summaries by site,
with trends, anomalies and table exports.

## Data

NC ECONet, State Climate Office of North Carolina
(https://econet.climate.ncsu.edu). Gaps were filled with MERRA-2 reanalysis
(https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/) calibrated against each
station. Every view reports how much of what is shown is imputed.

Calibration used generalised additive mixed models (R 4.5.3, mgcv 1.9-4), one
per variable across the whole network, with a shared correction curve and a
penalised per-station deviation. Rainfall used empirical quantile mapping by
station and month.

## Heat stress

Temperature-Humidity Index (THI) was calculated using dew point as source of humidity in the formula:

    THImax  = Tmax + 0.36 * Tdew + 41.2
    THImin  = Tmin + 0.36 * Tdew + 41.2
    THImean = (THImax + THImin) / 2

where Tdew is the daily average dew point. Day counts are based on THImax.

Thresholds for Heat Stress intensity of 75 (Alert), 79 (Danger) and 84 (Emergency) follow the Livestock
Weather Safety Index. The threshold were applied to the THImax (i.e. maximum daily intensity).

## Anomalies

Anomalies are computed against the 2006-2025 average of this dataset, not
against the 1991-2020 climatological reference period. The baseline is
recomputed over whichever months are selected.

## Running locally

    pip install -r requirements.txt
    streamlit run biomet_app.py