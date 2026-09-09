# Biometeorological Data Explorer

Dashboard over daily meteorological records from six weather stations in North
Carolina. Temperature, rainfall and heat-stress summaries by site, with trends,
anomalies and table exports, plus a Region matching section that extends the
comparison to the contiguous United States.

## Data

*Overview, Time series and Anomalies* use NC ECONet, State Climate Office of
North Carolina (https://econet.climate.ncsu.edu), 2006-2025, for the six
stations only. Gaps were filled with MERRA-2 reanalysis
(https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/) calibrated against each
station. Every view reports how much of what is shown is imputed.

Calibration used generalised additive mixed models (R 4.5.3, mgcv 1.9-4), one
per variable across the whole network, with a shared correction curve and a
penalised per-station deviation. Rainfall used empirical quantile mapping by
station and month.

*Region matching* uses the same MERRA-2 reanalysis, 1991-2025, retrieved
through NASA POWER (https://power.larc.nasa.gov), over 2863 cells at
0.625 x 0.5 degrees covering the contiguous United States.

## Heat stress

Temperature-Humidity Index (THI) was calculated using dew point as source of
humidity in the formula:

    THImax  = Tmax + 0.36 * Tdew + 41.2
    THImin  = Tmin + 0.36 * Tdew + 41.2
    THImean = (THImax + THImin) / 2

where Tdew is the daily average dew point. Day counts are based on THImax.

Thresholds for Heat Stress intensity of 75 (Alert), 79 (Danger) and 84
(Emergency) follow the Livestock Weather Safety Index. The threshold were
applied to the THImax (i.e. maximum daily intensity).

## Anomalies

Anomalies are computed against the 2006-2025 average of this dataset, not
against the 1991-2020 climatological reference period. The baseline is
recomputed over whichever months are selected.

## Region matching

Which North Carolina station is extrapolable to which part of the country, and
in which part of the year.

*Station reach* colours every grid cell by how different its climate is from a
selected station. *Advanced search* goes the other way: pick a region, a state
or a ZIP code and a radius, and rank the six stations against it, with the
station's window free to fall anywhere in the year.

Comparisons use mean temperature, diurnal temperature range, day-to-day
temperature change, dew point, rainfall, and days with THImax at or above 79.

The sigma dissimilarity index gives how many standard deviations apart two
locations are, across all variables at once, using a Mahalanobis distance in a
reduced principal-component space. Adapted from Mahony et al., 2017, Glob
Change Biol. 23, 3934-3955 (https://doi.org/10.1111/gcb.13645). Below 2 sigma
the climate is considered an acceptable analogue, following Chaudhary et al.,
2023, Sci Rep 13, 9317 (https://doi.org/10.1038/s41598-023-35887-x).

Per variable, the standardised difference is the cell or region minus the
station, divided by the station's own interannual standard deviation.

## Running locally

    pip install -r requirements.txt
    streamlit run biomet_app.py

## Development

Data processing and imputation were written in R. The Streamlit dashboard was
built with the assistance of Claude Code (Anthropic), under manual review.