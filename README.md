# Offline Reentry Risk / Exposure Prototype

This repository is a simplified academic prototype for demonstrating how public reentry metadata, optional historical TLE/GP history, and basic geospatial overlays can be combined into a coarse exposure workflow for a university presentation.

It is not an operational reentry forecast system.

It does not claim exact impact locations.

It estimates a coarse affected-region exposure from a reentry window, path, or buffered corridor, then summarizes land/ocean, country overlap, and a simplified population exposure proxy.

If historical TLE data is unavailable, the project still runs in a path-driven demo mode and shows how a reentry corridor can be translated into exposure summaries.

## What The Prototype Does

1. Downloads and caches public datasets:
   - Aerospace CORDS reentry metadata
   - Natural Earth land polygons
   - Natural Earth country polygons
2. Builds a cleaned local reentry table.
3. Selects a small presentation subset of historical reentries.
4. Optionally downloads Space-Track `gp_history` for selected NORAD IDs when credentials are available.
5. Supports manual GP/TLE history files in `data/manual_gp_history/`.
6. Builds a lightweight reentry time-window baseline from TLE trend features when enough GP history exists.
7. Converts either an input path or an optional TLE-derived ground track into a buffered corridor polygon.
8. Intersects that corridor with land and countries.
9. Produces coarse exposure summaries and presentation-ready maps/plots.

## Repo Layout

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── manual_gp_history/
│   └── sample_path.csv
├── notebooks/
│   └── demo_reentry_risk.ipynb
├── outputs/
│   ├── figures/
│   ├── maps/
│   └── tables/
├── scripts/
│   ├── download_data.py
│   ├── build_cases.py
│   ├── run_demo.py
│   └── corridor_from_path.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── io_utils.py
    ├── cords_loader.py
    ├── spacetrack_client.py
    ├── tle_features.py
    ├── time_window_model.py
    ├── corridor.py
    ├── exposure.py
    └── plotting.py
```

## Setup

Create and activate a virtual environment, then install the core dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The default `requirements.txt` now includes both `sgp4` and `rasterio`.

- `sgp4` enables the optional coarse path-from-TLE mode.
- `rasterio` enables population raster zonal summaries when you already have a raster on disk.

Copy `.env.example` to `.env` if you want credential-based features:

```bash
copy .env.example .env
```

## Configuration

Default settings live in [configs/default.yaml](/c:/Users/koonj/SPACESCI/configs/default.yaml).

Key settings:

- `selected_norad_ids`: optional fixed case list
- `corridor_width_km`: global path buffer width
- `use_spacetrack_if_available`: toggles credential-based GP history download
- `use_population_raster_if_available`: enables raster-based exposure if a raster path exists
- `country_population_fallback`: uses country population estimates when no raster exists
- `manual_path_file`: guaranteed demo path input

## Commands

Download and cache the public datasets:

```bash
python scripts/download_data.py
```

Build a small presentation case list:

```bash
python scripts/build_cases.py
```

Run the full demo pipeline:

```bash
python scripts/run_demo.py
```

Run a batch of selected cases, for example 20 satellites:

```bash
python scripts/run_demo.py --all-cases --limit 20
```

Build only a corridor from a CSV or GeoJSON path:

```bash
python scripts/corridor_from_path.py --input data/sample_path.csv --width-km 200
```

## Expected Outputs

The main outputs are written under `outputs/`.

When batch mode is used, each satellite also gets its own folder under `outputs/cases/<norad>_<object_slug>/`.

Tables:

- [outputs/tables/reentries_clean.csv](/c:/Users/koonj/SPACESCI/outputs/tables/reentries_clean.csv)
- [outputs/tables/selected_cases.csv](/c:/Users/koonj/SPACESCI/outputs/tables/selected_cases.csv)
- [outputs/tables/batch_exposure_summary.csv](/c:/Users/koonj/SPACESCI/outputs/tables/batch_exposure_summary.csv)
- [outputs/tables/time_window_predictions.csv](/c:/Users/koonj/SPACESCI/outputs/tables/time_window_predictions.csv)
- [outputs/tables/country_overlap.csv](/c:/Users/koonj/SPACESCI/outputs/tables/country_overlap.csv)
- [outputs/tables/exposure_summary.csv](/c:/Users/koonj/SPACESCI/outputs/tables/exposure_summary.csv)

Maps and figures:

- [outputs/maps/corridor.geojson](/c:/Users/koonj/SPACESCI/outputs/maps/corridor.geojson)
- [outputs/maps/corridor.html](/c:/Users/koonj/SPACESCI/outputs/maps/corridor.html)
- [outputs/figures/corridor_static.png](/c:/Users/koonj/SPACESCI/outputs/figures/corridor_static.png)
- [outputs/figures/country_overlap_top10.png](/c:/Users/koonj/SPACESCI/outputs/figures/country_overlap_top10.png)
- [outputs/figures/land_ocean_fraction.png](/c:/Users/koonj/SPACESCI/outputs/figures/land_ocean_fraction.png)
- [outputs/figures/population_exposure_summary.png](/c:/Users/koonj/SPACESCI/outputs/figures/population_exposure_summary.png)
- [outputs/figures/predicted_vs_actual_hours_to_decay.png](/c:/Users/koonj/SPACESCI/outputs/figures/predicted_vs_actual_hours_to_decay.png)

## Space-Track Support

If `SPACETRACK_USER` and `SPACETRACK_PASS` are set, the pipeline will try to:

1. log in to Space-Track,
2. download `gp_history` for selected NORAD IDs,
3. cache raw responses under `data/raw/spacetrack/`,
4. normalize them to `data/processed/gp_history/`.

If credentials are missing or login fails, the rest of the project still runs.

You can also provide manual history files in [data/manual_gp_history](/c:/Users/koonj/SPACESCI/data/manual_gp_history). Supported inputs include:

- Space-Track style CSV
- Space-Track style JSON
- plain text TLE history files

## Population Exposure Modes

Required fallback:

- Country-level coarse exposure uses overlap fraction times country population estimates.
- Natural Earth `POP_EST` is used when available.
- A custom CSV can be joined later if needed.

Optional raster mode:

- Drop a local population raster into `data/raw/population/` or `data/processed/population/`, or set `population_raster_path` in the config.
- The current implementation assumes a population count raster for simple zonal summation.
- If no raster is found, the pipeline automatically falls back to country-level exposure.

## Offline Use

After the first successful public-data download and any optional credential-based ingestion, the project can run offline using the cached files under `data/raw/` and `data/processed/`.

## Data Sources

- Aerospace CORDS reentries: <https://aerospace.org/reentries>
- Aerospace CORDS grid view: <https://aerospace.org/reentries/grid>
- Space-Track documentation: <https://www.space-track.org/documentation>
- NOAA SWPC data access: <https://www.swpc.noaa.gov/content/data-access>
- Natural Earth land: <https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/>
- Natural Earth countries: <https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/>
- GPW / Earthdata reference: <https://www.earthdata.nasa.gov/data/projects/gpw>

## Notes On Scientific Scope

This project intentionally avoids overstating precision.

- The time-window model is a lightweight baseline using TLE trend features with simple tabular regressors (linear regression, random forest, and extra trees).
- The corridor is a buffered path, not a validated probabilistic debris corridor.
- The exposure outputs are coarse indicators intended for presentation and discussion.
