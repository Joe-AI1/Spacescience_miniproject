from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config import AppConfig
from src.io_utils import download_file, find_first_file, unzip_archive, write_dataframe


LOGGER = logging.getLogger(__name__)
AREA_CRS = "EPSG:6933"


def ensure_natural_earth_layers(config: AppConfig, force: bool = False) -> dict[str, Path]:
    raw_dir = config.raw_dir / "natural_earth"
    processed_dir = config.processed_dir / "natural_earth"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    land_zip = raw_dir / "ne_10m_land.zip"
    countries_zip = raw_dir / "ne_10m_admin_0_countries.zip"
    land_dir = processed_dir / "land"
    countries_dir = processed_dir / "countries"

    download_file(config.natural_earth_land_url, land_zip, force=force)
    download_file(config.natural_earth_countries_url, countries_zip, force=force)
    unzip_archive(land_zip, land_dir, force=force)
    unzip_archive(countries_zip, countries_dir, force=force)

    land_shp = find_first_file(land_dir, ["*.shp"])
    countries_shp = find_first_file(countries_dir, ["*.shp"])
    if land_shp is None or countries_shp is None:
        raise FileNotFoundError("Natural Earth shapefiles were not found after extraction.")
    return {"land": land_shp, "countries": countries_shp}


def _load_population_overrides(csv_path: Path | None) -> pd.DataFrame | None:
    if csv_path is None or not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    lowered = {column.lower(): column for column in df.columns}
    iso_col = lowered.get("iso_a3") or lowered.get("adm0_a3")
    name_col = lowered.get("admin") or lowered.get("name")
    pop_col = lowered.get("population") or lowered.get("pop_est")
    if pop_col is None:
        return None
    output = df.copy()
    rename_map = {pop_col: "population_override"}
    if iso_col:
        rename_map[iso_col] = "iso_a3"
    if name_col:
        rename_map[name_col] = "country_name"
    output = output.rename(columns=rename_map)
    keep_cols = [column for column in ["iso_a3", "country_name", "population_override"] if column in output.columns]
    return output[keep_cols]


def _load_layers_if_available(config: AppConfig) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    try:
        paths = ensure_natural_earth_layers(config, force=False)
        land = gpd.read_file(paths["land"]).to_crs("EPSG:4326")
        countries = gpd.read_file(paths["countries"]).to_crs("EPSG:4326")
        return land, countries
    except Exception as exc:
        LOGGER.warning("Natural Earth layers unavailable: %s", exc)
        return None, None


def _safe_overlay(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if left.empty or right.empty:
        return gpd.GeoDataFrame(columns=list(left.columns) + list(right.columns), geometry=[], crs=left.crs)
    return gpd.overlay(left, right, how="intersection", keep_geom_type=False)


def _country_columns(countries: gpd.GeoDataFrame) -> tuple[str | None, str | None, str | None]:
    lowered = {column.lower(): column for column in countries.columns}
    iso_col = lowered.get("iso_a3") or lowered.get("adm0_a3")
    name_col = lowered.get("name") or lowered.get("admin") or lowered.get("name_long")
    pop_col = lowered.get("pop_est")
    return iso_col, name_col, pop_col


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _discover_population_raster_path(config: AppConfig) -> Path | None:
    candidates: list[Path] = []
    if config.population_raster_path is not None:
        candidates.append(config.population_raster_path)
    for directory in (config.raw_dir / "population", config.processed_dir / "population"):
        if directory.exists():
            for pattern in ("*.tif", "*.tiff", "*.vrt"):
                candidates.extend(sorted(directory.rglob(pattern)))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            LOGGER.info("Using population raster: %s", candidate)
            return candidate
    return None


def _compute_raster_population(corridor_gdf: gpd.GeoDataFrame, raster_path: Path | None) -> float | None:
    if raster_path is None or not raster_path.exists():
        return None
    try:
        import rasterio
        from rasterio.mask import mask
    except ImportError:
        LOGGER.info("rasterio is not installed; skipping raster population exposure.")
        return None

    with rasterio.open(raster_path) as src:
        corridor_in_raster_crs = corridor_gdf.to_crs(src.crs)
        data, _ = mask(src, corridor_in_raster_crs.geometry, crop=True, filled=False)
        valid = data[0]
        return float(valid[valid > 0].sum())


def run_exposure_analysis(
    corridor_gdf: gpd.GeoDataFrame,
    config: AppConfig,
    output_tables_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    land, countries = _load_layers_if_available(config)
    country_overlap = pd.DataFrame(
        columns=[
            "country_name",
            "iso_a3",
            "overlap_area_km2",
            "corridor_overlap_fraction",
            "country_overlap_fraction",
            "population_estimate",
            "coarse_population_exposure_score",
        ]
    )

    corridor_area_km2 = corridor_gdf.to_crs(AREA_CRS).geometry.area.sum() / 1_000_000.0
    summary = {
        "corridor_area_km2": corridor_area_km2,
        "land_area_km2": pd.NA,
        "land_fraction": pd.NA,
        "ocean_fraction": pd.NA,
        "coarse_population_exposure_score_total": pd.NA,
        "raster_population_exposure": pd.NA,
        "population_method": "country_fallback" if config.country_population_fallback else "none",
        "population_raster_available": False,
        "population_raster_path_used": pd.NA,
        "status": "ok",
    }

    if land is not None:
        land_overlap = _safe_overlay(corridor_gdf.to_crs(AREA_CRS), land.to_crs(AREA_CRS))
        land_area_km2 = land_overlap.geometry.area.sum() / 1_000_000.0
        summary["land_area_km2"] = land_area_km2
        summary["land_fraction"] = land_area_km2 / corridor_area_km2 if corridor_area_km2 > 0 else pd.NA
        if pd.notna(summary["land_fraction"]):
            land_fraction = max(0.0, min(1.0, float(summary["land_fraction"])))
            summary["land_fraction"] = land_fraction
            summary["ocean_fraction"] = max(0.0, 1.0 - land_fraction)
        else:
            summary["ocean_fraction"] = pd.NA
    else:
        summary["status"] = "missing_land_reference_data"

    if countries is not None:
        countries_projected = countries.to_crs(AREA_CRS)
        corridor_projected = corridor_gdf.to_crs(AREA_CRS)
        iso_col, name_col, pop_col = _country_columns(countries_projected)
        intersection = _safe_overlay(corridor_projected, countries_projected)
        if not intersection.empty:
            intersection["overlap_area_km2"] = intersection.geometry.area / 1_000_000.0
            if iso_col and name_col:
                country_area_lookup = countries_projected[[iso_col, name_col, "geometry"]].copy()
                country_area_lookup["country_area_km2"] = country_area_lookup.geometry.area / 1_000_000.0
                merge_keys = [column for column in [iso_col, name_col] if column in intersection.columns]
                intersection = intersection.merge(country_area_lookup.drop(columns="geometry"), on=merge_keys, how="left")
            else:
                intersection["country_area_km2"] = pd.NA

            overrides = _load_population_overrides(config.country_population_csv)
            if overrides is not None:
                if iso_col and "iso_a3" in overrides.columns and iso_col in intersection.columns:
                    intersection = intersection.merge(overrides, left_on=iso_col, right_on="iso_a3", how="left")
                elif name_col and "country_name" in overrides.columns and name_col in intersection.columns:
                    intersection = intersection.merge(overrides, left_on=name_col, right_on="country_name", how="left")

            pop_value_col = _first_existing_column(
                intersection,
                [column for column in [pop_col, f"{pop_col}_x" if pop_col else None, f"{pop_col}_y" if pop_col else None] if column],
            )
            if pop_value_col is not None:
                if "population_override" in intersection.columns:
                    intersection["population_estimate"] = pd.to_numeric(intersection["population_override"], errors="coerce").fillna(
                        pd.to_numeric(intersection[pop_value_col], errors="coerce")
                    )
                else:
                    intersection["population_estimate"] = pd.to_numeric(intersection[pop_value_col], errors="coerce")
            else:
                intersection["population_estimate"] = intersection.get("population_override", pd.NA)

            intersection["corridor_overlap_fraction"] = intersection["overlap_area_km2"] / corridor_area_km2 if corridor_area_km2 > 0 else pd.NA
            intersection["country_overlap_fraction"] = intersection["overlap_area_km2"] / intersection["country_area_km2"]
            intersection["coarse_population_exposure_score"] = intersection["population_estimate"] * intersection["country_overlap_fraction"]
            country_overlap = (
                intersection.assign(
                    country_name=intersection[name_col] if name_col and name_col in intersection.columns else "Unknown",
                    iso_a3=intersection[iso_col] if iso_col and iso_col in intersection.columns else None,
                )[
                    [
                        "country_name",
                        "iso_a3",
                        "overlap_area_km2",
                        "corridor_overlap_fraction",
                        "country_overlap_fraction",
                        "population_estimate",
                        "coarse_population_exposure_score",
                    ]
                ]
                .sort_values("overlap_area_km2", ascending=False)
                .reset_index(drop=True)
            )
            score_series = pd.to_numeric(country_overlap["coarse_population_exposure_score"], errors="coerce")
            summary["coarse_population_exposure_score_total"] = score_series.fillna(0.0).sum()
    else:
        summary["status"] = "missing_country_reference_data" if summary["status"] == "ok" else summary["status"]

    if config.use_population_raster_if_available:
        raster_path = _discover_population_raster_path(config)
        if raster_path is not None:
            summary["population_raster_available"] = True
            summary["population_raster_path_used"] = str(raster_path)
            raster_population = _compute_raster_population(corridor_gdf, raster_path)
            if raster_population is not None:
                summary["raster_population_exposure"] = raster_population
                summary["population_method"] = "raster_zonal_sum"
        else:
            LOGGER.info("No local population raster found. Using country-level fallback exposure.")

    summary_df = pd.DataFrame([summary])
    tables_dir = output_tables_dir or config.outputs_tables_dir
    write_dataframe(country_overlap, tables_dir / "country_overlap.csv")
    write_dataframe(summary_df, tables_dir / "exposure_summary.csv")
    return summary_df, country_overlap, land, countries
