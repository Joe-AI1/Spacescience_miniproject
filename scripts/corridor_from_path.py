from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.corridor import build_corridor_from_points, load_path_points, save_corridor_geojson, wrap_geometry_antimeridian
from src.exposure import run_exposure_analysis
from src.io_utils import configure_logging
from src.plotting import plot_corridor_static, plot_country_overlap, plot_land_ocean, plot_population_summary, save_corridor_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a buffered reentry corridor from an input path file.")
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--input", required=True, help="CSV or GeoJSON path file.")
    parser.add_argument("--width-km", type=float, default=None, help="Corridor half-width in kilometers.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    input_path = Path(args.input).resolve()
    width_km = float(args.width_km) if args.width_km is not None else config.corridor_width_km

    points_df = load_path_points(input_path)
    corridor_gdf, path_points_gdf = build_corridor_from_points(points_df, width_km=width_km)
    display_corridor_gdf, display_path_points_gdf = build_corridor_from_points(points_df, width_km=width_km, wrap_longitudes=False)
    map_corridor_gdf = display_corridor_gdf.copy()
    map_corridor_gdf["geometry"] = map_corridor_gdf.geometry.apply(wrap_geometry_antimeridian)
    save_corridor_geojson(map_corridor_gdf, config.outputs_maps_dir / "corridor.geojson")
    summary_df, country_overlap, land, countries = run_exposure_analysis(corridor_gdf, config)
    plot_corridor_static(
        map_corridor_gdf,
        config.outputs_figures_dir / "corridor_static.png",
        land,
        countries,
        display_path_points_gdf,
        summary_df=summary_df,
        country_overlap=country_overlap,
        title="Path-Based Reentry Corridor",
        input_label=f"Manual path file: {input_path.name}",
    )
    save_corridor_map(
        map_corridor_gdf,
        config.outputs_maps_dir / "corridor.html",
        display_path_points_gdf,
        summary_df=summary_df,
        country_overlap=country_overlap,
        map_title="Path-Based Reentry Corridor",
        input_label=f"Manual path file: {input_path.name}",
    )
    plot_country_overlap(country_overlap, config.outputs_figures_dir / "country_overlap_top10.png")
    plot_land_ocean(summary_df, config.outputs_figures_dir / "land_ocean_fraction.png")
    plot_population_summary(summary_df, config.outputs_figures_dir / "population_exposure_summary.png")

    print("Corridor built.")
    print(f"GeoJSON: {config.outputs_maps_dir / 'corridor.geojson'}")
    print(f"Exposure summary: {config.outputs_tables_dir / 'exposure_summary.csv'}")


if __name__ == "__main__":
    main()
