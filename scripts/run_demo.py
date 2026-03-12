from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.cords_loader import load_cords_reentries, select_presentation_cases
from src.corridor import build_corridor_from_points, build_path_from_tle_history, load_path_points, save_corridor_geojson, wrap_geometry_antimeridian
from src.exposure import run_exposure_analysis
from src.io_utils import configure_logging, write_dataframe
from src.plotting import (
    plot_corridor_static,
    plot_country_overlap,
    plot_land_ocean,
    plot_population_summary,
    plot_time_window_diagnostics,
    save_figure_gallery,
    save_corridor_map,
)
from src.spacetrack_client import collect_gp_history
from src.time_window_model import run_time_window_model


def _load_or_build_reentries(config, force_download: bool) -> pd.DataFrame:
    cached = config.outputs_tables_dir / "reentries_clean.csv"
    if cached.exists() and not force_download:
        return pd.read_csv(cached, parse_dates=["reentry_time_utc", "launch_date"])
    return load_cords_reentries(config, force=force_download)


def _select_cases(config, reentries: pd.DataFrame) -> pd.DataFrame:
    selected_path = config.outputs_tables_dir / "selected_cases.csv"
    if selected_path.exists():
        return pd.read_csv(selected_path, parse_dates=["reentry_time_utc", "launch_date"])
    return select_presentation_cases(reentries, config)


def _case_slug(selected_case: pd.Series) -> str:
    norad_id = int(selected_case["norad_id"])
    object_name = str(selected_case.get("object_name", f"norad_{norad_id}"))
    slug = re.sub(r"[^a-z0-9]+", "_", object_name.lower()).strip("_")
    return f"{norad_id}_{slug or 'case'}"


def _case_output_dirs(config, selected_case: pd.Series, batch_mode: bool) -> tuple[Path, Path, Path]:
    if not batch_mode:
        return config.outputs_maps_dir, config.outputs_figures_dir, config.outputs_tables_dir

    case_root = config.outputs_dir / "cases" / _case_slug(selected_case)
    maps_dir = case_root / "maps"
    figures_dir = case_root / "figures"
    tables_dir = case_root / "tables"
    maps_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return maps_dir, figures_dir, tables_dir


def _build_case_path_points(
    selected_case: pd.Series,
    gp_histories: dict[int, pd.DataFrame],
    config,
    path_file_arg: str | None,
) -> tuple[pd.DataFrame, str]:
    path_points = None
    input_label = "Unknown input"
    case_norad = int(selected_case["norad_id"])

    if config.use_tle_track_if_available and case_norad in gp_histories:
        reentry_time = pd.to_datetime(selected_case["reentry_time_utc"], utc=True, errors="coerce")
        path_points = build_path_from_tle_history(
            gp_histories[case_norad],
            reentry_time_utc=reentry_time,
            track_duration_hours=config.track_duration_hours,
            track_step_minutes=config.track_step_minutes,
        )
        if path_points is not None:
            input_label = "TLE-derived final-orbit track"

    if path_points is None:
        input_path = Path(path_file_arg).resolve() if path_file_arg else config.manual_path_file
        if input_path is None or not input_path.exists():
            raise FileNotFoundError("No path file was available for corridor generation.")
        path_points = load_path_points(input_path)
        input_label = f"Manual path file: {input_path.name}"

    return path_points, input_label


def _run_case(
    selected_case: pd.Series,
    gp_histories: dict[int, pd.DataFrame],
    config,
    width_km: float,
    path_file_arg: str | None,
    batch_mode: bool,
) -> dict[str, object]:
    maps_dir, figures_dir, tables_dir = _case_output_dirs(config, selected_case, batch_mode=batch_mode)
    path_points, input_label = _build_case_path_points(selected_case, gp_histories, config, path_file_arg)

    corridor_gdf, path_points_gdf = build_corridor_from_points(path_points, width_km=width_km)
    display_corridor_gdf, display_path_points_gdf = build_corridor_from_points(path_points, width_km=width_km, wrap_longitudes=False)
    map_corridor_gdf = display_corridor_gdf.copy()
    map_corridor_gdf["geometry"] = map_corridor_gdf.geometry.apply(wrap_geometry_antimeridian)

    save_corridor_geojson(map_corridor_gdf, maps_dir / "corridor.geojson")
    summary_df, country_overlap, land, countries = run_exposure_analysis(corridor_gdf, config, output_tables_dir=tables_dir)

    case_norad = int(selected_case["norad_id"])
    map_title = f"Reentry Corridor: {selected_case['object_name']} ({case_norad})"
    plot_corridor_static(
        map_corridor_gdf,
        figures_dir / "corridor_static.png",
        land,
        countries,
        display_path_points_gdf,
        summary_df=summary_df,
        country_overlap=country_overlap,
        title=map_title,
        input_label=input_label,
    )
    save_corridor_map(
        map_corridor_gdf,
        maps_dir / "corridor.html",
        display_path_points_gdf,
        summary_df=summary_df,
        country_overlap=country_overlap,
        map_title=map_title,
        input_label=input_label,
    )
    plot_country_overlap(country_overlap, figures_dir / "country_overlap_top10.png")
    plot_land_ocean(summary_df, figures_dir / "land_ocean_fraction.png")
    plot_population_summary(summary_df, figures_dir / "population_exposure_summary.png")
    save_figure_gallery(figures_dir, title=f"Figures: {selected_case['object_name']} ({case_norad})")

    summary_row = summary_df.iloc[0].to_dict()
    summary_row["case_id"] = selected_case.get("case_id")
    summary_row["object_name"] = selected_case.get("object_name")
    summary_row["norad_id"] = case_norad
    summary_row["reentry_time_utc"] = selected_case.get("reentry_time_utc")
    summary_row["maps_dir"] = str(maps_dir)
    summary_row["figures_dir"] = str(figures_dir)
    summary_row["tables_dir"] = str(tables_dir)
    summary_row["corridor_map_html"] = str(maps_dir / "corridor.html")
    return summary_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reduced offline reentry exposure demo pipeline.")
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--path-file", default=None, help="Optional CSV or GeoJSON path input.")
    parser.add_argument("--case-norad", type=int, default=None, help="Optional NORAD ID to prioritize.")
    parser.add_argument("--all-cases", action="store_true", help="Run every selected case instead of only the first one.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of selected cases to process.")
    parser.add_argument("--width-km", type=float, default=None, help="Override corridor width in kilometers.")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of public datasets.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    width_km = float(args.width_km) if args.width_km is not None else config.corridor_width_km

    reentries = _load_or_build_reentries(config, force_download=args.force_download)
    selected_cases = _select_cases(config, reentries)
    if args.case_norad is not None:
        filtered = selected_cases[selected_cases["norad_id"].astype("Int64") == int(args.case_norad)]
        if not filtered.empty:
            selected_cases = filtered.reset_index(drop=True)
    if args.limit is not None:
        selected_cases = selected_cases.head(int(args.limit)).reset_index(drop=True)
    if not args.all_cases:
        selected_cases = selected_cases.head(1).reset_index(drop=True)
    if selected_cases.empty:
        raise RuntimeError("No presentation cases are available after filtering.")

    gp_histories = collect_gp_history(selected_cases, config, force=False)
    model_result = run_time_window_model(gp_histories, selected_cases, config)
    if model_result is not None:
        plot_time_window_diagnostics(
            predictions=model_result.predictions,
            metrics=model_result.metrics,
            feature_importance=model_result.feature_importance,
            output_dir=config.outputs_figures_dir,
        )
    save_figure_gallery(config.outputs_figures_dir, title="Generated Figures")

    batch_mode = len(selected_cases) > 1
    batch_rows: list[dict[str, object]] = []
    for _, selected_case in selected_cases.iterrows():
        batch_rows.append(
            _run_case(
                selected_case=selected_case,
                gp_histories=gp_histories,
                config=config,
                width_km=width_km,
                path_file_arg=args.path_file,
                batch_mode=batch_mode,
            )
        )

    if batch_mode:
        batch_df = pd.DataFrame(batch_rows)
        batch_summary_path = config.outputs_tables_dir / "batch_exposure_summary.csv"
        write_dataframe(batch_df, batch_summary_path)
        print("Reduced reentry exposure batch complete.")
        print(f"Cases processed: {len(batch_df)}")
        print(f"Batch summary: {batch_summary_path}")
        print(f"Case outputs root: {config.outputs_dir / 'cases'}")
        return

    result = batch_rows[0]
    print("Reduced reentry exposure demo complete.")
    print(f"Selected case: {result['object_name']} ({int(result['norad_id'])})")
    print(f"Corridor area km^2: {float(result['corridor_area_km2']):.2f}")
    if pd.notna(result.get("land_fraction")):
        print(f"Land fraction: {float(result['land_fraction']):.3f}")
        print(f"Ocean fraction: {float(result['ocean_fraction']):.3f}")
    print(f"Exposure summary: {Path(result['tables_dir']) / 'exposure_summary.csv'}")
    print(f"Corridor map: {result['corridor_map_html']}")


if __name__ == "__main__":
    main()
