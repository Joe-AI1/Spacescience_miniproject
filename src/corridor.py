from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import transform, unary_union
from shapely.validation import make_valid


LOGGER = logging.getLogger(__name__)


def load_path_points(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(input_path)
        lon_col = next((column for column in frame.columns if column.lower() in {"lon", "longitude", "x"}), None)
        lat_col = next((column for column in frame.columns if column.lower() in {"lat", "latitude", "y"}), None)
        if lon_col is None or lat_col is None:
            raise ValueError(f"{input_path} must contain lon/lat columns")
        output = frame.rename(columns={lon_col: "lon", lat_col: "lat"}).copy()
        if "timestamp" in output.columns:
            output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
            return output[["timestamp", "lon", "lat"]]
        return output[["lon", "lat"]]

    gdf = gpd.read_file(input_path)
    if gdf.empty:
        raise ValueError(f"No geometries found in {input_path}")
    geometry = gdf.geometry.iloc[0]
    if geometry.geom_type == "LineString":
        coords = list(geometry.coords)
    elif geometry.geom_type == "MultiLineString":
        coords = [coord for line in geometry.geoms for coord in line.coords]
    else:
        coords = [(geom.x, geom.y) for geom in gdf.geometry if isinstance(geom, Point)]
    return pd.DataFrame(coords, columns=["lon", "lat"])


def _unwrap_longitudes(longitudes: list[float]) -> list[float]:
    if not longitudes:
        return []
    unwrapped = [float(longitudes[0])]
    for lon in longitudes[1:]:
        candidate = float(lon)
        previous = unwrapped[-1]
        while candidate - previous > 180.0:
            candidate -= 360.0
        while candidate - previous < -180.0:
            candidate += 360.0
        unwrapped.append(candidate)
    return unwrapped


def _wrap_longitude(value: float) -> float:
    wrapped = ((value + 180.0) % 360.0) - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _build_local_aeqd_crs(latitudes: pd.Series, longitudes: list[float]) -> CRS:
    center_lat = float(pd.Series(latitudes).mean())
    center_lon = _wrap_longitude(float(np.mean(longitudes)))
    return CRS.from_proj4(f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m +no_defs")


def _normalize_geometry_longitudes(geometry):
    def _wrap_xy(x: float, y: float, z: float | None = None):
        return _wrap_longitude(x), y

    return transform(_wrap_xy, geometry)


def wrap_geometry_antimeridian(geometry):
    if geometry.is_empty:
        return geometry
    if not geometry.is_valid:
        geometry = make_valid(geometry)

    min_x, _, max_x, _ = geometry.bounds
    start_zone = math.floor((min_x + 180.0) / 360.0)
    end_zone = math.floor((max_x + 180.0) / 360.0)
    pieces = []

    for zone in range(start_zone, end_zone + 1):
        min_lon = -180.0 + 360.0 * zone
        max_lon = 180.0 + 360.0 * zone
        clipped = geometry.intersection(box(min_lon, -90.0, max_lon, 90.0))
        if clipped.is_empty:
            continue
        shifted = transform(lambda x, y, z=None, zone=zone: (x - 360.0 * zone, y), clipped)
        pieces.append(shifted)

    if not pieces:
        return _normalize_geometry_longitudes(geometry)

    wrapped = unary_union(pieces)
    return _normalize_geometry_longitudes(wrapped)


def build_corridor_from_points(
    points_df: pd.DataFrame,
    width_km: float,
    wrap_longitudes: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    frame = points_df.copy()
    frame["lon_unwrapped"] = _unwrap_longitudes(frame["lon"].astype(float).tolist())
    line = LineString(list(zip(frame["lon_unwrapped"], frame["lat"].astype(float))))
    display_lon = frame["lon"] if wrap_longitudes else frame["lon_unwrapped"]
    path_points_gdf = gpd.GeoDataFrame(frame, geometry=gpd.points_from_xy(display_lon, frame["lat"]), crs="EPSG:4326")
    line_gdf = gpd.GeoSeries([line], crs="EPSG:4326")

    local_crs = _build_local_aeqd_crs(frame["lat"], frame["lon_unwrapped"].tolist())
    projected = line_gdf.to_crs(local_crs)
    corridor_geom = projected.buffer(width_km * 1000.0, cap_style=2).iloc[0]
    corridor = gpd.GeoDataFrame({"corridor_width_km": [width_km]}, geometry=[corridor_geom], crs=local_crs).to_crs("EPSG:4326")
    if wrap_longitudes:
        corridor["geometry"] = corridor.geometry.apply(_normalize_geometry_longitudes)
    return corridor, path_points_gdf


def save_corridor_geojson(corridor_gdf: gpd.GeoDataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corridor_gdf.to_file(output_path, driver="GeoJSON")
    return output_path


def _julian_date(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    year = dt.year
    month = dt.month
    day = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1_000_000.0) / 60.0) / 60.0) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    a = math.floor(year / 100)
    b = 2 - a + math.floor(a / 4)
    return math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + b - 1524.5


def _gmst_radians(dt: datetime) -> float:
    jd = _julian_date(dt)
    t = (jd - 2451545.0) / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * (t**2) - (t**3) / 38710000.0
    return math.radians(gmst_deg % 360.0)


def _ecef_to_geodetic(x_km: float, y_km: float, z_km: float) -> tuple[float, float]:
    a = 6378.137
    b = 6356.7523142
    e2 = 1.0 - (b**2) / (a**2)
    lon = math.atan2(y_km, x_km)
    p = math.sqrt(x_km * x_km + y_km * y_km)
    lat = math.atan2(z_km, p * (1.0 - e2))
    for _ in range(5):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z_km + e2 * n * sin_lat, p)
    return math.degrees(lat), _wrap_longitude(math.degrees(lon))


def _teme_to_lat_lon(position_teme_km: tuple[float, float, float], dt: datetime) -> tuple[float, float]:
    gmst = _gmst_radians(dt)
    x, y, z = position_teme_km
    x_ecef = x * math.cos(gmst) + y * math.sin(gmst)
    y_ecef = -x * math.sin(gmst) + y * math.cos(gmst)
    return _ecef_to_geodetic(x_ecef, y_ecef, z)


def build_path_from_tle_history(
    gp_history_df: pd.DataFrame,
    reentry_time_utc: pd.Timestamp,
    track_duration_hours: float,
    track_step_minutes: int,
) -> pd.DataFrame | None:
    try:
        from sgp4.api import Satrec, jday
    except ImportError:
        LOGGER.info("sgp4 is not installed; skipping TLE-derived path generation.")
        return None

    usable = gp_history_df.dropna(subset=["tle_line1", "tle_line2"]).sort_values("epoch")
    if usable.empty:
        return None

    latest = usable[usable["epoch"] <= reentry_time_utc].tail(1)
    if latest.empty:
        latest = usable.tail(1)
    tle_row = latest.iloc[0]
    sat = Satrec.twoline2rv(tle_row["tle_line1"], tle_row["tle_line2"])

    effective_duration_hours = float(track_duration_hours)
    if pd.notna(tle_row.get("mean_motion")) and float(tle_row["mean_motion"]) > 0:
        orbit_period_hours = 24.0 / float(tle_row["mean_motion"])
        effective_duration_hours = min(effective_duration_hours, orbit_period_hours * 1.25)

    start_time = max(pd.Timestamp(tle_row["epoch"]), pd.Timestamp(reentry_time_utc) - pd.Timedelta(hours=effective_duration_hours))
    end_time = pd.Timestamp(reentry_time_utc)
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f"{int(track_step_minutes)}min")
    rows: list[dict[str, object]] = []
    for timestamp in timestamps:
        dt = timestamp.to_pydatetime().astimezone(timezone.utc)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1_000_000.0)
        error_code, position, _ = sat.sgp4(jd, fr)
        if error_code != 0:
            continue
        lat, lon = _teme_to_lat_lon(position, dt)
        rows.append({"timestamp": dt.isoformat(), "lon": lon, "lat": lat})
    return pd.DataFrame(rows) if rows else None
