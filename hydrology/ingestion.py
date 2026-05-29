from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import warnings
from typing import Optional

import pandas as pd

from hydrology.config import BasinConfig, PipelineConfig


def _require_dependency(package_name: str, hint: str):
    try:
        module = importlib.import_module(package_name)
        return module
    except Exception as exc:
        raise ImportError(f"Missing optional dependency '{package_name}'. {hint}") from exc


@dataclass
class IngestionBundle:
    historical_rain_df: pd.DataFrame
    historical_rain_ds: "object"
    forecast_df: pd.DataFrame
    forecast_ds: "object"
    soil_moisture_df: pd.DataFrame
    cwc_labels_df: pd.DataFrame
    distance_to_river_meters: float


def get_historical_rainfall(config: PipelineConfig, basin: BasinConfig):
    imd = _require_dependency(
        "imdlib",
        "Install requirements-hydrology.txt and ensure IMD binary datasets are available.",
    )

    config.imd_data_dir.mkdir(parents=True, exist_ok=True)
    rain_dir = config.imd_data_dir / "rain"
    missing_years = [
        year
        for year in range(config.start_year, config.end_year + 1)
        if not (rain_dir / f"{year}.grd").exists()
    ]
    if missing_years:
        imd.get_data("rain", config.start_year, config.end_year, fn_format="yearwise", file_dir=str(config.imd_data_dir))
    data = imd.open_data("rain", config.start_year, config.end_year, "yearwise", str(config.imd_data_dir))
    ds = data.get_xarray()

    df = (
        ds.sel(lat=basin.target_lat, lon=basin.target_lon, method="nearest")
        .to_dataframe()
        .reset_index()
    )
    df = df[df["rain"] != -999.0].copy()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    return df, ds


def get_live_forecast(basin: BasinConfig, lead_hours: int = 24):
    herbie_mod = _require_dependency(
        "herbie",
        "Install requirements-hydrology.txt to fetch ECMWF open forecasts.",
    )
    Herbie = getattr(herbie_mod, "Herbie")

    base_time = pd.Timestamp.utcnow().normalize()
    last_error: Optional[Exception] = None

    # Try same-day then previous-day cycle to reduce operational failures.
    for day_offset in (0, -1):
        run_time = base_time + pd.Timedelta(days=day_offset)
        try:
            h = Herbie(
                date=run_time.strftime("%Y-%m-%d 00:00"),
                model="ecmwf",
                product="oper",
                fxx=lead_hours,
            )
            ds = h.xarray(search="(?i).*tp.*")
            df = (
                ds.sel(latitude=basin.target_lat, longitude=basin.target_lon, method="nearest")
                .to_dataframe()
                .reset_index()
            )
            return df, ds
        except Exception as exc:
            last_error = exc

    raise RuntimeError("Unable to fetch ECMWF forecast from Herbie fallback cycles.") from last_error


def get_soil_moisture(basin_polygon_geojson: dict, basin: BasinConfig):
    ee = _require_dependency(
        "ee",
        "Install requirements-hydrology.txt and run Earth Engine auth before pipeline execution.",
    )

    try:
        ee_project = os.getenv("EE_PROJECT")
        if ee_project:
            ee.Initialize(project=ee_project)
        else:
            ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Set EE_PROJECT to your Google Cloud project ID "
            "and run 'earthengine authenticate'. Example (PowerShell): "
            "$env:EE_PROJECT='my-ee-project-488121'; earthengine authenticate"
        ) from exc

    if isinstance(basin_polygon_geojson, str):
        basin_polygon_geojson = json.loads(basin_polygon_geojson)

    roi = ee.Geometry(basin_polygon_geojson)
    start_date, end_date = basin.soil_moisture_date_range

    smap = (
        ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture")
        .filterDate(start_date, end_date)
        .select("ssm")
    )

    def _get_basin_mean(image):
        mean_dict = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=basin.soil_moisture_scale_m,
            maxPixels=1e9,
        )
        return ee.Feature(
            None,
            {
                "date": image.date().format("YYYY-MM-dd"),
                "soil_moisture": mean_dict.get("ssm"),
            },
        )

    features = smap.map(_get_basin_mean).getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in features])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def get_river_distance_and_target(basin: BasinConfig):
    gpd = _require_dependency(
        "geopandas",
        "Install requirements-hydrology.txt for geospatial vector operations.",
    )
    shapely_geometry = _require_dependency("shapely.geometry", "Install shapely from requirements-hydrology.txt.")
    Point = getattr(shapely_geometry, "Point")

    rivers_gdf = gpd.read_file(basin.river_shapefile)
    if "UPLAND_SKM" in rivers_gdf.columns:
        major_rivers = rivers_gdf[rivers_gdf["UPLAND_SKM"] > 100].copy()
    elif "UP_AREA" in rivers_gdf.columns:
        warnings.warn(
            "River shapefile does not contain 'UPLAND_SKM'; using 'UP_AREA' > 100 as fallback.",
            RuntimeWarning,
        )
        major_rivers = rivers_gdf[rivers_gdf["UP_AREA"] > 100].copy()
    else:
        warnings.warn(
            "River shapefile has neither 'UPLAND_SKM' nor 'UP_AREA'; using all river geometries.",
            RuntimeWarning,
        )
        major_rivers = rivers_gdf.copy()

    if major_rivers.empty:
        warnings.warn(
            "Major river filter returned no rows; falling back to all river geometries.",
            RuntimeWarning,
        )
        major_rivers = rivers_gdf.copy()

    target_point = gpd.GeoSeries([Point(basin.target_lon, basin.target_lat)], crs="EPSG:4326")
    distance_m = target_point.to_crs("EPSG:32643").geometry.iloc[0].distance(
        major_rivers.to_crs("EPSG:32643").unary_union
    )

    df_cwc = pd.read_csv(basin.cwc_gauge_csv)
    df_cwc["date"] = pd.to_datetime(df_cwc["date"]).dt.tz_localize(None)
    df_cwc["flood_occurred"] = (df_cwc["water_level"] >= basin.cwc_danger_level_m).astype(int)
    return float(distance_m), df_cwc[["date", "water_level", "flood_occurred"]].copy()


def run_ingestion(
    config: PipelineConfig,
    basin: BasinConfig,
    basin_polygon_geojson: dict,
    lead_hours: int = 24,
    fetch_forecast: bool = True,
):
    historical_df, historical_ds = get_historical_rainfall(config, basin)
    forecast_df = pd.DataFrame()
    forecast_ds = None
    if fetch_forecast:
        try:
            forecast_df, forecast_ds = get_live_forecast(basin, lead_hours=lead_hours)
        except Exception as exc:
            warnings.warn(
                f"Live ECMWF forecast fetch failed and will be skipped for this run: {exc}",
                RuntimeWarning,
            )
    soil_df = get_soil_moisture(basin_polygon_geojson, basin)
    distance_m, cwc_df = get_river_distance_and_target(basin)
    return IngestionBundle(
        historical_rain_df=historical_df,
        historical_rain_ds=historical_ds,
        forecast_df=forecast_df,
        forecast_ds=forecast_ds,
        soil_moisture_df=soil_df,
        cwc_labels_df=cwc_df,
        distance_to_river_meters=distance_m,
    )

# earthengine authenticate
# $env:EE_PROJECT = "my-ee-project-488121"
# earthengine set_project $env:EE_PROJECT
# python -c "import os,ee; ee.Initialize(project=os.environ['EE_PROJECT']); print('EE init OK')"
# python -m hydrology.train_hydrology --basin-id jalandhar --basin-shapefile data/basins/jalandhar_basin.shp --river-shapefile data/hydrorivers/HydroRIVERS_v10_as.shp --cwc-gauge-csv data/cwc/jalandhar_cwc.csv --cwc-danger-level 212.0 --target-lat 31.3 --target-lon 75.5 --start-year 2010 --end-year 2020 --lead-hours 24 48 72 96 120 144 168
