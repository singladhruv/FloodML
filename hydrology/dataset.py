from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json
import pandas as pd

from hydrology.config import BasinConfig, PipelineConfig
from hydrology.features import add_antecedent_rainfall_features, add_generic_lag_features, compute_basin_map_timeseries
from hydrology.ingestion import run_ingestion


@dataclass
class DatasetBundle:
    base_df: pd.DataFrame
    per_lead: Dict[int, pd.DataFrame]
    feature_columns: List[str]


def _load_basin_geojson_from_shapefile(shapefile_path: Path) -> dict:
    import geopandas as gpd

    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    if gdf.empty:
        raise ValueError(f"No geometry found in shapefile: {shapefile_path}")
    return gdf.geometry.iloc[0].__geo_interface__


def assemble_hydrology_dataset(config: PipelineConfig, basin: BasinConfig) -> DatasetBundle:
    basin_polygon_geojson = _load_basin_geojson_from_shapefile(basin.basin_shapefile)
    ingested = run_ingestion(config, basin, basin_polygon_geojson, lead_hours=24, fetch_forecast=False)

    # Historical MAP features from IMD gridded rainfall.
    map_df = compute_basin_map_timeseries(
        ds_weather=ingested.historical_rain_ds,
        basin_shapefile_path=basin.basin_shapefile,
        rain_var="rain",
    )
    map_features = add_antecedent_rainfall_features(
        map_df,
        rain_col="basin_mean_rainfall",
        dropna=False,
    )
    map_features = add_antecedent_rainfall_features(
        map_features,
        rain_col="basin_max_rainfall",
        prefix="rain_max",
    )

    soil_df = ingested.soil_moisture_df.copy()
    soil_df["date"] = pd.to_datetime(soil_df["date"]).dt.floor("D")
    soil_df = soil_df.rename(columns={"date": "time"})

    cwc_df = ingested.cwc_labels_df.copy()
    cwc_df["date"] = pd.to_datetime(cwc_df["date"]).dt.floor("D")
    cwc_df = add_generic_lag_features(
        cwc_df,
        col="water_level",
        lags=(1, 2, 3),
        rolling_windows=(3, 7),
        prefix="wl",
    )
    cwc_df = cwc_df.rename(columns={"date": "time"})

    base_df = map_features.merge(soil_df[["time", "soil_moisture"]], on="time", how="left")
    base_df = base_df.merge(cwc_df, on="time", how="left")
    base_df["distance_to_river_meters"] = ingested.distance_to_river_meters
    base_df = base_df.sort_values("time").dropna().reset_index(drop=True)

    # Build lead-time specific targets for strict forecast settings.
    per_lead: Dict[int, pd.DataFrame] = {}
    for lead_hours in config.lead_hours:
        lead_days = max(1, lead_hours // 24)
        lead_df = base_df.copy()
        lead_df["target_flood"] = lead_df["flood_occurred"].shift(-lead_days)
        lead_df["lead_hours"] = lead_hours
        lead_df = lead_df.dropna(subset=["target_flood"]).copy()
        lead_df["target_flood"] = lead_df["target_flood"].astype(int)
        per_lead[lead_hours] = lead_df

    feature_columns = [
        c
        for c in base_df.columns
        if c
        not in {
            "time",
            "flood_occurred",
            "target_flood",
        }
    ]

    return DatasetBundle(base_df=base_df, per_lead=per_lead, feature_columns=feature_columns)


def save_dataset_bundle(bundle: DatasetBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.base_df.to_parquet(output_dir / "base_dataset.parquet", index=False)
    for lead_hours, df in bundle.per_lead.items():
        df.to_parquet(output_dir / f"dataset_lead_{lead_hours}h.parquet", index=False)
    metadata = {
        "feature_columns": bundle.feature_columns,
        "lead_hours": sorted(bundle.per_lead.keys()),
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
