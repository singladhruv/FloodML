from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
import warnings

import pandas as pd

from hydrology.config import DEFAULT_LEAD_HOURS, BasinConfig, PipelineConfig
from hydrology.dataset import assemble_hydrology_dataset, save_dataset_bundle
from hydrology.train_hydrology import _train_for_lead


REQUIRED_COLUMNS = {
    "basin_id",
    "basin_shapefile",
    "river_shapefile",
    "cwc_gauge_csv",
    "cwc_danger_level",
    "target_lat",
    "target_lon",
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Batch train hydrology basin models from a CSV registry.")
    parser.add_argument("--registry-csv", required=True, help="CSV with one basin per row.")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--lead-hours", nargs="+", type=int, default=list(DEFAULT_LEAD_HOURS))
    parser.add_argument("--artifacts-dir", default="artifacts/hydrology")
    parser.add_argument("--imd-data-dir", default="data/imd_data")
    return parser.parse_args()


def _validate_registry_columns(df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Registry CSV is missing required columns: {sorted(missing)}")


def _build_basin_config(row: pd.Series) -> BasinConfig:
    return BasinConfig(
        basin_id=str(row["basin_id"]),
        basin_shapefile=Path(row["basin_shapefile"]),
        river_shapefile=Path(row["river_shapefile"]),
        cwc_gauge_csv=Path(row["cwc_gauge_csv"]),
        cwc_danger_level_m=float(row["cwc_danger_level"]),
        target_lat=float(row["target_lat"]),
        target_lon=float(row["target_lon"]),
    )


def main():
    args = _parse_args()
    registry_df = pd.read_csv(args.registry_csv)
    _validate_registry_columns(registry_df)

    config = PipelineConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        imd_data_dir=Path(args.imd_data_dir),
        artifacts_dir=Path(args.artifacts_dir),
        lead_hours=tuple(args.lead_hours),
    )
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    summary_path = config.artifacts_dir / "batch_training_summary.json"
    if summary_path.exists():
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        run_summary = {}
    for _, row in registry_df.iterrows():
        basin = _build_basin_config(row)
        print(f"Training basin: {basin.basin_id}")

        dataset_bundle = assemble_hydrology_dataset(config, basin)
        basin_dataset_dir = config.artifacts_dir / basin.basin_id
        save_dataset_bundle(dataset_bundle, basin_dataset_dir)

        models_by_lead = {}
        results_by_lead = {}
        for lead_hours, lead_df in dataset_bundle.per_lead.items():
            class_count = lead_df["target_flood"].nunique(dropna=True)
            if class_count < 2:
                warnings.warn(
                    f"Skipping basin={basin.basin_id}, lead={lead_hours}h because target_flood has "
                    f"{class_count} class(es): {sorted(lead_df['target_flood'].unique().tolist())}",
                    RuntimeWarning,
                )
                continue
            summary, model = _train_for_lead(lead_df, dataset_bundle.feature_columns, config.random_state)
            models_by_lead[lead_hours] = model
            results_by_lead[lead_hours] = summary

        if not models_by_lead:
            warnings.warn(
                f"Skipping artifact write for basin={basin.basin_id}; no lead horizons had >=2 classes.",
                RuntimeWarning,
            )
            run_summary[basin.basin_id] = {
                "error": "No lead horizons had >=2 classes in target_flood.",
            }
            continue

        artifact = {
            "kind": "hydrology_flood_model_bundle",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "basin_config": {
                "basin_id": basin.basin_id,
                "basin_shapefile": str(basin.basin_shapefile),
                "river_shapefile": str(basin.river_shapefile),
                "cwc_gauge_csv": str(basin.cwc_gauge_csv),
                "cwc_danger_level_m": basin.cwc_danger_level_m,
                "target_lat": basin.target_lat,
                "target_lon": basin.target_lon,
                "soil_moisture_date_range": basin.soil_moisture_date_range,
                "soil_moisture_scale_m": basin.soil_moisture_scale_m,
            },
            "pipeline_config": {
                "start_year": config.start_year,
                "end_year": config.end_year,
                "imd_data_dir": str(config.imd_data_dir),
                "artifacts_dir": str(config.artifacts_dir),
                "random_state": config.random_state,
                "lead_hours": list(config.lead_hours),
            },
            "feature_columns": dataset_bundle.feature_columns,
            "models_by_lead": models_by_lead,
            "results_by_lead": results_by_lead,
        }

        model_path = config.artifacts_dir / f"{basin.basin_id}_hydrology_model_bundle.pkl"
        with model_path.open("wb") as f:
            pickle.dump(artifact, f)

        metrics_path = config.artifacts_dir / f"{basin.basin_id}_hydrology_metrics.json"
        metrics_path.write_text(json.dumps(results_by_lead, indent=2), encoding="utf-8")

        primary_lead = min(dataset_bundle.per_lead.keys())
        latest_row = (
            dataset_bundle.per_lead[primary_lead]
            .sort_values("time")
            .tail(1)[dataset_bundle.feature_columns]
            .copy()
        )
        latest_row_path = config.artifacts_dir / f"{basin.basin_id}_latest_feature_row.parquet"
        latest_row.to_parquet(latest_row_path, index=False)

        run_summary[basin.basin_id] = {
            "model_bundle": str(model_path),
            "metrics_json": str(metrics_path),
            "latest_feature_row": str(latest_row_path),
            "results_by_lead": results_by_lead,
        }

    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"Saved batch summary: {summary_path}")


if __name__ == "__main__":
    main()
