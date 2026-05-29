from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hydrology.config import DEFAULT_LEAD_HOURS, BasinConfig, PipelineConfig
from hydrology.dataset import assemble_hydrology_dataset, save_dataset_bundle


def _time_split(df: pd.DataFrame, time_col: str = "time", train_ratio: float = 0.7, val_ratio: float = 0.15):
    frame = df.sort_values(time_col).reset_index(drop=True)
    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return frame.iloc[:train_end], frame.iloc[train_end:val_end], frame.iloc[val_end:]


def _select_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
    }


def _build_candidates(random_state: int):
    return {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=random_state,
                    ),
                )
            ]
        ),
    }


def _train_for_lead(df: pd.DataFrame, feature_columns, random_state: int) -> Tuple[dict, object]:
    train_df, val_df, test_df = _time_split(df, time_col="time")
    x_train, y_train = train_df[feature_columns], train_df["target_flood"].astype(int)
    x_val, y_val = val_df[feature_columns], val_df["target_flood"].astype(int)
    x_test, y_test = test_df[feature_columns], test_df["target_flood"].astype(int)

    candidates = _build_candidates(random_state)
    best_name = None
    best_model = None
    best_auc = -1.0

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, val_prob)
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model

    calibrated = CalibratedClassifierCV(estimator=best_model, method="sigmoid", cv=5)
    calibrated.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))

    val_prob = calibrated.predict_proba(x_val)[:, 1]
    threshold = _select_threshold(y_val.to_numpy(), val_prob)
    test_prob = calibrated.predict_proba(x_test)[:, 1]

    summary = {
        "selected_model": best_name,
        "validation_auc": float(best_auc),
        "test_metrics": _metrics(y_test.to_numpy(), test_prob, threshold),
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    return summary, calibrated


def _parse_args():
    parser = argparse.ArgumentParser(description="Train hydrology-aware FloodML model bundle.")
    parser.add_argument("--basin-id", required=True)
    parser.add_argument("--basin-shapefile", required=True)
    parser.add_argument("--river-shapefile", required=True)
    parser.add_argument("--cwc-gauge-csv", required=True)
    parser.add_argument("--cwc-danger-level", type=float, required=True)
    parser.add_argument("--target-lat", type=float, required=True)
    parser.add_argument("--target-lon", type=float, required=True)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--lead-hours", nargs="+", type=int, default=list(DEFAULT_LEAD_HOURS))
    parser.add_argument("--artifacts-dir", default="artifacts/hydrology")
    parser.add_argument("--imd-data-dir", default="data/imd_data")
    return parser.parse_args()


def main():
    args = _parse_args()
    basin = BasinConfig(
        basin_id=args.basin_id,
        basin_shapefile=Path(args.basin_shapefile),
        river_shapefile=Path(args.river_shapefile),
        cwc_gauge_csv=Path(args.cwc_gauge_csv),
        cwc_danger_level_m=args.cwc_danger_level,
        target_lat=args.target_lat,
        target_lon=args.target_lon,
    )
    config = PipelineConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        imd_data_dir=Path(args.imd_data_dir),
        artifacts_dir=Path(args.artifacts_dir),
        lead_hours=tuple(args.lead_hours),
    )

    dataset_bundle = assemble_hydrology_dataset(config, basin)
    save_dataset_bundle(dataset_bundle, config.artifacts_dir)

    models_by_lead = {}
    results_by_lead = {}
    for lead_hours, lead_df in dataset_bundle.per_lead.items():
        summary, model = _train_for_lead(lead_df, dataset_bundle.feature_columns, config.random_state)
        models_by_lead[lead_hours] = model
        results_by_lead[lead_hours] = summary

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

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.artifacts_dir / f"{basin.basin_id}_hydrology_model_bundle.pkl"
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    metrics_path = config.artifacts_dir / f"{basin.basin_id}_hydrology_metrics.json"
    metrics_path.write_text(json.dumps(results_by_lead, indent=2), encoding="utf-8")

    print(f"Saved model bundle: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(results_by_lead, indent=2))


if __name__ == "__main__":
    main()
