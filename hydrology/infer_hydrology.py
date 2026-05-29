from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser(description="Run inference from a trained hydrology model bundle.")
    parser.add_argument("--model-bundle", required=True, help="Path to *_hydrology_model_bundle.pkl")
    parser.add_argument("--lead-hours", type=int, default=24)
    parser.add_argument(
        "--feature-row-parquet",
        required=True,
        help="Parquet containing one row of already engineered feature columns.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    bundle_path = Path(args.model_bundle)
    with bundle_path.open("rb") as f:
        artifact = pickle.load(f)

    models_by_lead = artifact["models_by_lead"]
    if args.lead_hours not in models_by_lead:
        raise KeyError(f"Lead {args.lead_hours}h not available. Options: {sorted(models_by_lead.keys())}")

    feature_columns = artifact["feature_columns"]
    feature_df = pd.read_parquet(args.feature_row_parquet)
    if len(feature_df) != 1:
        raise ValueError("feature-row-parquet must contain exactly one row.")

    missing = [c for c in feature_columns if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    model = models_by_lead[args.lead_hours]
    threshold = artifact["results_by_lead"][args.lead_hours]["test_metrics"]["threshold"]

    x = feature_df[feature_columns]
    prob = float(model.predict_proba(x)[:, 1][0])
    pred = int(prob >= threshold)

    output = {
        "lead_hours": args.lead_hours,
        "risk_probability": round(prob, 6),
        "threshold": round(float(threshold), 6),
        "alert": "UNSAFE" if pred == 1 else "SAFE",
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

