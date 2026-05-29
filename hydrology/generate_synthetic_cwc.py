from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CITIES_BASE_LEVELS = {
    "delhi": 205.0,
    "mumbai": 5.0,
    "kolkata": 6.0,
    "bengaluru": 850.0,
    "chennai": 5.0,
    "hyderabad": 500.0,
    "ahmedabad": 50.0,
    "pune": 550.0,
    "lucknow": 120.0,
    "jaipur": 400.0,
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic CWC-like daily water-level CSVs for 10 cities.")
    parser.add_argument("--output-dir", default="data/cwc")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2020-12-31")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = _parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start=args.start_date, end=args.end_date, freq="D")
    print("Generating synthetic CWC data for 10 cities...\n")

    for city, base_level in CITIES_BASE_LEVELS.items():
        df = pd.DataFrame({"date": dates})
        df["day_of_year"] = df["date"].dt.dayofyear

        # Monsoon-like seasonal pulse around early August.
        df["monsoon_surge"] = 8.0 * np.exp(-0.5 * ((df["day_of_year"] - 220) / 20) ** 2)
        df["noise"] = np.random.normal(0, 0.5, len(df))
        df["water_level"] = (base_level + df["monsoon_surge"] + df["noise"]).round(2)

        out_df = df[["date", "water_level"]]
        output_path = output_dir / f"{city}_cwc.csv"
        out_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path} ({len(out_df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
