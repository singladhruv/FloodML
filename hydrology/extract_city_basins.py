from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import geopandas as gpd
from shapely.geometry import Point


warnings.filterwarnings("ignore")


CITIES_DATA = {
    "delhi": {"id": "delhi", "coords": (77.2090, 28.6139)},
    "mumbai": {"id": "mumbai", "coords": (72.8777, 19.0760)},
    "kolkata": {"id": "kolkata", "coords": (88.3639, 22.5726)},
    "bengaluru": {"id": "bengaluru", "coords": (77.5946, 12.9716)},
    "chennai": {"id": "chennai", "coords": (80.2707, 13.0827)},
    "hyderabad": {"id": "hyderabad", "coords": (78.4867, 17.3850)},
    "ahmedabad": {"id": "ahmedabad", "coords": (72.5714, 23.0225)},
    "pune": {"id": "pune", "coords": (73.8567, 18.5204)},
    "lucknow": {"id": "lucknow", "coords": (80.9462, 26.8467)},
    "jaipur": {"id": "jaipur", "coords": (75.7873, 26.9124)},
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Extract one HydroBASINS polygon per target city.")
    parser.add_argument(
        "--hydrobasins-shp",
        default="data/hydrorivers/hybas_as_lev08_v1c.shp",
        help="Path to HydroBASINS Asia shapefile (e.g., hybas_as_lev08_v1c.shp).",
    )
    parser.add_argument("--output-dir", default="data/basins")
    return parser.parse_args()


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hydrobasins_path = Path(args.hydrobasins_shp)
    if not hydrobasins_path.exists():
        raise FileNotFoundError(
            f"HydroBASINS shapefile not found: {hydrobasins_path}. "
            "Download/extract it and pass the path via --hydrobasins-shp."
        )

    print("1. Loading HydroBASINS dataset...")
    basins = gpd.read_file(hydrobasins_path).to_crs("EPSG:4326")

    print("2. Extracting target watersheds for 10 cities...\n")
    for city, info in CITIES_DATA.items():
        basin_id = info["id"]
        lon, lat = info["coords"]
        point = Point(lon, lat)

        # intersects is safer than contains for boundary points.
        target = basins[basins.geometry.intersects(point)].copy()
        if target.empty:
            print(f"Warning: no basin found for {city} at ({lon}, {lat}).")
            continue

        target["basin_id"] = basin_id
        output_path = output_dir / f"{basin_id}_basin.shp"
        target.to_file(output_path)
        print(f"Saved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
