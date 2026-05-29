from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point


JALANDHAR_LON = 75.5762
JALANDHAR_LAT = 31.3260


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create jalandhar_basin.shp from HydroBASINS (no QGIS required)."
    )
    parser.add_argument(
        "--hydrobasins-shp",
        default=r"data\basins\hybas_as_lev08_v1c.shp",
        help="Path to HydroBASINS Asia Level-8 shapefile.",
    )
    parser.add_argument(
        "--output-shp",
        default=r"data\basins\jalandhar_basin.shp",
        help="Output shapefile path for extracted Jalandhar basin.",
    )
    parser.add_argument(
        "--basin-id",
        default="jalandhar_01",
        help="Value to write in basin_id field.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hydrobasins_path = Path(args.hydrobasins_shp)
    output_path = Path(args.output_shp)

    if not hydrobasins_path.exists():
        raise FileNotFoundError(f"HydroBASINS shapefile not found: {hydrobasins_path}")

    print("Loading HydroBASINS...")
    basins = gpd.read_file(hydrobasins_path)

    if basins.crs is None:
        raise ValueError("HydroBASINS has no CRS metadata. Expected EPSG:4326.")
    if str(basins.crs).upper() != "EPSG:4326":
        basins = basins.to_crs("EPSG:4326")

    point = Point(JALANDHAR_LON, JALANDHAR_LAT)
    print("Selecting basin that contains/intersects Jalandhar...")
    selected = basins[basins.geometry.intersects(point)].copy()

    if selected.empty:
        raise ValueError("No basin found for Jalandhar coordinates.")

    if len(selected) > 1:
        # Prefer smallest matching polygon when multiple candidates overlap.
        selected["__tmp_area"] = selected.geometry.area
        selected = selected.nsmallest(1, "__tmp_area").drop(columns=["__tmp_area"])

    selected["basin_id"] = args.basin_id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing output shapefile: {output_path}")
    selected.to_file(output_path)
    print("Success: Jalandhar basin shapefile generated.")


if __name__ == "__main__":
    main()

