from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import warnings


def _ensure_rio(ds_weather):
    if not hasattr(ds_weather, "rio"):
        raise ValueError("Dataset is missing rioxarray accessors. Install rioxarray and pass an xarray Dataset.")


def compute_basin_map_timeseries(ds_weather, basin_shapefile_path: Path, rain_var: str = "rain") -> pd.DataFrame:
    import geopandas as gpd
    import rioxarray  # noqa: F401
    from shapely.geometry import mapping
    from rioxarray.exceptions import NoDataInBounds

    if rain_var not in ds_weather:
        raise KeyError(f"Variable '{rain_var}' not found in weather dataset variables: {list(ds_weather.data_vars)}")

    basin_gdf = gpd.read_file(basin_shapefile_path).to_crs("EPSG:4326")
    _ensure_rio(ds_weather)

    x_dim = "lon" if "lon" in ds_weather.dims else "longitude"
    y_dim = "lat" if "lat" in ds_weather.dims else "latitude"

    ds_weather = ds_weather.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)
    ds_weather = ds_weather.rio.write_crs("EPSG:4326", inplace=False)

    try:
        clipped = ds_weather.rio.clip(
            basin_gdf.geometry.apply(mapping),
            basin_gdf.crs,
            drop=True,
            all_touched=True,
        )
        mean_series = clipped[rain_var].mean(dim=[y_dim, x_dim], skipna=True)
        max_series = clipped[rain_var].max(dim=[y_dim, x_dim], skipna=True)
    except NoDataInBounds:
        # Small polygons can miss grid-cell centers; fallback to centroid-nearest cell.
        centroid = basin_gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
        warnings.warn(
            f"No gridded rainfall cells found in basin bounds for {basin_shapefile_path}; "
            "falling back to centroid-nearest rainfall cell.",
            RuntimeWarning,
        )
        mean_series = ds_weather[rain_var].sel(
            {y_dim: centroid.y, x_dim: centroid.x},
            method="nearest",
        )
        max_series = mean_series

    df = mean_series.to_dataframe(name="basin_mean_rainfall").reset_index()
    max_df = max_series.to_dataframe(name="basin_max_rainfall").reset_index()
    time_col = "time" if "time" in df.columns else "valid_time"
    max_time_col = "time" if "time" in max_df.columns else "valid_time"
    df = df.rename(columns={time_col: "time"})
    max_df = max_df.rename(columns={max_time_col: "time"})
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    max_df["time"] = pd.to_datetime(max_df["time"]).dt.tz_localize(None)
    df = df.merge(max_df[["time", "basin_max_rainfall"]], on="time", how="left")
    return df[["time", "basin_mean_rainfall", "basin_max_rainfall"]].sort_values("time").reset_index(drop=True)


def add_antecedent_rainfall_features(
    df: pd.DataFrame,
    rain_col: str = "basin_mean_rainfall",
    prefix: str = "rain",
    lag_days: Sequence[int] = (1,),
    rolling_sum_days: Sequence[int] = (3, 7, 14),
    rolling_avg_days: Sequence[int] = (7,),
    dropna: bool = True,
) -> pd.DataFrame:
    frame = df.copy().sort_values("time")
    if rain_col not in frame.columns:
        raise KeyError(f"Column '{rain_col}' not found.")

    for lag in lag_days:
        frame[f"{prefix}_{lag}d_lag"] = frame[rain_col].shift(lag)

    for window in rolling_sum_days:
        frame[f"{prefix}_{window}d_sum"] = frame[rain_col].shift(1).rolling(window=window).sum()

    for window in rolling_avg_days:
        frame[f"{prefix}_{window}d_avg"] = frame[rain_col].shift(1).rolling(window=window).mean()

    if dropna:
        frame = frame.dropna()
    return frame.reset_index(drop=True)


def add_generic_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    prefix: str,
) -> pd.DataFrame:
    frame = df.copy().sort_values("date")
    for lag in lags:
        frame[f"{prefix}_{lag}d_lag"] = frame[col].shift(lag)
    for window in rolling_windows:
        frame[f"{prefix}_{window}d_avg"] = frame[col].shift(1).rolling(window=window).mean()
    return frame
