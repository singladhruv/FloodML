from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


DEFAULT_LEAD_HOURS: Tuple[int, ...] = (24, 48, 72, 96, 120, 144, 168)


@dataclass
class BasinConfig:
    basin_id: str
    basin_shapefile: Path
    river_shapefile: Path
    cwc_gauge_csv: Path
    cwc_danger_level_m: float
    target_lat: float
    target_lon: float
    soil_moisture_date_range: Tuple[str, str] = ("2015-04-01", "2020-12-31")
    soil_moisture_scale_m: int = 10000


@dataclass
class PipelineConfig:
    start_year: int = 2010
    end_year: int = 2020
    imd_data_dir: Path = Path("data/imd_data")
    artifacts_dir: Path = Path("artifacts/hydrology")
    random_state: int = 42
    lead_hours: Tuple[int, ...] = field(default_factory=lambda: DEFAULT_LEAD_HOURS)
