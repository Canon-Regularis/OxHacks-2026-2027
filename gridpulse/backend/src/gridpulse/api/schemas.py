"""Response shapes. Keep aligned with gridpulse/contracts/examples/ (a test checks they parse).

Time series are columnar: one shared `t` array of UTC bucket starts, and one `values` array
per series of the same length, with null for missing. This maps directly onto chart libraries
such as uPlot and keeps multi-thousand-point responses small.
Errors are `{"detail": {"code": "...", "message": "..."}}`.
"""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Resolution = Literal["30m", "1h", "1d"]


class Model(BaseModel):
    # Lets code use from_=... while JSON uses "from".
    model_config = ConfigDict(populate_by_name=True)


class Health(Model):
    status: Literal["ok", "degraded"]
    mode: Literal["live", "record", "replay"]
    database: Literal["ok", "unavailable", "not_configured", "not_used"]
    version: str


class SeriesInfo(Model):
    id: str
    label: str
    unit: str
    kind: Literal["settlement", "forecast", "outturn"]
    datasets: list[str]  # raw datasets it is built from; details in Catalog.datasets


class DatasetCoverage(Model):
    """One raw dataset, as recorded by the data loader in raw.dataset_catalog."""

    dataset: str
    source: str | None
    notes: str | None
    first_utc: datetime | None
    last_utc: datetime | None
    loaded_at_utc: datetime | None
    approx_rows: int | None


class RunInfo(Model):
    run_id: int
    algorithm_id: str
    algorithm_version: str
    params: dict  # every parameter the run used
    overrides: dict  # the params that differ from the algorithm's current defaults; {} for a default run
    datasets: dict[str, datetime | None]  # raw dataset read -> its load time then (None: not recorded)
    range_start_utc: datetime
    range_end_utc: datetime
    cutoff_utc: datetime | None
    finished_at_utc: datetime | None


class AlgorithmInfo(Model):
    id: str
    version: str
    description: str
    inputs: list[str]
    datasets: list[str]  # raw datasets behind the inputs
    outputs: dict[str, str]  # name -> unit
    tables: dict[str, str] = {}  # name -> description
    params: dict  # default parameters
    param_descriptions: dict[str, str]  # param name -> meaning
    limitations: str
    latest_run: RunInfo | None = None  # the run /api/analysis serves by default


class Limits(Model):
    max_points: int
    max_range_days: int


class Catalog(Model):
    series: list[SeriesInfo]
    datasets: list[DatasetCoverage]
    algorithms: list[AlgorithmInfo]
    limits: Limits
    data_updated_at_utc: datetime | None


class Column(Model):
    label: str
    unit: str
    values: list[float | None]


class SeriesResponse(Model):
    from_: date = Field(alias="from")
    to: date
    resolution: Resolution
    cutoff_utc: datetime | None
    t: list[datetime]
    settlement_date: list[date] | None  # only at 30m resolution
    settlement_period: list[int] | None
    series: dict[str, Column]


class TickerItem(Model):
    id: str
    label: str
    unit: str
    value: float | None
    previous: float | None
    change: float | None


class Snapshot(Model):
    as_of_utc: datetime
    settlement_date: date
    settlement_period: int
    items: list[TickerItem]
    data_updated_at_utc: datetime | None


class Extreme(Model):
    delivery_start_utc: datetime
    settlement_date: date
    settlement_period: int
    value: float


class ExtremesResponse(Model):
    series: SeriesInfo
    from_: date = Field(alias="from")
    to: date
    order: Literal["desc", "asc"]
    rows: list[Extreme]


class Vintage(Model):
    published_at_utc: datetime
    value: float | None


class PeriodIndicator(Model):
    algorithm_id: str
    algorithm_version: str
    run_id: int
    indicator: str
    unit: str | None
    value: float | None


class PeriodDetail(Model):
    settlement_date: date
    settlement_period: int
    delivery_start_utc: datetime
    delivery_end_utc: datetime
    values: dict[str, float | None]  # series id -> value; labels/units are in the catalog
    vintages: dict[str, list[Vintage]]  # forecast history for this period
    analysis: list[PeriodIndicator]


class AnalysisResponse(Model):
    algorithm: AlgorithmInfo
    run: RunInfo
    from_: date = Field(alias="from")
    to: date
    resolution: Resolution
    t: list[datetime]
    indicators: dict[str, Column]
    tables: list[str] = []  # stored table names; fetch from /api/analysis/runs/{run_id}/tables
    warnings: list[str]


class RunList(Model):
    algorithm: AlgorithmInfo
    runs: list[RunInfo]  # newest first; every version, params and cutoff


class Table(Model):
    description: str
    columns: list[str]
    rows: list[list[Any]]  # cells are numbers, strings, booleans or null; datetimes are UTC ISO strings


class RunTables(Model):
    run: RunInfo
    tables: dict[str, Table]


class WeatherLocation(Model):
    id: str
    label: str
    latitude: float
    longitude: float


class WeatherLocations(Model):
    locations: list[WeatherLocation]
    warnings: list[str] = []


class WeatherPoint(Model):
    # t is the END of a completed UTC hour. No later observations enter its statistics.
    t: datetime
    temperature: float | None
    pressure: float | None
    temperature_change: float | None
    pressure_change: float | None
    price_change: float | None
    temperature_correlation: float | None
    pressure_correlation: float | None
    temperature_samples: int
    pressure_samples: int


class WeatherResponse(Model):
    location: WeatherLocation
    points: list[WeatherPoint]
    source: str
    warnings: list[str]
    window_hours: int
