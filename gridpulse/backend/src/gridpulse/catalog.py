"""Series the API can serve, and where each one lives in Tiger Data.

Each series is one column of a `gridpulse.*_30m` view (one row per settlement period), created in
gridpulse/db/migrations/002_series_views.sql. Coarser resolutions are bucketed from it on the fly
with `agg`. To add a series: add the column to a view there, then add an entry here naming the raw
datasets the view reads.
"""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Series:
    label: str
    unit: str
    relation: str  # view in the gridpulse schema
    column: str  # also the column name in the algorithm input frame
    kind: str  # settlement (settled market values) | forecast | outturn
    datasets: tuple[str, ...]  # tables in Tiger's `raw` schema the view reads (raw.dataset_catalog names)
    agg: str = "avg"  # how 30-minute values combine into 1h/1d buckets: avg | sum | min | max

    def __post_init__(self) -> None:
        if self.agg not in ("avg", "sum", "min", "max"):  # interpolated into SQL, so keep it closed
            raise ValueError(f"unsupported aggregate {self.agg!r}")
        if self.kind not in ("settlement", "forecast", "outturn"):
            raise ValueError(f"unsupported kind {self.kind!r}")


PRICES = ("elexon_system_prices",)
WINDFOR = ("elexon_windfor",)
NDF = ("elexon_ndf",)

SERIES: dict[str, Series] = {
    "price": Series("Imbalance price", "GBP/MWh", "price_30m", "price_gbp_mwh", "settlement", PRICES),
    "price_min": Series(
        "Imbalance price, bucket low", "GBP/MWh", "price_30m", "price_gbp_mwh", "settlement", PRICES, "min"
    ),
    "price_max": Series(
        "Imbalance price, bucket high", "GBP/MWh", "price_30m", "price_gbp_mwh", "settlement", PRICES, "max"
    ),
    "niv": Series("Net imbalance volume", "MWh", "price_30m", "niv_mwh", "settlement", PRICES, "sum"),
    "wind": Series(
        "Wind forecast (WINDFOR, latest before delivery)", "MW", "wind_30m", "wind_mw", "forecast", WINDFOR
    ),
    "wind_outturn": Series(
        "Wind outturn (FUELHH)", "MW", "wind_outturn_30m", "wind_outturn_mw", "outturn", ("elexon_fuelhh",)
    ),
    "demand": Series(
        "National demand forecast (NDF, latest before delivery)", "MW", "demand_30m", "demand_mw", "forecast", NDF
    ),
    "demand_outturn": Series(
        "National demand outturn (NESO ND)", "MW", "demand_outturn_30m", "demand_outturn_mw", "outturn",
        ("neso_historic_demand",),
    ),
    "residual_demand": Series(
        "Residual demand (demand - wind forecast)", "MW", "residual_30m", "residual_demand_mw", "forecast",
        NDF + WINDFOR,
    ),
}


def series_info(series_id: str) -> dict:
    """The catalog fields of one series, as served in SeriesInfo."""
    s = SERIES[series_id]
    return {"id": series_id, "label": s.label, "unit": s.unit, "kind": s.kind, "datasets": list(s.datasets)}


def datasets_for(columns: Iterable[str]) -> list[str]:
    """Raw datasets behind these input-frame columns, sorted. Settlement date and period come
    from the clock, so they add none."""
    by_column = {s.column: s.datasets for s in SERIES.values()}
    return sorted({dataset for column in columns for dataset in by_column.get(column, ())})


# Shown in the terminal-style ticker strip by /api/snapshot.
SNAPSHOT_SERIES = ("price", "niv", "wind", "wind_outturn", "demand", "residual_demand")

# Views of every published forecast version, listed per settlement period by /api/periods.
FORECAST_VINTAGES = {
    "wind": ("wind_forecast_vintages", "wind_mw"),
    "demand": ("demand_forecast_vintages", "demand_mw"),
}
