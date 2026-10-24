"""The contract every algorithm implements.

Algorithms are pure functions: a pandas DataFrame in, a pandas DataFrame out. They never touch
the database, HTTP or files; analysis/service.py does that around them.

Input frame                                     Output frame
  index   delivery_start_utc (tz-aware UTC),      index   delivery_start_utc (tz-aware UTC)
          one row per 30-min settlement period,   columns exactly the declared `outputs`
          ascending, starting `warmup` early      values  float; NaN means missing and is
  columns the declared `inputs` (INPUT_COLUMNS)           stored as NULL, never as 0
  values  NaN where the source had no data

Rows outside [range start, range end + horizon) are dropped before storage.

Results that are not one value per period (ACF by lag, metric scorecards, fitted parameters,
residual diagnostics) go in named tables: return AlgorithmResult(series=..., tables={...}).
A table is any DataFrame of at most MAX_TABLE_ROWS rows. A non-default index is kept as a column,
datetimes must be tz-aware (stored as UTC ISO strings) and NaN is stored as null. Tables are
returned as computed, so restrict them to the requested range yourself (rows from
frame.index[0] + warmup onwards).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

# Every column an algorithm may request, with its meaning. Built by analysis/features.py.
INPUT_COLUMNS = {
    "settlement_date": "Settlement day (Europe/London date)",
    "settlement_period": "Settlement period number, 1..46/48/50",
    "price_gbp_mwh": "Imbalance price, GBP/MWh (may be negative)",
    "niv_mwh": "Net imbalance volume, MWh",
    "wind_mw": "Latest WINDFOR wind forecast for the period (latest by the cutoff in cutoff mode), MW",
    "wind_outturn_mw": "FUELHH wind outturn (excludes embedded wind), MW",
    "demand_mw": "Latest NDF national demand forecast for the period, MW",
    "demand_outturn_mw": "NESO national demand outturn (ND), MW; ~21 days in arrears",
    "residual_demand_mw": "demand_mw - wind_mw, MW",
}


MAX_TABLE_ROWS = 20_000  # keeps API responses small; aggregate before returning larger tables


@dataclass
class AlgorithmResult:
    """What run() returns when it has tables as well as, or instead of, per-period series."""

    series: pd.DataFrame | None = None  # the output frame described above
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)  # name -> table


@dataclass(frozen=True)
class Algorithm:
    id: str  # stable and URL-safe, e.g. "wind_price_correlation"
    version: str  # bump when results change; runs of older versions stop being served
    description: str  # one line, shown in the UI
    run: Callable[..., pd.DataFrame | AlgorithmResult]  # run(frame, **params)
    inputs: tuple[str, ...]  # columns needed from INPUT_COLUMNS
    outputs: dict[str, str]  # output column -> unit ("" if unitless); may be empty if tables are declared
    tables: dict[str, str] = field(default_factory=dict)  # table name -> one-line description
    params: dict[str, Any] = field(default_factory=dict)  # defaults; `gridpulse analyse --param` overrides
    warmup: timedelta = timedelta(0)  # history needed before the requested range
    horizon: timedelta = timedelta(0)  # how far past the range outputs may extend (predictions)
    limitations: str = ""  # shown next to results; say what the numbers do NOT mean
