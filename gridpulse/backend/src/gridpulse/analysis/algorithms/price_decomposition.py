"""Seasonal decomposition and centred smoothers of the imbalance price (P1 #18, #19).

STL and the classical decomposition split the most recent ``max_periods`` of the longest
gap-free stretch of the range into trend, seasonal and remainder, and the component strengths
come from the classical one. The smoothers run over every run of present prices. Everything
here is two-sided, so it is for charts and retrospective analysis, never a feature.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from gridpulse.analysis.algorithms import decomposition
from gridpulse.analysis.algorithms._checks import require_int
from gridpulse.analysis.algorithms._frames import as_floats, half_hourly, longest_complete
from gridpulse.analysis.algorithms.errors import AlgorithmError
from gridpulse.analysis.algorithms.evaluation import _status
from gridpulse.analysis.algorithms.price_driver_correlograms import SAMPLE_COLUMNS, sample_table
from gridpulse.analysis.algorithms.series import HALF_HOUR, HalfHourlySeries
from gridpulse.analysis.contract import Algorithm, AlgorithmResult

GBP = "GBP/MWh"
# Window lengths are part of the output names, so they are fixed rather than parameters.
MEDIAN, SAVGOL, SAVGOL_DEGREE, LOWESS = 9, 13, 2, 49
METHODS = ("stl", "classical")
COMPONENTS = ("trend", "seasonal", "remainder")
OUTPUTS = {f"{method}_{part}": GBP for method in METHODS for part in COMPONENTS} | {
    f"price_median_{MEDIAN}": GBP,
    f"price_savgol_{SAVGOL}": GBP,
    f"price_lowess_{LOWESS}": GBP,
}
INDEX_COLUMNS = ["phase", "first_utc", "stl_gbp_mwh", "classical_gbp_mwh"]
STRENGTH_COLUMNS = ["period", "trend_strength", "seasonal_strength", "status", "message"]
METHOD_COLUMNS = ["method", "status", "message"]
TABLE_COLUMNS = {
    "sample": SAMPLE_COLUMNS,
    "seasonal_indices": INDEX_COLUMNS,
    "strength": STRENGTH_COLUMNS,
    "methods": METHOD_COLUMNS,
}


def _smoothed(price: HalfHourlySeries) -> dict[str, HalfHourlySeries]:
    return {
        f"price_median_{MEDIAN}": decomposition.median_filter(price, MEDIAN),
        f"price_savgol_{SAVGOL}": decomposition.savitzky_golay(price, SAVGOL, SAVGOL_DEGREE),
        f"price_lowess_{LOWESS}": decomposition.lowess(price, LOWESS),
    }


def run(
    frame: pd.DataFrame,
    *,
    period: int,
    seasonal_window: int,
    trend_window: int | None,
    max_periods: int,
) -> AlgorithmResult:
    period = require_int("period", period, minimum=2)
    max_periods = require_int("max_periods", max_periods, minimum=1)
    out = pd.DataFrame(np.nan, index=frame.index, columns=list(OUTPUTS))
    for name, series in _smoothed(half_hourly(frame["price_gbp_mwh"])).items():
        out[name] = as_floats(series)

    used = longest_complete(frame[["price_gbp_mwh"]]).iloc[-max_periods:]
    first = [used.index[0] + j * HALF_HOUR if len(used) >= period else None for j in range(period)]
    indices = pd.DataFrame({"phase": range(period), "first_utc": first})
    methods: list[list[object]] = []
    strength: list[object] = [period, None, None, "insufficient_data", "no gap-free prices"]
    if len(used):
        sample = half_hourly(used["price_gbp_mwh"])
        decompose: dict[str, Callable[[], decomposition.Decomposition]] = {
            "stl": lambda: decomposition.stl_decomposition(
                sample, period=period, seasonal_window=seasonal_window, trend_window=trend_window
            ),
            "classical": lambda: decomposition.classical_decomposition(sample, period=period),
        }
        for method, make in decompose.items():
            try:
                result = make()
            except AlgorithmError as exc:  # too few complete cycles: this method stays empty
                methods.append([method, _status(exc), str(exc)])
                continue
            methods.append([method, "ok", None])
            for part in COMPONENTS:
                out.loc[used.index, f"{method}_{part}"] = as_floats(getattr(result, part))
            indices[f"{method}_gbp_mwh"] = result.seasonal_indices
        try:
            found = decomposition.component_strength(sample, period=period)
            strength = [period, found.trend, found.seasonal, "ok", None]
        except AlgorithmError as exc:
            strength = [period, None, None, _status(exc), str(exc)]
    else:
        methods = [[method, "insufficient_data", "no gap-free prices"] for method in METHODS]
    for method in METHODS:
        if f"{method}_gbp_mwh" not in indices:
            indices[f"{method}_gbp_mwh"] = None
    return AlgorithmResult(
        series=out,
        tables={
            "sample": sample_table(frame, used),
            "seasonal_indices": indices[INDEX_COLUMNS],
            "strength": pd.DataFrame([strength], columns=STRENGTH_COLUMNS),
            "methods": pd.DataFrame(methods, columns=METHOD_COLUMNS),
        },
    )


ALGORITHM = Algorithm(
    id="price_decomposition",
    version="0.1.0",
    description="Imbalance price split into trend, daily seasonal and remainder by STL and by "
    "classical decomposition, with trend and seasonal strengths and three centred smoothers "
    "(9-period median, 13-period Savitzky-Golay, 49-period LOWESS).",
    run=run,
    inputs=("price_gbp_mwh",),
    outputs=OUTPUTS,
    tables={
        "sample": "The gap-free stretch decomposed: its first and last period, its length and how "
        "many periods of the range were left out",
        "seasonal_indices": "Seasonal effect per phase in GBP/MWh by each method (STL's averaged "
        "over cycles), with the first period of each phase",
        "strength": "Trend and seasonal strengths from 0 to 1 of the classical decomposition",
        "methods": "Whether each decomposition ran; a row that did not explains why",
    },
    params={"period": 48, "seasonal_window": 7, "trend_window": None, "max_periods": 2688},
    param_descriptions={
        "period": "Seasonal period in half-hours (48 = daily, 336 = weekly)",
        "seasonal_window": "STL seasonal smoothing span in cycles; odd, at least 5. STL needs "
        "this many complete cycles",
        "trend_window": "STL trend smoothing span in periods; odd, at least 5. null takes "
        "Cleveland's default (93 at period 48)",
        "max_periods": "Most recent gap-free periods decomposed (2688 = 56 days)",
    },
    limitations="Retrospective and two-sided: every component and smoother uses later prices "
    "as well as earlier ones, so none was knowable at the time and none is a forecast or a "
    "feature. Phases count elapsed half-hours from the first decomposed period, so after a "
    "clock change the daily profile is an hour off the local clock. The classical trend and "
    "remainder are empty within half a period of the stretch's ends, and each smoother within "
    "half its window of any gap. Orchid's STL omits the low-pass filter, and a price spike tens "
    "of standard deviations high can leak into the seasonal. Strengths describe this stretch "
    "only.",
)
