"""Unit-root tests, differencing orders and the AR spectrum of the imbalance price (P1 #16, #17).

ADF and KPSS run on the price, its first difference and its daily difference over the most
recent ``max_periods`` of the longest gap-free stretch of the range. ``difference_orders``
shows what AutoARIMA would apply, and the Burg AR spectrum where the price's variance lies by
frequency. Tables only.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal

import pandas as pd

from gridpulse.analysis.algorithms import diagnostics
from gridpulse.analysis.algorithms._checks import require_int
from gridpulse.analysis.algorithms._frames import half_hourly, longest_complete
from gridpulse.analysis.algorithms.errors import (
    AlgorithmError,
    InsufficientDataError,
    InvalidInputError,
)
from gridpulse.analysis.algorithms.evaluation import _status
from gridpulse.analysis.algorithms.price_driver_correlograms import SAMPLE_COLUMNS, sample_table
from gridpulse.analysis.algorithms.series import HalfHourlySeries
from gridpulse.analysis.contract import Algorithm, AlgorithmResult

LEVELS = (0.01, 0.025, 0.05, 0.1)  # the tabulated levels; ADF has no 2.5% value
CRITICAL = [f"critical_{str(level * 100).removesuffix('.0').replace('.', '_')}" for level in LEVELS]
UNIT_ROOT_COLUMNS = [
    "series", "test", "regression", "null_hypothesis", "statistic", *CRITICAL,
    "rejected_at", "lags", "n", "status", "message",
]  # fmt: skip
ORDER_COLUMNS = ["season_length", "seasonal_choice", "seasonal", "regular", "status", "message"]
SPECTRUM_COLUMNS = ["frequency", "period_hours", "density"]
SPECTRUM_FIT_COLUMNS = ["order", "innovation_variance", "status", "message"]
TABLE_COLUMNS = {
    "sample": SAMPLE_COLUMNS,
    "unit_root": UNIT_ROOT_COLUMNS,
    "difference_orders": ORDER_COLUMNS,
    "spectrum": SPECTRUM_COLUMNS,
    "spectrum_fit": SPECTRUM_FIT_COLUMNS,
}
REGRESSIONS: tuple[Literal["c", "ct"], ...] = ("c", "ct")


def _transforms(price: HalfHourlySeries, season_length: int) -> dict[str, HalfHourlySeries]:
    """The price and its differences, each starting where its first value is defined."""
    values = price.complete_values()

    def differenced(lag: int) -> HalfHourlySeries:
        changes = [b - a for a, b in zip(values, values[lag:], strict=False)]
        return HalfHourlySeries(price.timestamp(lag), changes)

    return {
        "price": price,
        "price_change_1": differenced(1),
        f"price_change_{season_length}": differenced(season_length),
    }


def _need(n: int, required: int, what: str) -> None:
    """InsufficientDataError when a sample of ``n`` is too short for the configured lags or
    order, which the adapters would otherwise reject as invalid settings."""
    if n < required:
        raise InsufficientDataError(required, n, what=what)


Regression = Literal["c", "ct"]


def _adf(series: HalfHourlySeries, lags: int, regression: Regression) -> diagnostics.UnitRootTest:
    _need(len(series), 2 * lags + 5 + (regression == "ct"), f"values for {lags} ADF lags")
    return diagnostics.adf_test(series, lags=lags, regression=regression)


def _kpss(series: HalfHourlySeries, lags: int, regression: Regression) -> diagnostics.UnitRootTest:
    _need(len(series), 5 * lags, f"values for {lags} KPSS lags")
    return diagnostics.kpss_test(series, lags=lags, regression=regression)


def _unit_root(
    name: str,
    test: str,
    regression: str,
    run_test: Callable[[], diagnostics.UnitRootTest],
) -> list[object]:
    try:
        result = run_test()
    except AlgorithmError as exc:
        missing = [None] * (len(UNIT_ROOT_COLUMNS) - 5)
        return [name, test, regression, *missing, _status(exc), str(exc)]
    critical = [result.critical_values.get(level) for level in LEVELS]
    rejected = ", ".join(f"{level:.1%}".replace(".0%", "%") for level in result.rejected_at)
    found = [result.null_hypothesis, result.statistic, *critical, rejected or "none"]
    return [name, test, regression, *found, result.lags, result.nobs, "ok", None]


def _orders(price: HalfHourlySeries, season_length: int) -> list[list[object]]:
    rows: list[list[object]] = []
    for choice, fixed in (("tested", None), ("fixed", 1)):
        try:
            found = diagnostics.difference_orders(
                price, season_length=season_length, seasonal_differencing=fixed
            )
        except AlgorithmError as exc:
            rows.append([season_length, choice, fixed, None, _status(exc), str(exc)])
        else:
            rows.append([season_length, choice, found.seasonal, found.regular, "ok", None])
    return rows


def run(
    frame: pd.DataFrame,
    *,
    season_length: int,
    adf_lags: int,
    kpss_lags: int,
    regressions: Sequence[str],
    spectrum_order: int,
    nfft: int,
    max_periods: int,
) -> AlgorithmResult:
    season_length = require_int("season_length", season_length, minimum=3)
    max_periods = require_int("max_periods", max_periods, minimum=1)
    checked = [regression for regression in REGRESSIONS if regression in regressions]
    if not regressions or not set(regressions) <= set(REGRESSIONS):
        raise InvalidInputError(
            f"regressions must be some of {list(REGRESSIONS)}, got {regressions}"
        )
    used = longest_complete(frame[["price_gbp_mwh"]]).iloc[-max_periods:]
    tables = {name: pd.DataFrame(columns=columns) for name, columns in TABLE_COLUMNS.items()}
    tables["sample"] = sample_table(frame, used)
    if not len(used):
        return AlgorithmResult(tables=tables)
    price = half_hourly(used["price_gbp_mwh"])

    unit_root: list[list[object]] = []
    for name, series in _transforms(price, season_length).items():
        for regression in checked:
            adf = partial(_adf, series, adf_lags, regression)
            kpss = partial(_kpss, series, kpss_lags, regression)
            unit_root.append(_unit_root(name, "adf", regression, adf))
            unit_root.append(_unit_root(name, "kpss", regression, kpss))
    tables["unit_root"] = pd.DataFrame(unit_root, columns=UNIT_ROOT_COLUMNS)
    tables["difference_orders"] = pd.DataFrame(_orders(price, season_length), columns=ORDER_COLUMNS)

    try:
        _need(len(price), 2 * spectrum_order + 2, f"values for an AR({spectrum_order}) spectrum")
        spectrum = diagnostics.ar_spectrum(price, order=spectrum_order, nfft=nfft)
    except AlgorithmError as exc:
        tables["spectrum_fit"] = pd.DataFrame(
            [[spectrum_order, None, _status(exc), str(exc)]], columns=SPECTRUM_FIT_COLUMNS
        )
    else:
        tables["spectrum"] = pd.DataFrame(
            [
                [f, 1 / (2 * f) if f else None, d]
                for f, d in zip(spectrum.frequencies, spectrum.density, strict=True)
            ],
            columns=SPECTRUM_COLUMNS,
        )
        tables["spectrum_fit"] = pd.DataFrame(
            [[spectrum.order, spectrum.innovation_variance, "ok", None]],
            columns=SPECTRUM_FIT_COLUMNS,
        )
    return AlgorithmResult(tables=tables)


ALGORITHM = Algorithm(
    id="price_stationarity",
    version="0.1.0",
    description="ADF and KPSS tests on the imbalance price, its half-hourly change and its daily "
    "change; the differences AutoARIMA would apply; and the price's AR spectrum.",
    run=run,
    inputs=("price_gbp_mwh",),
    outputs={},
    tables={
        "sample": "The gap-free stretch tested: its first and last period, its length and how "
        "many periods of the range were left out",
        "unit_root": "ADF (null: unit root) and KPSS (null: stationary) per series and "
        "deterministic term: statistic, critical values at 1%, 2.5% (KPSS only), 5% and 10%, "
        "the levels at which the null is rejected, lags and observations",
        "difference_orders": "Seasonal and regular differences (0 or 1) AutoARIMA would apply, "
        "with the seasonal difference tested and with it fixed at 1",
        "spectrum": "AR spectral density in (GBP/MWh)^2 per cycle per half-hour, by frequency "
        "in cycles per half-hour, with the matching period in hours",
        "spectrum_fit": "The Burg AR order and innovation variance behind the spectrum, or why "
        "it could not be computed",
    },
    params={
        "season_length": 48,
        "adf_lags": 48,
        "kpss_lags": 48,
        "regressions": ["c", "ct"],
        "spectrum_order": 48,
        "nfft": 1344,
        "max_periods": 2688,
    },
    param_descriptions={
        "season_length": "Seasonal period in half-hours for the seasonal difference and its "
        "test (48 = daily)",
        "adf_lags": "Lagged differences in each ADF regression (48 = one day); fixed, not selected",
        "kpss_lags": "Autocovariances in KPSS's long-run variance; at least 48 for half-hourly "
        "data, or the test rejects stationarity too often",
        "regressions": "Deterministic terms tested: c (constant) and/or ct (constant and linear "
        "trend)",
        "spectrum_order": "AR order of the Burg fit; 48 can resolve every harmonic of the daily "
        "cycle",
        "nfft": "Frequency grid size; a multiple of 336 puts daily and weekly harmonics on it",
        "max_periods": "Most recent gap-free periods tested (2688 = 56 days)",
    },
    limitations="Retrospective: statistics of the whole stretch, with no pass/fail verdict. "
    "There are no p-values, only the tabulated levels at which each null is rejected; the "
    "tests assume a constant variance, which imbalance prices do not have. ADF and KPSS have "
    "opposite nulls, so read them together. difference_orders uses Orchid's default KPSS lags, "
    "which can over-difference persistent series, while 48 KPSS lags make it slow to reject on a few weeks of data. The spectrum is of an AR model fitted to the "
    "price, not a raw periodogram.",
)
