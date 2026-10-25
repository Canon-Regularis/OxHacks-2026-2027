"""Residual diagnostics of seasonal ETS and ARIMA fits to the imbalance price (P0 #14).

Each model is fitted once to the most recent ``max_periods`` of the longest gap-free stretch of
the range; its in-sample residuals then go through Ljung-Box, Durbin-Watson, Jarque-Bera and
ARCH-LM. Tables only.
"""

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pandas as pd

from gridpulse.analysis.algorithms import _parallel, diagnostics
from gridpulse.analysis.algorithms._backtest import build
from gridpulse.analysis.algorithms._checks import require_int
from gridpulse.analysis.algorithms._frames import half_hourly, longest_complete
from gridpulse.analysis.algorithms.errors import AlgorithmError, InvalidInputError
from gridpulse.analysis.algorithms.forecasting import ARIMA, ETS, Forecaster
from gridpulse.analysis.algorithms.price_driver_correlograms import SAMPLE_COLUMNS, sample_table
from gridpulse.analysis.contract import Algorithm, AlgorithmResult

MODELS: dict[str, Callable[..., Forecaster]] = {"ets": ETS, "arima": ARIMA}
LJUNG_BOX_COLUMNS = ["model", "lag", "statistic", "df", "p_value"]
TEST_COLUMNS = ["model", "test", "statistic", "p_value", "notes"]
TABLE_COLUMNS = {"sample": SAMPLE_COLUMNS, "ljung_box": LJUNG_BOX_COLUMNS, "tests": TEST_COLUMNS}


def run(
    frame: pd.DataFrame,
    *,
    models: Mapping[str, Mapping[str, Any]],
    model_df: Mapping[str, int],
    ljung_box_lags: int,
    arch_lags: int,
    max_periods: int,
) -> AlgorithmResult:
    forecasters = build(MODELS, models)
    if set(model_df) != set(MODELS):
        raise InvalidInputError(f"model_df must give {sorted(MODELS)}, got {sorted(model_df)}")
    max_periods = require_int("max_periods", max_periods, minimum=1)
    used = longest_complete(frame[["price_gbp_mwh"]]).iloc[-max_periods:]
    ljung_box: list[list[object]] = []
    tests: list[list[object]] = []
    if len(used):
        history = half_hourly(used["price_gbp_mwh"])
        fits = _parallel.forecast_all(list(forecasters.values()), history, 1)
        for name, fit in zip(forecasters, fits, strict=True):
            try:
                if isinstance(fit, AlgorithmError):
                    raise fit
                result = diagnostics.residual_diagnostics(
                    fit.residuals,
                    ljung_box_lags=ljung_box_lags,
                    model_df=model_df[name],
                    arch_lags=arch_lags,
                )
            except AlgorithmError as exc:  # too little data, or the fit failed
                tests.append([name, "not_computed", None, None, str(exc)])
                continue
            box = result.ljung_box
            for lag, statistic, df, p in zip(
                box.lags, box.statistics, box.df, box.p_values, strict=True
            ):
                ljung_box.append([name, lag, statistic, df if p is not None else None, p])
            tests.extend(_tests(name, result))
    return AlgorithmResult(
        tables={
            "sample": sample_table(frame, used),
            "ljung_box": pd.DataFrame(ljung_box, columns=LJUNG_BOX_COLUMNS),
            "tests": pd.DataFrame(tests, columns=TEST_COLUMNS),
        }
    )


def _tests(name: str, result: diagnostics.ResidualDiagnostics) -> Iterator[list[object]]:
    jb, arch = result.jarque_bera, result.arch_lm
    yield [name, "durbin_watson", result.durbin_watson, None, "Near 2 means little lag-1 "
           "autocorrelation"]  # fmt: skip
    yield [name, "jarque_bera", jb.statistic, jb.p_value,
           f"skew {jb.skew:.2f}, excess kurtosis {jb.excess_kurtosis:.2f}"]  # fmt: skip
    yield [name, "arch_lm", arch.statistic, arch.p_value, f"{arch.lags} lags"]


ALGORITHM = Algorithm(
    id="price_residual_diagnostics",
    version="0.1.0",
    description="Ljung-Box, Durbin-Watson, Jarque-Bera and ARCH-LM tests on the in-sample "
    "residuals of seasonal ETS and ARIMA fits to the imbalance price.",
    run=run,
    inputs=("price_gbp_mwh",),
    outputs={},
    tables={
        "sample": "The gap-free stretch the models were fitted to: its first and last period, its "
        "length and how many periods of the range were left out",
        "ljung_box": "Ljung-Box Q statistic, degrees of freedom and p-value per model at lags 1..K",
        "tests": "Durbin-Watson, Jarque-Bera and ARCH-LM statistics with p-values per model; a "
        "not_computed row explains a model that could not be tested",
    },
    params={
        "models": {
            "ets": {"trend": None, "damped": False, "seasonal": "additive", "season_length": 48},
            "arima": {"p": 2, "d": 0, "q": 1, "seasonal_differencing": 1, "season_length": 48},
        },
        "model_df": {"ets": 0, "arima": 3},
        "ljung_box_lags": 48,
        "arch_lags": 12,
        "max_periods": 2688,
    },
    param_descriptions={
        "models": "Settings of each fitted model, as in price_forecast_backtest",
        "model_df": "Fitted AR + MA terms per model, subtracted from the Ljung-Box degrees of "
        "freedom",
        "ljung_box_lags": "Largest Ljung-Box lag in periods (48 = one day)",
        "arch_lags": "Lags of squared residuals in the ARCH-LM regression",
        "max_periods": "Most recent gap-free periods each model is fitted to (2688 = 56 days)",
    },
    limitations="Retrospective: parameters are estimated on the whole stretch, so residuals are "
    "never features. Statistics and p-values only, with no pass/fail threshold. P-values below "
    "about 1e-16 show as 0.",
)
