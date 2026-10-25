"""Vector autoregression backed by Orchid (P1 #26).

A VAR(p) regresses each series on a constant and the previous p values of every series. Fits
run on z-scores and are mapped back to the data's units, so Orchid's absolute floors (a 1e-10
ridge, 1e-12 pivots) act alike in every unit. Lags count elapsed half-hours and there are no
seasonal terms: deseasonalise half-hourly data first, because small lag orders cannot represent
the daily cycle.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from types import MappingProxyType
from typing import Any

from gridpulse.analysis.algorithms._checks import orchid_errors, require_int, require_variation
from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
)
from gridpulse.analysis.algorithms.forecasting import (
    _MAX_REGRESSORS,
    DEFAULT_LEVELS,
    Forecast,
    PredictionInterval,
    _finite,
    _forecast,
    _prepare,
    _rescaled,
)
from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.series import HALF_HOUR, HalfHourlySeries

# Orchid floors forecast variances at 1e-12, here a share of the standardised history's variance.
_FORECAST_VARIANCE_FLOOR = 1e-12
# Orchid factors the standardised residual covariance after adding 1e-10 to its diagonal, so even
# a singular one factors. Each series' own innovations must carry 100 times that.
_MIN_INNOVATION_VARIANCE = 1e-8


@dataclass(frozen=True)
class GrangerTest:
    """F test that the lags of one series add nothing to the equation of another.

    ``df`` holds the numerator and denominator degrees of freedom.
    """

    statistic: float
    p_value: float
    df: tuple[int, int]


@dataclass(frozen=True)
class VarAnalysis:
    """A VAR fitted to the whole sample, in the data's units.

    ``coefficients`` maps each equation to its regressors, named "<series>.L<lag>", and
    ``residual_covariance`` is the maximum-likelihood estimate, which divides by ``nobs``.
    ``impulse_responses[h]`` maps each response series to its change h periods after a
    one-standard-deviation orthogonalised shock to each series, in the response's units. Shocks
    are orthogonalised by Cholesky in the series' order, so on impact a series responds only to
    its own shock and those of earlier series. ``variance_decomposition[s - 1]`` maps each
    response series to the share of its s-step forecast-error variance due to each shock.
    ``granger`` maps (caused, causing) to the test that the lags of ``causing`` add nothing to
    the equation of ``caused``.
    """

    names: tuple[str, ...]
    lags: int
    nobs: int
    intercept: Mapping[str, float]
    coefficients: Mapping[str, Mapping[str, float]]
    residual_covariance: Mapping[str, Mapping[str, float]]
    log_likelihood: float
    aic: float
    bic: float
    impulse_responses: tuple[Mapping[str, Mapping[str, float]], ...]
    variance_decomposition: tuple[Mapping[str, Mapping[str, float]], ...]
    granger: Mapping[tuple[str, str], GrangerTest]


@dataclass(frozen=True)
class VarLagSelection:
    """AIC and BIC of VAR(p) by p, every fit on the same last ``nobs`` periods."""

    nobs: int
    aic: Mapping[int, float]
    bic: Mapping[int, float]


@dataclass(frozen=True)
class VAR:
    """VAR(lags) forecast of the history, fitted to it and the drivers over its periods.

    Each driver must cover the history's periods and be complete over them. Values outside
    them are never used: the VAR forecasts the drivers along with the history.
    """

    drivers: Mapping[str, HalfHourlySeries]
    lags: int

    def __post_init__(self) -> None:
        drivers = dict(self.drivers)
        if not drivers:
            raise InvalidInputError("a VAR forecast needs at least one driver")
        _check_model(list(drivers), "lags", self.lags, k=1 + len(drivers))
        object.__setattr__(self, "drivers", MappingProxyType(drivers))

    def __reduce__(self) -> tuple[Any, ...]:  # a mappingproxy cannot be pickled
        return type(self), (dict(self.drivers), self.lags)

    @property
    def min_history(self) -> int:
        return _min_observations(1 + len(self.drivers), self.lags)

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        covered = [_covering(name, driver, history) for name, driver in self.drivers.items()]
        labels = ["history", *(f"driver {name}" for name in self.drivers)]
        means, scales, rows = _standardised(labels, [values, *covered])
        result = _fit(rows, self.lags)
        # The history is variable 0, the only one whose intervals Orchid's forecast gives.
        if result.sigma_u[0][0] <= _FORECAST_VARIANCE_FLOOR:
            raise DegenerateDataError(
                "the history is a linear function of past values to within 1e-12 of its variance"
            )
        with orchid_errors("VAR forecast"):
            output = result.forecast(horizon, level=levels)
        mean, scale = means[0], scales[0]
        intervals = tuple(
            PredictionInterval(
                level,
                _unstandardised(output.lower[level], mean, scale),
                _unstandardised(output.upper[level], mean, scale),
            )
            for level in levels
        )
        return _forecast(
            history,
            source="autoregressive.VAR",
            params={"drivers": tuple(self.drivers), "lags": self.lags},
            mean=_unstandardised(output.mean, mean, scale),
            intervals=intervals,
            interval_method="analytic",
            residuals=_rescaled([row[0] for row in result.resid], scale),
        )


def select_var_lags(series: Mapping[str, HalfHourlySeries], *, max_lags: int) -> VarLagSelection:
    """AIC and BIC of VAR(p) for p = 1..max_lags, each fitted to the last n - max_lags periods.

    The common sample keeps the criteria comparable across p. Like ``var_analysis``, this is
    retrospective.
    """
    names, columns = _columns(series, "max_lags", max_lags)
    _, scales, rows = _standardised(names, columns)
    nobs = len(rows) - max_lags
    # Standardising raises the Gaussian log-likelihood by nobs * sum(ln scale), so each
    # criterion is twice that higher in the data's units.
    shift = nobs * math.fsum(map(math.log, scales))
    aic, bic = {}, {}
    for p in range(1, max_lags + 1):
        result = _fit(rows[max_lags - p :], p)
        _check_innovations(names, result.sigma_u)
        aic[p] = result.aic + 2 * shift
        bic[p] = result.bic + 2 * shift
    return VarLagSelection(nobs, aic, bic)


def var_analysis(series: Mapping[str, HalfHourlySeries], *, lags: int, horizon: int) -> VarAnalysis:
    """A VAR(lags) of ``series``, with responses and decompositions up to ``horizon`` periods.

    Retrospective: every parameter is estimated from the whole sample, so no output was
    available at an earlier cutoff. The mapping's order is the Cholesky order of the shocks.
    Granger tests measure predictive association, not causation, and their p-values assume
    stationary series (check with ``adf_test`` and ``kpss_test``). Small lag orders cannot
    represent the daily cycle, so deseasonalise half-hourly data first.
    """
    require_int("horizon", horizon, minimum=1)
    names, columns = _columns(series, "lags", lags)
    means, s, rows = _standardised(names, columns)
    result = _fit(rows, lags)
    _check_innovations(names, result.sigma_u)
    k = len(names)
    pairs = [(i, j) for i in range(k) for j in range(k) if i != j]
    with orchid_errors("VAR analysis"):
        responses = result.irf(horizon).irf
        shares = result.fevd(horizon - 1).fevd  # fevd(h)[t] decomposes the (t + 1)-step error
        tests = [result.test_causality(i, causing=j) for i, j in pairs]
    # The data are means + s * z, so each lag matrix becomes S A S^-1, the covariance S Sigma S
    # and the responses of series i scale by s[i]; the intercepts absorb the means.
    coefs = [[[s[i] * m[i][j] / s[j] for j in range(k)] for i in range(k)] for m in result.coefs]
    intercept = [
        means[i]
        + s[i] * result.intercept[i]
        - math.fsum(coef[i][j] * means[j] for coef in coefs for j in range(k))
        for i in range(k)
    ]
    covariance = [[s[i] * s[j] * result.sigma_u[i][j] for j in range(k)] for i in range(k)]
    impulse = [[[s[i] * value for value in row] for i, row in enumerate(m)] for m in responses]
    shift = result.nobs * math.fsum(map(math.log, s))  # see select_var_lags
    criteria = [result.llf - shift, result.aic + 2 * shift, result.bic + 2 * shift]
    estimates = [
        intercept,
        criteria,
        *covariance,
        *(row for matrix in [*coefs, *impulse, *shares] for row in matrix),
        *([test.statistic, test.pvalue] for test in tests),
    ]
    _check_finite(estimates, "the VAR's estimates are not finite in the data's units")
    log_likelihood, aic, bic = criteria
    return VarAnalysis(
        names=tuple(names),
        lags=lags,
        nobs=result.nobs,
        intercept=dict(zip(names, intercept, strict=True)),
        coefficients={
            name: {
                f"{names[j]}.L{lag}": coef[i][j]
                for lag, coef in enumerate(coefs, start=1)
                for j in range(k)
            }
            for i, name in enumerate(names)
        },
        residual_covariance=_table(names, covariance),
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        impulse_responses=tuple(_table(names, matrix) for matrix in impulse),
        variance_decomposition=tuple(_table(names, table) for table in shares),
        granger={
            (names[i], names[j]): GrangerTest(test.statistic, test.pvalue, test.df)
            for (i, j), test in zip(pairs, tests, strict=True)
        },
    )


def _check_model(names: Sequence[object], lags_name: str, lags: object, *, k: int) -> None:
    """Non-empty string names, and a lag order that keeps the equations of a VAR of ``k``
    series within the regressor limit."""
    for name in names:
        if not isinstance(name, str) or not name:
            raise InvalidInputError(f"series names must be non-empty strings, got {name!r}")
    width = 1 + k * require_int(lags_name, lags, minimum=1)
    if width > _MAX_REGRESSORS:
        raise InvalidInputError(
            f"each equation has {width} regressors; at most {_MAX_REGRESSORS} are allowed"
        )


def _min_observations(k: int, lags: int) -> int:
    """The shortest sample for a VAR of ``k`` series: each equation fits 1 + k * lags
    coefficients to the periods after the first ``lags``, and its residuals need ``k`` degrees
    of freedom for the k x k residual covariance to be non-singular."""
    return (k + 1) * lags + k + 1


def _columns(
    series: Mapping[str, HalfHourlySeries], lags_name: str, lags: int
) -> tuple[list[str], list[list[float]]]:
    """The names and complete values of ``series``, checked for a VAR with ``lags`` lags."""
    names = list(series)
    if len(names) < 2:
        raise InvalidInputError(f"a VAR needs at least two series, got {len(names)}")
    _check_model(names, lags_name, lags, k=len(names))
    first = series[names[0]]
    for name, other in series.items():
        if (other.start, len(other)) != (first.start, len(first)):
            raise InvalidInputError(
                f"the series must cover the same periods: {name} has {len(other)} from "
                f"{other.start.isoformat()} and {names[0]} {len(first)} from "
                f"{first.start.isoformat()}"
            )
    columns = [other.complete_values() for other in series.values()]
    minimum = _min_observations(len(names), lags)
    if len(first) < minimum:
        raise InsufficientDataError(minimum, len(first))
    return names, columns


def _covering(name: str, driver: HalfHourlySeries, history: HalfHourlySeries) -> list[float]:
    """The driver's values over the history's periods, which it must cover."""
    first = (history.start - driver.start) // HALF_HOUR
    if first < 0 or first + len(history) > len(driver):
        raise InvalidInputError(
            f"driver {name} ({len(driver)} periods from {driver.start.isoformat()}) does not "
            f"cover the history ({len(history)} from {history.start.isoformat()})"
        )
    return driver.window(first, first + len(history)).complete_values()


def _standardised(
    labels: Sequence[str], columns: Sequence[list[float]]
) -> tuple[list[float], list[float], list[list[float]]]:
    """The mean and standard deviation of each column, and the z-scores as one row per period."""
    means, scales, scores = [], [], []
    for label, values in zip(labels, columns, strict=True):
        require_variation(label, values)
        spread = max(values) - min(values)
        if not math.isfinite(spread):
            raise CalculationError(f"the range of {label} overflows a float")
        unit = [value / spread for value in values]  # under 1e9 in size: the variation check
        centre = fmean(unit)
        scale = math.sqrt(fmean([(u - centre) ** 2 for u in unit]))  # at least sqrt(0.5 / n)
        means.append(centre * spread)
        scales.append(scale * spread)
        scores.append([(u - centre) / scale for u in unit])
    return means, scales, [list(row) for row in zip(*scores, strict=True)]


def _fit(rows: list[list[float]], lags: int) -> ar.VARResults:
    """Orchid's VAR(lags) of standardised rows, with finite estimates."""
    with orchid_errors("VAR"):
        result = ar.VAR(rows).fit(maxlags=lags)
    estimates = [
        result.intercept,
        [result.llf, result.aic, result.bic],
        *result.sigma_u,
        *result.resid,
        *(row for matrix in result.coefs for row in matrix),
    ]
    _check_finite(estimates, "Orchid VAR returned non-finite estimates")
    return result


def _check_innovations(names: Sequence[str], sigma: ar.Matrix) -> None:
    """``DegenerateDataError`` unless every series has innovations of its own.

    ``sigma`` is the residual covariance of the standardised series, and its squared Cholesky
    pivots are the variance of each series' innovations given those of earlier series. A pivot
    near zero, from a series that is a linear function of past values and of earlier series,
    makes ``sigma`` singular: the log-likelihood is then unbounded (Orchid floors it) and the
    orthogonalised shocks are not identified (Orchid's jitter picks them). Pairwise
    correlations miss identities among three or more series, such as residual demand =
    demand - wind.
    """
    factor: list[list[float]] = []
    for i, name in enumerate(names):
        row: list[float] = []
        for j in range(i):
            dot = math.fsum(a * b for a, b in zip(row, factor[j][:j], strict=True))
            row.append((sigma[i][j] - dot) / factor[j][j])
        variance = sigma[i][i] - math.fsum(value * value for value in row)
        if variance < _MIN_INNOVATION_VARIANCE:
            earlier = f" and of current {', '.join(names[:i])}" if i else ""
            raise DegenerateDataError(
                f"{name} is a linear function of past values{earlier} to within 1e-8 of its "
                "variance"
            )
        factor.append([*row, math.sqrt(variance)])


def _check_finite(rows: Iterable[Sequence[float]], message: str) -> None:
    if not all(math.isfinite(value) for row in rows for value in row):
        raise CalculationError(message)


def _unstandardised(values: Sequence[float] | None, mean: float, scale: float) -> tuple[float, ...]:
    """Orchid's forecasts of a standardised series, in the data's units."""
    return _finite([mean + scale * value for value in _finite(values)])


def _table(names: Sequence[str], matrix: ar.Matrix) -> dict[str, dict[str, float]]:
    """``matrix[i][j]`` keyed by the names of series i and j."""
    return {
        name: dict(zip(names, row, strict=True)) for name, row in zip(names, matrix, strict=True)
    }
