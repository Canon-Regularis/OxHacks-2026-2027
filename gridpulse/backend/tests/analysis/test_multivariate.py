import copy
import math
import pickle
import random
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime

import pytest

from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
    MissingDataError,
)
from gridpulse.analysis.algorithms.forecasting import Forecaster
from gridpulse.analysis.algorithms.multivariate import (
    VAR,
    VarAnalysis,
    select_var_lags,
    var_analysis,
)
from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.series import HalfHourlySeries

START = datetime(2026, 9, 1, tzinfo=UTC)
NAMES = ("price", "wind", "demand")
MakeSeries = Callable[..., HalfHourlySeries]


def _simulate(n: int) -> dict[str, list[float]]:
    """A VAR(1) in which prices, often negative, follow the previous period's wind and demand,
    which evolve on their own."""
    rng = random.Random(0)
    price, wind, demand = 0.0, 5000.0, 30000.0
    values: dict[str, list[float]] = {name: [] for name in NAMES}
    for t in range(n + 50):  # the first 50 periods are burn-in
        price = 0.5 * price - 0.004 * (wind - 5000) + 0.001 * (demand - 30000) + rng.gauss(0, 8)
        wind = 5000 + 0.8 * (wind - 5000) + rng.gauss(0, 600)
        demand = 30000 + 0.7 * (demand - 30000) + rng.gauss(0, 1500)
        if t >= 50:
            for name, value in zip(NAMES, (price, wind, demand), strict=True):
                values[name].append(value)
    return values


def _series(columns: Mapping[str, Sequence[float | None]]) -> dict[str, HalfHourlySeries]:
    return {name: HalfHourlySeries(START, values) for name, values in columns.items()}


DATA = _simulate(400)
ROWS = [list(row) for row in zip(*DATA.values(), strict=True)]  # Orchid's layout, price first
SERIES = _series(DATA)
PRICE, WIND, DEMAND = (SERIES[name] for name in NAMES)
DRIVERS = {"wind": WIND, "demand": DEMAND}
TWO = {"price": PRICE, "wind": WIND}
CONSTANT = HalfHourlySeries(START, [30.0] * 400)
GAPPY = HalfHourlySeries(START, [*DATA["wind"][:50], None, *DATA["wind"][51:]])
SINE = [math.sin(2 * math.pi * t / 48) for t in range(400)]  # an exact AR(2)


def test_forecast_matches_orchid_on_raw_data() -> None:
    forecaster = VAR(DRIVERS, 2)
    result = forecaster.forecast(PRICE, 24, levels=(0.95, 0.8))
    raw = ar.VAR(ROWS).fit(maxlags=2)
    expected = raw.forecast(24, level=(0.8, 0.95))

    assert result == forecaster.forecast(PRICE, 24, levels=(0.95, 0.8))
    assert (result.source, result.interval_method) == ("autoregressive.VAR", "analytic")
    assert result.params == {"drivers": ("wind", "demand"), "lags": 2}
    assert result.timestamps() == tuple(PRICE.timestamp(400 + i) for i in range(24))
    assert result.mean == pytest.approx(expected.mean, rel=1e-9, abs=1e-9)
    assert [interval.level for interval in result.intervals] == [0.8, 0.95]
    for interval in result.intervals:
        assert interval.lower == pytest.approx(expected.lower[interval.level], rel=1e-9)
        assert interval.upper == pytest.approx(expected.upper[interval.level], rel=1e-9)
    assert result.residuals == pytest.approx([row[0] for row in raw.resid], abs=1e-8)


def test_forecast_uses_only_the_drivers_values_over_the_history() -> None:
    history = PRICE.window(100, 300)
    expected = VAR({"wind": WIND.window(100, 300)}, 2).forecast(history, 12)
    rng = random.Random(1)
    wind: list[float | None] = [rng.gauss(5000, 600) for _ in range(100)]
    wind += [*DATA["wind"][100:300], None, *(rng.gauss(5000, 600) for _ in range(99))]
    assert VAR({"wind": HalfHourlySeries(START, wind)}, 2).forecast(history, 12) == expected


@pytest.mark.parametrize(("scale", "offset"), [(1e-7, 0.0), (1e3, -40.0)])
def test_forecast_follows_the_historys_units_and_ignores_the_drivers(
    scale: float, offset: float
) -> None:
    # Fitted directly, prices in units this small would meet Orchid's 1e-12 variance floor.
    expected = VAR(DRIVERS, 2).forecast(PRICE, 12)
    history = HalfHourlySeries(START, [offset + scale * value for value in DATA["price"]])
    drivers = {
        "wind": HalfHourlySeries(START, [value / 1e3 for value in DATA["wind"]]),
        "demand": HalfHourlySeries(START, [value - 3e4 for value in DATA["demand"]]),
    }
    result = VAR(drivers, 2).forecast(history, 12)

    def converted(values: Sequence[float]) -> object:
        return pytest.approx(
            [offset + scale * value for value in values], rel=1e-9, abs=1e-9 * scale
        )

    assert result.mean == converted(expected.mean)
    for interval, unscaled in zip(result.intervals, expected.intervals, strict=True):
        assert interval.lower == converted(unscaled.lower)
        assert interval.upper == converted(unscaled.upper)
    assert result.residuals == pytest.approx(
        [scale * value for value in expected.residuals], rel=1e-9, abs=1e-9 * scale
    )


def test_linearly_dependent_drivers_change_nothing() -> None:
    # Unlike var_analysis, a forecast needs no non-singular residual covariance: an identity
    # among the drivers leaves the least-squares projections, and so the forecast, unchanged.
    expected = VAR({"wind": WIND}, 2).forecast(PRICE, 6)
    doubled = HalfHourlySeries(START, [2 * value + 1 for value in DATA["wind"]])
    result = VAR({"wind": WIND, "doubled": doubled}, 2).forecast(PRICE, 6)
    assert result.mean == pytest.approx(expected.mean, rel=1e-9, abs=1e-9)
    for interval, single in zip(result.intervals, expected.intervals, strict=True):
        assert interval.lower == pytest.approx(single.lower, rel=1e-9)
        assert interval.upper == pytest.approx(single.upper, rel=1e-9)


@pytest.mark.parametrize(
    ("n_drivers", "lags", "minimum"),
    [(1, 1, 6), (2, 2, 12), (1, 3, 12)],
    ids=["k=2 lags=1", "k=3 lags=2", "k=2 lags=3"],
)
def test_min_history_is_exact(
    make_series: MakeSeries, n_drivers: int, lags: int, minimum: int
) -> None:
    # (k + 1) * lags + k + 1 periods leave each equation's residuals k degrees of freedom.
    drivers = {f"driver {i}": make_series(minimum, seed=i + 1) for i in range(n_drivers)}
    forecaster: Forecaster = VAR(drivers, lags)
    assert forecaster.min_history == minimum
    forecaster.forecast(make_series(minimum), 4)
    with pytest.raises(InsufficientDataError) as excinfo:
        forecaster.forecast(make_series(minimum - 1), 4)
    assert excinfo.value.required == minimum


@pytest.mark.parametrize(
    ("call", "error"),
    [
        pytest.param(lambda: VAR({}, 1), InvalidInputError, id="no drivers"),
        pytest.param(lambda: VAR({"": WIND}, 1), InvalidInputError, id="empty name"),
        pytest.param(
            lambda: VAR({1: WIND}, 1),  # type: ignore[dict-item]
            InvalidInputError,
            id="non-string name",
        ),
        pytest.param(lambda: VAR(DRIVERS, 0), InvalidInputError, id="lags=0"),
        pytest.param(
            lambda: VAR(DRIVERS, 1.0),  # type: ignore[arg-type]
            InvalidInputError,
            id="float lags",
        ),
        pytest.param(lambda: VAR(DRIVERS, True), InvalidInputError, id="bool lags"),
        pytest.param(lambda: VAR({"wind": WIND}, 32), InvalidInputError, id="65 regressors"),
        pytest.param(lambda: VAR(DRIVERS, 1).forecast(PRICE, 0), InvalidInputError, id="horizon=0"),
        pytest.param(
            lambda: VAR(DRIVERS, 1).forecast(PRICE, 1, levels=(95,)),
            InvalidInputError,
            id="level=95",
        ),
        pytest.param(
            lambda: VAR(DRIVERS, 1).forecast(GAPPY, 1), MissingDataError, id="gappy history"
        ),
        pytest.param(
            lambda: VAR({"wind": GAPPY}, 1).forecast(PRICE, 1), MissingDataError, id="gappy driver"
        ),
        pytest.param(
            lambda: VAR({"wind": WIND.window(1, 400)}, 1).forecast(PRICE, 1),
            InvalidInputError,
            id="driver starts late",
        ),
        pytest.param(
            lambda: VAR({"wind": WIND.window(0, 399)}, 1).forecast(PRICE, 1),
            InvalidInputError,
            id="driver ends early",
        ),
        pytest.param(
            lambda: VAR({"flat": CONSTANT}, 1).forecast(PRICE, 1),
            DegenerateDataError,
            id="constant driver",
        ),
        pytest.param(
            lambda: VAR(DRIVERS, 1).forecast(CONSTANT, 1),
            DegenerateDataError,
            id="constant history",
        ),
        pytest.param(  # Orchid would floor the forecast variance at 1e-12
            lambda: VAR(DRIVERS, 2).forecast(HalfHourlySeries(START, SINE), 1),
            DegenerateDataError,
            id="history fits exactly",
        ),
        pytest.param(
            lambda: VAR(DRIVERS, 1).forecast(HalfHourlySeries(START, [1.5e308, -1.5e308] * 200), 1),
            CalculationError,
            id="range overflows",
        ),
    ],
)
def test_forecaster_errors_are_classified(
    call: Callable[[], object], error: type[Exception]
) -> None:
    with pytest.raises(error) as excinfo:
        call()
    assert excinfo.type is error


def test_forecaster_survives_pickling_and_copying() -> None:
    # Process-pool batch jobs pickle the forecasters they send to workers.
    forecaster = VAR(DRIVERS, 2)
    for clone in (pickle.loads(pickle.dumps(forecaster)), copy.deepcopy(forecaster)):
        assert clone == forecaster
        with pytest.raises(TypeError):
            clone.drivers["wind"] = WIND  # type: ignore[index]


def test_analysis_matches_orchid_on_raw_data() -> None:
    result = var_analysis(SERIES, lags=2, horizon=6)
    raw = ar.VAR(ROWS).fit(maxlags=2)
    responses, shares = raw.irf(6).irf, raw.fevd(5).fevd

    assert (result.names, result.lags, result.nobs) == (NAMES, 2, 398)
    assert (result.log_likelihood, result.aic, result.bic) == pytest.approx(
        (raw.llf, raw.aic, raw.bic), rel=1e-12
    )
    for i, name in enumerate(NAMES):
        expected = {
            f"{other}.L{lag + 1}": raw.coefs[lag][i][j]
            for lag in range(2)
            for j, other in enumerate(NAMES)
        }
        assert result.coefficients[name] == pytest.approx(expected, rel=1e-6)
        assert result.intercept[name] == pytest.approx(raw.intercept[i], rel=1e-6)
        assert result.residual_covariance[name] == pytest.approx(
            dict(zip(NAMES, raw.sigma_u[i], strict=True)), rel=1e-9
        )
        for h, matrix in enumerate(responses):
            assert result.impulse_responses[h][name] == pytest.approx(
                dict(zip(NAMES, matrix[i], strict=True)), rel=1e-6, abs=1e-6
            )
        for step, table in enumerate(shares):
            assert result.variance_decomposition[step][name] == pytest.approx(
                dict(zip(NAMES, table[i], strict=True)), abs=1e-9
            )
    assert len(result.granger) == 6
    for (caused, causing), test in result.granger.items():
        oracle = raw.test_causality(NAMES.index(caused), causing=NAMES.index(causing))
        assert (test.statistic, test.p_value) == pytest.approx(
            (oracle.statistic, oracle.pvalue), rel=1e-6
        )
        assert test.df == oracle.df == (2, 391)


def test_responses_and_decompositions_are_consistent() -> None:
    result = var_analysis(SERIES, lags=2, horizon=5)
    impact = result.impulse_responses[0]
    assert (len(result.impulse_responses), len(result.variance_decomposition)) == (6, 5)
    for i, response in enumerate(NAMES):
        # On impact, a series responds only to its own shock and those of earlier series.
        assert [impact[response][shock] for shock in NAMES[i + 1 :]] == [0.0] * (2 - i)
        for other in NAMES:
            implied = math.fsum(impact[response][shock] * impact[other][shock] for shock in NAMES)
            assert implied == pytest.approx(result.residual_covariance[response][other], rel=1e-9)
        total = math.fsum(value * value for value in impact[response].values())
        first_step = {shock: impact[response][shock] ** 2 / total for shock in NAMES}
        assert result.variance_decomposition[0][response] == pytest.approx(first_step, rel=1e-9)
        for table in result.variance_decomposition:
            assert math.fsum(table[response].values()) == pytest.approx(1.0, abs=1e-12)


def _same_fit(result: VarAnalysis, expected: VarAnalysis) -> None:
    """The order-free estimates agree: everything but the orthogonalised shocks."""
    for name in NAMES:
        assert result.coefficients[name] == pytest.approx(expected.coefficients[name], rel=1e-9)
        assert result.intercept[name] == pytest.approx(expected.intercept[name], rel=1e-9)
        assert result.residual_covariance[name] == pytest.approx(
            expected.residual_covariance[name], rel=1e-9
        )
    assert (result.log_likelihood, result.aic, result.bic) == pytest.approx(
        (expected.log_likelihood, expected.aic, expected.bic), rel=1e-12
    )
    for key, test in expected.granger.items():
        assert result.granger[key].statistic == pytest.approx(test.statistic, rel=1e-9)


def test_analysis_is_equivariant_to_the_order_of_the_series() -> None:
    order = ("demand", "price", "wind")
    permuted = var_analysis({name: SERIES[name] for name in order}, lags=2, horizon=3)
    assert permuted.names == order
    _same_fit(permuted, var_analysis(SERIES, lags=2, horizon=3))


def test_analysis_follows_the_units_of_the_series() -> None:
    # Fitted directly, prices in units this small would meet Orchid's 1e-12 floors.
    factors = {"price": 1e-7, "wind": 1e3, "demand": 1.0}
    offsets = {"price": 0.0, "wind": -5e6, "demand": 1e4}
    expected = var_analysis(SERIES, lags=2, horizon=4)
    result = var_analysis(
        _series(
            {
                name: [offsets[name] + factors[name] * value for value in values]
                for name, values in DATA.items()
            }
        ),
        lags=2,
        horizon=4,
    )

    for name, factor in factors.items():
        assert result.coefficients[name] == pytest.approx(
            {
                regressor: factor / factors[regressor.rsplit(".L", 1)[0]] * value
                for regressor, value in expected.coefficients[name].items()
            },
            rel=1e-9,
        )
        assert result.residual_covariance[name] == pytest.approx(
            {
                other: factor * factors[other] * value
                for other, value in expected.residual_covariance[name].items()
            },
            rel=1e-9,
        )
        for scaled, responses in zip(
            result.impulse_responses, expected.impulse_responses, strict=True
        ):
            assert scaled[name] == pytest.approx(
                {shock: factor * value for shock, value in responses[name].items()},
                rel=1e-6,
                abs=1e-9 * factor,
            )
        for scaled, shares in zip(
            result.variance_decomposition, expected.variance_decomposition, strict=True
        ):
            assert scaled[name] == pytest.approx(shares[name], abs=1e-9)
    for key, test in expected.granger.items():
        assert (result.granger[key].statistic, result.granger[key].p_value) == pytest.approx(
            (test.statistic, test.p_value), rel=1e-9
        )
    # The log-likelihood falls by nobs * sum(ln factor), so AIC and BIC rise by twice that.
    log_factors = result.nobs * math.fsum(map(math.log, factors.values()))
    assert result.log_likelihood - expected.log_likelihood == pytest.approx(-log_factors, rel=1e-9)
    assert result.aic - expected.aic == pytest.approx(2 * log_factors, rel=1e-9)
    assert result.bic - expected.bic == pytest.approx(2 * log_factors, rel=1e-9)


def test_granger_tests_flag_exactly_the_simulated_links() -> None:
    granger = var_analysis(SERIES, lags=1, horizon=1).granger
    flagged = {pair for pair, test in granger.items() if test.p_value < 0.01}
    assert flagged == {("price", "wind"), ("price", "demand")}
    assert granger[("price", "wind")].df == (1, 395)  # 399 rows less 4 regressors


def test_lag_selection_matches_orchid_fits_on_a_common_sample() -> None:
    result = select_var_lags(SERIES, max_lags=3)
    fits = {p: ar.VAR(ROWS[3 - p :]).fit(maxlags=p) for p in (1, 2, 3)}
    assert result.nobs == 397
    assert all(fit.nobs == 397 for fit in fits.values())
    assert result.aic == pytest.approx({p: fit.aic for p, fit in fits.items()}, rel=1e-12)
    assert result.bic == pytest.approx({p: fit.bic for p, fit in fits.items()}, rel=1e-12)
    assert min(result.bic, key=result.bic.__getitem__) == 1  # the data come from a VAR(1)


SINGULAR: dict[str, tuple[dict[str, list[float]], str]] = {
    "collinear pair": (
        {"wind": DATA["wind"], "doubled": [2 * value + 1 for value in DATA["wind"]]},
        "doubled is a linear function of past values and of current wind",
    ),
    "identity of three": (
        {
            "demand": DATA["demand"],
            "wind": DATA["wind"],
            "residual": [d - w for d, w in zip(DATA["demand"], DATA["wind"], strict=True)],
        },
        "residual is a linear function of past values and of current demand, wind",
    ),
    "exact fit": ({"sine": SINE, "wind": DATA["wind"]}, "sine is a linear function of past values"),
}


@pytest.mark.parametrize(("columns", "message"), SINGULAR.values(), ids=SINGULAR.keys())
def test_singular_residual_covariance_is_degenerate(
    columns: dict[str, list[float]], message: str
) -> None:
    # No pairwise residual correlation is near 1 in the identity of three, yet Orchid would
    # factor its singular covariance after adding jitter, and floor its log-likelihood.
    series = _series(columns)
    with pytest.raises(DegenerateDataError, match=message) as excinfo:
        var_analysis(series, lags=2, horizon=2)
    assert excinfo.type is DegenerateDataError
    with pytest.raises(DegenerateDataError, match=message):
        select_var_lags(series, max_lags=2)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: VAR(DRIVERS, 1).forecast(PRICE, 1), id="VAR"),
        pytest.param(lambda: var_analysis(SERIES, lags=1, horizon=1), id="var_analysis"),
        pytest.param(lambda: select_var_lags(SERIES, max_lags=1), id="select_var_lags"),
    ],
)
def test_orchid_failures_become_calculation_errors(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[], object]
) -> None:
    def broken(self: ar.VAR, **options: object) -> ar.VARResults:
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(ar.VAR, "fit", broken)
    with pytest.raises(CalculationError) as excinfo:
        call()
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


@pytest.mark.parametrize(
    ("analyse", "k", "minimum"),
    [
        pytest.param(lambda s: var_analysis(s, lags=2, horizon=3), 3, 12, id="var_analysis"),
        pytest.param(lambda s: select_var_lags(s, max_lags=3), 2, 12, id="select_var_lags"),
    ],
)
def test_minimum_sample_is_exact(
    make_series: MakeSeries,
    analyse: Callable[[dict[str, HalfHourlySeries]], object],
    k: int,
    minimum: int,
) -> None:
    def sample(n: int) -> dict[str, HalfHourlySeries]:
        return {f"s{i}": make_series(n, amplitude=0.0, seed=i) for i in range(k)}

    analyse(sample(minimum))
    with pytest.raises(InsufficientDataError) as excinfo:
        analyse(sample(minimum - 1))
    assert (excinfo.value.required, excinfo.value.available) == (minimum, minimum - 1)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        pytest.param(
            lambda: var_analysis({"price": PRICE}, lags=1, horizon=1),
            InvalidInputError,
            id="one series",
        ),
        pytest.param(
            lambda: var_analysis({"price": PRICE, "": WIND}, lags=1, horizon=1),
            InvalidInputError,
            id="empty name",
        ),
        pytest.param(
            lambda: var_analysis(
                {"price": PRICE, 2: WIND},  # type: ignore[dict-item]
                lags=1,
                horizon=1,
            ),
            InvalidInputError,
            id="non-string name",
        ),
        pytest.param(
            lambda: var_analysis({"price": PRICE, "wind": WIND.window(1, 400)}, lags=1, horizon=1),
            InvalidInputError,
            id="different periods",
        ),
        pytest.param(lambda: var_analysis(TWO, lags=0, horizon=1), InvalidInputError, id="lags=0"),
        pytest.param(
            lambda: var_analysis(TWO, lags=1.0, horizon=1),  # type: ignore[arg-type]
            InvalidInputError,
            id="float lags",
        ),
        pytest.param(
            lambda: var_analysis(TWO, lags=32, horizon=1), InvalidInputError, id="65 regressors"
        ),
        pytest.param(
            lambda: var_analysis(TWO, lags=1, horizon=0), InvalidInputError, id="horizon=0"
        ),
        pytest.param(
            lambda: var_analysis(TWO, lags=1, horizon=True), InvalidInputError, id="bool horizon"
        ),
        pytest.param(lambda: select_var_lags(TWO, max_lags=0), InvalidInputError, id="max_lags=0"),
        pytest.param(
            lambda: select_var_lags(TWO, max_lags=2.0),  # type: ignore[arg-type]
            InvalidInputError,
            id="float max_lags",
        ),
        pytest.param(
            lambda: var_analysis({**TWO, "gappy": GAPPY}, lags=1, horizon=1),
            MissingDataError,
            id="missing value",
        ),
        pytest.param(
            lambda: select_var_lags({**TWO, "flat": CONSTANT}, max_lags=1),
            DegenerateDataError,
            id="constant series",
        ),
        pytest.param(
            lambda: var_analysis(
                {name: s.window(0, 5) for name, s in TWO.items()}, lags=1, horizon=1
            ),
            InsufficientDataError,
            id="too short",
        ),
        pytest.param(
            lambda: var_analysis(
                _series({name: [1e200 * value for value in DATA[name]] for name in TWO}),
                lags=1,
                horizon=1,
            ),
            CalculationError,
            id="covariance overflows",
        ),
    ],
)
def test_analysis_errors_are_classified(call: Callable[[], object], error: type[Exception]) -> None:
    with pytest.raises(error) as excinfo:
        call()
    assert excinfo.type is error
