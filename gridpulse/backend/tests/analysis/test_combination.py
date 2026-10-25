import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import pytest

from gridpulse.analysis.algorithms.combination import (
    CombinationMethod,
    Ensemble,
    combine_forecasts,
    inverse_error_weights,
)
from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    InsufficientDataError,
    InvalidInputError,
    MissingDataError,
)
from gridpulse.analysis.algorithms.evaluation import backtest
from gridpulse.analysis.algorithms.forecasting import (
    DEFAULT_LEVELS,
    Drift,
    Forecast,
    Forecaster,
    Naive,
    SeasonalNaive,
)
from gridpulse.analysis.algorithms.orchid import forecasting_baseline as fb
from gridpulse.analysis.algorithms.series import HALF_HOUR, HalfHourlySeries

START = datetime(2026, 9, 1, tzinfo=UTC)
ZERO_WIDTH = "zero-width intervals: residual variance is zero"
MakeSeries = Callable[..., HalfHourlySeries]
Rule = Callable[[list[list[float]]], list[float]]

LEVELS = (0.8, 0.95)
# Cheap members whose residual warm-ups differ: 1, 48 and 1 periods.
MEMBERS: tuple[Forecaster, ...] = (Naive(), SeasonalNaive(48), Drift())
SOURCES = tuple(f"forecasting_baseline.{name}" for name in ("Naive", "SeasonalNaive", "Drift"))
WEIGHTS = (3.0, 1.0, 2.0)
# Each method with its weights and the Orchid function it must apply to every array.
RULES: dict[CombinationMethod, tuple[Sequence[float] | None, Rule]] = {
    "mean": (None, fb.combine_mean),
    "median": (None, fb.combine_median),
    "weighted": (WEIGHTS, lambda vectors: fb.combine_weighted(vectors, WEIGHTS)),
}


@dataclass
class Spy:
    """Naive forecasts that record every history they are given."""

    min_history: int = 2
    histories: list[HalfHourlySeries] = field(default_factory=list)

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        self.histories.append(history)
        return Naive().forecast(history, horizon, levels=levels)


@pytest.fixture
def history(make_series: MakeSeries) -> HalfHourlySeries:
    return make_series(150, level=0.0)  # crosses zero, often negative


@pytest.fixture
def forecasts(history: HalfHourlySeries) -> list[Forecast]:
    """Each member's forecast of ``history``."""
    return [member.forecast(history, 24, levels=LEVELS) for member in MEMBERS]


def _shorter(forecast: Forecast) -> Forecast:
    """``forecast`` without its last period."""
    bands = tuple(
        replace(band, lower=band.lower[:-1], upper=band.upper[:-1]) for band in forecast.intervals
    )
    return replace(forecast, mean=forecast.mean[:-1], intervals=bands)


def _short_band(forecast: Forecast) -> Forecast:
    band, *others = forecast.intervals
    return replace(forecast, intervals=(replace(band, upper=band.upper[:-1]), *others))


@pytest.mark.parametrize("method", list(RULES))
def test_every_array_is_combined_by_the_orchid_rule(
    forecasts: list[Forecast], method: CombinationMethod
) -> None:
    weights, rule = RULES[method]
    result = combine_forecasts(forecasts, method=method, weights=weights)

    def expected(arrays: Sequence[Sequence[float]]) -> object:
        return pytest.approx(rule([list(array) for array in arrays]), rel=1e-12, abs=1e-12)

    assert result.mean == expected([forecast.mean for forecast in forecasts])
    assert [band.level for band in result.intervals] == list(LEVELS)
    for j, band in enumerate(result.intervals):
        assert band.lower == expected([forecast.intervals[j].lower for forecast in forecasts])
        assert band.upper == expected([forecast.intervals[j].upper for forecast in forecasts])
    tail = len(forecasts[1].residuals)  # SeasonalNaive's warm-up is the longest
    assert result.residuals == expected([forecast.residuals[-tail:] for forecast in forecasts])
    assert result.origin == forecasts[0].origin
    assert (result.source, result.interval_method) == (
        f"forecasting_baseline.combine_{method}",
        "quantile-averaging",
    )


@pytest.mark.parametrize("method", list(RULES))
def test_residuals_are_those_of_the_combined_fit(
    history: HalfHourlySeries, forecasts: list[Forecast], method: CombinationMethod
) -> None:
    weights, rule = RULES[method]
    residuals = combine_forecasts(forecasts, method=method, weights=weights).residuals
    tail = len(residuals)
    actual = history.complete_values()[-tail:]
    fits = [
        [value - error for value, error in zip(actual, forecast.residuals[-tail:], strict=True)]
        for forecast in forecasts
    ]
    expected = [value - fit for value, fit in zip(actual, rule(fits), strict=True)]
    assert residuals == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("method", list(RULES))
def test_bands_stay_ordered_and_nested(
    forecasts: list[Forecast], method: CombinationMethod
) -> None:
    result = combine_forecasts(forecasts, method=method, weights=RULES[method][0])
    inner, outer = result.intervals
    bands = list(zip(outer.lower, inner.lower, result.mean, inner.upper, outer.upper, strict=True))
    assert all(list(band) == sorted(band) for band in bands)


def test_bands_are_matched_by_level(forecasts: list[Forecast]) -> None:
    reversed_levels = replace(forecasts[1], intervals=forecasts[1].intervals[::-1])
    expected = combine_forecasts(forecasts[:2])
    assert combine_forecasts([forecasts[0], reversed_levels]) == expected


def test_a_member_without_residuals_leaves_no_common_tail(forecasts: list[Forecast]) -> None:
    result = combine_forecasts([forecasts[0], replace(forecasts[2], residuals=())])
    assert result.residuals == ()


@pytest.mark.parametrize(
    ("method", "weights", "used"),
    [
        ("mean", None, None),
        ("median", None, None),
        ("weighted", (2.0, 2.0, 4.0), (0.25, 0.25, 0.5)),
    ],
)
def test_params_record_the_rule_and_the_members(
    forecasts: list[Forecast],
    method: CombinationMethod,
    weights: tuple[float, ...] | None,
    used: tuple[float, ...] | None,
) -> None:
    result = combine_forecasts(forecasts, method=method, weights=weights)
    assert result.params == {"method": method, "weights": used, "members": SOURCES}


def test_warnings_are_the_members_in_order_without_repeats(forecasts: list[Forecast]) -> None:
    notes = [("a", "b"), ("b",), ("c", "a")]
    tagged = [replace(forecast, warnings=w) for forecast, w in zip(forecasts, notes, strict=True)]
    assert combine_forecasts(tagged).warnings == ("a", "b", "c")


@pytest.mark.parametrize(
    ("members", "method", "expected"),
    [
        pytest.param((Naive(), SeasonalNaive(48)), "mean", (), id="one zero-width member"),
        pytest.param(
            (SeasonalNaive(48), SeasonalNaive(96)), "mean", (ZERO_WIDTH,), id="both zero-width"
        ),
        # The median of two zero-width members and a wider one has zero width.
        pytest.param(
            (Naive(), SeasonalNaive(48), SeasonalNaive(96)),
            "median",
            (ZERO_WIDTH,),
            id="median of three",
        ),
    ],
)
def test_zero_width_warning_follows_the_combined_intervals(
    members: tuple[Forecaster, ...], method: CombinationMethod, expected: tuple[str, ...]
) -> None:
    # Daily-periodic to within rounding: the seasonal naive widths, about 1e-15 of the
    # amplitude, count as zero though they exceed an absolute 1e-12.
    periodic = HalfHourlySeries(START, [1e4 * math.sin(2 * math.pi * t / 48) for t in range(160)])
    assert Ensemble(members, method=method).forecast(periodic, 4).warnings == expected


def test_only_the_ratios_of_the_weights_matter(forecasts: list[Forecast]) -> None:
    tiny = (1e-13, 2e-13, 4e-13)
    with pytest.raises(fb.ForecastError):  # Orchid takes their sum, 7e-13, for zero
        fb.combine_weighted([list(forecast.mean) for forecast in forecasts], tiny)
    expected = combine_forecasts(forecasts, method="weighted", weights=(1.0, 2.0, 4.0))
    assert combine_forecasts(forecasts, method="weighted", weights=tiny) == expected


@pytest.mark.parametrize(
    ("malform", "error"),
    [
        pytest.param(
            lambda f: replace(f, origin=f.origin + HALF_HOUR), InvalidInputError, id="later origin"
        ),
        pytest.param(_shorter, InvalidInputError, id="shorter horizon"),
        pytest.param(_short_band, InvalidInputError, id="short band"),
        pytest.param(
            lambda f: replace(f, intervals=f.intervals[:1]), InvalidInputError, id="fewer levels"
        ),
        pytest.param(
            lambda f: replace(f, intervals=(replace(f.intervals[0], level=0.5), f.intervals[1])),
            InvalidInputError,
            id="other level",
        ),
        pytest.param(
            lambda f: replace(f, intervals=(f.intervals[0], f.intervals[0])),
            InvalidInputError,
            id="repeated level",
        ),
        pytest.param(
            lambda f: replace(f, mean=(math.nan, *f.mean[1:])), MissingDataError, id="NaN mean"
        ),
        pytest.param(
            lambda f: replace(f, residuals=(*f.residuals[:-1], math.inf)),
            InvalidInputError,
            id="infinite residual",
        ),
    ],
)
def test_rejects_unusable_members(
    forecasts: list[Forecast], malform: Callable[[Forecast], Forecast], error: type[Exception]
) -> None:
    with pytest.raises(error) as excinfo:
        combine_forecasts([forecasts[0], malform(forecasts[1])])
    assert excinfo.type is error


@pytest.mark.parametrize(
    ("call", "error"),
    [
        pytest.param(lambda f: combine_forecasts(f[:1]), InsufficientDataError, id="one forecast"),
        pytest.param(
            lambda f: combine_forecasts(f, method="trimmed"),  # type: ignore[arg-type]
            InvalidInputError,
            id="unknown method",
        ),
        pytest.param(
            lambda f: combine_forecasts(f, weights=WEIGHTS), InvalidInputError, id="mean weights"
        ),
        pytest.param(
            lambda f: combine_forecasts(f, method="median", weights=WEIGHTS),
            InvalidInputError,
            id="median weights",
        ),
        pytest.param(
            lambda f: combine_forecasts(f, method="weighted"),
            InvalidInputError,
            id="no weights",
        ),
        pytest.param(
            lambda f: combine_forecasts(f, method="weighted", weights=WEIGHTS[:2]),
            InvalidInputError,
            id="too few weights",
        ),
        pytest.param(
            lambda f: combine_forecasts(f, method="weighted", weights=(*WEIGHTS, 1.0)),
            InvalidInputError,
            id="too many weights",
        ),
        pytest.param(
            lambda f: combine_forecasts(f, method="weighted", weights=(1.0, 0.0, 1.0)),
            InvalidInputError,
            id="zero weight",
        ),
        pytest.param(
            lambda f: combine_forecasts(f, method="weighted", weights=(1.0, math.nan, 1.0)),
            InvalidInputError,
            id="NaN weight",
        ),
        pytest.param(
            lambda f: combine_forecasts(f, method="weighted", weights=(1.0, True, 1.0)),
            InvalidInputError,
            id="bool weight",
        ),
    ],
)
def test_combine_forecasts_rejects_invalid_arguments(
    forecasts: list[Forecast],
    call: Callable[[list[Forecast]], object],
    error: type[Exception],
) -> None:
    with pytest.raises(error) as excinfo:
        call(forecasts)
    assert excinfo.type is error


@pytest.mark.parametrize("method", ["mean", "median"])
def test_overflow_raises_calculation_error(
    forecasts: list[Forecast], method: CombinationMethod
) -> None:
    # Both rules add the two means: fsum raises OverflowError, the median returns inf.
    huge = [replace(forecast, mean=(1.7e308,) * len(forecast.mean)) for forecast in forecasts[:2]]
    with pytest.raises(CalculationError):
        combine_forecasts(huge, method=method)


@pytest.mark.parametrize("method", list(RULES))
def test_ensemble_backtests_as_the_combination_of_its_members(
    make_series: MakeSeries, method: CombinationMethod
) -> None:
    weights = RULES[method][0]
    series = make_series(200, level=0.0)
    ensemble = Ensemble(MEMBERS, method=method, weights=weights)
    report = backtest(ensemble, series, initial=100, horizon=12, step=24, levels=LEVELS)
    assert [fold.status for fold in report.folds] == ["ok"] * 4
    assert {f"coverage_{level:g}" for level in LEVELS} <= set(report.metrics)
    for fold in report.folds:
        history = series.window(0, fold.train_size)  # expanding windows start at the first value
        members = [member.forecast(history, 12, levels=LEVELS) for member in MEMBERS]
        assert fold.forecast == combine_forecasts(members, method=method, weights=weights)


@pytest.mark.parametrize(("members", "minimum"), [((Naive(), Drift()), 3), (MEMBERS, 49)], ids=repr)
def test_ensemble_min_history_is_exact(
    make_series: MakeSeries, members: tuple[Forecaster, ...], minimum: int
) -> None:
    ensemble = Ensemble(members)
    assert ensemble.min_history == minimum
    ensemble.forecast(make_series(minimum), 4)
    with pytest.raises(InsufficientDataError) as excinfo:
        ensemble.forecast(make_series(minimum - 1), 4)
    assert excinfo.value.required == minimum


@pytest.mark.parametrize(
    ("history", "horizon", "levels", "error"),
    [
        pytest.param(
            HalfHourlySeries(START, [1.0] * 48), 4, LEVELS, InsufficientDataError, id="short"
        ),
        pytest.param(
            HalfHourlySeries(START, [*([1.0] * 59), None]), 4, LEVELS, MissingDataError, id="gap"
        ),
        pytest.param(
            HalfHourlySeries(START, [1.0] * 60), 0, LEVELS, InvalidInputError, id="horizon 0"
        ),
        pytest.param(
            HalfHourlySeries(START, [1.0] * 60), 4, (95,), InvalidInputError, id="level 95"
        ),
    ],
)
def test_ensemble_fits_no_member_for_a_rejected_call(
    history: HalfHourlySeries, horizon: int, levels: Sequence[float], error: type[Exception]
) -> None:
    spy = Spy()
    with pytest.raises(error) as excinfo:
        Ensemble((spy, SeasonalNaive(48))).forecast(history, horizon, levels=levels)
    assert excinfo.type is error
    assert spy.histories == []


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: Ensemble(()), id="no members"),
        pytest.param(lambda: Ensemble((Naive(),)), id="one member"),
        pytest.param(
            lambda: Ensemble(MEMBERS, method="trimmed"),  # type: ignore[arg-type]
            id="unknown method",
        ),
        pytest.param(lambda: Ensemble(MEMBERS, weights=WEIGHTS), id="mean weights"),
        pytest.param(lambda: Ensemble(MEMBERS, method="weighted"), id="no weights"),
        pytest.param(
            lambda: Ensemble(MEMBERS, method="weighted", weights=WEIGHTS[:2]),
            id="too few weights",
        ),
        pytest.param(
            lambda: Ensemble(MEMBERS, method="weighted", weights=(1.0, -1.0, 1.0)),
            id="negative weight",
        ),
    ],
)
def test_ensemble_rejects_invalid_settings(build: Callable[[], Ensemble]) -> None:
    with pytest.raises(InvalidInputError) as excinfo:
        build()
    assert excinfo.type is InvalidInputError


@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_inverse_error_weights_match_orchid(power: float) -> None:
    errors = (2.0, 3.0, 5.0)
    weights = inverse_error_weights(errors, power=power)
    assert weights == pytest.approx(fb.inverse_error_weights(errors, power), rel=1e-12)
    assert math.fsum(weights) == pytest.approx(1.0, rel=1e-15)


def test_inverse_error_weights_do_not_depend_on_units() -> None:
    errors = [2.0, 3.0, 5.0]
    expected = inverse_error_weights(errors, power=2.0)
    tiny, huge = ([error * scale for error in errors] for scale in (1e-14, 1e200))
    assert inverse_error_weights(tiny, power=2.0) == pytest.approx(expected, rel=1e-12)
    assert inverse_error_weights(huge, power=2.0) == pytest.approx(expected, rel=1e-12)
    # Orchid alone gives every error under 1e-12 a weight of 1e18, and squaring 2e200 overflows.
    assert fb.inverse_error_weights(tiny, 2.0) == pytest.approx([1 / 3] * 3)
    with pytest.raises(OverflowError):
        fb.inverse_error_weights(huge, 2.0)


def test_inverse_error_weights_hold_for_errors_over_1e12_apart() -> None:
    # Divided by the larger error, the smaller would fall under Orchid's 1e-12 floor.
    expected = (1 / (1 + 1e-13), 1e-13 / (1 + 1e-13))
    assert inverse_error_weights((1.0, 1e13)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("errors", "power"),
    [
        pytest.param((1.0, 0.0), 1.0, id="zero error"),
        pytest.param((1.0, -2.0), 1.0, id="negative error"),
        pytest.param((1.0, math.nan), 1.0, id="NaN error"),
        pytest.param((1.0, math.inf), 1.0, id="infinite error"),
        pytest.param((1.0, None), 1.0, id="missing error"),
        pytest.param((1.0,), 1.0, id="one error"),
        pytest.param((1.0, 2.0), 0.0, id="power 0"),
        pytest.param((1.0, 2.0), -1.0, id="negative power"),
        pytest.param((1.0, 2.0), math.inf, id="infinite power"),
        pytest.param((1.0, 2.0), math.nan, id="NaN power"),
        pytest.param((1.0, 2.0), True, id="bool power"),
        pytest.param((1.0, 2.0), "2", id="string power"),
    ],
)
def test_inverse_error_weights_reject_unusable_inputs(
    errors: Sequence[float], power: float
) -> None:
    with pytest.raises(InvalidInputError) as excinfo:
        inverse_error_weights(errors, power=power)
    assert excinfo.type is InvalidInputError


def test_inverse_error_weights_that_overflow_raise_calculation_error() -> None:
    with pytest.raises(CalculationError) as excinfo:  # 1e10 ** 40 overflows a float
        inverse_error_weights((1.0, 1e10), power=40.0)
    assert isinstance(excinfo.value.__cause__, OverflowError)
    with pytest.raises(CalculationError):  # so does the ratio of these errors
        inverse_error_weights((5e-324, 1e308))
