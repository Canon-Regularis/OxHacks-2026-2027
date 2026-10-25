import dataclasses
import functools
import math
import random
import statistics
from collections.abc import Callable

import pytest

from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
    MissingDataError,
)
from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.volatility import GarchFit, garch

HORIZON = 48
NOT_MEAN_REVERTING = (
    "variance not mean-reverting: persistence is within 2e-6 of 1 or above, so there is no "
    "unconditional variance"
)
OMEGA_COLLAPSED = (
    "omega collapsed: omega / (1 - persistence) is under 1e-6 of the residuals' variance, so "
    "there is no unconditional variance and variance forecasts decay towards zero"
)
Change = Callable[[ar.GARCHResult], ar.GARCHResult]


def _gjr_shocks(n: int, *, seed: int) -> list[float]:
    """GJR-GARCH(1, 1) shocks with a leverage effect, scaled to unit standard deviation."""
    rng = random.Random(seed)
    shocks, variance, shock = [], 1.0, 0.0
    for _ in range(n):
        leverage = 0.1 if shock < 0 else 0.0
        variance = 0.1 + (0.08 + leverage) * shock * shock + 0.8 * variance
        shock = math.sqrt(variance) * rng.gauss(0.0, 1.0)
        shocks.append(shock)
    scale = statistics.pstdev(shocks)
    return [shock / scale for shock in shocks]


RESIDUALS = _gjr_shocks(200, seed=2)  # each fit takes about a second

# Orchid results the adapter must refuse to report.
UNUSABLE: dict[str, Change] = {
    "infinite log-likelihood": lambda result: dataclasses.replace(result, loglike=-math.inf),
    "NaN parameter": lambda result: dataclasses.replace(
        result, params={**result.params, "beta1": math.nan}
    ),
    "NaN persistence": lambda result: dataclasses.replace(result, persistence=math.nan),
    "NaN standardized residual": lambda result: dataclasses.replace(
        result, std_resid=[math.nan, *result.std_resid[1:]]
    ),
    "zero conditional variance": lambda result: dataclasses.replace(
        result, conditional_variance=[0.0, *result.conditional_variance[1:]]
    ),
    "infinite conditional variance": lambda result: dataclasses.replace(
        result, conditional_variance=[math.inf, *result.conditional_variance[1:]]
    ),
}


@functools.cache
def _fitted(leverage: bool) -> GarchFit:
    """``garch`` of ``RESIDUALS``, shared between tests because each fit takes seconds."""
    return garch(RESIDUALS, horizon=HORIZON, leverage=leverage)


def _patch_fit(monkeypatch: pytest.MonkeyPatch, change: Change) -> None:
    """Make Orchid's fits cheap and pass their results through ``change``."""
    fit = ar.GARCH.fit

    def patched(self: ar.GARCH, *, seed: int = 0, maxiter: int = 1500) -> ar.GARCHResult:
        return change(fit(self, seed=seed, maxiter=1))  # one iteration: only the output matters

    monkeypatch.setattr(ar.GARCH, "fit", patched)


def _unreachable(*args: object, **kwargs: object) -> None:
    raise AssertionError("Orchid ran on unusable input")


@pytest.mark.parametrize(
    ("leverage", "model", "names"),
    [
        (False, "garch", ["mu", "omega", "alpha1", "beta1"]),
        (True, "gjr", ["mu", "omega", "alpha1", "beta1", "gamma1"]),
    ],
    ids=["garch", "gjr"],
)
def test_matches_orchid_at_unit_scale(leverage: bool, model: str, names: list[str]) -> None:
    # RESIDUALS have unit standard deviation, so the adapter's rescaling is the identity up to
    # rounding, which moves Orchid's Nelder-Mead estimates by at most a few parts in 1e7.
    orchid = ar.GARCH(RESIDUALS, p=1, q=1, model=model)
    expected = orchid.fit(seed=0)
    result = _fitted(leverage)
    assert result.leverage is leverage
    assert list(result.params) == names
    assert result.params == pytest.approx(expected.params, rel=1e-5)
    assert (result.log_likelihood, result.aic, result.bic) == pytest.approx(
        (expected.loglike, expected.aic, expected.bic), rel=1e-9
    )
    assert result.persistence == pytest.approx(expected.persistence, rel=1e-5)
    assert result.unconditional_variance == pytest.approx(expected.unconditional_variance, rel=1e-5)
    assert result.conditional_variance == pytest.approx(expected.conditional_variance, rel=1e-5)
    assert result.standardized_residuals == pytest.approx(expected.std_resid, rel=1e-5, abs=1e-5)
    assert result.variance_forecast == pytest.approx(orchid.forecast(expected, HORIZON), rel=1e-5)
    assert result.warnings == ()


def test_results_scale_with_the_data() -> None:
    # Fitted directly, residuals this small fall below Orchid's absolute 1e-12 variance floor.
    # Scaling them rounds them, which moves the estimates by at most a few parts in 1e7.
    scale = 1e-7
    tiny = garch([value * scale for value in RESIDUALS], horizon=HORIZON)
    unit = _fitted(False)
    mu, omega = unit.params["mu"], unit.params["omega"]
    assert tiny.params == pytest.approx(
        {**unit.params, "mu": mu * scale, "omega": omega * scale**2}, rel=1e-5
    )
    shift = len(RESIDUALS) * math.log(scale)
    assert (tiny.log_likelihood, tiny.aic, tiny.bic) == pytest.approx(
        (unit.log_likelihood - shift, unit.aic + 2 * shift, unit.bic + 2 * shift), rel=1e-9
    )
    assert tiny.persistence == pytest.approx(unit.persistence, rel=1e-5)
    assert tiny.standardized_residuals == pytest.approx(
        unit.standardized_residuals, rel=1e-5, abs=1e-5
    )
    assert unit.unconditional_variance is not None
    assert tiny.unconditional_variance == pytest.approx(
        unit.unconditional_variance * scale**2, rel=1e-5
    )
    for variances, expected in [
        (tiny.conditional_variance, unit.conditional_variance),
        (tiny.variance_forecast, unit.variance_forecast),
    ]:
        assert variances == pytest.approx([value * scale**2 for value in expected], rel=1e-5)


@pytest.mark.parametrize("leverage", [False, True], ids=["garch", "gjr"])
def test_variance_forecast_reverts_to_the_unconditional_variance(leverage: bool) -> None:
    result = _fitted(leverage)
    params, forecast = result.params, result.variance_forecast
    z, h = result.standardized_residuals[-1], result.conditional_variance[-1]
    assert z < 0  # so the leverage term enters the first step
    alpha = params["alpha1"] + params.get("gamma1", 0.0)
    first = params["omega"] + alpha * z * z * h + params["beta1"] * h
    assert forecast[0] == pytest.approx(first, rel=1e-12)
    # Later shocks are unknown, so each step multiplies the gap to u by the persistence.
    u = result.unconditional_variance
    assert u is not None
    gaps = [value - u for value in forecast]
    assert gaps[1:] == pytest.approx(
        [result.persistence * gap for gap in gaps[:-1]], rel=1e-9, abs=1e-12
    )
    assert len(forecast) == HORIZON
    assert min(forecast) > 0


def test_a_fall_in_volatility_meets_orchids_persistence_cap() -> None:
    # GARCH reads a drop in volatility as an integrated variance. Orchid stops alpha1 + beta1 at
    # 1 - 1e-6, where omega / (1 - persistence) would be set by the cap, not by the data.
    rng = random.Random(0)
    shocks = [rng.gauss(0.0, 1.0) for _ in range(50)] + [rng.gauss(0.0, 0.1) for _ in range(50)]
    result = garch(shocks, horizon=1)
    assert result.persistence == pytest.approx(1 - 1e-6, abs=1e-9)
    assert result.unconditional_variance is None
    assert result.warnings == (NOT_MEAN_REVERTING,)


def test_without_arch_effects_omega_can_collapse_to_zero() -> None:
    # White noise leaves omega unidentified. On this sample Orchid's search drives it to 0.0
    # with persistence below the cap.
    rng = random.Random(31)
    result = garch([rng.gauss(0.0, 1.0) for _ in range(200)], horizon=HORIZON)
    assert result.params["omega"] == 0.0
    assert result.unconditional_variance is None
    assert result.warnings == (OMEGA_COLLAPSED,)


@pytest.mark.parametrize(
    ("persistence", "mean_reverting"),
    [(1 - 3e-6, True), (1 - 2e-6, False), (1.0, False), (1.2, False)],
    ids=["below the limit", "at the limit", "one", "explosive"],
)
def test_persistence_near_or_above_one_leaves_no_unconditional_variance(
    monkeypatch: pytest.MonkeyPatch, persistence: float, mean_reverting: bool
) -> None:
    # Orchid leaves GJR persistence unconstrained, and above 1 omega / (1 - persistence) < 0.
    def set_persistence(result: ar.GARCHResult) -> ar.GARCHResult:
        return dataclasses.replace(
            result,
            persistence=persistence,
            unconditional_variance=result.params["omega"] / ((1 - persistence) or 1e-12),
        )

    _patch_fit(monkeypatch, set_persistence)
    result = garch(RESIDUALS, horizon=1)
    assert result.persistence == persistence
    assert (result.unconditional_variance is not None) is mean_reverting
    assert result.warnings == (() if mean_reverting else (NOT_MEAN_REVERTING,))


@pytest.mark.parametrize("omega", [1e-15, 0.0], ids=["collapsed", "underflowed"])
def test_a_collapsed_omega_leaves_no_unconditional_variance(
    monkeypatch: pytest.MonkeyPatch, omega: float
) -> None:
    # Orchid's omega is the exponential of its search variable, so it can underflow to 0.0.
    def collapse(result: ar.GARCHResult) -> ar.GARCHResult:
        return dataclasses.replace(
            result,
            params={**result.params, "omega": omega},
            unconditional_variance=omega / (1 - result.persistence),
        )

    _patch_fit(monkeypatch, collapse)
    result = garch(RESIDUALS, horizon=1)
    assert result.unconditional_variance is None
    assert result.warnings == (OMEGA_COLLAPSED,)


def test_needs_100_residuals() -> None:
    garch(RESIDUALS[:100], horizon=1)
    with pytest.raises(InsufficientDataError) as excinfo:
        garch(RESIDUALS[:99], horizon=1)
    assert (excinfo.value.required, excinfo.value.available) == (100, 99)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        pytest.param(lambda: garch(RESIDUALS, horizon=0), InvalidInputError, id="horizon=0"),
        pytest.param(
            lambda: garch(RESIDUALS, horizon=1.0),  # type: ignore[arg-type]
            InvalidInputError,
            id="float horizon",
        ),
        pytest.param(lambda: garch(RESIDUALS, horizon=True), InvalidInputError, id="bool horizon"),
        pytest.param(  # Orchid silently fits a symmetric GARCH for model="GJR"
            lambda: garch(RESIDUALS, horizon=1, leverage="GJR"),  # type: ignore[arg-type]
            InvalidInputError,
            id="leverage='GJR'",
        ),
        pytest.param(
            lambda: garch(RESIDUALS, horizon=1, leverage=1),  # type: ignore[arg-type]
            InvalidInputError,
            id="leverage=1",
        ),
        pytest.param(
            lambda: garch([*RESIDUALS, None], horizon=1),  # type: ignore[list-item]
            MissingDataError,
            id="None",
        ),
        pytest.param(lambda: garch([*RESIDUALS, math.nan], horizon=1), MissingDataError, id="NaN"),
        pytest.param(lambda: garch([*RESIDUALS, math.inf], horizon=1), InvalidInputError, id="inf"),
        pytest.param(
            lambda: garch([*RESIDUALS, "0.5"], horizon=1),  # type: ignore[list-item]
            InvalidInputError,
            id="string",
        ),
        pytest.param(lambda: garch([*RESIDUALS, True], horizon=1), InvalidInputError, id="bool"),
        pytest.param(lambda: garch([-2.5] * 200, horizon=1), DegenerateDataError, id="constant"),
        pytest.param(
            lambda: garch([1e200, -1e200] * 100, horizon=1),
            CalculationError,
            id="variance overflows",
        ),
        pytest.param(
            lambda: garch([1e-170, -1e-170] * 100, horizon=1),
            CalculationError,
            id="variance underflows",
        ),
    ],
)
def test_unusable_input_is_rejected_before_orchid_runs(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[], object], error: type[Exception]
) -> None:
    monkeypatch.setattr(ar, "GARCH", _unreachable)
    with pytest.raises(error) as excinfo:
        call()
    assert excinfo.type is error


def test_orchid_failures_become_calculation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: ar.GARCH, *, seed: int = 0, maxiter: int = 1500) -> ar.GARCHResult:
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(ar.GARCH, "fit", broken)
    with pytest.raises(CalculationError) as excinfo:
        garch(RESIDUALS, horizon=1)
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


@pytest.mark.parametrize("change", UNUSABLE.values(), ids=UNUSABLE.keys())
def test_unusable_orchid_output_raises_calculation_error(
    monkeypatch: pytest.MonkeyPatch, change: Change
) -> None:
    _patch_fit(monkeypatch, change)
    with pytest.raises(CalculationError):
        garch(RESIDUALS, horizon=1)
