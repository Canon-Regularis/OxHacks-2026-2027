"""Conditional-variance models backed by Orchid (P1 #27)."""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from gridpulse.analysis.algorithms._checks import (
    finite_values,
    orchid_errors,
    require_int,
    require_variation,
)
from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    InsufficientDataError,
    InvalidInputError,
)
from gridpulse.analysis.algorithms.forecasting import _finite
from gridpulse.analysis.algorithms.orchid import autoregressive as ar

_MIN_RESIDUALS = 100
# Orchid keeps alpha1 + beta1 below 1 - 1e-6, so a fit pressed against that cap reports a
# persistence just under 1 and an unconditional variance that the cap sets. Persistence within
# a further 1e-6 of the cap counts as reaching 1.
_PERSISTENCE_LIMIT = 1 - 2e-6
_NOT_MEAN_REVERTING = (
    "variance not mean-reverting: persistence is within 2e-6 of 1 or above, so there is no "
    "unconditional variance"
)
# omega is weakly identified (not at all without ARCH effects), and Orchid's search can drive
# it towards zero with persistence below the cap. In simulations omega / (1 - persistence) then
# falls under 1e-8 of the residuals' variance, while other fits stay above 1e-3 of it.
_COLLAPSE_LIMIT = 1e-6
_OMEGA_COLLAPSED = (
    "omega collapsed: omega / (1 - persistence) is under 1e-6 of the residuals' variance, so "
    "there is no unconditional variance and variance forecasts decay towards zero"
)


@dataclass(frozen=True)
class GarchFit:
    """A GARCH(1, 1) or GJR-GARCH(1, 1) fit in the residuals' units.

    ``params`` holds ``mu`` (the constant mean), ``omega`` (in squared units), ``alpha1``,
    ``beta1`` and, with leverage, ``gamma1``: the extra response to negative shocks, which
    Orchid keeps between 0 and 0.5. Negate the residuals to test for a larger response to
    positive price spikes. ``persistence`` is alpha1 + beta1 + gamma1 / 2, and
    ``unconditional_variance`` is omega / (1 - persistence). It is ``None``, with a warning,
    when persistence is within 2e-6 of 1 or above, or when it is under 1e-6 of the residuals'
    variance because omega has collapsed towards zero. ``conditional_variance`` starts from the
    residuals' variance. It and ``standardized_residuals`` use parameters estimated on every
    residual, so they are retrospective: use them for diagnostics, never as causal features.
    """

    leverage: bool
    params: Mapping[str, float]
    persistence: float
    unconditional_variance: float | None
    log_likelihood: float
    aic: float
    bic: float
    conditional_variance: tuple[float, ...]
    standardized_residuals: tuple[float, ...]
    variance_forecast: tuple[float, ...]
    warnings: tuple[str, ...]


def garch(residuals: Sequence[float], *, horizon: int, leverage: bool = False) -> GarchFit:
    """GARCH(1, 1), or GJR-GARCH(1, 1) with ``leverage``, by Gaussian quasi-maximum likelihood.

    Give at least 100 complete innovations or residuals of a mean model, not price levels:
    Orchid fits only a constant mean. Test them for ARCH effects first (the ARCH-LM test in
    ``residual_diagnostics``): without any, omega and beta1 are not identified. Order (1, 1) is
    the standard choice; higher orders cost more without a demonstrated need. ``variance_forecast``
    covers the ``horizon`` steps after the last residual. The fit runs on the residuals divided
    by their standard deviation, so Orchid's absolute 1e-12 variance floor does not depend on
    the units. Orchid's Nelder-Mead search can stop at a local optimum: a leverage fit can then
    have a lower log-likelihood than the symmetric fit it nests. Fits to four weeks of half-hours
    take seconds, so run them in batch jobs.
    """
    require_int("horizon", horizon, minimum=1)
    if not isinstance(leverage, bool):
        raise InvalidInputError(f"leverage must be a bool, got {leverage!r}")
    values = finite_values("residuals", residuals)
    if len(values) < _MIN_RESIDUALS:
        raise InsufficientDataError(_MIN_RESIDUALS, len(values), what="residuals")
    require_variation("residual series", values)
    sigma = statistics.pstdev(values)
    variance = sigma * sigma
    if not 0 < variance < math.inf:
        raise CalculationError("the residuals' variance overflows or underflows a float")
    scaled = [value / sigma for value in values]
    with orchid_errors("GARCH"):
        model = ar.GARCH(scaled, p=1, q=1, model="gjr" if leverage else "garch")
        result = model.fit(seed=0)
        # Orchid forecasts from the instance's last fit, whatever result it is passed.
        forecast = model.forecast(result, horizon)
    units = {"mu": sigma, "omega": variance}  # alpha1, beta1 and gamma1 are unit-free
    params = {name: value * units.get(name, 1.0) for name, value in result.params.items()}
    # In the data's units the Gaussian log-likelihood falls by n ln(sigma), so AIC and BIC rise
    # by twice that.
    shift = len(values) * math.log(sigma)
    log_likelihood = result.loglike - shift
    aic, bic = result.aic + 2 * shift, result.bic + 2 * shift
    _finite([*params.values(), log_likelihood, aic, bic, result.persistence])
    warnings: tuple[str, ...] = ()
    if result.persistence >= _PERSISTENCE_LIMIT:
        warnings = (_NOT_MEAN_REVERTING,)
    elif result.unconditional_variance < _COLLAPSE_LIMIT:  # in units of the residuals' variance
        warnings = (_OMEGA_COLLAPSED,)
    return GarchFit(
        leverage=leverage,
        params=params,
        persistence=result.persistence,
        unconditional_variance=(
            None if warnings else _variances([result.unconditional_variance], variance)[0]
        ),
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        conditional_variance=_variances(result.conditional_variance, variance),
        standardized_residuals=_finite(result.std_resid),
        variance_forecast=_variances(forecast, variance),
        warnings=warnings,
    )


def _variances(values: Sequence[float], variance: float) -> tuple[float, ...]:
    """Variances of the unit-variance fit in the data's units; each must be a positive float."""
    rescaled = tuple(value * variance for value in values)
    if not all(0 < value < math.inf for value in rescaled):
        raise CalculationError("a GARCH variance is not a positive float in the data's units")
    return rescaled
