"""State-space models backed by Orchid (P1 #20, #21, #22, #23).

Structural models run on values standardised to zero mean and unit standard deviation,
because Orchid's start values, variance floors and smoother are absolute, and every result
is mapped back to the data's units. Orchid's exact-diffuse initialisation misclassifies the
diffuse updates of seasonal models (its tests are absolute), so every state instead starts
from a proper vague prior with variance kappa = 1e4 in standardised units.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Literal, cast

from gridpulse.analysis.algorithms._checks import (
    orchid_errors,
    require_fraction,
    require_int,
    require_levels,
    require_real,
    require_variation,
)
from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
)
from gridpulse.analysis.algorithms.forecasting import (
    DEFAULT_LEVELS,
    Forecast,
    PredictionInterval,
    _finite,
    _forecast,
    _prepare,
)
from gridpulse.analysis.algorithms.orchid import state_space as ss
from gridpulse.analysis.algorithms.series import HalfHourlySeries
from gridpulse.analysis.algorithms.signals import _result

_KAPPA = ss._BIG  # every state's prior variance, in standardised units
# The data identify a state once they have cut its prior variance at least 100-fold.
_IDENTIFIED = 1e-2 * _KAPPA
_MAX_STATES = 13  # the pure-Python filter costs about 2 ms per period at 13 states
_LOG_BOUND = 40.0  # Orchid's clamp on log variances
_LOGIT_BOUND = 30.0  # keeps the damping strictly inside (0, 1) in floating point
# A smoothed disturbance whose variance is within this fraction of the disturbance variance
# is uninformed by the data: its auxiliary residual would divide rounding error.
_INFORMED = 1e-9
_PRIOR_SPREAD = 100.0  # discount DLM state priors: 100 times the warm-up variance


@dataclass(frozen=True)
class Estimate:
    """Means and standard deviations aligned with the input; ``None`` where unavailable."""

    mean: HalfHourlySeries
    sd: HalfHourlySeries


@dataclass(frozen=True)
class StudentT:
    """Student-t means, scales and degrees of freedom aligned with the input.

    The standard deviation is ``scale * sqrt(dof / (dof - 2))``, defined only for dof > 2.
    """

    mean: HalfHourlySeries
    scale: HalfHourlySeries
    dof: HalfHourlySeries


@dataclass(frozen=True)
class Band:
    """Central prediction intervals at coverage ``level``."""

    level: float
    lower: HalfHourlySeries
    upper: HalfHourlySeries


@dataclass(frozen=True, kw_only=True)
class StructuralModel:
    """A level or trend, trigonometric seasons and regressors, plus a white-noise irregular.

    ``trend`` is a random-walk level, a local linear trend, or a trend whose slope is
    multiplied by a damping factor each period. ``seasons`` holds (period, harmonics) pairs
    with periods of at least 3 half-hours and 1 <= harmonics <= (period - 1) // 2: Orchid
    would turn 0 harmonics into period // 2, and an even period's last harmonic adds a state
    the data cannot identify. For the same reason no two seasons may share a harmonic
    frequency j / period, as (48, 2) and (24, 1) do. ``regressors`` names driver series
    with constant coefficients; ``dynamic_regression`` makes the coefficient a random walk,
    for a single driver only, because Orchid gives every dynamic coefficient one shared
    variance, which no choice of units makes meaningful for several. The state dimension d
    (1 for a level, 2 for a trend, 2 per harmonic and 1 per regressor) is at most 13.
    """

    trend: Literal["level", "linear", "damped"] = "level"
    seasons: tuple[tuple[int, int], ...] = ()
    regressors: tuple[str, ...] = ()
    dynamic_regression: bool = False

    def __post_init__(self) -> None:
        if self.trend not in ("level", "linear", "damped"):
            raise InvalidInputError(
                f"trend must be 'level', 'linear' or 'damped', got {self.trend!r}"
            )
        seasons = tuple(_season(i, season) for i, season in enumerate(self.seasons))
        frequencies = [
            Fraction(j, period) for period, harmonics in seasons for j in range(1, harmonics + 1)
        ]
        if len(set(frequencies)) < len(frequencies):
            raise InvalidInputError(
                f"seasons must not share a harmonic frequency j / period, got {seasons}"
            )
        if isinstance(self.regressors, str):
            raise InvalidInputError("regressors must be a tuple of names, not one string")
        regressors = tuple(self.regressors)
        named = all(isinstance(name, str) and name for name in regressors)
        if not named or len(set(regressors)) < len(regressors):
            raise InvalidInputError(
                f"regressors must be distinct non-empty names, got {regressors}"
            )
        if not isinstance(self.dynamic_regression, bool):
            raise InvalidInputError(
                f"dynamic_regression must be a bool, got {self.dynamic_regression!r}"
            )
        if self.dynamic_regression and len(regressors) != 1:
            raise InvalidInputError("dynamic_regression needs exactly one regressor")
        object.__setattr__(self, "seasons", seasons)
        object.__setattr__(self, "regressors", regressors)
        if self.state_dimension > _MAX_STATES:
            raise InvalidInputError(
                f"the model has {self.state_dimension} states; at most {_MAX_STATES} are allowed"
            )

    @property
    def state_dimension(self) -> int:
        trend = 1 if self.trend == "level" else 2
        return trend + 2 * sum(harmonics for _, harmonics in self.seasons) + len(self.regressors)

    @property
    def variance_names(self) -> tuple[str, ...]:
        """The keys of ``StructuralParameters.variances``, in Orchid's parameter order."""
        names = ["level"]
        if self.trend != "level":
            names.append("slope")
        names += [f"seasonal_{period}" for period, _ in self.seasons]
        if self.dynamic_regression:
            names.append("regression")
        return (*names, "irregular")


@dataclass(frozen=True)
class StructuralParameters:
    """Disturbance variances in the data's units, and the trend's damping.

    ``variances`` has exactly the keys of ``model.variance_names``: per-period variances of
    the level, slope, each season's states and the irregular, in squared data units, and of
    the dynamic coefficient (``regression``), in squared series units per squared regressor
    unit. They are finite and non-negative, and at least one is positive. Give ``damping``,
    in (0, 1), exactly when the trend is damped.
    """

    model: StructuralModel
    variances: Mapping[str, float]
    damping: float | None = None

    def __post_init__(self) -> None:
        names = self.model.variance_names
        if set(self.variances) != set(names):
            raise InvalidInputError(
                f"variances must have exactly the keys {list(names)}, got {list(self.variances)}"
            )
        variances = {
            name: require_real(f"variances[{name!r}]", self.variances[name]) for name in names
        }
        if not all(0 <= value < math.inf for value in variances.values()) or not any(
            variances.values()
        ):
            raise InvalidInputError(
                f"variances must be finite, non-negative and not all zero, got {variances}"
            )
        if (self.model.trend == "damped") != (self.damping is not None):
            raise InvalidInputError("give damping exactly when the trend is damped")
        damping = None if self.damping is None else require_fraction("damping", self.damping)
        object.__setattr__(self, "variances", MappingProxyType(variances))
        object.__setattr__(self, "damping", damping)

    def __reduce__(self) -> tuple[Any, ...]:  # pickle and deepcopy cannot copy a mappingproxy
        return type(self), (self.model, dict(self.variances), self.damping)


@dataclass(frozen=True)
class StructuralFit:
    """Maximum-likelihood parameters, with the log-likelihood in the data's units.

    The log-likelihood follows Orchid's exact-diffuse convention, which leaves out the d
    observations that identify the initial state, so fits with different state dimensions
    are not comparable: their difference depends on the data's units. Regressors enter
    standardised, so the log-likelihood does not depend on their units.
    """

    parameters: StructuralParameters
    log_likelihood: float
    aic: float
    bic: float
    nobs: int


@dataclass(frozen=True)
class StructuralFilter:
    """Estimates from the values up to each period, in the data's units.

    ``seasons`` maps each period to its seasonal contribution and ``regression`` is the
    regressors' total contribution, sum b_t * x_t. ``prediction`` and ``bands`` give the
    one-step predictive distribution of each period's value, made at the period before;
    ``standardized_innovation`` divides its error by its standard deviation and is standard
    normal under the model.
    """

    level: Estimate
    slope: Estimate | None
    seasons: Mapping[int, Estimate]
    regression: Estimate | None
    prediction: Estimate
    bands: tuple[Band, ...]
    standardized_innovation: HalfHourlySeries


@dataclass(frozen=True)
class StructuralSmooth:
    """Estimates from the whole series, in the data's units: retrospective, never features.

    Components are as in ``StructuralFilter``; ``irregular`` is the smoothed observation
    noise. The scores are auxiliary residuals (Harvey and Koopman), standard normal under
    the model: ``outlier_score`` standardises the irregular, and ``level_break_score[t]``
    the smoothed level disturbance that moves the level from period t - 1 to t, so a level
    shift scores at the first period at its new level; ``slope_break_score`` likewise for
    the slope. Scores are missing at gaps and where the data do not inform the disturbance,
    including components with zero variance, and break scores at the first observed period.
    """

    level: Estimate
    slope: Estimate | None
    seasons: Mapping[int, Estimate]
    regression: Estimate | None
    irregular: Estimate
    outlier_score: HalfHourlySeries
    level_break_score: HalfHourlySeries
    slope_break_score: HalfHourlySeries | None


@dataclass(frozen=True)
class DiscountFilter:
    """Discount DLM estimates from the values up to each period, in the data's units.

    ``level`` and ``slope`` are posteriors, ``prediction`` is the one-step predictive
    distribution made at the period before, and ``bands`` are its central intervals.
    ``observation_sd`` estimates the observation noise; ``standardized_error`` divides each
    prediction error by the predictive scale.
    """

    level: StudentT
    slope: StudentT | None
    observation_sd: HalfHourlySeries
    prediction: StudentT
    bands: tuple[Band, ...]
    standardized_error: HalfHourlySeries


def fit_structural(
    series: HalfHourlySeries,
    model: StructuralModel,
    *,
    regressors: Mapping[str, HalfHourlySeries] | None = None,
) -> StructuralFit:
    """Maximum-likelihood variances (and damping) by Orchid's Nelder-Mead search.

    Gaps are allowed. With k estimated parameters, the fit needs max(d + k + 1, twice the
    longest season) observed values; ``regressors`` must cover every period of ``series``.
    A local level over four weeks takes seconds and a seasonal model tens of seconds, so run
    fits in batch jobs.
    """
    data = _standardise(series, model, regressors, minimum=_fit_minimum(model))
    components, observations = _unobserved(model, data), _observations(data)
    damping = 2 if model.trend == "damped" else None  # Orchid orders level, slope, damping

    def build(natural: Sequence[float]) -> ss.LinearGaussianSSM:
        return _vague(components.build(natural), model, data)

    def transform(free: Sequence[float]) -> list[float]:
        return [_logistic(u) if i == damping else _exp(u) for i, u in enumerate(free)]

    start = components.start_params()
    free = [math.log(p / (1 - p)) if i == damping else math.log(p) for i, p in enumerate(start)]
    # Identification is a property of the design, so the check switches off every state
    # disturbance: at the start values the drift of a slow harmonic alone would fail it.
    static = [p if i in (damping, len(start) - 1) else 0.0 for i, p in enumerate(start)]
    with orchid_errors("kalman_filter"):
        initial = ss.kalman_filter(build(static), observations)
    _check_identified(initial.filtered_cov[data.observed[-1]], "the observed values")
    with orchid_errors("fit_ssm_mle"):
        fit = ss.fit_ssm_mle(
            build, free, observations, transform=transform, maxiter=2000, restarts=1, seed=0
        )
    d, nobs = model.state_dimension, len(data.observed)
    # The prior's density costs each diffuse state ln(2 pi kappa) / 2 against Orchid's
    # exact-diffuse convention, and in the data's units every observation after the diffuse
    # phase loses ln(sd). Regressors stay standardised, so the value does not depend on
    # their units; Orchid's on the raw regressors is lower by the sum of ln sd(x).
    log_likelihood = (
        fit.loglike + d / 2 * math.log(2 * math.pi * _KAPPA) - (nobs - d) * math.log(data.sd)
    )
    if not math.isfinite(log_likelihood):
        raise CalculationError(f"Orchid's fit returned a log-likelihood of {fit.loglike}")
    k = len(fit.params)
    return StructuralFit(
        parameters=_parameters(model, fit.params, data),
        log_likelihood=log_likelihood,
        aic=-2 * log_likelihood + 2 * k,
        bic=-2 * log_likelihood + k * math.log(nobs),
        nobs=nobs,
    )


def filter_structural(
    series: HalfHourlySeries,
    parameters: StructuralParameters,
    *,
    regressors: Mapping[str, HalfHourlySeries] | None = None,
    prior_periods: int = 48,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> StructuralFilter:
    """Kalman-filtered components and one-step predictions; causal given the parameters.

    Values and regressors are standardised by the first ``prior_periods`` observed values,
    which must cover the longest season and at least max(2, d) values, and be followed by
    another observed value. States start at the last of them and predictions one period
    later, because a prediction for that period would see its own value through the
    standardisation. Regressors are known inputs: each prediction uses their values at the
    period predicted. At gaps, states repeat their prediction and innovations are missing.
    """
    model = parameters.model
    minimum = max(2, model.state_dimension, _longest(model))
    require_int("prior_periods", prior_periods, minimum=minimum)
    checked_levels = require_levels(levels)
    data = _standardise(series, model, regressors, minimum=prior_periods + 1, sample=prior_periods)
    natural = _natural(parameters, data)
    with orchid_errors("kalman_filter"):
        components = _unobserved(model, data)
        orchid_model = _vague(components.build(natural), model, data)
        filtered = ss.kalman_filter(orchid_model, _observations(data))
    ready = data.observed[prior_periods - 1]
    _check_identified(filtered.filtered_cov[ready], "the prior window's values")
    level, slope, seasons, regression = _components(
        series, data, model, filtered.filtered_state, filtered.filtered_cov, start=ready
    )
    quantiles = [ss.norm_ppf(0.5 + coverage / 2) for coverage in checked_levels]
    n = len(data.values)
    predictions: list[tuple[float, float, list[float]] | None] = [None] * n
    innovations: list[float | None] = [None] * n
    for t in range(ready + 1, n):
        signal = _dot(orchid_model.Z_t(t)[0], filtered.predicted_state[t])
        variance = filtered.forecast_error_cov[t][0][0]
        scale = data.sd * math.sqrt(max(variance, 0.0))
        predictions[t] = (data.mean + data.sd * signal, scale, [q * scale for q in quantiles])
        if data.values[t] is not None and variance > 0:
            innovations[t] = filtered.forecast_error[t][0] / math.sqrt(variance)
    means, scales = _columns(series, data.first, predictions)
    return StructuralFilter(
        level=level,
        slope=slope,
        seasons=seasons,
        regression=regression,
        prediction=Estimate(means, scales),
        bands=_bands(series, data.first, checked_levels, predictions),
        standardized_innovation=_aligned(series, data.first, innovations),
    )


def smooth_structural(
    series: HalfHourlySeries,
    parameters: StructuralParameters,
    *,
    regressors: Mapping[str, HalfHourlySeries] | None = None,
) -> StructuralSmooth:
    """Smoothed components, irregular and auxiliary residuals from the whole series.

    Retrospective: every estimate uses values after its period. Needs max(d + 1, longest
    season) observed values.
    """
    model = parameters.model
    data = _standardise(series, model, regressors, minimum=_fixed_minimum(model))
    natural = _natural(parameters, data)
    with orchid_errors("disturbance_smoother"):
        components = _unobserved(model, data)
        orchid_model = _vague(components.build(natural), model, data)
        smoothed = ss.disturbance_smoother(orchid_model, _observations(data))
    _check_identified(smoothed.smoothed_state_cov[0], "the observed values")
    level, slope, seasons, regression = _components(
        series, data, model, smoothed.smoothed_state, smoothed.smoothed_state_cov, start=0
    )
    n = len(data.values)
    means: list[float | None] = [None] * n
    sds: list[float | None] = [None] * n
    outliers: list[float | None] = [None] * n
    for t in data.observed:
        estimate = smoothed.smoothed_obs_disturbance[t][0]
        variance = smoothed.smoothed_obs_disturbance_cov[t][0][0]
        means[t], sds[t] = data.sd * estimate, data.sd * math.sqrt(max(variance, 0.0))
        outliers[t] = _auxiliary(estimate, natural[-1], variance)
    return StructuralSmooth(
        level=level,
        slope=slope,
        seasons=seasons,
        regression=regression,
        irregular=Estimate(_aligned(series, data.first, means), _aligned(series, data.first, sds)),
        outlier_score=_aligned(series, data.first, outliers),
        level_break_score=_breaks(series, data, smoothed, 0, natural[0]),
        slope_break_score=(
            None if model.trend == "level" else _breaks(series, data, smoothed, 1, natural[1])
        ),
    )


@dataclass(frozen=True)
class StructuralForecaster:
    """Structural-model forecasts by ``ss.forecast``, refitted to each history unless
    ``parameters`` are given.

    Models with regressors are rejected, since their future values are unknown. Identifying
    every state takes d values and a full cycle of the longest season; residuals start after
    them, so with fixed parameters the history needs one value more.
    """

    model: StructuralModel
    _: KW_ONLY
    parameters: StructuralParameters | None = None

    def __post_init__(self) -> None:
        if self.model.regressors:
            raise InvalidInputError("forecasts need a model without regressors")
        if self.parameters is not None and self.parameters.model != self.model:
            raise InvalidInputError("parameters were given for a different model")

    @property
    def min_history(self) -> int:
        if self.parameters is None:
            return _fit_minimum(self.model)
        return _warm_up(self.model) + 1

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        model = self.model
        horizon, checked_levels, _ = _prepare(history, horizon, levels, minimum=self.min_history)
        params: dict[str, object] = {"trend": model.trend, "seasons": model.seasons}
        if self.parameters is None:
            fit = fit_structural(history, model)
            parameters = fit.parameters
            params |= {"log_likelihood": fit.log_likelihood, "aic": fit.aic, "bic": fit.bic}
        else:
            parameters = self.parameters
        params["variances"] = dict(parameters.variances)
        if parameters.damping is not None:
            params["damping"] = parameters.damping
        data = _standardise(history, model, None, minimum=self.min_history)
        natural = _natural(parameters, data)
        with orchid_errors("forecast"):
            components = _unobserved(model, data)
            orchid_model = _vague(components.build(natural), model, data)
            result = ss.forecast(orchid_model, data.values, horizon, level=checked_levels)
            filtered = ss.kalman_filter(orchid_model, _observations(data))

        def unscaled(standardised: Sequence[Sequence[float]]) -> tuple[float, ...]:
            return _finite([data.mean + data.sd * value[0] for value in standardised])

        intervals = tuple(
            PredictionInterval(
                coverage, unscaled(result.lower[coverage]), unscaled(result.upper[coverage])
            )
            for coverage in checked_levels
        )
        errors = filtered.forecast_error[_warm_up(model) :]
        return _forecast(
            history,
            source="state_space.forecast",
            params=params,
            mean=unscaled(result.mean),
            intervals=intervals,
            interval_method="analytic",
            residuals=_finite([data.sd * error[0] for error in errors]),
            seasonal=bool(model.seasons),
        )


def discount_filter(
    series: HalfHourlySeries,
    *,
    trend: Literal["level", "linear"] = "level",
    discount: float = 0.95,
    variance_discount: float = 1.0,
    prior_periods: int = 48,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> DiscountFilter:
    """West and Harrison's discount DLM with a level, or a level and slope; causal.

    Each period the state's prior variance grows by 1 / ``discount``, and the observation
    variance is learned online, discounting past evidence by ``variance_discount`` (1 keeps
    it all). The priors come from the first ``prior_periods`` observed values, which must
    vary and be followed by another observed value: the level at their mean (the slope at
    zero) with 100 times their variance, which also seeds the observation variance with one
    degree of freedom. States start at the last of these values and predictions one period
    later, as in ``filter_structural``. At gaps, states repeat their prediction.
    """
    discount, variance_discount = _check_discount(trend, discount, variance_discount, prior_periods)
    checked_levels = require_levels(levels)
    observed = [t for t, value in enumerate(series.values) if value is not None]
    if len(observed) <= prior_periods:
        raise InsufficientDataError(prior_periods + 1, len(observed), what="observed values")
    first = observed[0]
    values = series.values[first:]
    window = [value for value in values if value is not None][:prior_periods]
    dlm = _discount_model(trend, discount, variance_discount, window)
    with orchid_errors("DiscountDLM"):
        result = dlm.filter(_orchid_values(values))
    ready = observed[prior_periods - 1] - first

    def from_ready(column: Sequence[float]) -> HalfHourlySeries:
        return _aligned(series, first, [*([None] * ready), *column[ready:]])

    def posterior(state: int) -> StudentT:
        means = [mean[state] for mean in result.filtered_mean]
        scales = [math.sqrt(max(cov[state][state], 0.0)) for cov in result.filtered_cov]
        return StudentT(from_ready(means), from_ready(scales), from_ready(result.dof))

    n = len(values)
    predictions: list[tuple[float, float, list[float]] | None] = [None] * n
    dofs: list[float | None] = [None] * n
    errors: list[float | None] = [None] * n
    for t in range(ready + 1, n):
        mean, scale = result.forecast[t], math.sqrt(result.forecast_var[t])
        dof = variance_discount * result.dof[t - 1]  # the last posterior's, discounted once
        widths = [_t_quantile(0.5 + coverage / 2, dof) * scale for coverage in checked_levels]
        predictions[t], dofs[t] = (mean, scale, widths), dof
        value = values[t]
        if value is not None:
            errors[t] = (value - mean) / scale
    means, scales = _columns(series, first, predictions)
    return DiscountFilter(
        level=posterior(0),
        slope=posterior(1) if trend == "linear" else None,
        observation_sd=from_ready([math.sqrt(scale) for scale in result.obs_scale]),
        prediction=StudentT(means, scales, _aligned(series, first, dofs)),
        bands=_bands(series, first, checked_levels, predictions),
        standardized_error=_aligned(series, first, errors),
    )


@dataclass(frozen=True, kw_only=True)
class DiscountForecaster:
    """Discount DLM forecasts (West and Harrison), so the DLM can be backtested like the
    fitted forecasters.

    The history is filtered as in ``discount_filter``. Forecasts then hold the evolution
    variance at W = (1 - discount) / discount * G C G' from the final posterior, rather than
    compounding 1 / discount every step as Orchid's own forecast does. Intervals are Student-t
    with the one-step degrees of freedom, ``variance_discount`` times the final posterior's,
    likewise held rather than discounted again every step.
    """

    trend: Literal["level", "linear"] = "level"
    discount: float = 0.95
    variance_discount: float = 1.0
    prior_periods: int = 48

    def __post_init__(self) -> None:
        discount, variance_discount = _check_discount(
            self.trend, self.discount, self.variance_discount, self.prior_periods
        )
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "variance_discount", variance_discount)

    @property
    def min_history(self) -> int:
        return self.prior_periods + 1

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        horizon, checked_levels, values = _prepare(
            history, horizon, levels, minimum=self.min_history
        )
        dlm = _discount_model(
            self.trend, self.discount, self.variance_discount, values[: self.prior_periods]
        )
        with orchid_errors("DiscountDLM"):
            result = dlm.filter(values)
        transposed = ss.transpose(dlm.G)
        spread = ss.matmul(ss.matmul(dlm.G, result.filtered_cov[-1]), transposed)  # G C G'
        evolution = ss.mat_scale(spread, (1 - self.discount) / self.discount)
        state, prior = result.filtered_mean[-1], ss.mat_scale(spread, 1 / self.discount)
        noise, dof = result.obs_scale[-1], result.dof[-1]
        means, scales = [], []
        for step in range(horizon):
            state = ss.matvec(dlm.G, state)
            if step:
                prior = ss.mat_add(ss.matmul(ss.matmul(dlm.G, prior), transposed), evolution)
            means.append(_dot(dlm.F, state))
            scales.append(math.sqrt(_dot(dlm.F, ss.matvec(prior, dlm.F)) + noise))
        intervals = []
        for coverage in checked_levels:
            q = _t_quantile(0.5 + coverage / 2, self.variance_discount * dof)
            lower = _finite([m - q * s for m, s in zip(means, scales, strict=True)])
            upper = _finite([m + q * s for m, s in zip(means, scales, strict=True)])
            intervals.append(PredictionInterval(coverage, lower, upper))
        residuals = [values[t] - result.forecast[t] for t in range(self.prior_periods, len(values))]
        return _forecast(
            history,
            source="state_space.DiscountDLM",
            params={
                "trend": self.trend,
                "discount": self.discount,
                "variance_discount": self.variance_discount,
                "prior_periods": self.prior_periods,
                "observation_variance": noise,
                "dof": dof,
            },
            mean=_finite(means),
            intervals=tuple(intervals),
            interval_method="student-t",
            residuals=_finite(residuals),
        )


@dataclass(frozen=True)
class _Data:
    """A series from its first observed period, standardised, with its regressors."""

    first: int  # the model starts at the first observed period
    mean: float
    sd: float
    values: list[float | None]
    observed: list[int]  # positions of the observed values in ``values``
    rows: list[list[float]]  # standardised regressors, one row per period
    centres: tuple[float, ...]
    scales: tuple[float, ...]


def _season(index: int, season: tuple[int, int]) -> tuple[int, int]:
    try:
        period, harmonics = season
    except (TypeError, ValueError):
        raise InvalidInputError(
            f"seasons[{index}] must be a (period, harmonics) pair, got {season!r}"
        ) from None
    require_int(f"seasons[{index}] period", period, minimum=3)
    require_int(f"seasons[{index}] harmonics", harmonics, minimum=1, maximum=(period - 1) // 2)
    return period, harmonics


def _longest(model: StructuralModel) -> int:
    return max((period for period, _ in model.seasons), default=0)


def _fit_minimum(model: StructuralModel) -> int:
    """One more observed value than states plus parameters, and two cycles of each season."""
    estimated = len(model.variance_names) + (model.trend == "damped")
    return max(model.state_dimension + estimated + 1, 2 * _longest(model))


def _fixed_minimum(model: StructuralModel) -> int:
    """d + 1 observed values, and a full cycle of the longest season: with fewer, the prior
    dominates the slowest harmonics (after three periods a 48-period harmonic keeps over a
    quarter of its prior variance)."""
    return max(model.state_dimension + 1, _longest(model))


def _warm_up(model: StructuralModel) -> int:
    """The one-step errors the prior dominates: identifying every state takes d values and a
    full cycle of the longest season."""
    return max(model.state_dimension, _longest(model))


def _standardise(
    series: HalfHourlySeries,
    model: StructuralModel,
    regressors: Mapping[str, HalfHourlySeries] | None,
    *,
    minimum: int,
    sample: int | None = None,
) -> _Data:
    """The series and regressors scaled by the mean and standard deviation of the first
    ``sample`` observed periods (all of them by default), once ``minimum`` are observed.

    Centring the regressors keeps the states modest, so the vague prior stays vague whatever
    the regressors' origin; the level state then carries b * mean(x) (see ``_vague``).
    """
    columns = _regressor_columns(series, model, regressors)
    observed = [(t, value) for t, value in enumerate(series.values) if value is not None]
    if len(observed) < minimum:
        raise InsufficientDataError(minimum, len(observed), what="observed values")
    used = observed[:sample]
    where = "the series" if sample is None else "the prior window"
    mean, variance = _moments(where, [value for _, value in used])
    sd = math.sqrt(variance)
    centres, scales = [], []
    for name, column in zip(model.regressors, columns, strict=True):
        centre, spread = _moments(f"regressor {name!r} in {where}", [column[t] for t, _ in used])
        centres.append(centre)
        scales.append(math.sqrt(spread))
    first = observed[0][0]
    return _Data(
        first=first,
        mean=mean,
        sd=sd,
        values=[None if value is None else (value - mean) / sd for value in series.values[first:]],
        observed=[t - first for t, _ in observed],
        rows=[
            [(column[t] - c) / s for column, c, s in zip(columns, centres, scales, strict=True)]
            for t in range(first, len(series))
        ],
        centres=tuple(centres),
        scales=tuple(scales),
    )


def _regressor_columns(
    series: HalfHourlySeries,
    model: StructuralModel,
    regressors: Mapping[str, HalfHourlySeries] | None,
) -> list[list[float]]:
    """The model's drivers, which must cover every period of ``series`` completely."""
    given = regressors or {}
    if set(given) != set(model.regressors):
        raise InvalidInputError(
            f"regressors must be exactly {list(model.regressors)}, got {list(given)}"
        )
    columns = []
    for name in model.regressors:
        driver = given[name]
        if driver.start != series.start or len(driver) != len(series):
            raise InvalidInputError(f"regressor {name!r} must cover the same periods as the series")
        columns.append(driver.complete_values())
    return columns


def _moments(name: str, values: Sequence[float]) -> tuple[float, float]:
    """The mean and sample variance of values that vary."""
    require_variation(name, values)
    try:
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    except OverflowError:
        variance = math.inf
    if not 0 < variance < math.inf:
        raise CalculationError(f"the variance of {name} overflows or underflows a float")
    return mean, variance


def _unobserved(model: StructuralModel, data: _Data) -> ss.UnobservedComponents:
    """Orchid's components for ``model``: trend, seasons, regression, irregular."""
    trend: ss.Component = (
        ss.LocalLevel()
        if model.trend == "level"
        else ss.LocalLinearTrend(damped=model.trend == "damped")
    )
    components = [trend]
    components += [ss.TrigonometricSeasonal(p, harmonics=h) for p, h in model.seasons]
    if model.regressors:
        components.append(ss.Regression(data.rows, dynamic=model.dynamic_regression))
    return ss.UnobservedComponents(_orchid_values(data.values), [*components, ss.Irregular()])


def _orchid_values(values: Sequence[float | None]) -> list[float]:
    """``values`` as Orchid annotates its inputs; it reads ``None`` as a missing value."""
    return cast(list[float], list(values))


def _observations(data: _Data) -> list[list[float]]:
    """The standardised values as Orchid's one-element observation vectors."""
    return [[value] for value in _orchid_values(data.values)]


def _vague(
    orchid_model: ss.LinearGaussianSSM, model: StructuralModel, data: _Data
) -> ss.LinearGaussianSSM:
    """``orchid_model`` with prior variance kappa for every state instead of Orchid's exact
    diffuse initialisation; every state of these models is diffuse, with no proper prior part.

    Centring a dynamic regressor puts b * mean(x) in the level state, so the coefficient's
    disturbance moves the level too. The change of state has unit determinant, which leaves
    the model and its exact-diffuse log-likelihood as they were.
    """
    loadings = orchid_model.R_t(0)
    if model.dynamic_regression:  # R[0][-1] loads the coefficient's disturbance on the level
        loadings = [row[:] for row in loadings]
        loadings[0][-1] = data.centres[0] / data.scales[0]
    return orchid_model.copy_with(
        P1=ss.eye(orchid_model.k_states, _KAPPA), P1_infinity=None, R=loadings
    )


def _factors(model: StructuralModel, data: _Data) -> list[float]:
    """Each variance's factor from the data's units to standardised ones."""
    unit = 1 / (data.sd * data.sd)
    return [
        unit * data.scales[0] ** 2 if name == "regression" else unit
        for name in model.variance_names
    ]


def _natural(parameters: StructuralParameters, data: _Data) -> list[float]:
    """Orchid's parameters in standardised units: the variances, with the damping after the
    slope's."""
    model = parameters.model
    natural = [
        parameters.variances[name] * factor
        for name, factor in zip(model.variance_names, _factors(model, data), strict=True)
    ]
    if not all(map(math.isfinite, natural)):
        raise CalculationError("the variances overflow a float in standardised units")
    if parameters.damping is not None:
        natural.insert(2, parameters.damping)
    return natural


def _parameters(
    model: StructuralModel, natural: Sequence[float], data: _Data
) -> StructuralParameters:
    """The inverse of ``_natural``."""
    variances = list(natural)
    damping = variances.pop(2) if model.trend == "damped" else None
    unscaled = {
        name: value / factor
        for name, value, factor in zip(
            model.variance_names, variances, _factors(model, data), strict=True
        )
    }
    if not all(map(math.isfinite, unscaled.values())):
        raise CalculationError("the fitted variances overflow a float in the data's units")
    return StructuralParameters(model, unscaled, damping)


def _exp(u: float) -> float:
    return math.exp(max(-_LOG_BOUND, min(_LOG_BOUND, u)))


def _logistic(u: float) -> float:
    return 1 / (1 + math.exp(-max(-_LOGIT_BOUND, min(_LOGIT_BOUND, u))))


def _check_identified(covariance: Sequence[Sequence[float]], source: str) -> None:
    if max(covariance[i][i] for i in range(len(covariance))) > _IDENTIFIED:
        raise DegenerateDataError(
            f"{source} do not identify every state: use more values, or check the regressors "
            "for collinearity or constant stretches"
        )


def _components(
    series: HalfHourlySeries,
    data: _Data,
    model: StructuralModel,
    states: Sequence[Sequence[float]],
    covariances: Sequence[Sequence[Sequence[float]]],
    *,
    start: int,
) -> tuple[Estimate, Estimate | None, dict[int, Estimate], Estimate | None]:
    """Level, slope, seasonal and regression estimates from ``start`` on, in data units.

    Each is a weighted sum of states. Centring the regressors moves b * mean(x) into the
    level state, so the level takes it back out and the regression stays sum b * x.
    """
    n = len(data.values)

    def estimate(weights: Sequence[Sequence[tuple[int, float]]], offset: float = 0.0) -> Estimate:
        means: list[float | None] = [None] * n
        sds: list[float | None] = [None] * n
        for t in range(start, n):
            mean, sd = _combination(weights[t], states[t], covariances[t])
            means[t], sds[t] = offset + data.sd * mean, data.sd * sd
        return Estimate(_aligned(series, data.first, means), _aligned(series, data.first, sds))

    position = 1 if model.trend == "level" else 2
    seasons = {}
    for period, harmonics in model.seasons:  # each harmonic's first state is the observed one
        seasons[period] = estimate([[(position + 2 * i, 1.0) for i in range(harmonics)]] * n)
        position += 2 * harmonics
    ratios = [centre / scale for centre, scale in zip(data.centres, data.scales, strict=True)]
    level = estimate(
        [[(0, 1.0), *((position + j, -r) for j, r in enumerate(ratios) if r)]] * n, data.mean
    )
    regression = None
    if model.regressors:
        loadings = [
            [(position + j, x + r) for j, (x, r) in enumerate(zip(row, ratios, strict=True))]
            for row in data.rows
        ]
        regression = estimate(loadings)
    slope = None if model.trend == "level" else estimate([[(1, 1.0)]] * n)
    return level, slope, seasons, regression


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _combination(
    weights: Sequence[tuple[int, float]],
    state: Sequence[float],
    covariance: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """The mean and standard deviation of a weighted sum of states."""
    mean = math.fsum(w * state[i] for i, w in weights)
    variance = math.fsum(v * w * covariance[i][j] for i, v in weights for j, w in weights)
    return mean, math.sqrt(max(variance, 0.0))  # rounding can leave a tiny negative


def _auxiliary(estimate: float, variance: float, posterior: float) -> float | None:
    """A smoothed disturbance over its own standard deviation, sqrt(variance - posterior
    variance), or ``None`` when the data barely inform it."""
    spread = variance - posterior
    return estimate / math.sqrt(spread) if spread > _INFORMED * variance else None


def _breaks(
    series: HalfHourlySeries,
    data: _Data,
    smoothed: ss.SmootherResult,
    index: int,
    variance: float,
) -> HalfHourlySeries:
    """Auxiliary residuals of state disturbance ``index``. Orchid's disturbance at t moves
    the state from t to t + 1, so each score is placed at the period it arrives at."""
    scores: list[float | None] = [None] * len(data.values)
    for t in data.observed[1:]:
        disturbance = smoothed.smoothed_state_disturbance[t - 1][index]
        posterior = smoothed.smoothed_state_disturbance_cov[t - 1][index][index]
        scores[t] = _auxiliary(disturbance, variance, posterior)
    return _aligned(series, data.first, scores)


def _columns(
    series: HalfHourlySeries, first: int, rows: Sequence[tuple[float, float, list[float]] | None]
) -> tuple[HalfHourlySeries, HalfHourlySeries]:
    """The means and scales of (mean, scale, half-widths) rows, as aligned series."""
    means = [None if row is None else row[0] for row in rows]
    scales = [None if row is None else row[1] for row in rows]
    return _aligned(series, first, means), _aligned(series, first, scales)


def _bands(
    series: HalfHourlySeries,
    first: int,
    levels: Sequence[float],
    rows: Sequence[tuple[float, float, list[float]] | None],
) -> tuple[Band, ...]:
    """One band per level from (mean, scale, half-width per level) rows."""

    def bound(i: int, sign: float) -> HalfHourlySeries:
        values = [None if row is None else row[0] + sign * row[2][i] for row in rows]
        return _aligned(series, first, values)

    return tuple(Band(level, bound(i, -1.0), bound(i, 1.0)) for i, level in enumerate(levels))


def _aligned(series: HalfHourlySeries, first: int, values: list[float | None]) -> HalfHourlySeries:
    """Results for the periods from ``first`` on, aligned with ``series``."""
    return _result(series, [*([None] * first), *values])


def _check_discount(
    trend: str, discount: float, variance_discount: float, prior_periods: int
) -> tuple[float, float]:
    """The validated discount and variance discount, as floats."""
    if trend not in ("level", "linear"):
        raise InvalidInputError(f"trend must be 'level' or 'linear', got {trend!r}")
    checked = []
    for name, value in (("discount", discount), ("variance_discount", variance_discount)):
        number = require_real(name, value)
        if not 0 < number <= 1:
            raise InvalidInputError(f"{name} must be in (0, 1], got {value!r}")
        checked.append(number)
    require_int("prior_periods", prior_periods, minimum=2)
    return checked[0], checked[1]


def _discount_model(
    trend: str, discount: float, variance_discount: float, window: Sequence[float]
) -> ss.DiscountDLM:
    """Orchid's DLM with its priors in the data's units, from the warm-up values."""
    mean, variance = _moments("the prior window", window)
    if trend == "level":
        loading, evolution, start = [1.0], [[1.0]], [mean]
    else:
        loading, evolution, start = [1.0, 0.0], [[1.0, 1.0], [0.0, 1.0]], [mean, 0.0]
    return ss.DiscountDLM(
        loading,
        evolution,
        delta=discount,
        m0=start,
        C0=ss.eye(len(loading), _PRIOR_SPREAD * variance),
        n0=1.0,
        s0=variance,
        beta=variance_discount,
    )


def _t_quantile(probability: float, dof: float) -> float:
    """``ss.t_ppf(probability, df=dof)`` for a probability above 0.5, by Newton's method.

    ``ss.t_ppf`` bisects 200 times (several milliseconds a call), which would dominate a
    filter that needs a quantile for every period. Above zero the t CDF is concave and below
    the normal CDF, so from the normal quantile the iterates rise monotonically to the root.
    """
    x = ss.norm_ppf(probability)
    if math.isinf(x):  # a probability that rounds to 1
        return x
    constant = math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2) - math.log(math.pi * dof) / 2
    for _ in range(100):
        density = math.exp(constant - (dof + 1) / 2 * math.log1p(x * x / dof))
        if not density:
            raise CalculationError(
                f"the t quantile at probability {probability} with {dof} degrees of freedom "
                "overflows a float"
            )
        step = (probability - ss.t_cdf(x, df=dof)) / density
        x += step
        if step <= 1e-12 * x:
            return x
    raise CalculationError(f"the t quantile at probability {probability} did not converge")
