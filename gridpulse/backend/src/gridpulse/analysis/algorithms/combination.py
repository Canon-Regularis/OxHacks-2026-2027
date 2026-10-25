"""Forecast combination backed by Orchid (P1 #24)."""

import math
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Literal

from gridpulse.analysis.algorithms._checks import (
    finite_values,
    orchid_errors,
    require_levels,
    require_real,
    require_weights,
)
from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    InsufficientDataError,
    InvalidInputError,
)
from gridpulse.analysis.algorithms.forecasting import (
    _ZERO_WIDTH,
    DEFAULT_LEVELS,
    Forecast,
    Forecaster,
    PredictionInterval,
    _finite,
    _prepare,
)
from gridpulse.analysis.algorithms.orchid import forecasting_baseline as fb
from gridpulse.analysis.algorithms.series import HalfHourlySeries

CombinationMethod = Literal["mean", "median", "weighted"]

_Combiner = Callable[[list[list[float]]], list[float]]


def combine_forecasts(
    forecasts: Sequence[Forecast],
    *,
    method: CombinationMethod = "mean",
    weights: Sequence[float] | None = None,
) -> Forecast:
    """One forecast from two or more forecasts with the same origin, horizon and levels.

    The rule combines the means and, separately, each bound of each interval: quantile
    averaging (Vincentization), under which ordered, nested member bands stay ordered and
    nested. Their coverage is not recalibrated, so check it with a backtest. ``weights``, one
    positive weight per forecast, are required for, and only allowed with, "weighted". Each
    rule commutes with ``x -> actual - x``, so the rule applied to the members' common
    residual tail gives the combined fit's residuals when the members were fitted to the same
    series. Member warnings are kept, each once, but the zero-width warning is given only when
    the combined intervals have zero width.
    """
    if len(forecasts) < 2:
        raise InsufficientDataError(2, len(forecasts), what="forecasts")
    checked = _check_method(method, weights, len(forecasts))
    combine, used = _combiner(method, checked)
    first = forecasts[0]
    levels = require_levels([band.level for band in first.intervals])
    tail = min(len(forecast.residuals) for forecast in forecasts)
    arrays = [_arrays(i, forecast, first, levels, tail) for i, forecast in enumerate(forecasts)]
    with orchid_errors(f"combine_{method}"):
        combined = [combine(list(vectors)) for vectors in zip(*arrays, strict=True)]
    mean, *bounds, residuals = map(_finite, combined)
    intervals = tuple(
        PredictionInterval(level, lower, upper)
        for level, lower, upper in zip(levels, bounds[::2], bounds[1::2], strict=True)
    )
    # forecasting's zero-width test, relative to the combined values as there is no history.
    magnitude = max((abs(value) for array in (mean, *bounds) for value in array), default=0.0)
    zero_width = all(
        up - low <= 1e-12 * magnitude
        for interval in intervals
        for low, up in zip(interval.lower, interval.upper, strict=True)
    )
    notes = [note for forecast in forecasts for note in forecast.warnings if note != _ZERO_WIDTH]
    return Forecast(
        source=f"forecasting_baseline.combine_{method}",
        params={
            "method": method,
            "weights": used,
            "members": tuple(forecast.source for forecast in forecasts),
        },
        origin=first.origin,
        mean=mean,
        intervals=intervals,
        interval_method="quantile-averaging",
        residuals=residuals,
        warnings=tuple(dict.fromkeys([_ZERO_WIDTH, *notes] if zero_width else notes)),
    )


@dataclass(frozen=True)
class Ensemble:
    """A combination of its members' forecasts by ``combine_forecasts``.

    Members are refitted to every history, so each backtest fold combines forecasts from its
    own training window alone. Fixed ``weights`` must come from data before every period the
    ensemble is evaluated on; see ``inverse_error_weights``.
    """

    members: Sequence[Forecaster]
    _: KW_ONLY
    method: CombinationMethod = "mean"
    weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        members = tuple(self.members)
        if len(members) < 2:
            raise InvalidInputError(f"an ensemble needs two or more members, got {len(members)}")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "weights", _check_method(self.method, self.weights, len(members)))

    @property
    def min_history(self) -> int:
        return max(member.min_history for member in self.members)

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        # Checked once up front, so a history too short for any member fits none of them.
        horizon, levels, _ = _prepare(history, horizon, levels, minimum=self.min_history)
        forecasts = [member.forecast(history, horizon, levels=levels) for member in self.members]
        return combine_forecasts(forecasts, method=self.method, weights=self.weights)


def inverse_error_weights(errors: Sequence[float], *, power: float = 1.0) -> tuple[float, ...]:
    """Weights proportional to ``error ** -power`` that sum to one, for method "weighted".

    Give one positive error per member, such as its backtest MAE, in member order; only their
    ratios matter. Weights derived from the backtest that then evaluates the ensemble have
    seen its outcomes, so that evaluation flatters the ensemble: derive them from earlier data.
    """
    checked = [require_real(f"errors[{i}]", error) for i, error in enumerate(errors)]
    if len(checked) < 2 or not all(0 < error < math.inf for error in checked):
        raise InvalidInputError(
            f"errors must be two or more positive, finite numbers, got {list(errors)}"
        )
    exponent = require_real("power", power)
    if not 0 < exponent < math.inf:
        raise InvalidInputError(f"power must be positive and finite, got {power!r}")
    # Relative to the smallest error every error is at least 1, so Orchid never applies its
    # 1e-12 floor (it gives smaller errors a weight of 1e18) and its powers cannot underflow.
    smallest = min(checked)
    relative = [error / smallest for error in checked]
    if not math.isfinite(max(relative)):
        raise CalculationError("the ratio of the largest to the smallest error overflows a float")
    with orchid_errors("inverse_error_weights"):
        weights = fb.inverse_error_weights(relative, power=exponent)
    return _finite(weights)


def _check_method(
    method: object, weights: Sequence[float] | None, count: int
) -> tuple[float, ...] | None:
    """The validated weights: one per member, given exactly when ``method`` is "weighted"."""
    if method not in ("mean", "median", "weighted"):
        raise InvalidInputError(f"method must be 'mean', 'median' or 'weighted', got {method!r}")
    if (method == "weighted") != (weights is not None):
        raise InvalidInputError("weights must be given exactly when method is 'weighted'")
    if weights is None:
        return None
    checked = require_weights(weights)
    if len(checked) != count:
        raise InvalidInputError(f"needs one weight per member, got {len(checked)} for {count}")
    return checked


def _combiner(
    method: CombinationMethod, weights: tuple[float, ...] | None
) -> tuple[_Combiner, tuple[float, ...] | None]:
    """Orchid's combining function for ``method`` and the normalised weights it applies."""
    if weights is None:
        return (fb.combine_median if method == "median" else fb.combine_mean), None
    # Orchid rejects weights that sum to under 1e-12, so pass them relative to the largest.
    largest = max(weights)
    relative = [weight / largest for weight in weights]
    total = math.fsum(relative)  # as Orchid normalises them

    def combine(vectors: list[list[float]]) -> list[float]:
        return fb.combine_weighted(vectors, relative)

    return combine, tuple(weight / total for weight in relative)


def _arrays(
    index: int, forecast: Forecast, first: Forecast, levels: tuple[float, ...], tail: int
) -> list[list[float]]:
    """The mean, both bounds at each level and the last ``tail`` residuals of the member at
    ``index``, which must share the first member's origin, horizon and levels."""
    name = f"forecasts[{index}]"
    if forecast.origin != first.origin:
        raise InvalidInputError(
            f"{name} starts at {forecast.origin.isoformat()}, not {first.origin.isoformat()}"
        )
    own_levels = require_levels([band.level for band in forecast.intervals])
    if own_levels != levels:
        raise InvalidInputError(f"{name} has levels {list(own_levels)}, not {list(levels)}")
    bands = sorted(forecast.intervals, key=lambda band: band.level)
    arrays = [forecast.mean, *(bound for band in bands for bound in (band.lower, band.upper))]
    horizon = len(first.mean)
    if any(len(array) != horizon for array in arrays):
        raise InvalidInputError(f"{name} must have {horizon} values in its mean and every bound")
    residuals = forecast.residuals[len(forecast.residuals) - tail :]  # [-0:] would keep them all
    return [finite_values(name, array) for array in (*arrays, residuals)]
