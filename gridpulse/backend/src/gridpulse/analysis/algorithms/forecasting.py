"""Univariate forecasters backed by Orchid (P0 #2, #3, #6, #9, #10, #11)."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from datetime import datetime
from typing import Literal, Protocol

from gridpulse.analysis.algorithms._checks import (
    orchid_errors,
    require_int,
    require_levels,
    require_weights,
)
from gridpulse.analysis.algorithms.calendar import (
    DAY_SLOTS,
    WEEK_SLOTS,
    FeatureMatrix,
    fourier_terms,
)
from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
)
from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.orchid import forecasting_baseline as fb
from gridpulse.analysis.algorithms.series import HALF_HOUR, HalfHourlySeries, crosses_clock_change

DEFAULT_LEVELS = (0.8, 0.95)


@dataclass(frozen=True)
class PredictionInterval:
    level: float
    lower: tuple[float, ...]
    upper: tuple[float, ...]


@dataclass(frozen=True)
class Forecast:
    """Point forecasts and central prediction intervals from ``origin`` onwards.

    ``residuals`` are in-sample residuals (actual minus fitted) after the model's warm-up,
    aligned to the end of the history. Their parameters were estimated on the whole history, so use them
    for diagnostics only, never as causal features.
    """

    source: str
    params: Mapping[str, object]
    origin: datetime
    mean: tuple[float, ...]
    intervals: tuple[PredictionInterval, ...]
    interval_method: str
    residuals: tuple[float, ...]
    warnings: tuple[str, ...] = ()

    def timestamps(self) -> tuple[datetime, ...]:
        return tuple(self.origin + i * HALF_HOUR for i in range(len(self.mean)))


class Forecaster(Protocol):
    @property
    def min_history(self) -> int:
        """The shortest complete history ``forecast`` accepts."""
        ...

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast: ...


_ZERO_WIDTH = "zero-width intervals: residual variance is zero"
_CLOCK_CHANGE = "seasonal model spans a GB clock change: seasonal lags count UTC half-hours"
_NOT_ADJUSTED = "seasonal adjustment not applied: no significant seasonality detected"
_MAX_REGRESSORS = 64  # least squares costs O(n * columns^2) in pure Python

_Output = tuple[tuple[float, ...], tuple[PredictionInterval, ...], tuple[float, ...]]


@dataclass(frozen=True)
class Naive:
    """Random walk: every step repeats the last value."""

    @property
    def min_history(self) -> int:
        return 2

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        mean, intervals, residuals = _fit(fb.Naive(), values, horizon, levels)
        intervals, residuals = _post_warm_up(mean, intervals, residuals, 1)
        return _forecast(
            history,
            source="forecasting_baseline.Naive",
            params={},
            mean=mean,
            intervals=intervals,
            interval_method="random-walk",
            residuals=residuals,
        )


@dataclass(frozen=True)
class SeasonalNaive:
    """Seasonal random walk: every step repeats the value one season earlier."""

    season_length: int

    def __post_init__(self) -> None:
        require_int("season_length", self.season_length, minimum=2)

    @property
    def min_history(self) -> int:
        return self.season_length + 1

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        m = self.season_length
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        mean, intervals, residuals = _fit(fb.SeasonalNaive(m), values, horizon, levels)
        intervals, residuals = _post_warm_up(mean, intervals, residuals, m)
        return _forecast(
            history,
            source="forecasting_baseline.SeasonalNaive",
            params={"season_length": m},
            mean=mean,
            intervals=intervals,
            interval_method="seasonal-random-walk",
            residuals=residuals,
            seasonal=True,
        )


@dataclass(frozen=True)
class Drift:
    """Random walk with drift along the line through the first and last values."""

    @property
    def min_history(self) -> int:
        return 3

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        mean, intervals, residuals = _fit(fb.Drift(), values, horizon, levels)
        intervals, residuals = _post_warm_up(mean, intervals, residuals, 1)
        return _forecast(
            history,
            source="forecasting_baseline.Drift",
            params={},
            mean=mean,
            intervals=intervals,
            interval_method="sqrt-horizon",
            residuals=residuals,
        )


@dataclass(frozen=True)
class Mean:
    """Flat forecast at the historical mean."""

    @property
    def min_history(self) -> int:
        return 2

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        mean, intervals, residuals = _fit(fb.MeanForecast(), values, horizon, levels)
        return _forecast(
            history,
            source="forecasting_baseline.MeanForecast",
            params={},
            mean=mean,
            intervals=intervals,
            interval_method="analytic",
            residuals=residuals,
        )


@dataclass(frozen=True)
class Naive2:
    """Naive forecast of the additively seasonally adjusted history (M4's Naive2).

    Seasonal adjustment runs only when the lag-``season_length`` autocorrelation is
    significant; the history must span more than two seasons so that it can be tested.
    """

    season_length: int

    def __post_init__(self) -> None:
        require_int("season_length", self.season_length, minimum=2)

    @property
    def min_history(self) -> int:
        return 2 * self.season_length + 1

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        model = fb.Naive2(self.season_length, model="additive")
        mean, intervals, residuals = _fit(model, values, horizon, levels)
        intervals, residuals = _post_warm_up(mean, intervals, residuals, 1)
        adjusted = model.seasonal_ is not None
        return _forecast(
            history,
            source="forecasting_baseline.Naive2",
            params={"season_length": self.season_length, "seasonal_adjustment": adjusted},
            mean=mean,
            intervals=intervals,
            interval_method="sqrt-horizon",
            residuals=residuals,
            seasonal=adjusted,
            notes=() if adjusted else (_NOT_ADJUSTED,),
        )


@dataclass(frozen=True)
class MovingAverage:
    """Flat forecast at the mean of the last ``window`` values."""

    window: int

    def __post_init__(self) -> None:
        require_int("window", self.window, minimum=1)

    @property
    def min_history(self) -> int:
        return self.window + 1

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        window = self.window
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        model = fb.MovingAverageForecaster(window)
        mean, intervals, residuals = _fit(model, values, horizon, levels)
        # Orchid fits its first `window` values from a placeholder and shorter windows.
        intervals, residuals = _post_warm_up(mean, intervals, residuals, window)
        return _forecast(
            history,
            source="forecasting_baseline.MovingAverageForecaster",
            params={"window": window},
            mean=mean,
            intervals=intervals,
            interval_method="sqrt-horizon",
            residuals=residuals,
        )


@dataclass(frozen=True)
class WeightedMovingAverage:
    """Flat forecast at a weighted mean of the last ``len(weights)`` values.

    Weights are positive, ordered oldest to newest and normalised to sum to one.
    """

    weights: Sequence[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", require_weights(self.weights))

    @property
    def min_history(self) -> int:
        return len(self.weights) + 1

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        model = fb.WeightedMovingAverage(self.weights)
        mean, intervals, residuals = _fit(model, values, horizon, levels)
        # Orchid fits its first k values with the placeholder y[0].
        intervals, residuals = _post_warm_up(mean, intervals, residuals, len(self.weights))
        return _forecast(
            history,
            source="forecasting_baseline.WeightedMovingAverage",
            params={"weights": tuple(model.weights)},
            mean=mean,
            intervals=intervals,
            interval_method="sqrt-horizon",
            residuals=residuals,
        )


@dataclass(frozen=True, kw_only=True)
class ETS:
    """Exponential smoothing with additive errors; multiplicative seasons need positive data."""

    trend: Literal["additive"] | None = None
    damped: bool = False
    seasonal: Literal["additive", "multiplicative"] | None = None
    season_length: int | None = None

    def __post_init__(self) -> None:
        if self.trend not in (None, "additive"):
            raise InvalidInputError(f"trend must be None or 'additive', got {self.trend!r}")
        if not isinstance(self.damped, bool):
            raise InvalidInputError(f"damped must be a bool, got {self.damped!r}")
        if self.damped and self.trend is None:
            raise InvalidInputError("damped requires a trend")
        if self.seasonal not in (None, "additive", "multiplicative"):
            raise InvalidInputError(
                f"seasonal must be None, 'additive' or 'multiplicative', got {self.seasonal!r}"
            )
        if (self.seasonal is None) != (self.season_length is None):
            raise InvalidInputError("seasonal and season_length must be given together")
        _check_season_length(self.season_length)

    @property
    def min_history(self) -> int:
        if self.season_length is not None:
            return 2 * self.season_length
        return 2 + (self.trend is not None) + self.damped

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        m = self.season_length
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        multiplicative = self.seasonal == "multiplicative"
        if multiplicative and min(values) <= 0:
            raise InvalidInputError("multiplicative seasonality needs a strictly positive history")
        model = fb.ETSForecaster(
            trend="add" if self.trend else None,
            damped=self.damped,
            seasonal="mul" if multiplicative else ("add" if self.seasonal else None),
            seasonal_periods=m,
            n_restarts=6,
            seed=0,
        )
        mean, intervals, residuals = _fit(model, values, horizon, levels)
        return _forecast(
            history,
            source="forecasting_baseline.ETSForecaster",
            params=_ets_params(model, m),
            mean=mean,
            intervals=intervals,
            interval_method="sqrt-horizon" if multiplicative else "analytic",
            residuals=residuals,
            seasonal=m is not None,
        )


@dataclass(frozen=True, kw_only=True)
class AutoETS:
    """Additive ETS chosen by AICc over trend, damping and (with ``season_length``) season."""

    season_length: int | None = None

    def __post_init__(self) -> None:
        _check_season_length(self.season_length)

    @property
    def min_history(self) -> int:
        return 4 if self.season_length is None else 2 * self.season_length

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        m = self.season_length
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        with orchid_errors("auto_ets"):
            model = fb.auto_ets(
                values,
                seasonal_periods=m,
                allow_multiplicative=False,
                damped_candidates=(False, True),
                criterion="aicc",
                seed=0,
            )
        mean, intervals, residuals = _predict(model, horizon, levels)
        return _forecast(
            history,
            source="forecasting_baseline.auto_ets",
            params=_ets_params(model, m),
            mean=mean,
            intervals=intervals,
            interval_method="analytic",
            residuals=residuals,
            seasonal=model.seasonal is not None,
        )


@dataclass(frozen=True)
class ARIMA:
    """ARIMA(p, d, q) with optional seasonal differencing, fitted by conditional least squares.

    The model has a constant only when it is undifferenced; there are no seasonal AR or MA
    terms.
    """

    p: int
    d: int
    q: int
    _: KW_ONLY
    seasonal_differencing: int = 0
    season_length: int | None = None

    def __post_init__(self) -> None:
        require_int("p", self.p, minimum=0, maximum=3)
        require_int("d", self.d, minimum=0, maximum=1)
        require_int("q", self.q, minimum=0, maximum=2)
        _check_seasonal_differencing(self.seasonal_differencing, self.season_length)

    @property
    def min_history(self) -> int:
        seasonal_span = self.seasonal_differencing * (self.season_length or 0)
        return self.d + seasonal_span + _arima_observations(self.p, self.q)

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        seasonal_d = self.seasonal_differencing
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        with orchid_errors("diff"):
            differenced = ar.diff(values, d=self.d, D=seasonal_d, m=self.season_length or 1)
        _require_variation(differenced, "differenced history" if self.d + seasonal_d else "history")
        scale, scaled = _scaled_by_range(values, differenced)
        order = (self.p, self.d, self.q)
        seasonal_order = (0, seasonal_d, 0, self.season_length or 0)
        trend = "c" if self.d + seasonal_d == 0 else "n"
        with orchid_errors("ARIMA"):
            result = ar.ARIMA(
                scaled,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                boxcox_lambda=None,
            ).fit(method="css", enforce_stationarity=True, enforce_invertibility=True, seed=0)
        mean, intervals = _arima_forecast(result, horizon, levels, scale)
        return _forecast(
            history,
            source="autoregressive.ARIMA",
            params={
                "order": order,
                "seasonal_order": seasonal_order,
                "trend": trend,
                **_arima_fit(result, scale),
            },
            mean=mean,
            intervals=intervals,
            interval_method="analytic",
            residuals=_rescaled(result.resid, scale),
            seasonal=seasonal_d == 1,
        )


@dataclass(frozen=True, kw_only=True)
class AutoARIMA:
    """Stepwise AICc search over ARIMA(p, d, q) after testing for seasonal and regular differences.

    ``seasonal_differencing=None`` tests for a seasonal difference when ``season_length``
    is given; the regular difference is then tested on the seasonally differenced history.
    Orchid's seasonal test (STL seasonal strength of at least 0.64) also fires on integrated
    series without a seasonal cycle, so give ``seasonal_differencing`` when it is known.
    """

    season_length: int | None = None
    seasonal_differencing: int | None = None
    max_p: int = 3
    max_q: int = 2

    def __post_init__(self) -> None:
        if self.seasonal_differencing is not None:
            _check_seasonal_differencing(self.seasonal_differencing, self.season_length)
        elif self.season_length is not None:
            # At season length 2 Orchid's seasonal test fires for every series.
            require_int("season_length", self.season_length, minimum=3)
        require_int("max_p", self.max_p, minimum=0, maximum=3)
        require_int("max_q", self.max_q, minimum=0, maximum=2)

    @property
    def min_history(self) -> int:
        """Long enough for the worst case the tests can choose (d = 1 and, with a season,
        D = 1) and, when D is tested, six seasons: over fewer, Orchid's seasonal test often
        picks D = 1 even for white noise."""
        observations = _arima_observations(self.max_p, self.max_q)
        m = self.season_length
        if m is None:
            return 1 + observations
        minimum = 1 + m + observations
        return minimum if self.seasonal_differencing == 1 else max(minimum, 6 * m)

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        m = self.season_length
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        _require_variation(values, "history")
        _, unit = _scaled_by_range(values, values)  # Orchid's tests also have absolute floors
        seasonal_d = self.seasonal_differencing
        with orchid_errors("difference tests"):
            if seasonal_d is None:
                seasonal_d = ar.nsdiffs(unit, m=m, max_D=1) if m is not None else 0
            period = (m or 1) if seasonal_d else 1
            d = ar.ndiffs(ar.diff(unit, d=0, D=seasonal_d, m=period), max_d=1)
            differenced = ar.diff(values, d=d, D=seasonal_d, m=period)
        if d + seasonal_d:
            _require_variation(differenced, "differenced history")
        scale, scaled = _scaled_by_range(values, differenced)
        trend = "c" if d + seasonal_d == 0 else "n"
        with orchid_errors("auto_arima"):
            selection = ar.auto_arima(
                scaled,
                seasonal=seasonal_d == 1,
                m=period,
                d=d,
                D=seasonal_d,
                max_p=self.max_p,
                max_q=self.max_q,
                max_P=0,
                max_Q=0,
                information_criterion="aicc",
                stepwise=True,
                trend=trend,
                method="css",
                seed=0,
            )
        result = selection.results
        mean, intervals = _arima_forecast(result, horizon, levels, scale)
        return _forecast(
            history,
            source="autoregressive.auto_arima",
            params={
                "season_length": m,
                "seasonal_differencing": self.seasonal_differencing,  # None: chosen by a test
                "max_p": self.max_p,
                "max_q": self.max_q,
                "order": result.model.order,
                "seasonal_order": result.model.seasonal_order if seasonal_d else (0, 0, 0, 0),
                "trend": trend,
                **_arima_fit(result, scale),
                "n_models": selection.n_models,
            },
            mean=mean,
            intervals=intervals,
            interval_method="analytic",
            residuals=_rescaled(result.resid, scale),
            seasonal=seasonal_d == 1,
        )


@dataclass(frozen=True, kw_only=True)
class TrendRegression:
    """Least-squares polynomial trend in the period index."""

    degree: int = 1

    def __post_init__(self) -> None:
        require_int("degree", self.degree, minimum=0, maximum=2)

    @property
    def min_history(self) -> int:
        return self.degree + 2

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        mean, intervals, residuals = _fit(fb.TrendForecaster(self.degree), values, horizon, levels)
        return _forecast(
            history,
            source="forecasting_baseline.TrendForecaster",
            params={"degree": self.degree},
            mean=mean,
            intervals=intervals,
            interval_method="sqrt-horizon",
            residuals=residuals,
        )


@dataclass(frozen=True)
class SeasonalDummyRegression:
    """Polynomial trend plus m - 1 seasonal dummies (phase m - 1, counted from the first value,
    is the reference); ``season_length`` is at most 48."""

    season_length: int
    _: KW_ONLY
    degree: int = 1

    def __post_init__(self) -> None:
        require_int("season_length", self.season_length, minimum=2, maximum=DAY_SLOTS)
        require_int("degree", self.degree, minimum=0, maximum=2)

    @property
    def min_history(self) -> int:
        return self.season_length + self.degree + 1

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        m, degree = self.season_length, self.degree
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        model = fb.SeasonalDummyForecaster(m, degree)
        mean, intervals, residuals = _fit(model, values, horizon, levels)
        return _forecast(
            history,
            source="forecasting_baseline.SeasonalDummyForecaster",
            params={"season_length": m, "degree": degree},
            mean=mean,
            intervals=intervals,
            interval_method="sqrt-horizon",
            residuals=residuals,
            seasonal=True,
        )


@dataclass(frozen=True)
class FourierRegression:
    """Polynomial trend plus ``harmonics`` sine-cosine pairs of period ``season_length``."""

    season_length: int
    harmonics: int
    _: KW_ONLY
    degree: int = 1

    def __post_init__(self) -> None:
        require_int("season_length", self.season_length, minimum=2)
        require_int("harmonics", self.harmonics, minimum=1, maximum=self.season_length // 2)
        require_int("degree", self.degree, minimum=0, maximum=2)
        nyquist = self.season_length % 2 == 0 and self.harmonics == self.season_length // 2
        width = self.degree + 1 + 2 * self.harmonics - nyquist
        if width > _MAX_REGRESSORS:
            raise InvalidInputError(
                f"the design has {width} columns; at most {_MAX_REGRESSORS} are allowed"
            )

    @property
    def min_history(self) -> int:
        """At least one full season, and the 2 * harmonics + degree + 2 values Orchid requires."""
        return max(self.season_length, 2 * self.harmonics + self.degree + 2)

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        m, harmonics, degree = self.season_length, self.harmonics, self.degree
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        model = fb.FourierForecaster(m, harmonics, degree)
        mean, intervals, residuals = _fit(model, values, horizon, levels)
        return _forecast(
            history,
            source="forecasting_baseline.FourierForecaster",
            params={"season_length": m, "harmonics": harmonics, "degree": degree},
            mean=mean,
            intervals=intervals,
            interval_method="sqrt-horizon",
            residuals=residuals,
            seasonal=True,
        )


@dataclass(frozen=True)
class CalendarRegression:
    """Polynomial trend plus Fourier terms of the GB local clock, exact across clock changes.

    The design is limited to 64 columns.
    """

    daily_harmonics: int
    _: KW_ONLY
    weekly_harmonics: int = 0
    degree: int = 1

    def __post_init__(self) -> None:
        require_int("degree", self.degree, minimum=0, maximum=2)
        if self._width() > _MAX_REGRESSORS:  # fourier_terms validates the harmonics
            raise InvalidInputError(
                f"the design has {self._width()} columns; at most {_MAX_REGRESSORS} are allowed"
            )

    @property
    def min_history(self) -> int:
        """At least one full week (or day) for weekly (or daily) terms, and one more value
        than there are coefficients. With 23 or more daily harmonics and no weekly terms, a
        history spanning a clock change can still have too few distinct clock slots."""
        cycle = WEEK_SLOTS if self.weekly_harmonics else DAY_SLOTS
        return max(cycle, self._width() + 1)

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        horizon, levels, values = _prepare(history, horizon, levels, minimum=self.min_history)
        # Each distinct clock slot gives one distinct calendar row. A clock change leaves a
        # day of periods 46 distinct slots, and Orchid's solver silently zeroes coefficients
        # that too few slots cannot identify.
        calendar = self._calendar(history.timestamps())
        slots = len(set(calendar.rows))
        if slots <= len(calendar.columns):
            raise InsufficientDataError(
                len(calendar.columns) + 1, slots, what="distinct local clock slots"
            )

        def features(rows: int, start: int) -> list[list[float]]:
            trend = fb.trend_features(rows, self.degree, start)
            calendar = self._calendar([history.timestamp(t) for t in range(start, start + rows)])
            return [row + list(terms) for row, terms in zip(trend, calendar.rows, strict=True)]

        model = fb.LinearRegressionForecaster(features)
        mean, intervals, residuals = _fit(model, values, horizon, levels)
        return _forecast(
            history,
            source="forecasting_baseline.LinearRegressionForecaster",
            params={
                "daily_harmonics": self.daily_harmonics,
                "weekly_harmonics": self.weekly_harmonics,
                "degree": self.degree,
            },
            mean=mean,
            intervals=intervals,
            interval_method="sqrt-horizon",
            residuals=residuals,
        )

    def _width(self) -> int:
        return self.degree + 1 + len(self._calendar(()).columns)

    def _calendar(self, timestamps: Sequence[datetime]) -> FeatureMatrix:
        return fourier_terms(
            timestamps,
            daily_harmonics=self.daily_harmonics,
            weekly_harmonics=self.weekly_harmonics,
        )


def _prepare(
    history: HalfHourlySeries, horizon: int, levels: Sequence[float], *, minimum: int
) -> tuple[int, tuple[float, ...], list[float]]:
    """The validated horizon, levels and history values, checked before Orchid runs."""
    checked_horizon = require_int("horizon", horizon, minimum=1)
    checked_levels = require_levels(levels)
    values = history.complete_values()
    if len(values) < minimum:
        raise InsufficientDataError(minimum, len(values))
    return checked_horizon, checked_levels, values


def _fit(
    model: fb.BaseForecaster, values: list[float], horizon: int, levels: Sequence[float]
) -> _Output:
    with orchid_errors(type(model).__name__):
        model.fit(values)
    return _predict(model, horizon, levels)


def _predict(model: fb.BaseForecaster, horizon: int, levels: Sequence[float]) -> _Output:
    """Point forecasts, one interval per level and every residual of a fitted model."""
    with orchid_errors(type(model).__name__):
        results = [model.forecast(horizon, level) for level in levels]
    intervals = tuple(
        PredictionInterval(level, _finite(result.lower), _finite(result.upper))
        for level, result in zip(levels, results, strict=True)
    )
    return _finite(results[0].mean), intervals, _finite(results[0].residuals)


def _scaled_by_range(
    values: Sequence[float], modelled: Sequence[float]
) -> tuple[float, list[float]]:
    """The range of ``modelled`` and ``values`` divided by it.

    ARIMA fits run on data scaled so that the series Orchid models (the differenced history)
    has unit range. Its absolute 1e-12 floors on variances then depend on neither the units
    nor a trend that differencing removes. The range must be non-zero.
    """
    scale = max(modelled) - min(modelled)
    if not math.isfinite(scale * scale):
        raise CalculationError("the history's range overflows a float")
    return scale, [value / scale for value in values]


def _arima_forecast(
    result: ar.ARIMAResults, horizon: int, levels: Sequence[float], scale: float
) -> tuple[tuple[float, ...], tuple[PredictionInterval, ...]]:
    """Analytic forecasts, in the data's units, from a fit on data divided by ``scale`` that
    passes a health check (Orchid never flags one)."""
    # Orchid floors sigma2 at 1e-12, so a unit-range fit at the floor has degenerated.
    healthy = (
        all(map(math.isfinite, (*result.params, result.aicc))) and 1e-12 < result.sigma2 < math.inf
    )
    if not healthy:
        raise CalculationError(
            "ARIMA fit has non-finite parameters or AICc, or sigma2 at Orchid's 1e-12 floor"
        )
    with orchid_errors("ARIMA forecast"):
        output = result.forecast(horizon, level=tuple(levels), method="analytic")
    intervals = tuple(
        PredictionInterval(
            level, _rescaled(output.lower[level], scale), _rescaled(output.upper[level], scale)
        )
        for level in levels
    )
    return _rescaled(output.mean, scale), intervals


def _arima_fit(result: ar.ARIMAResults, scale: float) -> dict[str, object]:
    """The fitted values of a fit on data divided by ``scale``, in the data's units.

    The constant scales with the data and sigma2 with its square. The Gaussian
    log-likelihood falls by nobs * ln(scale), so AICc rises by twice that.
    """
    sigma2 = result.sigma2 * scale * scale
    if not math.isfinite(sigma2):
        raise CalculationError("ARIMA sigma2 overflows a float in the data's units")
    return {
        "coefficients": {
            name: value * scale if name == "const" else value
            for name, value in zip(result.param_names, result.params, strict=True)
        },
        "sigma2": sigma2,
        "aicc": result.aicc + 2 * result.nobs * math.log(scale),
    }


def _rescaled(values: Sequence[float] | None, scale: float) -> tuple[float, ...]:
    return _finite([value * scale for value in _finite(values)])


def _forecast(
    history: HalfHourlySeries,
    *,
    source: str,
    params: Mapping[str, object],
    mean: tuple[float, ...],
    intervals: tuple[PredictionInterval, ...],
    interval_method: str,
    residuals: tuple[float, ...],
    seasonal: bool = False,
    notes: tuple[str, ...] = (),
) -> Forecast:
    # Least-squares rounding leaves "zero" widths of about 1e-15 times the data's magnitude.
    magnitude = max(abs(value) for value in (*history.values, *mean) if value is not None)
    tolerance = 1e-12 * magnitude
    zero_width = all(
        up - low <= tolerance
        for interval in intervals
        for low, up in zip(interval.lower, interval.upper, strict=True)
    )
    warnings = [_ZERO_WIDTH] if zero_width else []
    stop = history.timestamp(len(history) + len(mean))
    if seasonal and crosses_clock_change(history.start, stop):
        warnings.append(_CLOCK_CHANGE)
    return Forecast(
        source=source,
        params=params,
        origin=history.end,
        mean=mean,
        intervals=intervals,
        interval_method=interval_method,
        residuals=residuals,
        warnings=(*warnings, *notes),
    )


def _post_warm_up(
    mean: tuple[float, ...],
    intervals: tuple[PredictionInterval, ...],
    residuals: tuple[float, ...],
    warm_up: int,
) -> tuple[tuple[PredictionInterval, ...], tuple[float, ...]]:
    """Drop ``warm_up`` placeholder residuals and rescale the intervals to match.

    Orchid's base forecasters take sigma^2 as the mean square of all n residuals, placeholders
    included, so each half-width scales by sqrt(mean square after the warm-up / mean square).
    """
    kept = residuals[warm_up:]
    total = _mean_square(residuals)
    factor = math.sqrt(_mean_square(kept) / total) if total else 1.0
    return _rescale(mean, intervals, factor), kept


def _rescale(
    mean: tuple[float, ...], intervals: tuple[PredictionInterval, ...], factor: float
) -> tuple[PredictionInterval, ...]:
    """The intervals with every half-width multiplied by ``factor``."""
    return tuple(
        PredictionInterval(
            interval.level,
            _finite([c - factor * (c - low) for c, low in zip(mean, interval.lower, strict=True)]),
            _finite([c + factor * (up - c) for c, up in zip(mean, interval.upper, strict=True)]),
        )
        for interval in intervals
    )


def _finite(values: Sequence[float] | None) -> tuple[float, ...]:
    """Orchid output as a tuple; ``CalculationError`` if it is missing or non-finite."""
    if values is None or not all(map(math.isfinite, values)):
        raise CalculationError("Orchid returned missing or non-finite output")
    return tuple(values)


def _mean_square(values: Sequence[float]) -> float:
    return math.fsum(value * value for value in values) / len(values)


def _ets_params(model: fb.ETSForecaster, season_length: int | None) -> dict[str, object]:
    """The method label and smoothing parameters of the components the model has.

    ``state_`` holds placeholder values for absent components, so those are left out.
    """
    state = model.state_
    params: dict[str, object] = {
        "season_length": season_length,
        "method": model.method,
        "alpha": state.alpha,
    }
    if model.trend is not None:
        params["beta"] = state.beta
    if model.seasonal is not None:
        params["gamma"] = state.gamma
    if model.damped:
        params["phi"] = state.phi
    return params


def _check_season_length(season_length: int | None) -> None:
    if season_length is not None:
        require_int("season_length", season_length, minimum=2)


def _check_seasonal_differencing(seasonal_differencing: int, season_length: int | None) -> None:
    require_int("seasonal_differencing", seasonal_differencing, minimum=0, maximum=1)
    _check_season_length(season_length)
    if (season_length is None) == (seasonal_differencing == 1):
        raise InvalidInputError(
            "season_length must be given exactly when seasonal_differencing is 1"
        )


def _arima_observations(p: int, q: int) -> int:
    """Observations an ARIMA fit needs after differencing."""
    return max(100, 10 * (p + q + 1))


def _require_variation(values: Sequence[float], name: str) -> None:
    low, high = min(values), max(values)
    if high - low <= 1e-9 * max(abs(low), abs(high)):
        raise DegenerateDataError(f"{name} is constant to within 1e-9 of its magnitude")
