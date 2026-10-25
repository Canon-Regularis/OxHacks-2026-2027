import copy
import math
import pickle
import random
import statistics
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from types import SimpleNamespace

import pytest

from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
    MissingDataError,
)
from gridpulse.analysis.algorithms.forecasting import Forecaster
from gridpulse.analysis.algorithms.orchid import state_space as ss
from gridpulse.analysis.algorithms.series import HALF_HOUR, HalfHourlySeries
from gridpulse.analysis.algorithms.state_space import (
    DiscountForecaster,
    StructuralFit,
    StructuralForecaster,
    StructuralModel,
    StructuralParameters,
    _t_quantile,
    discount_filter,
    filter_structural,
    fit_structural,
    smooth_structural,
)

START = datetime(2026, 9, 1, tzinfo=UTC)
CLOCK_CHANGE_DAY = datetime(2026, 10, 25, tzinfo=UTC)  # clocks go back at 01:00 UTC
CLOCK_CHANGE = "seasonal model spans a GB clock change: seasonal lags count UTC half-hours"
MakeSeries = Callable[..., HalfHourlySeries]

LEVEL = StructuralModel()
TREND = StructuralModel(trend="linear")
DAMPED = StructuralModel(trend="damped")
DAILY = StructuralModel(seasons=((12, 1),))
WIND = StructuralModel(regressors=("wind",))
DYNAMIC = StructuralModel(regressors=("wind",), dynamic_regression=True)
LEVEL_PARAMETERS = StructuralParameters(LEVEL, {"level": 0.01, "irregular": 1.0})
TREND_PARAMETERS = StructuralParameters(TREND, {"level": 0.01, "slope": 1e-4, "irregular": 1.0})
DAILY_PARAMETERS = StructuralParameters(
    DAILY, {"level": 0.01, "seasonal_12": 1e-4, "irregular": 1.0}
)
WIND_PARAMETERS = StructuralParameters(WIND, {"level": 0.01, "irregular": 1.0})


def _local_level(
    n: int, *, q: float = 0.01, h: float = 1.0, seed: int = 0, start: float = 50.0
) -> list[float]:
    """A random-walk level with variance ``q`` per period, observed with noise variance ``h``."""
    rng = random.Random(seed)
    level, values = start, []
    for _ in range(n):
        level += rng.gauss(0.0, math.sqrt(q))
        values.append(level + rng.gauss(0.0, math.sqrt(h)))
    return values


SERIES = HalfHourlySeries(START, _local_level(60, seed=1))
WINDY = HalfHourlySeries(START, [math.sin(t / 5) for t in range(60)])


def _moments(values: Sequence[float]) -> tuple[float, float]:
    """The mean and sample variance, computed as the adapter standardises."""
    mean = math.fsum(values) / len(values)
    return mean, math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)


def _vague(model: ss.LinearGaussianSSM) -> ss.LinearGaussianSSM:
    """The adapter's prior: variance kappa = 1e4 for every standardised state."""
    return model.copy_with(P1=ss.eye(model.k_states, ss._BIG), P1_infinity=None)


def _at(series: HalfHourlySeries, t: int) -> float:
    value = series.values[t]
    assert value is not None
    return value


def _assert_moved(
    moved: HalfHourlySeries, base: HalfHourlySeries, scale: float = 1.0, offset: float = 0.0
) -> None:
    """``moved`` is ``base`` times ``scale`` plus ``offset``, missing in the same places."""
    assert [v is None for v in moved.values] == [v is None for v in base.values]
    expected = [offset + scale * v for v in base.values if v is not None]
    actual = [v for v in moved.values if v is not None]
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9 * scale)


def test_model_normalises_its_layout() -> None:
    seasons, regressors = [[48, 2], (336, 1)], ["wind"]  # lists, normalised to tuples
    model = StructuralModel(
        trend="damped",
        seasons=seasons,  # type: ignore[arg-type]
        regressors=regressors,  # type: ignore[arg-type]
        dynamic_regression=True,
    )
    assert model == StructuralModel(
        trend="damped", seasons=((48, 2), (336, 1)), regressors=("wind",), dynamic_regression=True
    )
    assert model.state_dimension == 2 + 4 + 2 + 1
    assert model.variance_names == (
        "level",
        "slope",
        "seasonal_48",
        "seasonal_336",
        "regression",
        "irregular",
    )
    with pytest.raises(TypeError):
        LEVEL_PARAMETERS.variances["level"] = 2.0  # type: ignore[index]


def test_fits_and_forecasters_pickle_and_deepcopy() -> None:
    # Batch jobs return fits from worker processes, and a read-only mapping cannot be pickled.
    parameters = StructuralParameters(DAMPED, {"level": 0.01, "slope": 1e-4, "irregular": 1.0}, 0.9)
    fit = StructuralFit(parameters, log_likelihood=-10.0, aic=26.0, bic=30.0, nobs=20)
    for value in (fit, StructuralForecaster(DAMPED, parameters=parameters)):
        assert pickle.loads(pickle.dumps(value)) == value
        assert copy.deepcopy(value) == value


@pytest.fixture(scope="module")
def level_fit() -> tuple[list[float], StructuralFit]:
    values = _local_level(200, q=0.05, seed=1)
    return values, fit_structural(HalfHourlySeries(START, values), LEVEL)


def test_local_level_fit_is_orchids_mle_on_standardised_values(
    level_fit: tuple[list[float], StructuralFit],
) -> None:
    values, fit = level_fit
    mean, variance = _moments(values)
    sd = math.sqrt(variance)
    standardised = [(v - mean) / sd for v in values]
    rows = [[v] for v in standardised]
    components = ss.UnobservedComponents(standardised, [ss.LocalLevel(), ss.Irregular()])
    oracle = ss.fit_ssm_mle(
        lambda natural: _vague(components.build(natural)),
        [0.0, 0.0],  # the logs of Orchid's start values
        rows,
        transform=lambda free: [math.exp(max(-40.0, min(40.0, u))) for u in free],
        maxiter=2000,
        restarts=1,
        seed=0,
    )
    level, noise = (value * variance for value in oracle.params)
    assert dict(fit.parameters.variances) == pytest.approx(
        {"level": level, "irregular": noise}, rel=1e-12
    )
    # Orchid's exact-diffuse filter is reliable for a local level.
    exact = ss.kalman_filter(components.build(oracle.params), rows)
    assert fit.log_likelihood == pytest.approx(exact.loglike - 199 * math.log(sd), abs=1e-3)
    assert fit.nobs == 200
    assert fit.aic == pytest.approx(-2 * fit.log_likelihood + 4, rel=1e-12)
    assert fit.bic == pytest.approx(-2 * fit.log_likelihood + 2 * math.log(200), rel=1e-12)


@pytest.mark.parametrize(("scale", "offset"), [(1e-6, 0.0), (1.0, 3e4)], ids=["scaled", "shifted"])
def test_fit_scales_with_the_data(
    level_fit: tuple[list[float], StructuralFit], scale: float, offset: float
) -> None:
    values, fit = level_fit
    moved = fit_structural(HalfHourlySeries(START, [v * scale + offset for v in values]), LEVEL)
    expected = {name: v * scale * scale for name, v in fit.parameters.variances.items()}
    assert dict(moved.parameters.variances) == pytest.approx(expected, rel=1e-9)
    # Every observation after the diffuse phase gains -ln(scale).
    shift = -(fit.nobs - 1) * math.log(scale)
    assert moved.log_likelihood == pytest.approx(fit.log_likelihood + shift, abs=1e-9)


def test_fit_starts_at_the_first_observed_value(
    level_fit: tuple[list[float], StructuralFit],
) -> None:
    values, fit = level_fit
    padded = HalfHourlySeries(START - 3 * HALF_HOUR, [None, None, None, *values])
    assert fit_structural(padded, LEVEL) == fit


def test_damping_at_its_bound_stays_below_one() -> None:
    # This history's likelihood rises towards an undamped trend; Orchid's own logistic,
    # clamped at 40, would round the damping to exactly 1.
    rng = random.Random(0)
    level, slope, values = 50.0, 0.5, []
    for _ in range(40):
        slope = 0.8 * slope + rng.gauss(0.0, 0.05)
        level += slope + rng.gauss(0.0, 0.1)
        values.append(level + rng.gauss(0.0, 0.5))
    damping = fit_structural(HalfHourlySeries(START, values), DAMPED).parameters.damping
    assert damping is not None
    assert 1 - 1e-12 < damping < 1


@pytest.mark.parametrize(
    ("model", "minimum"),
    [(LEVEL, 4), (DAMPED, 7), (StructuralModel(seasons=((12, 1),)), 24), (WIND, 5)],
    ids=["level", "damped", "season", "regression"],
)
def test_fit_needs_enough_observed_values(model: StructuralModel, minimum: int) -> None:
    values: list[float | None] = [*_local_level(minimum - 1, seed=3), None]  # a gap
    regressors = {"wind": WINDY.window(0, minimum)} if model.regressors else None
    with pytest.raises(InsufficientDataError) as excinfo:
        fit_structural(HalfHourlySeries(START, values), model, regressors=regressors)
    assert (excinfo.value.required, excinfo.value.available) == (minimum, minimum - 1)


def test_fit_accepts_weekly_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    # At Orchid's start values a weekly harmonic drifts too fast ever to look identified.
    # Only the identification check is under test, so the search is stubbed.
    monkeypatch.setattr(
        ss,
        "fit_ssm_mle",
        lambda build, free, y, *, transform, **options: SimpleNamespace(
            params=transform(free), loglike=0.0
        ),
    )
    rng = random.Random(0)
    values = [50 + 6 * math.sin(2 * math.pi * t / 336) + rng.gauss(0, 2) for t in range(672)]
    model = StructuralModel(seasons=((48, 1), (336, 1)))
    assert fit_structural(HalfHourlySeries(START, values), model).nobs == 672


def test_filter_follows_the_local_level_recursion() -> None:
    values = _local_level(80, seed=2)
    result = filter_structural(
        HalfHourlySeries(START, values), LEVEL_PARAMETERS, prior_periods=10, levels=(0.9,)
    )
    level, prediction = result.level, result.prediction
    z = statistics.NormalDist().inv_cdf(0.95)
    for t in range(10, 80):
        # One step ahead the level is unchanged, with variance q added; the value adds h.
        mean, sd = _at(prediction.mean, t), math.sqrt(_at(level.sd, t - 1) ** 2 + 0.01 + 1.0)
        assert mean == pytest.approx(_at(level.mean, t - 1), rel=1e-12)
        assert _at(prediction.sd, t) == pytest.approx(sd, rel=1e-9)
        assert _at(result.bands[0].upper, t) == pytest.approx(mean + z * sd, rel=1e-9)
        innovation = _at(result.standardized_innovation, t)
        assert innovation == pytest.approx((values[t] - mean) / sd, rel=1e-9)
    assert (result.slope, result.regression, result.seasons) == (None, None, {})


def test_filter_unscales_orchids_filtered_states() -> None:
    rng = random.Random(3)
    values = [0.1 * t + 2 * math.sin(2 * math.pi * t / 6) + rng.gauss(0.0, 0.3) for t in range(30)]
    model = StructuralModel(trend="linear", seasons=((6, 2),))
    variances = {"level": 0.01, "slope": 1e-4, "seasonal_6": 1e-3, "irregular": 0.09}
    result = filter_structural(
        HalfHourlySeries(START, values), StructuralParameters(model, variances), prior_periods=6
    )
    mean, variance = _moments(values[:6])
    sd = math.sqrt(variance)
    standardised = [(v - mean) / sd for v in values]
    components = ss.UnobservedComponents(
        standardised,
        [ss.LocalLinearTrend(), ss.TrigonometricSeasonal(6, harmonics=2), ss.Irregular()],
    )
    orchid_model = _vague(components.build([v / variance for v in variances.values()]))
    orchid = ss.kalman_filter(orchid_model, [[v] for v in standardised])
    seasonal = result.seasons[6]
    assert result.slope is not None
    for t in range(5, 30):
        state, cov = orchid.filtered_state[t], orchid.filtered_cov[t]
        assert _at(result.level.mean, t) == pytest.approx(mean + sd * state[0], rel=1e-12)
        assert _at(result.level.sd, t) == pytest.approx(sd * math.sqrt(cov[0][0]), rel=1e-9)
        assert _at(result.slope.mean, t) == pytest.approx(sd * state[1], rel=1e-9, abs=1e-12)
        # The cosine states of the two harmonics carry the seasonal contribution.
        contribution = sd * (state[2] + state[4])
        assert _at(seasonal.mean, t) == pytest.approx(contribution, rel=1e-9, abs=1e-12)
        spread = cov[2][2] + 2 * cov[2][4] + cov[4][4]
        assert _at(seasonal.sd, t) == pytest.approx(sd * math.sqrt(spread), rel=1e-9)


SEASONAL_DYNAMIC_PARAMETERS = StructuralParameters(
    StructuralModel(seasons=((6, 1),), regressors=("wind",), dynamic_regression=True),
    {"level": 0.01, "seasonal_6": 1e-3, "regression": 1e-4, "irregular": 1.0},
)


@pytest.mark.parametrize(
    ("parameters", "regressors"),
    [
        (TREND_PARAMETERS, None),
        (
            SEASONAL_DYNAMIC_PARAMETERS,
            {"wind": HalfHourlySeries(START, [10 + math.sin(t / 5) for t in range(40)])},
        ),
    ],
    ids=["trend", "season and regressor"],
)
def test_filter_outputs_use_only_earlier_values(
    parameters: StructuralParameters, regressors: dict[str, HalfHourlySeries] | None
) -> None:
    values, other = _local_level(40, seed=4), _local_level(40, seed=5, start=60.0)
    baseline = filter_structural(
        HalfHourlySeries(START, values), parameters, regressors=regressors, prior_periods=10
    )
    for t in range(1, 40):
        futures: list[list[float | None]] = [[*other[t:]], [None, *other[t + 1 :]]]
        for future in futures:
            series = HalfHourlySeries(START, [*values[:t], *future])
            result = filter_structural(series, parameters, regressors=regressors, prior_periods=10)
            # States at t see the value at t; predictions for t are made at t - 1.
            assert result.level.mean.values[:t] == baseline.level.mean.values[:t]
            assert result.level.sd.values[:t] == baseline.level.sd.values[:t]
            assert (
                result.prediction.mean.values[: t + 1] == baseline.prediction.mean.values[: t + 1]
            )
            assert result.prediction.sd.values[: t + 1] == baseline.prediction.sd.values[: t + 1]
            innovations = result.standardized_innovation.values[:t]
            assert innovations == baseline.standardized_innovation.values[:t]


def test_filter_masks_the_prior_window_gaps_and_diffuse_phase() -> None:
    values: list[float | None] = [None, None, *_local_level(30, seed=6)]
    values[8] = values[20] = None
    result = filter_structural(HalfHourlySeries(START, values), LEVEL_PARAMETERS, prior_periods=10)
    ready = 12  # the 10th observed value, after two leading gaps and one in the window
    assert result.level.mean.values[:ready] == (None,) * ready
    assert result.prediction.mean.values[: ready + 1] == (None,) * (ready + 1)
    assert result.bands[0].lower.values[: ready + 1] == (None,) * (ready + 1)
    innovations = result.standardized_innovation.values
    assert [t for t, value in enumerate(innovations) if value is not None] == [
        t for t in range(ready + 1, 32) if t != 20
    ]
    # At a gap the state is its prediction, so the value's prediction adds only the noise.
    assert result.level.mean.values[20] == result.prediction.mean.values[20]
    assert _at(result.prediction.sd, 20) == pytest.approx(
        math.sqrt(_at(result.level.sd, 20) ** 2 + 1.0), rel=1e-9
    )
    # A two-period window ends the trend's two-value diffuse phase; innovations follow it.
    trend = filter_structural(SERIES, TREND_PARAMETERS, prior_periods=2)
    assert trend.standardized_innovation.values[:2] == (None, None)
    assert trend.standardized_innovation.values[2] is not None


@pytest.mark.parametrize(("scale", "offset"), [(1e-6, 0.0), (1.0, 3e4)], ids=["scaled", "shifted"])
def test_filter_and_smoother_scale_with_the_data(scale: float, offset: float) -> None:
    values: list[float | None] = [*_local_level(50, seed=7)]
    values[30] = None
    moved_values = [None if v is None else v * scale + offset for v in values]
    variances = {name: v * scale * scale for name, v in TREND_PARAMETERS.variances.items()}
    moved_parameters = StructuralParameters(TREND, variances)
    series, moved = HalfHourlySeries(START, values), HalfHourlySeries(START, moved_values)

    base = filter_structural(series, TREND_PARAMETERS, prior_periods=10)
    result = filter_structural(moved, moved_parameters, prior_periods=10)
    assert base.slope is not None
    assert result.slope is not None
    _assert_moved(result.level.mean, base.level.mean, scale, offset)
    _assert_moved(result.level.sd, base.level.sd, scale)
    _assert_moved(result.slope.mean, base.slope.mean, scale)
    _assert_moved(result.prediction.mean, base.prediction.mean, scale, offset)
    _assert_moved(result.bands[1].upper, base.bands[1].upper, scale, offset)
    _assert_moved(result.standardized_innovation, base.standardized_innovation)

    smooth = smooth_structural(series, TREND_PARAMETERS)
    smooth_moved = smooth_structural(moved, moved_parameters)
    _assert_moved(smooth_moved.level.mean, smooth.level.mean, scale, offset)
    _assert_moved(smooth_moved.irregular.sd, smooth.irregular.sd, scale)
    _assert_moved(smooth_moved.outlier_score, smooth.outlier_score)
    _assert_moved(smooth_moved.level_break_score, smooth.level_break_score)


def test_auxiliary_residuals_are_standard_normal_under_the_model() -> None:
    # Orchid's own auxiliary residuals divide by the posterior sd: here its outlier scores
    # have sd 4.4, half beyond 3, and its level-break scores sd 0.24.
    values = _local_level(1000, q=0.01, h=1.0, seed=7)
    result = smooth_structural(HalfHourlySeries(START, values), LEVEL_PARAMETERS)
    for scores in (result.outlier_score, result.level_break_score):
        present = [s for s in scores.values if s is not None]
        assert len(present) >= 999
        assert 0.9 <= statistics.pstdev(present) <= 1.1
        assert sum(abs(s) > 3 for s in present) < 0.01 * len(present)


def test_scores_locate_a_level_shift_and_an_outlier() -> None:
    values = _local_level(300, q=1e-4, seed=3)
    values[150:] = [v + 8.0 for v in values[150:]]
    values[80] += 10.0
    result = smooth_structural(HalfHourlySeries(START, values), LEVEL_PARAMETERS)

    def peak(scores: HalfHourlySeries) -> int:
        return max(range(300), key=lambda t: abs(scores.values[t] or 0.0))

    assert peak(result.level_break_score) == 150  # the first period at the new level
    assert peak(result.outlier_score) == 80
    assert result.slope_break_score is None


def test_smoother_masks_gaps_and_uninformed_disturbances() -> None:
    values: list[float | None] = [None, *_local_level(40, seed=8)]
    values[20] = None
    series = HalfHourlySeries(START, values)
    result = smooth_structural(series, TREND_PARAMETERS)
    observed = [t for t in range(41) if t not in (0, 20)]
    assert [t for t, v in enumerate(result.irregular.mean.values) if v is not None] == observed
    assert [t for t, v in enumerate(result.outlier_score.values) if v is not None] == observed
    # No disturbance arrives at the first observed period, and a slope change arriving at
    # the last one has yet to move any observed value.
    assert result.slope_break_score is not None
    level_breaks = [t for t, v in enumerate(result.level_break_score.values) if v is not None]
    slope_breaks = [t for t, v in enumerate(result.slope_break_score.values) if v is not None]
    assert (level_breaks, slope_breaks) == (observed[1:], observed[1:-1])
    assert result.level.mean.values[0] is None  # the model starts at the first observation
    assert result.level.mean.values[20] is not None  # smoothing interpolates gaps
    fixed_level = StructuralParameters(TREND, {"level": 0.0, "slope": 1e-4, "irregular": 1.0})
    flat = smooth_structural(series, fixed_level)
    assert set(flat.level_break_score.values) == {None}


def test_duplicate_seasons_are_split_and_sum_to_the_series() -> None:
    rng = random.Random(5)
    daily = [3 * math.sin(2 * math.pi * t / 48) for t in range(500)]
    weekly = [2 * math.cos(2 * math.pi * t / 336) for t in range(500)]
    values = [10 + d + w + rng.gauss(0.0, 0.5) for d, w in zip(daily, weekly, strict=True)]
    model = StructuralModel(seasons=((48, 1), (336, 1)))
    variances = {"level": 1e-4, "seasonal_48": 1e-5, "seasonal_336": 1e-6, "irregular": 0.25}
    result = smooth_structural(
        HalfHourlySeries(START, values), StructuralParameters(model, variances)
    )
    assert list(result.seasons) == [48, 336]
    for season, truth, tolerance in ((48, daily, 0.3), (336, weekly, 0.2)):
        estimate = [_at(result.seasons[season].mean, t) for t in range(500)]
        assert max(abs(e - x) for e, x in zip(estimate, truth, strict=True)) < tolerance
    parts = (result.level, result.seasons[48], result.seasons[336], result.irregular)
    total = [math.fsum(_at(part.mean, t) for part in parts) for t in range(500)]
    assert total == pytest.approx(values, abs=1e-9)


def _wind_and_price(n: int) -> tuple[list[float], list[float]]:
    """Wind in MW, and a price falling 0.5 per MW about a random-walk level."""
    rng = random.Random(9)
    wind = [100 + 20 * math.sin(t / 7) + rng.gauss(0.0, 5.0) for t in range(n)]
    level = _local_level(n, q=0.01, h=0.0, seed=4, start=20.0)
    price = [x - 0.5 * w + rng.gauss(0.0, 1.0) for x, w in zip(level, wind, strict=True)]
    return wind, price


def test_static_regression_contribution_is_recovered() -> None:
    wind, price = _wind_and_price(300)
    result = smooth_structural(
        HalfHourlySeries(START, price),
        WIND_PARAMETERS,
        regressors={"wind": HalfHourlySeries(START, wind)},
    )
    assert result.regression is not None
    contribution = [_at(result.regression.mean, t) for t in range(300)]
    coefficients = [c / w for c, w in zip(contribution, wind, strict=True)]
    assert coefficients == pytest.approx([coefficients[0]] * 300, rel=1e-9)
    assert coefficients[0] == pytest.approx(-0.5, abs=0.02)
    parts = (result.level, result.regression, result.irregular)
    total = [math.fsum(_at(part.mean, t) for part in parts) for t in range(300)]
    assert total == pytest.approx(price, abs=1e-9)


def test_regression_log_likelihood_does_not_depend_on_the_regressors_units() -> None:
    wind, price = _wind_and_price(48)
    mw, kw = (
        fit_structural(
            HalfHourlySeries(START, price),
            WIND,
            regressors={"wind": HalfHourlySeries(START, [scale * w for w in wind])},
        )
        for scale in (1.0, 1000.0)
    )
    assert kw.log_likelihood == pytest.approx(mw.log_likelihood, rel=1e-9)


def test_dynamic_regression_variance_is_per_squared_regressor_unit() -> None:
    wind, price = _wind_and_price(200)
    series = HalfHourlySeries(START, price)
    in_mw = StructuralParameters(DYNAMIC, {"level": 0.01, "regression": 1e-6, "irregular": 1.0})
    in_kw = StructuralParameters(DYNAMIC, {"level": 0.01, "regression": 1e-12, "irregular": 1.0})
    mw = smooth_structural(series, in_mw, regressors={"wind": HalfHourlySeries(START, wind)})
    kw_wind = HalfHourlySeries(START, [1000 * w for w in wind])
    kw = smooth_structural(series, in_kw, regressors={"wind": kw_wind})
    assert mw.regression is not None
    assert kw.regression is not None
    _assert_moved(kw.regression.mean, mw.regression.mean)
    _assert_moved(kw.level.mean, mw.level.mean)
    coefficients = {round(_at(mw.regression.mean, t) / w, 6) for t, w in enumerate(wind)}
    assert len(coefficients) > 1  # the coefficient moves


def test_dynamic_regression_is_one_model_whatever_the_prior_window() -> None:
    # Each window centres the regressor on its own mean. Unless the coefficient's disturbance
    # also moves the level, that changes the model, and the two filters differ by about 1e-3
    # rather than the vague prior's 1e-8.
    wind, price = _wind_and_price(200)
    parameters = StructuralParameters(
        DYNAMIC, {"level": 0.01, "regression": 1e-6, "irregular": 1.0}
    )
    short, long = (
        filter_structural(
            HalfHourlySeries(START, price),
            parameters,
            regressors={"wind": HalfHourlySeries(START, wind)},
            prior_periods=prior_periods,
        )
        for prior_periods in (24, 96)
    )
    assert short.regression is not None
    assert long.regression is not None
    for a, b in (
        (short.regression.mean, long.regression.mean),
        (short.level.mean, long.level.mean),
    ):
        assert a.values[150:] == pytest.approx(b.values[150:], rel=1e-4)


def test_dynamic_regression_without_variance_is_the_static_one() -> None:
    # The driver's mean is about 700 of its sds, which must not make the vague prior
    # informative.
    wind, price = _wind_and_price(200)
    series = HalfHourlySeries(START, price)
    driver = {"wind": HalfHourlySeries(START, [w + 1e4 for w in wind])}
    fixed = StructuralParameters(DYNAMIC, {"level": 0.01, "regression": 0.0, "irregular": 1.0})
    dynamic = smooth_structural(series, fixed, regressors=driver)
    static = smooth_structural(series, WIND_PARAMETERS, regressors=driver)
    assert dynamic.regression is not None
    assert static.regression is not None
    for a, b in (
        (dynamic.level, static.level),
        (dynamic.regression, static.regression),
        (dynamic.irregular, static.irregular),
    ):
        assert a.mean.values == pytest.approx(b.mean.values, rel=1e-12)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda m, r: fit_structural(SERIES, m, regressors=r), id="fit"),
        pytest.param(
            lambda m, r: filter_structural(
                SERIES, StructuralParameters(m, {"level": 0.01, "irregular": 1.0}), regressors=r
            ),
            id="filter",
        ),
        pytest.param(
            lambda m, r: smooth_structural(
                SERIES, StructuralParameters(m, {"level": 0.01, "irregular": 1.0}), regressors=r
            ),
            id="smooth",
        ),
    ],
)
def test_collinear_regressors_do_not_identify_their_coefficients(
    call: Callable[[StructuralModel, dict[str, HalfHourlySeries]], object],
) -> None:
    model = StructuralModel(regressors=("wind", "copy"))
    with pytest.raises(DegenerateDataError) as excinfo:
        call(model, {"wind": WINDY, "copy": WINDY})
    assert excinfo.type is DegenerateDataError


def test_forecaster_unscales_orchids_forecast() -> None:
    rng = random.Random(3)
    values = [0.1 * t + 2 * math.sin(2 * math.pi * t / 6) + rng.gauss(0.0, 0.3) for t in range(40)]
    model = StructuralModel(trend="linear", seasons=((6, 2),))
    variances = {"level": 0.01, "slope": 1e-4, "seasonal_6": 1e-3, "irregular": 0.09}
    forecaster = StructuralForecaster(model, parameters=StructuralParameters(model, variances))
    result = forecaster.forecast(HalfHourlySeries(START, values), 10, levels=(0.95, 0.8))
    mean, variance = _moments(values)
    sd = math.sqrt(variance)
    standardised = [(v - mean) / sd for v in values]
    components = ss.UnobservedComponents(
        standardised,
        [ss.LocalLinearTrend(), ss.TrigonometricSeasonal(6, harmonics=2), ss.Irregular()],
    )
    orchid_model = _vague(components.build([v / variance for v in variances.values()]))
    orchid = ss.forecast(orchid_model, standardised, 10, level=(0.8, 0.95))
    assert result.mean == pytest.approx([mean + sd * m[0] for m in orchid.mean], rel=1e-12)
    for interval in result.intervals:
        lower = [mean + sd * v[0] for v in orchid.lower[interval.level]]
        upper = [mean + sd * v[0] for v in orchid.upper[interval.level]]
        assert interval.lower == pytest.approx(lower, rel=1e-12)
        assert interval.upper == pytest.approx(upper, rel=1e-12)
    filtered = ss.kalman_filter(orchid_model, [[v] for v in standardised])
    errors = filtered.forecast_error[6:]  # after the warm-up: d and the season are both 6
    assert result.residuals == pytest.approx([sd * e[0] for e in errors], rel=1e-12)
    assert (result.source, result.interval_method) == ("state_space.forecast", "analytic")
    assert result.params == {"trend": "linear", "seasons": ((6, 2),), "variances": variances}


def test_residuals_start_after_a_full_cycle_of_the_season() -> None:
    # The prior dominates a harmonic's errors for a cycle, beyond the d = 3 values.
    history = HalfHourlySeries(START, _local_level(40, seed=3))
    result = StructuralForecaster(DAILY, parameters=DAILY_PARAMETERS).forecast(history, 1)
    assert len(result.residuals) == 40 - 12


def test_fitted_forecaster_forecasts_with_the_fit(
    level_fit: tuple[list[float], StructuralFit],
) -> None:
    values, fit = level_fit
    history = HalfHourlySeries(START, values)
    fitted = StructuralForecaster(LEVEL).forecast(history, 6)
    fixed = StructuralForecaster(LEVEL, parameters=fit.parameters).forecast(history, 6)
    assert (fitted.mean, fitted.intervals, fitted.residuals) == (
        fixed.mean,
        fixed.intervals,
        fixed.residuals,
    )
    assert fitted.params["variances"] == dict(fit.parameters.variances)
    assert (fitted.params["aic"], fitted.params["bic"]) == (fit.aic, fit.bic)


@pytest.mark.parametrize(
    ("forecaster", "minimum"),
    [
        (StructuralForecaster(LEVEL, parameters=LEVEL_PARAMETERS), 2),
        (StructuralForecaster(TREND, parameters=TREND_PARAMETERS), 3),
        (StructuralForecaster(DAILY, parameters=DAILY_PARAMETERS), 13),  # a season, a residual
        (StructuralForecaster(LEVEL), 4),
        (DiscountForecaster(prior_periods=10), 11),
        (DiscountForecaster(trend="linear", prior_periods=2), 3),
    ],
    ids=["level", "trend", "season", "fitted level", "discount level", "discount trend"],
)
def test_min_history_is_exact(
    make_series: MakeSeries, forecaster: Forecaster, minimum: int
) -> None:
    assert forecaster.min_history == minimum
    result = forecaster.forecast(make_series(minimum, level=0.0), 4, levels=(0.8, 0.95))
    assert result.residuals
    inner, outer = result.intervals
    bands = zip(outer.lower, inner.lower, result.mean, inner.upper, outer.upper, strict=True)
    assert all(list(band) == sorted(band) for band in bands)
    with pytest.raises(InsufficientDataError) as excinfo:
        forecaster.forecast(make_series(minimum - 1), 4)
    assert excinfo.value.required == minimum


def test_seasonal_forecasts_warn_across_a_clock_change() -> None:
    history = HalfHourlySeries(CLOCK_CHANGE_DAY - 48 * HALF_HOUR, _local_level(48, seed=2))
    seasonal = StructuralForecaster(DAILY, parameters=DAILY_PARAMETERS).forecast(history, 4)
    assert seasonal.warnings == (CLOCK_CHANGE,)
    plain = StructuralForecaster(LEVEL, parameters=LEVEL_PARAMETERS).forecast(history, 4)
    assert plain.warnings == ()


def test_discount_filter_matches_orchid_with_the_same_priors() -> None:
    values: list[float | None] = [*_local_level(80, seed=8)]
    values[50] = None
    result = discount_filter(
        HalfHourlySeries(START, values),
        trend="linear",
        discount=0.9,
        variance_discount=0.98,
        prior_periods=12,
        levels=(0.9,),
    )
    mean, variance = _moments([v for v in values[:12] if v is not None])
    model = ss.DiscountDLM(
        [1.0, 0.0],
        [[1.0, 1.0], [0.0, 1.0]],
        delta=0.9,
        m0=[mean, 0.0],
        C0=ss.eye(2, 100 * variance),
        n0=1.0,
        s0=variance,
        beta=0.98,
    )
    orchid = model.filter(values)  # type: ignore[arg-type]  # Orchid reads None as missing
    assert result.slope is not None
    ready = 11
    for state, estimate in enumerate((result.level, result.slope)):
        assert estimate.mean.values[:ready] == (None,) * ready
        means = [m[state] for m in orchid.filtered_mean[ready:]]
        scales = [math.sqrt(c[state][state]) for c in orchid.filtered_cov[ready:]]
        assert estimate.mean.values[ready:] == pytest.approx(means, rel=1e-12)
        assert estimate.scale.values[ready:] == pytest.approx(scales, rel=1e-12)
        assert estimate.dof.values[ready:] == pytest.approx(orchid.dof[ready:], rel=1e-12)
    observation_sd = [math.sqrt(s) for s in orchid.obs_scale[ready:]]
    assert result.observation_sd.values[ready:] == pytest.approx(observation_sd, rel=1e-12)
    prediction = result.prediction
    assert prediction.mean.values[: ready + 1] == (None,) * (ready + 1)
    for t in range(ready + 1, 80):
        scale = math.sqrt(orchid.forecast_var[t])
        assert _at(prediction.mean, t) == pytest.approx(orchid.forecast[t], rel=1e-12)
        assert _at(prediction.scale, t) == pytest.approx(scale, rel=1e-12)
        # The prior for t discounts the degrees of freedom at t - 1 once.
        assert _at(prediction.dof, t) == 0.98 * orchid.dof[t - 1]
        value = values[t]
        if value is None:
            assert result.standardized_error.values[t] is None
        else:
            error = (value - orchid.forecast[t]) / scale
            assert _at(result.standardized_error, t) == pytest.approx(error, rel=1e-12)
    for t in range(ready + 1, 80, 7):  # Orchid's quantile takes milliseconds
        quantile = ss.t_ppf(0.95, df=0.98 * orchid.dof[t - 1])
        half_width = quantile * math.sqrt(orchid.forecast_var[t])
        assert _at(result.bands[0].upper, t) == pytest.approx(orchid.forecast[t] + half_width)
        assert _at(result.bands[0].lower, t) == pytest.approx(orchid.forecast[t] - half_width)


def test_discount_filter_uses_only_earlier_values() -> None:
    values: list[float | None] = [None, None, *_local_level(30, seed=4)]
    values[5] = None  # inside the prior window, which then ends at its 8th observed value, 10
    other = _local_level(32, seed=5, start=60.0)
    baseline = discount_filter(HalfHourlySeries(START, values), trend="linear", prior_periods=8)
    errors = baseline.standardized_error.values
    assert baseline.level.mean.values[:10] == (None,) * 10
    assert [t for t, value in enumerate(errors) if value is not None] == list(range(11, 32))
    for t in range(1, 32):
        futures: list[list[float | None]] = [[*other[t:]], [None, *other[t + 1 :]]]
        for future in futures:
            series = HalfHourlySeries(START, [*values[:t], *future])
            result = discount_filter(series, trend="linear", prior_periods=8)
            assert result.level.mean.values[:t] == baseline.level.mean.values[:t]
            assert result.observation_sd.values[:t] == baseline.observation_sd.values[:t]
            assert (
                result.prediction.mean.values[: t + 1] == baseline.prediction.mean.values[: t + 1]
            )
            assert result.bands[0].lower.values[: t + 1] == baseline.bands[0].lower.values[: t + 1]
            assert result.standardized_error.values[:t] == baseline.standardized_error.values[:t]


@pytest.mark.parametrize(("scale", "offset"), [(1e-6, 0.0), (1.0, 3e4)], ids=["scaled", "shifted"])
def test_discount_filter_scales_with_the_data(scale: float, offset: float) -> None:
    values: list[float | None] = [None, *_local_level(60, seed=9)]
    values[30] = None
    moved = [None if v is None else v * scale + offset for v in values]
    base = discount_filter(HalfHourlySeries(START, values), trend="linear", prior_periods=10)
    result = discount_filter(HalfHourlySeries(START, moved), trend="linear", prior_periods=10)
    assert base.slope is not None
    assert result.slope is not None
    _assert_moved(result.level.mean, base.level.mean, scale, offset)
    _assert_moved(result.slope.scale, base.slope.scale, scale)
    _assert_moved(result.observation_sd, base.observation_sd, scale)
    _assert_moved(result.prediction.mean, base.prediction.mean, scale, offset)
    _assert_moved(result.bands[1].upper, base.bands[1].upper, scale, offset)
    _assert_moved(result.standardized_error, base.standardized_error)
    assert result.prediction.dof == base.prediction.dof


@pytest.mark.parametrize("variance_discount", [1.0, 0.8])
def test_discount_forecaster_holds_the_evolution_variance(variance_discount: float) -> None:
    values = _local_level(60, seed=10)
    forecaster = DiscountForecaster(
        discount=0.9, variance_discount=variance_discount, prior_periods=12
    )
    result = forecaster.forecast(HalfHourlySeries(START, values), 24, levels=(0.95,))
    mean, variance = _moments(values[:12])
    model = ss.DiscountDLM(
        [1.0],
        [[1.0]],
        delta=0.9,
        m0=[mean],
        C0=[[100 * variance]],
        s0=variance,
        beta=variance_discount,
    )
    orchid = model.filter(values)
    level, spread = orchid.filtered_mean[-1][0], orchid.filtered_cov[-1][0][0]
    noise, dof = orchid.obs_scale[-1], orchid.dof[-1]
    # West and Harrison: R_k = C (1 + (k - 1)(1 - delta)) / delta, plus the observation noise.
    variances = [spread * (1 + (k - 1) * 0.1) / 0.9 + noise for k in range(1, 25)]
    quantile = ss.t_ppf(0.975, df=variance_discount * dof)  # the one-step degrees of freedom
    half_widths = [quantile * math.sqrt(v) for v in variances]
    assert result.mean == pytest.approx([level] * 24, rel=1e-12)
    widths = [u - m for u, m in zip(result.intervals[0].upper, result.mean, strict=True)]
    assert widths == pytest.approx(half_widths, rel=1e-9)
    assert variances[-1] < model.forecast(values, 24)[1][-1]  # Orchid compounds 1 / delta
    errors = [v - f for v, f in zip(values[12:], orchid.forecast[12:], strict=True)]
    assert result.residuals == pytest.approx(errors, rel=1e-12)
    assert result.params == {
        "trend": "level",
        "discount": 0.9,
        "variance_discount": variance_discount,
        "prior_periods": 12,
        "observation_variance": noise,
        "dof": dof,
    }
    assert (result.source, result.interval_method) == ("state_space.DiscountDLM", "student-t")


def test_discount_forecaster_extends_a_linear_trend() -> None:
    values = [2.0 * t + math.sin(t) for t in range(40)]
    result = DiscountForecaster(trend="linear", discount=0.9, prior_periods=10).forecast(
        HalfHourlySeries(START, values), 24, levels=(0.95,)
    )
    steps = [b - a for a, b in zip(result.mean, result.mean[1:], strict=False)]
    assert steps == pytest.approx([steps[0]] * 23, rel=1e-9)
    assert steps[0] == pytest.approx(2.0, abs=0.2)
    mean, variance = _moments(values[:10])
    model = ss.DiscountDLM(
        [1.0, 0.0],
        [[1.0, 1.0], [0.0, 1.0]],
        delta=0.9,
        m0=[mean, 0.0],
        C0=ss.eye(2, 100 * variance),
        s0=variance,
    )
    orchid = model.filter(values)
    (c00, c01), (_, c11) = orchid.filtered_cov[-1]
    p00, p01, p11 = c00 + 2 * c01 + c11, c01 + c11, c11  # P = G C G'

    def level_variance(j: int) -> float:
        """The level's variance in G^j P G'^j."""
        return p00 + 2 * j * p01 + j * j * p11

    # R_1 = P / delta and R_k = G R_(k-1) G' + W, with W = P (1 - delta) / delta, so
    # R_k = G^(k-1) P G'^(k-1) / delta + the sum of G^i W G'^i for i < k - 1.
    variances = [
        level_variance(k - 1) / 0.9 + 0.1 / 0.9 * math.fsum(map(level_variance, range(k - 1)))
        for k in range(1, 25)
    ]
    quantile = ss.t_ppf(0.975, df=orchid.dof[-1])
    half_widths = [quantile * math.sqrt(v + orchid.obs_scale[-1]) for v in variances]
    widths = [u - m for u, m in zip(result.intervals[0].upper, result.mean, strict=True)]
    assert widths == pytest.approx(half_widths, rel=1e-9)


def test_discount_forecaster_stores_float_discounts() -> None:
    forecaster = DiscountForecaster(
        discount=Fraction(19, 20),  # type: ignore[arg-type]
        variance_discount=Fraction(9, 10),  # type: ignore[arg-type]
    )
    assert type(forecaster.discount) is float
    assert type(forecaster.variance_discount) is float
    assert forecaster.forecast(SERIES, 1).params["discount"] == 0.95


def test_t_quantile_matches_the_cauchy_closed_form() -> None:
    # Newton takes the most steps at small degrees of freedom and high levels.
    assert _t_quantile(0.9995, 1.0) == pytest.approx(math.tan(math.pi * 0.4995), rel=1e-12)


def test_orchid_failures_raise_calculation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise ZeroDivisionError("boom")

    with monkeypatch.context() as patch:
        patch.setattr(ss, "kalman_filter", broken)
        with pytest.raises(CalculationError) as excinfo:
            filter_structural(SERIES, LEVEL_PARAMETERS, prior_periods=10)
        assert isinstance(excinfo.value.__cause__, ZeroDivisionError)
    with monkeypatch.context() as patch:  # a CDF that never reaches 0.975 stalls Newton
        patch.setattr(ss, "t_cdf", lambda x, df: 0.975 - 1e-6)
        with pytest.raises(CalculationError):
            discount_filter(SERIES, prior_periods=10, levels=(0.95,))
    # The largest level below 1 has an infinite quantile.
    extreme = (0.9999999999999999,)
    with pytest.raises(CalculationError):
        filter_structural(SERIES, LEVEL_PARAMETERS, prior_periods=10, levels=extreme)
    with pytest.raises(CalculationError):
        discount_filter(SERIES, prior_periods=10, levels=extreme)
    with pytest.raises(CalculationError):  # at 0.01 degrees of freedom the quantile overflows
        discount_filter(SERIES, variance_discount=0.01, prior_periods=10, levels=(0.999,))
    with pytest.raises(CalculationError):  # the variance overflows a float
        smooth_structural(HalfHourlySeries(START, [1e200, -1e200] * 5), LEVEL_PARAMETERS)


CONSTANT = HalfHourlySeries(START, [42.0] * 60)
GAPPY = HalfHourlySeries(START, [*WINDY.values[:10], None, *WINDY.values[11:]])
DAMPED_VARIANCES = {"level": 0.01, "slope": 1e-4, "irregular": 1.0}


@pytest.mark.parametrize(
    ("call", "error"),
    [
        pytest.param(lambda: StructuralModel(trend="cubic"), InvalidInputError, id="trend"),  # type: ignore[arg-type]
        pytest.param(lambda: StructuralModel(seasons=((2, 1),)), InvalidInputError, id="period 2"),
        pytest.param(
            lambda: StructuralModel(seasons=((48, 0),)), InvalidInputError, id="no harmonics"
        ),
        pytest.param(
            lambda: StructuralModel(seasons=((48, 24),)), InvalidInputError, id="Nyquist harmonic"
        ),
        pytest.param(
            lambda: StructuralModel(seasons=((48.0, 1),)),  # type: ignore[arg-type]
            InvalidInputError,
            id="float period",
        ),
        pytest.param(
            lambda: StructuralModel(seasons=((48, True),)), InvalidInputError, id="bool harmonics"
        ),
        pytest.param(
            lambda: StructuralModel(seasons=(48, 1)),  # type: ignore[arg-type]
            InvalidInputError,
            id="unpaired season",
        ),
        pytest.param(
            lambda: StructuralModel(seasons=((48, 1), (48, 2))),
            InvalidInputError,
            id="repeated period",
        ),
        pytest.param(
            lambda: StructuralModel(seasons=((48, 2), (24, 1))),
            InvalidInputError,
            id="shared harmonic",
        ),
        pytest.param(
            lambda: StructuralModel(regressors="wind"),  # type: ignore[arg-type]
            InvalidInputError,
            id="regressors string",
        ),
        pytest.param(
            lambda: StructuralModel(regressors=("wind", "wind")),
            InvalidInputError,
            id="repeated regressor",
        ),
        pytest.param(lambda: StructuralModel(regressors=("",)), InvalidInputError, id="no name"),
        pytest.param(
            lambda: StructuralModel(dynamic_regression=1),  # type: ignore[arg-type]
            InvalidInputError,
            id="dynamic int",
        ),
        pytest.param(
            lambda: StructuralModel(dynamic_regression=True),
            InvalidInputError,
            id="dynamic without regressors",
        ),
        pytest.param(
            lambda: StructuralModel(regressors=("a", "b"), dynamic_regression=True),
            InvalidInputError,
            id="two dynamic regressors",
        ),
        pytest.param(
            lambda: StructuralModel(trend="linear", seasons=((48, 6),)),
            InvalidInputError,
            id="14 states",
        ),
        pytest.param(
            lambda: StructuralParameters(LEVEL, {"level": 1.0}), InvalidInputError, id="no key"
        ),
        pytest.param(
            lambda: StructuralParameters(LEVEL, {"level": 1.0, "irregular": 1.0, "slope": 1.0}),
            InvalidInputError,
            id="extra key",
        ),
        pytest.param(
            lambda: StructuralParameters(LEVEL, {"level": -1.0, "irregular": 1.0}),
            InvalidInputError,
            id="negative variance",
        ),
        pytest.param(
            lambda: StructuralParameters(LEVEL, {"level": math.inf, "irregular": 1.0}),
            InvalidInputError,
            id="infinite variance",
        ),
        pytest.param(
            lambda: StructuralParameters(LEVEL, {"level": math.nan, "irregular": 1.0}),
            InvalidInputError,
            id="NaN variance",
        ),
        pytest.param(
            lambda: StructuralParameters(LEVEL, {"level": "1", "irregular": 1.0}),  # type: ignore[dict-item]
            InvalidInputError,
            id="string variance",
        ),
        pytest.param(
            lambda: StructuralParameters(LEVEL, {"level": 0.0, "irregular": 0.0}),
            InvalidInputError,
            id="all variances zero",
        ),
        pytest.param(
            lambda: StructuralParameters(DAMPED, DAMPED_VARIANCES),
            InvalidInputError,
            id="damped without damping",
        ),
        pytest.param(
            lambda: StructuralParameters(TREND, DAMPED_VARIANCES, damping=0.9),
            InvalidInputError,
            id="damping without damped trend",
        ),
        pytest.param(
            lambda: StructuralParameters(DAMPED, DAMPED_VARIANCES, damping=1.0),
            InvalidInputError,
            id="damping 1",
        ),
        pytest.param(
            lambda: filter_structural(SERIES, LEVEL_PARAMETERS, prior_periods=1),
            InvalidInputError,
            id="prior_periods 1",
        ),
        pytest.param(
            lambda: filter_structural(SERIES, LEVEL_PARAMETERS, prior_periods=10.0),  # type: ignore[arg-type]
            InvalidInputError,
            id="float prior_periods",
        ),
        pytest.param(
            lambda: filter_structural(SERIES, DAILY_PARAMETERS, prior_periods=11),
            InvalidInputError,
            id="window shorter than the season",
        ),
        pytest.param(
            lambda: filter_structural(
                SERIES,
                StructuralParameters(
                    StructuralModel(trend="linear", regressors=("wind",)), DAMPED_VARIANCES
                ),
                regressors={"wind": WINDY},
                prior_periods=2,
            ),
            InvalidInputError,
            id="window shorter than the state dimension",
        ),
        pytest.param(
            lambda: filter_structural(SERIES, LEVEL_PARAMETERS, levels=(95,)),
            InvalidInputError,
            id="level 95",
        ),
        pytest.param(
            lambda: filter_structural(SERIES, LEVEL_PARAMETERS, prior_periods=60),
            InsufficientDataError,
            id="nothing after the window",
        ),
        pytest.param(
            lambda: filter_structural(CONSTANT, LEVEL_PARAMETERS),
            DegenerateDataError,
            id="constant window",
        ),
        pytest.param(
            lambda: filter_structural(
                SERIES, WIND_PARAMETERS, regressors={"wind": HalfHourlySeries(START, [0.0] * 60)}
            ),
            DegenerateDataError,
            id="regressor constant over the window",
        ),
        pytest.param(
            lambda: smooth_structural(SERIES, WIND_PARAMETERS),
            InvalidInputError,
            id="missing regressor",
        ),
        pytest.param(
            lambda: smooth_structural(SERIES, LEVEL_PARAMETERS, regressors={"wind": WINDY}),
            InvalidInputError,
            id="unexpected regressor",
        ),
        pytest.param(
            lambda: smooth_structural(
                SERIES, WIND_PARAMETERS, regressors={"wind": WINDY.window(1, 60)}
            ),
            InvalidInputError,
            id="misaligned regressor",
        ),
        pytest.param(
            lambda: smooth_structural(SERIES, WIND_PARAMETERS, regressors={"wind": GAPPY}),
            MissingDataError,
            id="incomplete regressor",
        ),
        pytest.param(
            lambda: smooth_structural(SERIES.window(0, 11), DAILY_PARAMETERS),
            InsufficientDataError,
            id="smoothing less than a season",
        ),
        pytest.param(
            lambda: StructuralForecaster(WIND), InvalidInputError, id="forecaster regressors"
        ),
        pytest.param(
            lambda: StructuralForecaster(LEVEL, parameters=TREND_PARAMETERS),
            InvalidInputError,
            id="parameters for another model",
        ),
        pytest.param(
            lambda: StructuralForecaster(LEVEL, parameters=LEVEL_PARAMETERS).forecast(SERIES, 0),
            InvalidInputError,
            id="horizon 0",
        ),
        pytest.param(
            lambda: StructuralForecaster(LEVEL, parameters=LEVEL_PARAMETERS).forecast(
                HalfHourlySeries(START, [1.0, None, 2.0]), 1
            ),
            MissingDataError,
            id="forecast history with a gap",
        ),
        pytest.param(
            lambda: discount_filter(SERIES, trend="damped"),  # type: ignore[arg-type]
            InvalidInputError,
            id="discount trend",
        ),
        pytest.param(
            lambda: discount_filter(SERIES, discount=0.0), InvalidInputError, id="discount 0"
        ),
        pytest.param(
            lambda: discount_filter(SERIES, discount=1.5), InvalidInputError, id="discount 1.5"
        ),
        pytest.param(
            lambda: discount_filter(SERIES, discount=math.nan), InvalidInputError, id="NaN discount"
        ),
        pytest.param(
            lambda: discount_filter(SERIES, discount="0.9"),  # type: ignore[arg-type]
            InvalidInputError,
            id="string discount",
        ),
        pytest.param(
            lambda: discount_filter(SERIES, variance_discount=0.0),
            InvalidInputError,
            id="variance_discount 0",
        ),
        pytest.param(
            lambda: discount_filter(SERIES, prior_periods=1), InvalidInputError, id="warm-up of 1"
        ),
        pytest.param(
            lambda: discount_filter(SERIES, prior_periods=True),
            InvalidInputError,
            id="bool warm-up",
        ),
        pytest.param(lambda: discount_filter(SERIES, levels=()), InvalidInputError, id="no levels"),
        pytest.param(
            lambda: discount_filter(SERIES, prior_periods=60),
            InsufficientDataError,
            id="nothing after the warm-up",
        ),
        pytest.param(lambda: discount_filter(CONSTANT), DegenerateDataError, id="constant warm-up"),
        pytest.param(
            lambda: DiscountForecaster(variance_discount=1.1),
            InvalidInputError,
            id="forecaster variance_discount",
        ),
        pytest.param(
            lambda: DiscountForecaster(prior_periods=10).forecast(CONSTANT, 1),
            DegenerateDataError,
            id="forecaster constant warm-up",
        ),
    ],
)
def test_errors_are_classified(call: Callable[[], object], error: type[InvalidInputError]) -> None:
    with pytest.raises(error) as excinfo:
        call()
    assert excinfo.type is error
