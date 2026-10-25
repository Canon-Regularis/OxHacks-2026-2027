"""Day-ahead imbalance price backtest of the P1 forecasters (P1 #15, #16, #21, #22, #24, #25).

Theta, a low-order AR, the discount DLM, a local-level structural model and an ensemble of
the cheap ones, against a daily seasonal naive benchmark. The fold machinery is in
``_backtest``; the ``dm_tests`` table then compares each model's errors at chosen lead steps
with the benchmark's by the Diebold-Mariano test.
"""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from gridpulse.analysis.algorithms import _backtest, evaluation
from gridpulse.analysis.algorithms._backtest import WINDOW
from gridpulse.analysis.algorithms._checks import require_int
from gridpulse.analysis.algorithms.combination import Ensemble
from gridpulse.analysis.algorithms.errors import (
    AlgorithmError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
)
from gridpulse.analysis.algorithms.forecasting import AR, Forecaster, SeasonalNaive, Theta
from gridpulse.analysis.algorithms.series import HALF_HOUR
from gridpulse.analysis.algorithms.state_space import (
    DiscountForecaster,
    StructuralForecaster,
    StructuralModel,
)
from gridpulse.analysis.contract import Algorithm, AlgorithmResult


def _structural(*, trend: Literal["level", "linear", "damped"]) -> StructuralForecaster:
    """A structural model without seasons, refitted at every origin: a seasonal fit takes
    14-50 s, too long to repeat at each of 56 origins."""
    return StructuralForecaster(StructuralModel(trend=trend))


MEMBERS: dict[str, Callable[..., Forecaster]] = {
    "seasonal_naive": SeasonalNaive,
    "theta": Theta,
    "ar": AR,
    "discount_dlm": DiscountForecaster,
    "structural": _structural,
}
MODELS = (*MEMBERS, "ensemble")  # the ensemble combines members configured beside it
BANDED = MODELS[1:]  # the seasonal naive's intervals are scored but not stored as series
OUTPUTS = _backtest.outputs(MODELS, BANDED)
DM_COLUMNS = [
    "model", "benchmark", "horizon_step", "lead_hours", "loss", "n",
    "mean_loss_model", "mean_loss_benchmark", "statistic", "p_value", "status", "message",
]  # fmt: skip
TABLE_COLUMNS = _backtest.TABLE_COLUMNS | {"dm_tests": DM_COLUMNS}
LOSSES: tuple[evaluation.Loss, ...] = ("squared", "absolute")
__all__ = ["ALGORITHM", "MODELS", "OUTPUTS", "TABLE_COLUMNS", "WINDOW", "build", "run"]


def build(models: Mapping[str, Mapping[str, Any]]) -> dict[str, Forecaster]:
    """One forecaster per model; the ensemble's ``members`` name models configured beside it,
    which it uses with their settings there."""
    if set(models) != set(MODELS):
        raise InvalidInputError(
            f"models must configure exactly {sorted(MODELS)}, got {sorted(models)}"
        )
    forecasters = _backtest.build(MEMBERS, {name: models[name] for name in MEMBERS})
    settings = dict(models["ensemble"])
    members = settings.pop("members", None)
    if not isinstance(members, (list, tuple)) or not all(isinstance(m, str) for m in members):
        raise InvalidInputError(f"ensemble members must be a list of model names, got {members!r}")
    if not set(members) <= set(MEMBERS) or len(set(members)) < len(members):
        raise InvalidInputError(
            f"ensemble members must be distinct models among {sorted(MEMBERS)}, got {members}"
        )
    ensemble = Ensemble(members=[forecasters[name] for name in members], **settings)
    return forecasters | {"ensemble": ensemble}


def run(
    frame: pd.DataFrame,
    *,
    horizon: int,
    window: int,
    max_folds: int,
    season_length: int,
    models: Mapping[str, Mapping[str, Any]],
    dm_benchmark: str,
    dm_leads: Sequence[int],
    dm_loss: str,
    memo: MutableMapping[str, str] | None = None,
) -> AlgorithmResult:
    forecasters = build(models)
    horizon = require_int("horizon", horizon, minimum=1)
    if dm_benchmark not in MODELS:
        raise InvalidInputError(f"dm_benchmark must be one of {list(MODELS)}, got {dm_benchmark!r}")
    leads = [require_int("dm_leads", lead, minimum=1, maximum=horizon) for lead in dm_leads]
    if dm_loss not in LOSSES:
        raise InvalidInputError(f"dm_loss must be one of {list(LOSSES)}, got {dm_loss!r}")
    result = _backtest.run_backtest(
        frame,
        forecasters,
        banded=BANDED,
        horizon=horizon,
        window=window,
        max_folds=max_folds,
        season_length=season_length,
        memo=memo,
        fold_params=True,
    )
    assert result.series is not None
    actual = frame["price_gbp_mwh"].iloc[WINDOW:]
    result.tables["dm_tests"] = dm_tests(
        result.series,
        actual,
        horizon=horizon,
        benchmark=dm_benchmark,
        leads=leads,
        loss=dm_loss,
    )
    return result


def dm_tests(
    forecasts: pd.DataFrame,
    actual: pd.Series,
    *,
    horizon: int,
    benchmark: str,
    leads: Sequence[int],
    loss: evaluation.Loss,
) -> pd.DataFrame:
    """Diebold-Mariano tests of every model against ``benchmark`` at each lead step.

    ``forecasts`` holds a ``<model>_forecast`` column per model, aligned with ``actual`` and
    starting at the first origin, so that row i was forecast ``i % horizon + 1`` steps ahead.
    Origins are ``horizon`` apart and every lead is at most ``horizon``, so the errors at one
    lead come from separate origins and the test's horizon is ceil(lead / horizon) = 1. The
    benchmark's errors go first, so a positive statistic means the model was more accurate.
    """
    step = np.arange(len(forecasts)) % horizon + 1
    truth = actual.to_numpy(dtype=float)
    power = 2 if loss == "squared" else 1
    rows: list[list[object]] = []
    for lead in leads:
        at = step == lead
        base = truth[at] - forecasts[f"{benchmark}_forecast"].to_numpy(dtype=float)[at]
        for name in MODELS:
            if name == benchmark:
                continue
            errors = truth[at] - forecasts[f"{name}_forecast"].to_numpy(dtype=float)[at]
            both = np.isfinite(base) & np.isfinite(errors)  # periods of excluded folds drop out
            a, b = base[both], errors[both]
            n = len(a)
            means = [float(np.mean(np.abs(e) ** power)) if n else None for e in (b, a)]
            row: list[object] = [name, benchmark, lead, lead / 2, loss, n, *means]
            try:
                test = evaluation.diebold_mariano(list(a), list(b), horizon=1, loss=loss)
            except InsufficientDataError as exc:
                rows.append([*row, None, None, "insufficient_data", str(exc)])
            except DegenerateDataError as exc:
                rows.append([*row, None, None, "degenerate_data", str(exc)])
            except AlgorithmError as exc:
                rows.append([*row, None, None, "failed", str(exc)])
            else:
                message = "; ".join(test.warnings) or None
                rows.append([*row, test.statistic, test.p_value, "ok", message])
    return pd.DataFrame(rows, columns=DM_COLUMNS)


ALGORITHM = Algorithm(
    id="price_p1_backtest",
    version="0.1.0",
    description="Day-ahead imbalance price from Theta, AR(2), a discount DLM, a local-level "
    "structural model and their ensemble, against a daily seasonal naive benchmark, scored by "
    "a rolling-origin backtest with Diebold-Mariano tests.",
    run=run,
    inputs=("price_gbp_mwh",),
    outputs=OUTPUTS,
    tables=_backtest.TABLES
    | {
        "fold_params": "Fitted parameters per model and origin (Theta's alpha and drift, AR "
        "coefficients, the DLM's observation variance, structural variances and likelihood)",
        "dm_tests": "Diebold-Mariano test of each model against the benchmark at each chosen "
        "lead step: mean losses in GBP/MWh (absolute) or its square (squared), statistic "
        "(positive when the model was more accurate) and p-value",
    },
    params=_backtest.SETTINGS
    | {
        "models": {
            "seasonal_naive": {"season_length": 48},
            "theta": {"season_length": 48, "theta": 2.0, "seasonal_model": "additive"},
            "ar": {"p": 2},
            "discount_dlm": {
                "trend": "level",
                "discount": 0.95,
                "variance_discount": 1.0,
                "prior_periods": 48,
            },
            "structural": {"trend": "level"},
            "ensemble": {"members": ["theta", "ar", "discount_dlm"], "method": "mean"},
        },
        "dm_benchmark": "seasonal_naive",
        "dm_leads": [1, 48],
        "dm_loss": "absolute",
    },
    param_descriptions=_backtest.SETTING_DESCRIPTIONS
    | {
        "dm_benchmark": "Model every other model is tested against in dm_tests",
        "dm_leads": "Lead steps tested in dm_tests, each at most horizon (1 = 30 minutes ahead, "
        "48 = 24 hours ahead)",
        "dm_loss": "Loss the Diebold-Mariano test compares: absolute (as MAE) or squared (as RMSE)",
    },
    warmup=WINDOW * HALF_HOUR,
    memo=True,
    limitations=_backtest.LIMITATIONS
    + " The seasonal naive's intervals are scored but not stored as series. The ensemble "
    "averages each interval bound (quantile averaging), which does not recalibrate coverage. "
    "Diebold-Mariano p-values test equal accuracy over these folds only; with about 56 errors "
    "per lead and heavy-tailed prices they are approximate, and testing several models and "
    "leads at once makes some small p-values likely by chance.",
)
