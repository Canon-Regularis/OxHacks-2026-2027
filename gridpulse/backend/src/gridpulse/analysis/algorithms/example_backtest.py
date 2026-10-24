"""Reference algorithm: day-ahead price backtest of two naive baselines.

Shows the fixed-horizon pattern for rolling-origin backtests (docs/gridpulse/BACKEND_DESIGN.md):
every target period t gets the forecast made at origin t - horizon, so forecasts and actuals share
one time axis and chart directly, while scorecards go in tables. It uses settled prices only, so no
forecast-publication timing is involved. Replace with Orchid's SeasonalNaive/backtest when ready.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from gridpulse.analysis.contract import Algorithm, AlgorithmResult

WEEK_PERIODS = 336  # 7 x 48 settlement periods = 168 hours (an hour off across a clock change)
WARMUP = timedelta(days=7)  # history the weekly baseline needs before the first target


def run(frame: pd.DataFrame, *, horizon_periods: int) -> AlgorithmResult:
    if not 1 <= horizon_periods < WEEK_PERIODS:
        raise ValueError(f"horizon_periods must be between 1 and {WEEK_PERIODS - 1}")
    price = frame["price_gbp_mwh"]

    # Each model only uses prices already settled at the origin, i.e. before period t - horizon.
    forecasts = pd.DataFrame(
        {
            "persistence_forecast": price.shift(horizon_periods + 1),  # last settled price at the origin
            "seasonal_naive_forecast": price.shift(WEEK_PERIODS),  # same period one week earlier
        }
    )

    # Score only the requested range; the first WARMUP of the frame is history.
    evaluated = forecasts.loc[frame.index[0] + WARMUP :]
    actual = price.loc[evaluated.index]
    errors = evaluated.sub(actual, axis=0)

    metrics = pd.DataFrame(
        [
            {
                "model": model,
                "horizon_periods": horizon_periods,
                "n": int(error.notna().sum()),
                "mae_gbp_mwh": error.abs().mean(),
                "rmse_gbp_mwh": np.sqrt((error**2).mean()),
                "bias_gbp_mwh": error.mean(),
            }
            for model, error in errors.items()
        ]
    )
    worst = errors["seasonal_naive_forecast"].abs().nlargest(10)
    worst_misses = pd.DataFrame(
        {
            "actual_gbp_mwh": actual.loc[worst.index],
            "forecast_gbp_mwh": evaluated.loc[worst.index, "seasonal_naive_forecast"],
            "abs_error_gbp_mwh": worst,
        }
    )  # indexed by delivery_start_utc, which becomes the first column

    return AlgorithmResult(series=forecasts, tables={"metrics": metrics, "worst_misses": worst_misses})


ALGORITHM = Algorithm(
    id="price_naive_backtest",
    version="0.1.0",
    description="Day-ahead imbalance price from persistence and same-period-last-week baselines, with error scorecards.",
    run=run,
    inputs=("price_gbp_mwh",),
    outputs={"persistence_forecast": "GBP/MWh", "seasonal_naive_forecast": "GBP/MWh"},
    tables={
        "metrics": "MAE, RMSE and bias per model over the requested range",
        "worst_misses": "Ten periods with the largest seasonal-naive error",
    },
    params={"horizon_periods": 48},
    warmup=WARMUP,
    limitations="Numerical price baselines for comparison, not spike probabilities or a trading signal.",
)
