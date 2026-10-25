"""Pure calculations and Orchid adapters. Accept prepared data; do not access HTTP or the database.

To add an algorithm: create a module here that defines `ALGORITHM = Algorithm(...)` (see
analysis/contract.py and example_correlation.py), then add it to the list below. The tests in
tests/analysis/ run every registered algorithm against the contract automatically.
"""

from gridpulse.analysis.algorithms import (
    calendar_fourier_terms,
    example_backtest,
    example_correlation,
    price_baseline_backtest,
    price_day_ahead_forecast,
    price_decomposition,
    price_driver_correlograms,
    price_forecast_backtest,
    price_p1_backtest,
    price_regression_backtest,
    price_residual_diagnostics,
    price_rolling_signals,
    price_stationarity,
)

ALGORITHMS = {
    algorithm.id: algorithm
    for algorithm in [
        example_correlation.ALGORITHM,
        example_backtest.ALGORITHM,
        price_forecast_backtest.ALGORITHM,
        price_baseline_backtest.ALGORITHM,
        price_regression_backtest.ALGORITHM,
        price_day_ahead_forecast.ALGORITHM,
        price_rolling_signals.ALGORITHM,
        price_driver_correlograms.ALGORITHM,
        price_residual_diagnostics.ALGORITHM,
        calendar_fourier_terms.ALGORITHM,
        price_p1_backtest.ALGORITHM,
        price_decomposition.ALGORITHM,
        price_stationarity.ALGORITHM,
    ]
}
