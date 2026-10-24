"""Pure calculations and Orchid adapters. Accept prepared data; do not access HTTP or the database.

To add an algorithm: create a module here that defines `ALGORITHM = Algorithm(...)` (see
analysis/contract.py and example_correlation.py), then add it to the list below. The tests in
tests/analysis/ run every registered algorithm against the contract automatically.
"""

from gridpulse.analysis.algorithms import example_backtest, example_correlation, price_forecast_backtest

ALGORITHMS = {
    algorithm.id: algorithm
    for algorithm in [example_correlation.ALGORITHM, example_backtest.ALGORITHM, price_forecast_backtest.ALGORITHM]
}
