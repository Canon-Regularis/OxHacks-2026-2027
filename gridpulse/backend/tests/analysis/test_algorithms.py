"""Contract checks run against every registered algorithm. New algorithms are covered automatically."""

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from gridpulse.analysis.algorithms import ALGORITHMS
from gridpulse.analysis.contract import INPUT_COLUMNS, MAX_TABLE_ROWS, Algorithm
from gridpulse.analysis.service import AlgorithmError, check_output, check_tables, split_result, table_data

algorithms = pytest.mark.parametrize("algorithm", ALGORITHMS.values(), ids=list(ALGORITHMS))


def run(algorithm, input_frame):
    """Run as the service does: declared inputs only, default params, split into series and tables."""
    return split_result(algorithm, algorithm.run(input_frame[list(algorithm.inputs)].copy(), **algorithm.params))


def test_input_frame_matches_contract(input_frame):
    assert list(input_frame.columns) == list(INPUT_COLUMNS)
    assert str(input_frame.index.tz) == "UTC"
    assert input_frame["price_gbp_mwh"].isna().any()
    assert input_frame["settlement_period"].between(1, 50).all()


@algorithms
def test_declares_known_inputs_and_some_result(algorithm):
    assert set(algorithm.inputs) <= set(INPUT_COLUMNS)
    assert algorithm.outputs or algorithm.tables, "declare at least one output or table"


@algorithms
def test_output_honours_contract(algorithm, input_frame):
    frame = input_frame[list(algorithm.inputs)]
    before = frame.copy()
    start = frame.index[0] + algorithm.warmup
    end = frame.index[-1] + timedelta(minutes=30)

    series, tables = split_result(algorithm, algorithm.run(frame, **algorithm.params))
    series = check_output(algorithm, series, start, end + algorithm.horizon)
    stored_tables = check_tables(algorithm, tables)

    pd.testing.assert_frame_equal(frame, before)  # inputs are not mutated
    if algorithm.outputs:
        assert not series.empty
        assert series.index.min() >= start
    for data in stored_tables.values():
        json.dumps(data, allow_nan=False)  # what the API will serve


@algorithms
def test_is_deterministic(algorithm, input_frame):
    (series_a, tables_a), (series_b, tables_b) = run(algorithm, input_frame), run(algorithm, input_frame)
    if algorithm.outputs:
        pd.testing.assert_frame_equal(series_a, series_b)
    assert check_tables(algorithm, tables_a) == check_tables(algorithm, tables_b)


def test_check_output_rejects_undeclared_columns(input_frame):
    algorithm = ALGORITHMS["wind_price_correlation"]
    bad = pd.DataFrame({"surprise": 1.0}, index=input_frame.index)
    with pytest.raises(AlgorithmError, match="declared"):
        check_output(algorithm, bad, input_frame.index[0], input_frame.index[-1])


def test_check_output_rejects_naive_index(input_frame):
    algorithm = ALGORITHMS["wind_price_correlation"]
    naive = pd.DataFrame({name: 1.0 for name in algorithm.outputs}, index=input_frame.index.tz_localize(None))
    with pytest.raises(AlgorithmError, match="tz-aware"):
        check_output(algorithm, naive, input_frame.index[0], input_frame.index[-1])


# --- Tables -----------------------------------------------------------------------------------

TABLES_ONLY = Algorithm(
    id="acf_example",
    version="0",
    description="",
    run=lambda frame: None,
    inputs=("price_gbp_mwh",),
    outputs={},
    tables={"acf": "Autocorrelation by lag"},
)


def test_tables_only_algorithm_needs_no_series():
    assert check_output(TABLES_ONLY, None, pd.Timestamp("2026-10-01", tz="UTC"), pd.Timestamp("2026-10-02", tz="UTC")).empty


def test_undeclared_or_missing_tables_rejected():
    with pytest.raises(AlgorithmError, match="declared"):
        check_tables(TABLES_ONLY, {"pacf": pd.DataFrame()})
    with pytest.raises(AlgorithmError, match="declared"):
        check_tables(TABLES_ONLY, {})


def test_table_serialisation():
    acf = pd.DataFrame(
        {"acf": [1.0, np.nan, np.inf], "computed_at": pd.to_datetime(["2026-10-01T00:00Z"] * 3, utc=True)},
        index=pd.Index([0, 1, 2], name="lag"),
    )
    assert table_data(TABLES_ONLY, "acf", acf) == {
        "columns": ["lag", "acf", "computed_at"],  # a meaningful index is kept as a column
        "rows": [
            [0, 1.0, "2026-10-01T00:00:00Z"],
            [1, None, "2026-10-01T00:00:00Z"],  # NaN -> null
            [2, None, "2026-10-01T00:00:00Z"],  # inf -> null
        ],
    }


def test_table_rejects_naive_datetimes_and_oversized_tables():
    with pytest.raises(AlgorithmError, match="timezone-naive"):
        table_data(TABLES_ONLY, "acf", pd.DataFrame({"t": pd.to_datetime(["2026-10-01"])}))
    with pytest.raises(AlgorithmError, match="limit"):
        table_data(TABLES_ONLY, "acf", pd.DataFrame({"x": range(MAX_TABLE_ROWS + 1)}))


def test_backtest_forecasts_only_use_prices_settled_at_the_origin(input_frame):
    algorithm = ALGORITHMS["price_naive_backtest"]
    series, _ = run(algorithm, input_frame)
    horizon = algorithm.params["horizon_periods"]
    price = input_frame["price_gbp_mwh"]
    target = input_frame.index[400]
    assert series.at[target, "persistence_forecast"] == pytest.approx(price.iloc[400 - horizon - 1], nan_ok=True)
    assert series.at[target, "seasonal_naive_forecast"] == pytest.approx(price.iloc[400 - 336], nan_ok=True)
