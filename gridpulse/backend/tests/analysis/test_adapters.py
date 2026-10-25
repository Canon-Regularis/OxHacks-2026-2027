"""The P0 adapters beyond price_forecast_backtest, on complete synthetic data (fast settings)."""

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gridpulse.analysis.algorithms import (
    calendar_fourier_terms,
    price_baseline_backtest,
    price_day_ahead_forecast,
    price_driver_correlograms,
    price_regression_backtest,
    price_residual_diagnostics,
    price_rolling_signals,
)
from gridpulse.analysis.algorithms._backtest import WINDOW
from gridpulse.analysis.algorithms._frames import longest_complete
from gridpulse.analysis.contract import AlgorithmResult
from gridpulse.analysis.service import check_tables


def prices(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 80 + 30 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 8, n)


def frame_of(columns: dict[str, Any], start: str = "2026-09-01") -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    index = pd.date_range(start, periods=n, freq="30min", tz="UTC", name="delivery_start_utc")
    return pd.DataFrame(columns, index=index)


def series(result: AlgorithmResult) -> pd.DataFrame:
    assert result.series is not None
    return result.series


def assert_tables(module: Any, result: AlgorithmResult) -> None:
    check_tables(module.ALGORITHM, result.tables)
    assert {name: list(t.columns) for name, t in result.tables.items()} == module.TABLE_COLUMNS


# --- Backtests ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "window"),
    [(price_baseline_backtest, 672), (price_regression_backtest, 336)],
    ids=["baseline", "regression"],
)
def test_backtest_scores_every_model(module: Any, window: int) -> None:
    frame = frame_of({"price_gbp_mwh": prices(WINDOW + 48)})
    params = {**module.ALGORITHM.params, "window": window, "max_folds": 1}
    result = module.run(frame, **params)
    assert_tables(module, result)
    assert series(result).notna().all().all()
    folds = result.tables["folds"]
    assert set(folds["status"]) == {"ok"}
    assert list(folds["model"]) == list(module.MODELS)
    assert result.tables["metrics"]["mae_gbp_mwh"].notna().all()


def test_weekly_seasonal_naive_repeats_last_week() -> None:
    values = prices(WINDOW + 48)
    frame = frame_of({"price_gbp_mwh": values})
    params = {**price_baseline_backtest.ALGORITHM.params, "window": 672, "max_folds": 1}
    out = series(price_baseline_backtest.run(frame, **params))
    week = out["seasonal_naive_week_forecast"].to_numpy()
    assert week == pytest.approx(values[WINDOW - 336 : WINDOW - 336 + 48])
    assert out["naive_forecast"].to_numpy() == pytest.approx([values[WINDOW - 1]] * 48)


# --- Rolling signals ----------------------------------------------------------------------------


def test_rolling_signals_match_direct_calculations() -> None:
    values = prices(400)
    values[390] = np.nan
    frame = frame_of({"price_gbp_mwh": values})
    out = price_rolling_signals.run(frame)
    assert list(out.columns) == list(price_rolling_signals.OUTPUTS)
    t = 360
    window = values[t - 47 : t + 1]
    assert out["price_mean_48"].iloc[t] == pytest.approx(window.mean())
    assert out["price_std_48"].iloc[t] == pytest.approx(window.std(ddof=1))
    assert out["price_median_48"].iloc[t] == pytest.approx(np.median(window))
    assert out["price_rms_48"].iloc[t] == pytest.approx(math.sqrt((window**2).mean()))
    weights = np.arange(1, 49, dtype=float)
    assert out["price_weighted_mean_48"].iloc[t] == pytest.approx(
        (window * weights).sum() / weights.sum()
    )
    week = values[t - 335 : t + 1]
    assert out["price_p10_336"].iloc[t] == pytest.approx(np.quantile(week, 0.1))
    assert out["price_change_1"].iloc[t] == pytest.approx(values[t] - values[t - 1])
    assert out["price_change_48"].iloc[t] == pytest.approx(values[t] - values[t - 48])
    assert math.isnan(out["price_p10_336"].iloc[334])  # the 7-day window is not full yet
    windowed = [name for name in out.columns if not name.startswith(("price_change", "price_ewma"))]
    assert out[windowed].iloc[390:].isna().all().all()  # every 24-hour window touches the gap
    assert out["price_change_1"].iloc[390:392].isna().all()  # needs the missing price
    assert out["price_change_1"].iloc[392:].notna().all()
    assert math.isnan(out["price_ewma_48"].iloc[391])  # the EWMA restarts after the gap


# --- Fourier terms ------------------------------------------------------------------------------


def test_fourier_terms_repeat_the_clock_change_hour() -> None:
    frame = frame_of({"settlement_period": range(1, 51)}, start="2026-10-24T23:00Z")  # 25 Oct
    out = calendar_fourier_terms.run(frame)
    assert list(out.columns) == list(calendar_fourier_terms.OUTPUTS)
    assert len(out) == 50
    assert out.iloc[2].equals(out.iloc[4])  # 01:00 BST and 01:00 GMT share a slot
    assert out.iloc[3].equals(out.iloc[5])
    assert out["daily_cos_1"].iloc[0] == pytest.approx(1.0)  # local midnight


# --- Correlograms -------------------------------------------------------------------------------


def driver_frame(n: int, lead: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    wind = rng.normal(8000, 2000, n + lead)
    price = 200 - 0.01 * wind[:-lead] + rng.normal(0, 3, n)  # wind leads price by `lead`
    return frame_of(
        {
            "price_gbp_mwh": price,
            "residual_demand_mw": 30000 - wind[lead:] + rng.normal(0, 500, n),
            "wind_mw": wind[lead:],
        }
    )


def lag_table(result: AlgorithmResult, name: str) -> pd.DataFrame:
    return result.tables[name].set_index("lag_periods")


def test_correlograms_find_the_lead() -> None:
    frame = driver_frame(600)
    result = price_driver_correlograms.run(frame, max_lag=12, alpha=0.05)
    assert_tables(price_driver_correlograms, result)
    ccf = lag_table(result, "ccf_wind")
    assert list(ccf.index) == list(range(-12, 13))
    assert ccf["value"].abs().idxmax() == 4  # driver leads by 4 half-hours
    assert ccf.at[4, "value"] < 0
    assert len(lag_table(result, "acf_price")) == 12
    sample = result.tables["sample"].iloc[0]
    assert (sample["periods"], sample["excluded_periods"]) == (600, 0)


def test_correlograms_use_the_longest_gap_free_stretch() -> None:
    frame = driver_frame(600)
    frame.iloc[100, 2] = np.nan  # a gap in the wind forecast only
    result = price_driver_correlograms.run(frame, max_lag=12, alpha=0.05)
    sample = result.tables["sample"].iloc[0]
    assert sample["first_utc"] == frame.index[101]
    assert (sample["periods"], sample["excluded_periods"]) == (499, 101)
    short = price_driver_correlograms.run(frame.iloc[:20], max_lag=12, alpha=0.05)
    assert short.tables["acf_price"].empty
    assert short.tables["sample"].iloc[0]["periods"] == 20


def test_longest_complete_prefers_the_longest_then_earliest() -> None:
    frame = frame_of({"x": [1, np.nan, 1, 1, np.nan, 1, 1, np.nan]})
    assert list(longest_complete(frame).index) == list(frame.index[2:4])
    assert longest_complete(frame_of({"x": [np.nan, np.nan]})).empty


# --- Residual diagnostics -----------------------------------------------------------------------


def diagnostics_params(**overrides: Any) -> dict[str, Any]:
    return {**price_residual_diagnostics.ALGORITHM.params, "max_periods": 672, **overrides}


def test_residual_diagnostics_per_model() -> None:
    frame = frame_of({"price_gbp_mwh": prices(800)})
    result = price_residual_diagnostics.run(frame, **diagnostics_params())
    assert_tables(price_residual_diagnostics, result)
    box = result.tables["ljung_box"]
    assert box.groupby("model").size().to_dict() == {"arima": 48, "ets": 48}
    arima = box[box["model"] == "arima"].set_index("lag")
    assert arima.loc[1:3, "p_value"].isna().all()  # df = lag - model_df <= 0: not testable
    assert arima.loc[4:, "p_value"].notna().all()
    tests = result.tables["tests"]
    assert sorted(tests["test"].unique()) == ["arch_lm", "durbin_watson", "jarque_bera"]
    assert result.tables["sample"].iloc[0]["periods"] == 672  # the most recent max_periods


def test_residual_diagnostics_explain_models_they_cannot_test() -> None:
    frame = frame_of({"price_gbp_mwh": prices(120)})  # enough for ETS (96), not ARIMA (148)
    result = price_residual_diagnostics.run(frame, **diagnostics_params())
    tests = result.tables["tests"].set_index("model")
    assert tests.loc["arima", "test"] == "not_computed"
    assert "148" in tests.loc["arima", "notes"]
    assert set(tests.loc["ets", "test"]) == {"durbin_watson", "jarque_bera", "arch_lm"}
    assert set(result.tables["ljung_box"]["model"]) == {"ets"}


# --- Day-ahead forecast -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("end", "day", "periods"),
    [
        ("2026-10-23T23:00Z", "2026-10-24", 48),  # BST midnight
        ("2026-10-24T23:00Z", "2026-10-25", 50),  # clocks go back
        ("2026-03-28T00:00Z", "2026-03-28", 48),
        ("2026-03-29T00:00Z", "2026-03-29", 46),  # clocks go forward
        ("2026-10-24T12:00Z", "2026-10-24", 22),  # mid-day: the rest of the day
    ],
)
def test_forecast_day(end: str, day: str, periods: int) -> None:
    instant = datetime.fromisoformat(end.replace("Z", "+00:00")).astimezone(UTC)
    assert price_day_ahead_forecast.forecast_day(instant) == (day, periods)


def day_ahead_params() -> dict[str, Any]:
    models = price_day_ahead_forecast.ALGORITHM.params["models"]
    fast = models | {"auto_arima": {**models["auto_arima"], "max_p": 1, "max_q": 1}}
    return {"history_periods": 336, "models": fast}


def test_day_ahead_forecasts_the_next_settlement_day() -> None:
    values = prices(WINDOW)
    frame = frame_of({"price_gbp_mwh": values}, start="2026-09-25T23:00Z")  # ends 23 Oct 23:00Z
    result = price_day_ahead_forecast.run(frame, **day_ahead_params())
    assert_tables(price_day_ahead_forecast, result)
    out = series(result)
    assert out.index[0] == frame.index[-1] + pd.Timedelta(minutes=30)
    assert len(out) == 48
    assert out.notna().all().all()
    assert out["seasonal_naive_forecast"].to_numpy() == pytest.approx(values[-48:])
    day = result.tables["forecast_day"].iloc[0]
    assert (day["settlement_date"], day["periods"]) == ("2026-10-24", 48)
    assert day["history_first_utc"] == frame.index[-336]
    selected = result.tables["selected_models"]
    assert {"method", "alpha"} <= set(selected[selected["model"] == "auto_ets"]["parameter"])
    assert "order" in set(selected[selected["model"] == "auto_arima"]["parameter"])


def test_day_ahead_reports_models_it_cannot_fit() -> None:
    values = prices(WINDOW)
    values[-5] = np.nan
    frame = frame_of({"price_gbp_mwh": values}, start="2026-09-25T23:00Z")
    result = price_day_ahead_forecast.run(frame, **day_ahead_params())
    assert series(result).isna().all().all()
    warnings = result.tables["fit_warnings"]
    assert list(warnings["model"]) == list(price_day_ahead_forecast.MODELS)
    assert warnings["warning"].str.startswith("not forecast:").all()
