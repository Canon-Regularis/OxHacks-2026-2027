"""Numbers for the maths slides, on the deck's own sample (1 Sep-23 Oct 2026, 53 midnight origins).

Runs origin/main's registered algorithms to reproduce MODELLING_NOTES, then the engine-only
adapters (Diebold-Mariano, GARCH, Kalman filter/smoother) on the same window.
"""

import csv
import json
import math
import statistics as st
import sys
import time
from datetime import datetime

import pandas as pd

from gridpulse.analysis.algorithms import (
    diagnostics,
    evaluation,
    forecasting as fc,
    price_baseline_backtest,
    price_forecast_backtest,
    price_residual_diagnostics,
    state_space,
    volatility,
)
from gridpulse.analysis.algorithms.series import HalfHourlySeries

WARMUP, SAMPLE, H = 1344, 2544, 48


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    index = pd.to_datetime([r["start_utc"] for r in rows], utc=True).rename("delivery_start_utc")
    return pd.DataFrame({"price_gbp_mwh": [float(r["price"]) for r in rows]}, index=index)


def main(data_dir, out_path):
    frame = load(f"{data_dir}/system_prices.csv")
    first = pd.Timestamp("2026-08-31T23:00Z")  # 1 Sep 00:00 BST, the deck's first origin
    frame = frame.loc[first - pd.Timedelta(minutes=30 * WARMUP): pd.Timestamp("2026-10-23T22:30Z")]
    assert len(frame) == WARMUP + SAMPLE and frame["price_gbp_mwh"].notna().all(), len(frame)
    sample = frame.iloc[WARMUP:]
    out = {"frame": [str(frame.index[0]), str(frame.index[-1]), len(frame)]}

    # 1. Reproduce the team's backtests with the registered algorithms.
    tables = {}
    for alg in (price_forecast_backtest.ALGORITHM, price_baseline_backtest.ALGORITHM):
        t0 = time.perf_counter()
        result = alg.run(frame[list(alg.inputs)].copy(), **alg.params)
        tables[alg.id] = result.tables
        print(alg.id, f"{time.perf_counter() - t0:.0f}s", flush=True)
    metrics = pd.concat([t["metrics"] for t in tables.values()])
    by_h = pd.concat([t["by_horizon"] for t in tables.values()])
    folds = pd.concat([t["folds"] for t in tables.values()])
    keep = ["model", "n_ok_folds", "mae_gbp_mwh", "mase", "coverage_80", "coverage_95", "interval_score_80"]
    out["metrics"] = metrics[keep].round(3).to_dict("records")
    step = lambda m, s: float(by_h[(by_h.model == m) & (by_h.horizon_step == s)].mae_gbp_mwh.iloc[0])
    out["check"] = {"naive_30min": step("naive", 1), "mean_24h": step("mean", 48),
                    "ets_30min": step("ets", 1), "ets_24h": step("ets", 48)}
    naive_h = by_h[by_h.model == "naive"].set_index("horizon_step")
    mean_h = by_h[by_h.model == "mean"].set_index("horizon_step")
    flips = [int(s) for s in naive_h.index if mean_h.mae_gbp_mwh[s] < naive_h.mae_gbp_mwh[s]]
    out["mean_beats_naive_first_step"] = flips[0] if flips else None
    out["mean_beats_naive_steps"] = flips

    # 2. Diebold-Mariano on per-day MASE (same scale as the MASE shown), ETS vs runner-up.
    prices = frame["price_gbp_mwh"].tolist()
    ok = folds[folds.status == "ok"]
    by_origin = {m: ok[ok.model == m].set_index("origin_utc").mae_gbp_mwh for m in ("ets", "naive2", "naive", "mean")}
    origins = sorted(set(by_origin["ets"].index) & set(by_origin["naive2"].index))
    def scale(origin):
        i = frame.index.get_loc(pd.Timestamp(origin))
        train = prices[i - WARMUP:i]
        return st.fmean(abs(train[t] - train[t - 48]) for t in range(48, len(train)))
    s = {o: scale(o) for o in origins}
    mase = {m: [by_origin[m][o] / s[o] for o in origins] for m in ("ets", "naive2")}
    d = [a - b for a, b in zip(mase["ets"], mase["naive2"])]
    dm1 = evaluation.diebold_mariano(mase["ets"], mase["naive2"], horizon=1, loss="absolute")
    dm2 = evaluation.diebold_mariano(mase["ets"], mase["naive2"], horizon=2, loss="absolute")
    raw = evaluation.diebold_mariano([by_origin["ets"][o] for o in origins], [by_origin["naive2"][o] for o in origins], horizon=1, loss="absolute")
    dbar = st.fmean(d)
    lag1 = sum((d[i] - dbar) * (d[i - 1] - dbar) for i in range(1, len(d))) / sum((x - dbar) ** 2 for x in d)
    out["dm"] = {"n": len(origins), "mean_mase_ets": st.fmean(mase["ets"]), "mean_mase_naive2": st.fmean(mase["naive2"]),
                 "h1": [dm1.statistic, dm1.p_value], "h2": [dm2.statistic, dm2.p_value, list(dm2.warnings)],
                 "raw_mae_h1": [raw.statistic, raw.p_value], "d_lag1_acf": lag1,
                 "days_ets_better": sum(x < 0 for x in d)}

    # 3. ACF and the rho > 1/2 rule.
    series = HalfHourlySeries(sample.index[0].to_pydatetime(), sample["price_gbp_mwh"].tolist())
    acf = diagnostics.autocorrelation(series, max_lag=48)
    out["acf"] = [round(v, 4) for v in acf.values]
    below = [lag for lag, v in zip(acf.lags, acf.values) if v < 0.5]
    out["acf_first_below_half"] = below[0] if below else None

    # 4. Residual tests (registered algorithm), then GARCH on raw and slot-scaled ETS residuals.
    rd_alg = price_residual_diagnostics.ALGORITHM
    rd = rd_alg.run(sample[list(rd_alg.inputs)].copy(), **rd_alg.params)
    out["residual_tests"] = rd.tables["tests"].round(4).to_dict("records")
    resid = list(fc.ETS(seasonal="additive", season_length=48).forecast(series, 1).residuals)
    offset = SAMPLE - len(resid)
    slots = [[] for _ in range(48)]
    for i, r in enumerate(resid):
        slots[(offset + i) % 48].append(r)
    sd = [st.pstdev(v) for v in slots]
    scaled = [r / sd[(offset + i) % 48] for i, r in enumerate(resid)]
    for name, values in (("raw", resid), ("slot_scaled", scaled)):
        diag = diagnostics.residual_diagnostics(values, ljung_box_lags=48, arch_lags=12)
        g = volatility.garch(values, horizon=48)
        p = g.persistence
        out[f"garch_{name}"] = {"arch_lm": diag.arch_lm.statistic, "excess_kurtosis": diag.jarque_bera.excess_kurtosis,
                                "alpha1": g.params["alpha1"], "beta1": g.params["beta1"], "persistence": p,
                                "half_life_periods": math.log(2) / -math.log(p) if 0 < p < 1 else None,
                                "warnings": list(g.warnings)}
        print("garch", name, out[f"garch_{name}"], flush=True)

    # 5. Kalman filter and disturbance smoother: level + daily cycle (3 harmonics).
    t0 = time.perf_counter()
    model = state_space.StructuralModel(trend="level", seasons=((48, 3),))
    fit = state_space.fit_structural(series, model)
    filt = state_space.filter_structural(series, fit.parameters)
    smooth = state_space.smooth_structural(series, fit.parameters)
    z = [v for v in filt.standardized_innovation.values if v is not None]
    outl = smooth.outlier_score.values
    lvl = smooth.level_break_score.values
    def events(values, k=6):
        top = sorted((i for i, v in enumerate(values) if v is not None), key=lambda i: -abs(values[i]))[:k]
        return [[str(sample.index[i]), round(values[i], 2), sample["price_gbp_mwh"].iloc[i]] for i in top]
    out["kalman"] = {"seconds": round(time.perf_counter() - t0), "variances": dict(fit.parameters.variances),
                     "share_abs_z_gt_3": sum(abs(v) > 3 for v in z) / len(z), "n_z": len(z),
                     "spikes_abs_gt_3": sum(v is not None and abs(v) > 3 for v in outl),
                     "shifts_abs_gt_3": sum(v is not None and abs(v) > 3 for v in lvl),
                     "top_spikes": events(outl), "top_shifts": events(lvl)}
    print(json.dumps(out, indent=1, default=str))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
