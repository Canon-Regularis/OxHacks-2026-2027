"""Bounded tools reuse the published API implementation and canonical runner."""
import json
import math
from fastapi import HTTPException
from gridpulse.analysis import service
from gridpulse.analysis.algorithms import ALGORITHMS
from gridpulse.api import market, system, analysis
from gridpulse.api.deps import DateRange
from gridpulse.catalog import SERIES
from gridpulse.settlement import range_utc

# Conservative initial automatic tier. Expensive or unfamiliar algorithms require approval.
AUTOMATIC = {"calendar_fourier_terms", "price_rolling_signals", "wind_price_correlation"}


def validate(spec, settings):
    if (spec.to_date - spec.from_date).days + 1 > min(settings.max_range_days, 366):
        raise ValueError("Agent analysis requests are limited to 366 days.")
    if spec.run_id is not None and spec.params:
        raise ValueError("Choose a stored run ID or parameter overrides, not both.")
    if spec.tool in {"runs", "analysis", "execute"}:
        if spec.algorithm not in ALGORITHMS:
            raise ValueError("Unknown algorithm; choose from the capability catalogue.")
        algo = ALGORITHMS[spec.algorithm]
        service.with_params(algo, spec.params)
        for key, value in spec.params.items():
            default = algo.params[key]
            if isinstance(default, bool):
                valid = type(value) is bool
            elif isinstance(default, (int, float)):
                valid = type(value) in (int, float) and math.isfinite(value) and abs(value) <= 10000
            elif isinstance(default, str):
                valid = isinstance(value, str) and len(value) <= 100
            elif isinstance(default, (list, tuple)):
                valid = isinstance(value, list) and len(value) <= 32 and all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 10000 for v in value)
            else:
                valid = value == default
            if not valid:
                raise ValueError(f"Invalid or unbounded parameter {key}.")
    if spec.tool == "series" and (not spec.ids or any(s not in SERIES for s in spec.ids)):
        raise ValueError("Unknown or empty source series selection.")


def read(pool, spec, settings):
    validate(spec, settings)
    if pool is None:
        return {"error": "Database unavailable; no results were retrieved."}
    start, end = range_utc(spec.from_date, spec.to_date)
    dates = DateRange(spec.from_date, spec.to_date, start, end)
    with pool.connection(timeout=5) as conn:
        try:
            if spec.tool == "coverage":
                result = system.catalog(conn)
            elif spec.tool == "series":
                # Model gets bounded evidence; UI loads its own higher-resolution data.
                result = market.series(dates, ",".join(spec.ids), "1d" if (spec.to_date-spec.from_date).days > 2 else "30m", None, conn)
            elif spec.tool == "runs":
                result = analysis.runs(ALGORITHMS[spec.algorithm], 10, conn)
            elif spec.tool == "analysis":
                result = analysis.analysis(analysis.Selection(ALGORITHMS[spec.algorithm], spec.run_id, spec.params), dates, "auto", conn)
            elif spec.tool == "tables":
                if spec.run_id is None:
                    raise ValueError("A run_id is required.")
                result = analysis.run_tables(spec.run_id, spec.table or None, conn)
            elif spec.tool == "period":
                result = market.period(spec.from_date, spec.period, conn)
            else:
                raise ValueError("Execution requires the job controller.")
            value = result.model_dump(mode="json", by_alias=True)
            if spec.tool == "analysis":
                value["indicators"] = {key: {"unit": col["unit"], "sample": col["values"][:24], "points": len(col["values"])} for key, col in value["indicators"].items()}
                value["t"] = value["t"][:24]
                value["sample_note"] = "Only first 24 points shown; not full-range statistics."
            if spec.tool == "tables":
                for table in value["tables"].values():
                    table["total_rows"] = len(table["rows"])
                    table["rows"] = table["rows"][:30]
            if len(json.dumps(value)) > 45000:
                return {"error": "Evidence too large; request a narrower range or specific table."}
            return value
        except HTTPException as exc:
            return {"error": exc.detail}


def capabilities():
    return {
        "version": 1,
        "series": [{"id": sid, "label": s.label, "unit": s.unit, "kind": s.kind} for sid, s in SERIES.items()],
        "algorithms": [analysis.algorithm_info(a).model_dump(mode="json") for a in ALGORITHMS.values()],
        "widgets": ["chart", "table", "comparison", "inspector", "context"],
    }


def validate_view(view, evidence):
    if view is None:
        return
    runs = {}
    for item in evidence:
        data = item["data"]
        run = data.get("run")
        if run:
            runs[run["run_id"]] = run["algorithm_id"]
        if "run_id" in data and "algorithm" in data:
            runs[data["run_id"]] = data["algorithm"]
        for run in data.get("runs", []):
            runs[run["run_id"]] = run["algorithm_id"]
    for widget in view.widgets:
        if any(s not in SERIES for s in widget.series):
            raise ValueError("View references an unknown series.")
        if widget.algorithm:
            algo = ALGORITHMS.get(widget.algorithm)
            if not algo or runs.get(widget.run_id) != widget.algorithm:
                raise ValueError("Inspect a stored run before displaying its results.")
            if any(output not in algo.outputs for output in widget.outputs) or (widget.table and widget.table not in algo.tables):
                raise ValueError("View references unknown algorithm outputs.")
