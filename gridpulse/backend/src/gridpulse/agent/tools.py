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
                    table["numeric_columns"] = [column for i, column in enumerate(table["columns"]) if any(type(row[i]) in (int, float) for row in table["rows"])]
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
        "widgets": {
            "chart": {"bindings": ["source series", "stored algorithm outputs"], "use": "Time-series lines on a wide horizontal chart; combine wind and demand (MW), price may use a second GBP/MWh axis."},
            "table": {"bindings": ["source series", "named stored algorithm table"], "use": "Raw source data rows with UTC timestamps and units, OR an exact named algorithm table. Source tables do not need an algorithm."},
            "comparison": {"bindings": ["named stored algorithm table"], "use": "Horizontal bars comparing a VERIFIED numeric metric column across categories. Never use for raw wind/demand series. Read tables first; select table and metric."},
            "inspector": {"bindings": ["linked selected settlement period"], "use": "Period details only after a chart point is selected; not a general summary card. Avoid adding an empty inspector."},
            "context": {"bindings": ["notes"], "use": "Concise editable notes with meaningful nonempty text, not a live statistic card."},
        },
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
    inspected_tables = {}
    for item in evidence:
        data = item["data"]
        if data.get("run") and data.get("tables") and isinstance(data["tables"], dict):
            inspected_tables[data["run"]["run_id"]] = data["tables"]
    for widget in view.widgets:
        if any(s not in SERIES for s in widget.series):
            raise ValueError("View references an unknown series.")
        if not widget.algorithm and (widget.table or any(output not in widget.series for output in widget.outputs)):
            raise ValueError("Source widgets use source IDs for outputs and no named algorithm table; leave table empty.")
        if widget.type in {"chart", "table"} and not widget.algorithm and not widget.series:
            raise ValueError(f"A {widget.type} needs source series or a valid algorithm binding.")
        if widget.type == "comparison" and not widget.algorithm:
            raise ValueError("Comparison widgets need a stored algorithm table, not raw source series. Use chart or table for wind/demand.")
        if widget.algorithm:
            algo = ALGORITHMS.get(widget.algorithm)
            if not algo or runs.get(widget.run_id) != widget.algorithm:
                raise ValueError("Inspect a stored run before displaying its results.")
            if any(output not in algo.outputs for output in widget.outputs) or (widget.table and widget.table not in algo.tables):
                raise ValueError("View references unknown algorithm outputs.")
            if widget.type == "chart" and not algo.outputs and not widget.series:
                raise ValueError("This algorithm has tables only; choose a named table widget instead of an empty chart.")
            if widget.type in {"table", "comparison"} and (not widget.table or widget.table not in algo.tables):
                raise ValueError("This algorithm widget requires an exact available table name; use chart for time-series outputs.")
            if widget.type == "comparison":
                table = inspected_tables.get(widget.run_id, {}).get(widget.table)
                if not table:
                    raise ValueError("Read the algorithm table before choosing a comparison metric.")
                numeric = table.get("numeric_columns") or [column for i, column in enumerate(table["columns"]) if any(type(row[i]) in (int, float) for row in table["rows"])]
                if not numeric or (widget.metric and widget.metric not in numeric):
                    raise ValueError("Comparison requires an available numeric metric column.")
                widget.metric = widget.metric or next((name for name in ("mase", "mae_gbp_mwh", "rmse_gbp_mwh", "score") if name in numeric), numeric[0])
                if widget.category and widget.category not in table["columns"]:
                    raise ValueError("Unknown comparison category column.")
                widget.category = widget.category or next((name for name in table["columns"] if name not in numeric), table["columns"][0])

        if widget.type == "chart":
            algo = ALGORITHMS.get(widget.algorithm) if widget.algorithm else None
            sources = widget.series if algo or not widget.outputs else [sid for sid in widget.series if sid in widget.outputs]
            units = {SERIES[sid].unit for sid in sources}
            if algo:
                units.update(algo.outputs[key] for key in (widget.outputs or list(algo.outputs)))
            if len(units) > 2:
                raise ValueError("A chart supports at most two distinct units; split this into aligned charts.")
