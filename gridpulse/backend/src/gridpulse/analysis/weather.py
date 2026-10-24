"""Descriptive weather/price change statistics. No training and no causal interpretation.

Correlations use up to 168 consecutive hourly slots ending at each point, counting only
finite paired differences. Reindex before differencing: gaps must not become multi-hour
changes. t is the completed hour END, so a 17:00 selection never includes 17:30 prices.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from gridpulse.api.schemas import WeatherPoint

WINDOW_HOURS = 168
MIN_PAIRS = 24

@dataclass(frozen=True)
class WeatherMetric:
    id: str
    unit: str

# Extend this registry with a schema/UI definition when adding another metric.
WEATHER_METRICS = (WeatherMetric("temperature", "°C"), WeatherMetric("pressure", "hPa"))


def _finite(value) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def weather_statistics(weather: list[dict], prices: list[dict], start: datetime, end: datetime) -> list[WeatherPoint]:
    index = pd.date_range(start - timedelta(hours=WINDOW_HOURS + 1), end, freq="h", inclusive="left")
    frame = pd.DataFrame(index=index)
    for rows, columns in [(weather, [m.id for m in WEATHER_METRICS]), (prices, ["price"])]:
        source = pd.DataFrame(rows)
        if not source.empty:
            source = source.set_index("bucket")
        for column in columns:
            frame[column] = source[column].reindex(index) if not source.empty else np.nan
    differences = frame.diff()
    output = []
    for i, bucket in enumerate(index):
        completed_at = bucket.to_pydatetime() + timedelta(hours=1)
        if completed_at < start or completed_at >= end:
            continue
        values = {"t": completed_at, "price_change": _finite(differences.iloc[i]["price"])}
        # At most 168 hourly differences, including the one ending at completed_at.
        history = differences.iloc[max(0, i - WINDOW_HOURS + 1): i + 1]
        for metric in WEATHER_METRICS:
            paired = history[["price", metric.id]].replace([np.inf, -np.inf], np.nan).dropna()
            enough = len(paired) >= MIN_PAIRS and (paired.std(ddof=0) > 1e-12).all()
            correlation = paired["price"].corr(paired[metric.id]) if enough else np.nan
            values.update({metric.id: _finite(frame.iloc[i][metric.id]),
                           f"{metric.id}_change": _finite(differences.iloc[i][metric.id]),
                           f"{metric.id}_correlation": _finite(correlation),
                           f"{metric.id}_samples": len(paired)})
        output.append(WeatherPoint(**values))
    return output
