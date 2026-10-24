"""Builds the 30-minute input frame every algorithm receives (columns in contract.INPUT_COLUMNS).

The database selects the latest forecast per period (or the latest published by a cutoff) and
joins prices onto a complete period grid, so missing periods appear as NaN rather than gaps.
"""

from datetime import datetime

import pandas as pd
from psycopg import Connection

from gridpulse.analysis.contract import INPUT_COLUMNS
from gridpulse.db import queries
from gridpulse.settlement import settlement_period_of


def build_input_frame(
    conn: Connection, start: datetime, end: datetime, cutoff: datetime | None = None
) -> pd.DataFrame:
    return input_frame_from_rows(queries.input_rows(conn, start, end, cutoff))


def input_frame_from_rows(rows: list[tuple]) -> pd.DataFrame:
    """Turn database rows (columns as queries.INPUT_ROW_COLUMNS) into the algorithm input frame.

    Separate from the query so tests and notebooks can build frames without a database.
    """
    frame = pd.DataFrame(rows, columns=queries.INPUT_ROW_COLUMNS).set_index("delivery_start_utc")
    frame.index = pd.to_datetime(frame.index, utc=True).rename("delivery_start_utc")
    frame = frame.astype("float64")

    frame["residual_demand_mw"] = frame["demand_mw"] - frame["wind_mw"]
    periods = [settlement_period_of(t) for t in frame.index]
    frame["settlement_date"] = [d for d, _ in periods]
    frame["settlement_period"] = [p for _, p in periods]
    return frame[list(INPUT_COLUMNS)]
