"""Sine and cosine terms of the Europe/London clock (P0 #6), for the Methods page.

They are the seasonal basis of CalendarRegression. The harmonics are part of the output names,
so they are fixed rather than parameters.
"""

from datetime import UTC, datetime

import pandas as pd

from gridpulse.analysis.algorithms.calendar import fourier_terms
from gridpulse.analysis.contract import Algorithm

DAILY_HARMONICS = 2
WEEKLY_HARMONICS = 1
_COLUMNS = fourier_terms(
    [datetime(2026, 1, 1, tzinfo=UTC)],
    daily_harmonics=DAILY_HARMONICS,
    weekly_harmonics=WEEKLY_HARMONICS,
).columns
OUTPUTS = dict.fromkeys(_COLUMNS, "")


def run(frame: pd.DataFrame) -> pd.DataFrame:
    terms = fourier_terms(
        list(frame.index.to_pydatetime()),
        daily_harmonics=DAILY_HARMONICS,
        weekly_harmonics=WEEKLY_HARMONICS,
    )
    return pd.DataFrame(list(terms.rows), index=frame.index, columns=list(terms.columns))


ALGORITHM = Algorithm(
    id="calendar_fourier_terms",
    version="0.1.0",
    description="Sine and cosine terms of each period's Europe/London clock time (daily and weekly "
    "harmonics), the seasonal basis behind the calendar regression.",
    run=run,
    inputs=("settlement_period",),  # only the index is used: the terms come from the clock
    outputs=OUTPUTS,
    limitations="Deterministic functions of the clock, not data. On a 50-period day the repeated "
    "hour reuses its slots, so two periods share values; on a 46-period day the skipped hour "
    "never occurs.",
)
