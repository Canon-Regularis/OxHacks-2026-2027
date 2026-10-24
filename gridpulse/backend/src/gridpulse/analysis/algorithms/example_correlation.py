"""Reference algorithm: trailing correlation between the wind forecast and the imbalance price.

It proves the import -> analyse -> API path end to end and shows the contract in
analysis/contract.py. Keep, replace or delete it once the real algorithms land.
"""

from datetime import timedelta

import pandas as pd

from gridpulse.analysis.contract import Algorithm


def run(frame: pd.DataFrame, *, window_periods: int, min_periods: int) -> pd.DataFrame:
    corr = frame["wind_mw"].rolling(window_periods, min_periods=min_periods).corr(frame["price_gbp_mwh"])
    return corr.to_frame("wind_price_corr")


ALGORITHM = Algorithm(
    id="wind_price_correlation",
    version="0.1.0",
    description="Trailing 7-day Pearson correlation between the latest wind forecast and the imbalance price.",
    run=run,
    inputs=("wind_mw", "price_gbp_mwh"),
    outputs={"wind_price_corr": ""},
    params={"window_periods": 336, "min_periods": 240},
    param_descriptions={
        "window_periods": "Trailing window length in settlement periods (336 = 7 days)",
        "min_periods": "Fewest non-missing periods in the window for a value; fewer gives null",
    },
    warmup=timedelta(days=7),  # keep equal to window_periods x 30 min
    limitations="Descriptive statistic over past periods. Not a forecast, a probability or evidence of cause.",
)
