import numpy as np
import pandas as pd
import pytest

from gridpulse.analysis.features import input_frame_from_rows


@pytest.fixture
def input_frame() -> pd.DataFrame:
    """Two synthetic weeks shaped exactly like build_input_frame output, with some gaps."""
    index = pd.date_range("2026-10-10", periods=48 * 14, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    n = len(index)
    price = rng.normal(80, 40, n)
    price[::50] = np.nan  # missing periods must stay missing
    rows = zip(
        index.to_pydatetime(),
        price,
        rng.normal(0, 300, n),  # niv_mwh
        rng.uniform(1_000, 15_000, n),  # wind_mw
        rng.uniform(1_000, 15_000, n),  # wind_outturn_mw
        rng.uniform(20_000, 40_000, n),  # demand_mw
        rng.uniform(20_000, 40_000, n),  # demand_outturn_mw
    )
    return input_frame_from_rows(list(rows))
