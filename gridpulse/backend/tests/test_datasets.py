"""Dataset lineage: which raw datasets each series, algorithm and run depends on."""

from datetime import UTC, datetime

import pytest

from gridpulse.analysis import service
from gridpulse.analysis.algorithms import ALGORITHMS
from gridpulse.analysis.contract import INPUT_COLUMNS
from gridpulse.catalog import SERIES, datasets_for


def test_every_series_names_its_datasets():
    for series in SERIES.values():
        assert series.datasets and all(name == name.strip() and name for name in series.datasets)


def test_datasets_for_columns():
    assert datasets_for(["wind_mw", "price_gbp_mwh"]) == ["elexon_system_prices", "elexon_windfor"]
    assert datasets_for(["residual_demand_mw"]) == ["elexon_ndf", "elexon_windfor"]
    assert datasets_for(["settlement_date", "settlement_period"]) == []  # from the clock


def test_every_data_input_column_maps_to_a_dataset():
    clock = {"settlement_date", "settlement_period"}
    for column in set(INPUT_COLUMNS) - clock:
        assert datasets_for([column]), f"{column} has no series naming its datasets"


@pytest.mark.parametrize("algorithm", ALGORITHMS.values(), ids=list(ALGORITHMS))
def test_algorithms_depend_on_their_inputs_datasets(algorithm):
    assert datasets_for(algorithm.inputs) == sorted(datasets_for(algorithm.inputs))
    assert datasets_for(algorithm.inputs) or set(algorithm.inputs) <= {"settlement_date", "settlement_period"}


def run(datasets):
    return {"run_id": 7, "algorithm_id": "wind_price_correlation", "datasets": datasets}


def test_run_datasets_falls_back_for_unrecorded_runs():
    algorithm = ALGORITHMS["wind_price_correlation"]
    recorded = {"elexon_windfor": "2026-10-24T14:17:00+00:00"}
    assert service.run_datasets(algorithm, run(recorded)) == recorded
    assert service.run_datasets(algorithm, run(None)) == {"elexon_system_prices": None, "elexon_windfor": None}
    assert service.run_datasets(None, run(None)) == {}


def test_reloaded_warnings(monkeypatch):
    loads = {
        "elexon_windfor": datetime(2026, 10, 24, 18, 0, tzinfo=UTC),  # reloaded after the run
        "elexon_system_prices": datetime(2026, 10, 24, 14, 17, tzinfo=UTC),  # unchanged
    }
    monkeypatch.setattr(service.queries, "dataset_loads", lambda conn, names: {n: loads.get(n) for n in names})
    used = {"elexon_windfor": "2026-10-24T14:17:00+00:00", "elexon_system_prices": "2026-10-24T14:17:00+00:00"}
    warnings = service.reloaded_warnings(None, run(used))
    assert len(warnings) == 1 and warnings[0].startswith("elexon_windfor was loaded again at 2026-10-24 18:00 UTC")
    assert service.reloaded_warnings(None, run(None)) == []
    assert service.reloaded_warnings(None, run({"elexon_windfor": None})) == []
