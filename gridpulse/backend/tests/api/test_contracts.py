"""The frontend builds against gridpulse/contracts/examples/; keep them parseable by the real schemas."""

import json

import pytest

from gridpulse.api import schemas
from gridpulse.config import APP_DIR

EXAMPLES_DIR = APP_DIR / "contracts" / "examples"

MODELS = {
    "health.json": schemas.Health,
    "catalog.json": schemas.Catalog,
    "snapshot.json": schemas.Snapshot,
    "series.json": schemas.SeriesResponse,
    "extremes.json": schemas.ExtremesResponse,
    "period.json": schemas.PeriodDetail,
    "analysis.json": schemas.AnalysisResponse,
    "analysis_tables.json": schemas.RunTables,
}


def load(name: str) -> dict:
    return json.loads((EXAMPLES_DIR / name).read_text())


@pytest.mark.parametrize("name", MODELS)
def test_example_parses(name):
    body = load(name)
    assert body.pop("_fixture"), "label every example as synthetic"
    MODELS[name].model_validate(body)


@pytest.mark.parametrize("name, columns_key", [("series.json", "series"), ("analysis.json", "indicators")])
def test_columnar_arrays_share_the_time_axis(name, columns_key):
    body = load(name)
    for column in body[columns_key].values():
        assert len(column["values"]) == len(body["t"])


def test_table_rows_match_columns():
    for table in load("analysis_tables.json")["tables"].values():
        assert all(len(row) == len(table["columns"]) for row in table["rows"])


def test_error_example_shape():
    assert set(load("error.json")["detail"]) == {"code", "message"}


def test_every_example_is_covered():
    assert {p.name for p in EXAMPLES_DIR.glob("*.json")} == set(MODELS) | {"error.json"}
