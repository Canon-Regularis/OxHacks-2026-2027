"""The frontend builds against gridpulse/contracts/examples/; keep them parseable by the real schemas."""

import json

import pytest

from gridpulse.analysis.algorithms import ALGORITHMS
from gridpulse.api import schemas
from gridpulse.api.analysis import algorithm_info
from gridpulse.catalog import datasets_for
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
    "analysis_runs.json": schemas.RunList,
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


# Planned algorithm adapters: algorithms/<id>.analysis.json and, if it declares tables, <id>.tables.json.
ALGORITHM_EXAMPLES = sorted((EXAMPLES_DIR / "algorithms").glob("*.analysis.json"))


@pytest.mark.parametrize("path", ALGORITHM_EXAMPLES, ids=lambda p: p.name)
def test_algorithm_example(path):
    body = json.loads(path.read_text())
    assert body.pop("_fixture"), "label every example as synthetic"
    analysis = schemas.AnalysisResponse.model_validate(body)
    algorithm_id = path.name.removesuffix(".analysis.json")
    assert analysis.algorithm.id == analysis.run.algorithm_id == algorithm_id
    assert set(analysis.indicators) == set(analysis.algorithm.outputs)
    assert all(len(column.values) == len(analysis.t) for column in analysis.indicators.values())
    assert analysis.tables == list(analysis.algorithm.tables)
    assert set(analysis.algorithm.param_descriptions) == set(analysis.algorithm.params)
    assert analysis.run.params == analysis.algorithm.params and analysis.run.overrides == {}
    assert analysis.algorithm.datasets == datasets_for(analysis.algorithm.inputs)
    assert list(analysis.run.datasets) == analysis.algorithm.datasets

    tables_path = path.with_name(f"{algorithm_id}.tables.json")
    if not analysis.tables:
        assert not tables_path.exists()
        return
    body = json.loads(tables_path.read_text())
    assert body.pop("_fixture"), "label every example as synthetic"
    tables = schemas.RunTables.model_validate(body)
    assert tables.run == analysis.run
    assert list(tables.tables) == analysis.tables
    for name, table in tables.tables.items():
        assert table.description == analysis.algorithm.tables[name]
        assert all(len(row) == len(table.columns) for row in table.rows)


def test_examples_of_implemented_algorithms_match_the_registry():
    implemented = 0
    for path in ALGORITHM_EXAMPLES:
        example = json.loads(path.read_text())["algorithm"]
        algorithm = ALGORITHMS.get(example["id"])
        if algorithm is None:
            continue  # planned, not built yet
        served = algorithm_info(algorithm).model_dump(mode="json", exclude={"latest_run"})
        assert {k: v for k, v in example.items() if k != "latest_run"} == served, example["id"]
        implemented += 1
    assert implemented, "at least one planned algorithm is implemented"


def test_every_algorithm_example_is_paired():
    names = {p.name for p in (EXAMPLES_DIR / "algorithms").glob("*.json")}
    ids = {p.name.removesuffix(".analysis.json") for p in ALGORITHM_EXAMPLES}
    assert names <= {f"{i}.analysis.json" for i in ids} | {f"{i}.tables.json" for i in ids}


def test_every_param_is_described():
    algorithms = load("catalog.json")["algorithms"] + [load("analysis_runs.json")["algorithm"]]
    for algorithm in algorithms:
        assert set(algorithm["param_descriptions"]) == set(algorithm["params"]), algorithm["id"]


def test_run_list_marks_overrides():
    body = load("analysis_runs.json")
    defaults = body["algorithm"]["params"]
    for run in body["runs"]:
        assert run["overrides"] == {k: v for k, v in run["params"].items() if defaults.get(k) != v}


def test_dataset_lineage_is_consistent():
    catalog = load("catalog.json")
    loaded = {d["dataset"] for d in catalog["datasets"]}
    for series in catalog["series"]:
        assert set(series["datasets"]) <= loaded, series["id"]
    algorithms = catalog["algorithms"] + [load("analysis.json")["algorithm"], load("analysis_runs.json")["algorithm"]]
    for algorithm in algorithms:
        assert algorithm["datasets"] == datasets_for(algorithm["inputs"]), algorithm["id"]
        if algorithm.get("latest_run"):
            assert list(algorithm["latest_run"]["datasets"]) == algorithm["datasets"]


def test_error_example_shape():
    assert set(load("error.json")["detail"]) == {"code", "message"}


def test_every_example_is_covered():
    assert {p.name for p in EXAMPLES_DIR.glob("*.json")} == set(MODELS) | {"error.json"}
