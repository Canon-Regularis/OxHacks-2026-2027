import pytest

from gridpulse.catalog import FORECAST_VINTAGES, SERIES
from gridpulse.config import MIGRATIONS_DIR
from gridpulse.db.migrate import split_statements

MIGRATIONS = sorted(MIGRATIONS_DIR.glob("*.sql"))


def test_split_statements_drops_comment_only_chunks():
    text = "-- header\nCREATE TABLE a (x int);\n\n-- note\nSELECT 1;\n-- trailing comment\n"
    assert split_statements(text) == ["-- header\nCREATE TABLE a (x int)", "-- note\nSELECT 1"]


@pytest.mark.parametrize("path", MIGRATIONS, ids=lambda p: p.name)
def test_migrations_split_into_single_statements(path):
    statements = split_statements(path.read_text())
    assert statements
    for statement in statements:
        code = "\n".join(line for line in statement.splitlines() if not line.strip().startswith("--"))
        assert ";" not in code, f"{path.name}: put each statement's `;` at the end of a line"


def test_every_catalog_view_is_created_by_a_migration():
    sql = "\n".join(path.read_text() for path in MIGRATIONS)
    views = {s.relation for s in SERIES.values()} | {view for view, _ in FORECAST_VINTAGES.values()}
    for view in views:
        assert f"VIEW gridpulse.{view} AS" in sql
