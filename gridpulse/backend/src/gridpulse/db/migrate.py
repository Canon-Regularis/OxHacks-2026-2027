"""Applies gridpulse/db/migrations/*.sql in filename order.

Migrations are written to be re-runnable, so there is no tracking table: every run applies
every file. Statements run one at a time in autocommit mode because TimescaleDB refuses to
create continuous aggregates inside a transaction.
"""

import re
from pathlib import Path

import psycopg


def split_statements(text: str) -> list[str]:
    """Split on a `;` that ends a line. Chunks containing only comments are dropped."""
    chunks = re.split(r";[ \t]*$", text, flags=re.MULTILINE)
    statements = []
    for chunk in chunks:
        code = [line for line in chunk.splitlines() if line.strip() and not line.strip().startswith("--")]
        if code:
            statements.append(chunk.strip())
    return statements


def run_migrations(conn: psycopg.Connection, directory: Path) -> list[str]:
    applied = []
    for path in sorted(directory.glob("*.sql")):
        for statement in split_statements(path.read_text()):
            conn.execute(statement)
        applied.append(path.name)
    return applied
