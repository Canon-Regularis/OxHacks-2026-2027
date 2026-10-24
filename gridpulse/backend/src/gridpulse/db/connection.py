"""Tiger Data (PostgreSQL + TimescaleDB) connections. Nothing connects at import time."""

import psycopg
from psycopg_pool import ConnectionPool


def open_pool(database_url: str, query_timeout_seconds: int) -> ConnectionPool:
    """Pool for API requests.

    Connections are autocommit (reads need no transaction) and carry a statement timeout,
    so one accidental unbounded query cannot tie up the database during the demo.
    """

    def configure(conn: psycopg.Connection) -> None:
        conn.execute("SET TIME ZONE 'UTC'")
        conn.execute(f"SET statement_timeout = '{int(query_timeout_seconds)}s'")

    pool = ConnectionPool(
        database_url,
        min_size=1,
        max_size=10,
        open=False,
        configure=configure,
        kwargs={"autocommit": True},
    )
    pool.open(wait=False)  # keep starting even if the database is down; /api/health reports it
    return pool


def connect(database_url: str) -> psycopg.Connection:
    """Direct connection for CLI jobs (migrations, imports, analysis runs). No statement timeout."""
    conn = psycopg.connect(database_url, autocommit=True)
    conn.execute("SET TIME ZONE 'UTC'")
    return conn
