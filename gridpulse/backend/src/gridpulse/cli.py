"""Batch commands. Installed as `gridpulse`; run `gridpulse --help`.

    gridpulse migrate
    gridpulse analyse wind_price_correlation --from 2026-09-01 --to 2026-10-23
    gridpulse algorithms
    gridpulse refresh                     rerun whatever is stale, up to the latest settled day
"""

import argparse
import json
import sys
from datetime import date, datetime

from gridpulse.analysis.algorithms import ALGORITHMS
from gridpulse.analysis.refresh import refresh
from gridpulse.analysis.service import run_algorithm
from gridpulse.catalog import datasets_for
from gridpulse.config import MIGRATIONS_DIR, get_settings
from gridpulse.db.connection import connect
from gridpulse.db.migrate import run_migrations
from gridpulse.settlement import range_utc


def _connect():
    url = get_settings().database_url
    if not url:
        sys.exit("DATABASE_URL is not set (see gridpulse/.env.example).")
    return connect(url)


def _aware(value: str) -> datetime:
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None:
        raise argparse.ArgumentTypeError("give an explicit offset, e.g. 2026-10-24T12:00Z")
    return instant


def _param(value: str) -> tuple[str, object]:
    name, _, raw = value.partition("=")
    try:
        return name, json.loads(raw)
    except json.JSONDecodeError:
        return name, raw


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gridpulse", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("migrate", help="apply gridpulse/db/migrations (safe to repeat)")

    p = commands.add_parser("analyse", help="run an algorithm and store its results")
    p.add_argument("algorithm", choices=ALGORITHMS)
    p.add_argument("--from", dest="from_date", type=date.fromisoformat, required=True)
    p.add_argument("--to", dest="to_date", type=date.fromisoformat, required=True)
    p.add_argument("--cutoff", type=_aware, help="use only data available at this instant")
    p.add_argument("--param", type=_param, action="append", default=[], help="override a parameter: name=value")
    p.add_argument("--no-memo", action="store_true", help="refit everything instead of reusing stored backtest folds")

    commands.add_parser("algorithms", help="list registered algorithms")

    p = commands.add_parser("refresh", help="rerun algorithms whose results are missing or stale")
    p.add_argument("algorithms", nargs="*", metavar="ALGORITHM", help="algorithm ids (default: all)")
    p.add_argument("--through", type=date.fromisoformat, help="last settlement day (default: latest complete)")
    p.add_argument("--force", action="store_true", help="rerun even if up to date, refitting everything")

    args = parser.parse_args(argv)

    if args.command == "algorithms":
        for a in ALGORITHMS.values():
            print(f"{a.id} v{a.version}: {a.description}\n  inputs={list(a.inputs)} datasets={datasets_for(a.inputs)}")
            print(f"  outputs={a.outputs} params={a.params}")

    elif args.command == "migrate":
        with _connect() as conn:
            for name in run_migrations(conn, MIGRATIONS_DIR):
                print(f"applied {name}")

    elif args.command == "analyse":
        start, end = range_utc(args.from_date, args.to_date)
        with _connect() as conn:
            run_id = run_algorithm(
                conn, ALGORITHMS[args.algorithm], start, end, dict(args.param), args.cutoff, reuse=not args.no_memo
            )
        print(f"run {run_id} succeeded")

    elif args.command == "refresh":
        unknown = sorted(set(args.algorithms) - set(ALGORITHMS))
        if unknown:
            parser.error(f"unknown algorithms {unknown}; choose from {list(ALGORITHMS)}")
        chosen = [ALGORITHMS[a] for a in args.algorithms] or list(ALGORITHMS.values())
        with _connect() as conn:
            outcomes = refresh(conn, chosen, args.through, args.force)
        for o in outcomes:
            run = f"run {o.run_id}" if o.run_id else ""
            print(f"{o.algorithm_id:28} {o.from_date}..{o.to_date}  {o.status:9} {run:8} {o.seconds:6.1f}s  {o.detail}")
        if any(o.status == "failed" for o in outcomes):
            sys.exit(1)


if __name__ == "__main__":
    main()
