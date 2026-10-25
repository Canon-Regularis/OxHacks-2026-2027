"""Record API responses for the offline demo (GRIDPULSE_MODE=replay) without clicking through a UI.

Runs the API in-process in record mode against the configured database and requests what a demo
of one date range needs: catalog, snapshot, series, extremes, every algorithm's results, tables
and run list, the day-ahead forecast's day, and the period inspector for the top price periods.
Replay matches a request by its path and exact query values (in any order), so pass the
frontend's own requests with --urls once they are known. Commit the output directory to bake the
responses into the backend image (gridpulse/deploy/README.md).

    cd gridpulse && backend/.venv/bin/python scripts/record_demo.py --from 2026-10-17 --to 2026-10-23
    backend/.venv/bin/python scripts/record_demo.py --from ... --to ... --urls demo_urls.txt --out /tmp/demo
"""

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--from", dest="from_date", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="to_date", type=date.fromisoformat, required=True)
    parser.add_argument("--urls", type=Path, help="file of extra request paths, one per line, e.g. /api/series?ids=price&from=...")
    parser.add_argument("--out", type=Path, help="output directory (default: the app's DEMO_CACHE_DIR)")
    args = parser.parse_args()

    os.environ["GRIDPULSE_MODE"] = "record"
    if args.out:
        os.environ["DEMO_CACHE_DIR"] = str(args.out.resolve())

    from fastapi.testclient import TestClient  # after the environment is set: settings read it once

    from gridpulse.config import get_settings
    from gridpulse.main import create_app

    get_settings.cache_clear()
    out = get_settings().demo_cache_dir
    span = {"from": args.from_date.isoformat(), "to": args.to_date.isoformat()}
    recorded: list[str] = []
    failed: list[str] = []

    with TestClient(create_app()) as client:

        def get(path: str, **query: object) -> dict | None:
            response = client.get(path, params=query)
            label = str(response.request.url).removeprefix(str(client.base_url))
            (recorded if response.status_code == 200 else failed).append(label)
            if response.status_code != 200:
                print(f"  {response.status_code} {label}: {response.text[:160]}", file=sys.stderr)
                return None
            return response.json()

        catalog = get("/api/catalog")
        if catalog is None:
            sys.exit("The catalog failed; check the database settings.")
        get("/api/snapshot")
        series_ids = ",".join(s["id"] for s in catalog["series"])
        for ids in (series_ids, "price", "price,wind,demand"):
            get("/api/series", ids=ids, **span)
        for order in ("desc", "asc"):
            extremes = get("/api/extremes", order=order, **span)
            for row in (extremes or {}).get("rows", [])[:10]:
                get(f"/api/periods/{row['settlement_date']}/{row['settlement_period']}")

        for algorithm in catalog["algorithms"]:
            run = algorithm["latest_run"]
            if run is None:
                continue
            get("/api/analysis", algorithm=algorithm["id"], **span)
            get("/api/analysis/runs", algorithm=algorithm["id"])
            tables = get(f"/api/analysis/runs/{run['run_id']}/tables")
            day = ((tables or {}).get("tables", {}).get("forecast_day") or {}).get("rows")
            if day:  # the day-ahead forecast lies after the range: record its own day too
                forecast = {"from": day[0][0], "to": day[0][0]}
                get("/api/analysis", algorithm=algorithm["id"], **forecast)
                get("/api/series", ids="price", **forecast)

        for line in (args.urls.read_text().splitlines() if args.urls else []):
            if line.strip() and not line.startswith("#"):
                response = client.get(line.strip())
                (recorded if response.status_code == 200 else failed).append(line.strip())

    manifest = {
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "range": span,
        "data_updated_at_utc": catalog["data_updated_at_utc"],
        "runs": {a["id"]: a["latest_run"] and a["latest_run"]["run_id"] for a in catalog["algorithms"]},
        "requests": recorded,
        "failed": failed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Recorded {len(recorded)} responses in {out} ({len(failed)} failed); see manifest.json.")


if __name__ == "__main__":
    main()
