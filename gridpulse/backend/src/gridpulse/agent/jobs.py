"""Killable analysis workers. A disconnected turn cannot leave unbounded CPU work behind."""
import asyncio
import multiprocessing as mp
import os
import tempfile
import json
from pathlib import Path


def _worker(database_url, query_timeout, spec, output):
    import signal
    os.environ["GRIDPULSE_WORKERS"] = "1"  # parent deadline controls the entire job, no detached fit pools
    def cancelled(signum, frame):
        raise RuntimeError("Analysis cancelled")
    signal.signal(signal.SIGTERM, cancelled)
    # Spawned process owns its connection; no API pool or model credentials are passed.
    from gridpulse.analysis.algorithms import ALGORITHMS
    from gridpulse.analysis.service import run_algorithm
    from gridpulse.db.connection import connect
    from gridpulse.settlement import range_utc
    from datetime import date
    try:
        start, end = range_utc(date.fromisoformat(spec["from_date"]), date.fromisoformat(spec["to_date"]))
        with connect(database_url) as conn:
            conn.execute(f"SET statement_timeout = '{int(query_timeout)}s'")
            run = run_algorithm(conn, ALGORITHMS[spec["algorithm"]], start, end, spec["params"])
        result = {"run_id": run, "algorithm": spec["algorithm"]}
    except Exception:
        result = {"error": "Analysis failed; check input coverage and parameter requirements."}
    Path(output).write_text(json.dumps(result))


async def execute(settings, spec):
    fd, path = tempfile.mkstemp(prefix="eisengrau-job-", suffix=".json")
    os.close(fd)
    process = mp.get_context("spawn").Process(
        target=_worker,
        args=(settings.database_url, settings.query_timeout_seconds, spec.model_dump(mode="json"), path),
    )
    try:
        process.start()
        async with asyncio.timeout(settings.agent_job_timeout_seconds):
            while process.is_alive():
                await asyncio.sleep(0.1)
        process.join()
        return json.loads(Path(path).read_text())
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 2)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 2)
            process.close()
        Path(path).unlink(missing_ok=True)
