"""Independent fits run in a bounded process pool.

Orchid is pure Python, so threads would share one interpreter lock; processes do not. Every
task is a deterministic function of its arguments, so results are the same in any order and
with any number of workers. ``GRIDPULSE_WORKERS`` sets the number of processes (1 runs every
task in this process); by default it is the CPUs this process may use, at most
``DEFAULT_MAX_WORKERS``, since `gridpulse refresh` shares a small VM with the API.
"""

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import TypeVar

from gridpulse.analysis.algorithms.errors import AlgorithmError
from gridpulse.analysis.algorithms.forecasting import DEFAULT_LEVELS, Forecast, Forecaster
from gridpulse.analysis.algorithms.series import HalfHourlySeries

WORKERS_ENV = "GRIDPULSE_WORKERS"
DEFAULT_MAX_WORKERS = 4

_Task = TypeVar("_Task")
_Result = TypeVar("_Result")


def workers() -> int:
    """Processes ``run_all`` may use."""
    configured = os.environ.get(WORKERS_ENV, "").strip()
    if not configured:
        return min(_cpus(), DEFAULT_MAX_WORKERS)
    try:
        count = int(configured)
    except ValueError:
        raise ValueError(f"{WORKERS_ENV} must be a whole number, got {configured!r}") from None
    if count < 1:
        raise ValueError(f"{WORKERS_ENV} must be at least 1, got {count}")
    return count


def run_all(function: Callable[[_Task], _Result], tasks: Sequence[_Task]) -> list[_Result]:
    """``[function(task) for task in tasks]``, spread over ``workers()`` processes.

    ``function`` and the tasks must pickle, so ``function`` is a module-level function or a
    ``functools.partial`` of one. An exception from a task propagates, as in the loop.
    """
    count = min(workers(), len(tasks))
    if count <= 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(count, mp_context=_context()) as pool:
        return list(pool.map(function, tasks))


def forecast_all(
    forecasters: Sequence[Forecaster],
    history: HalfHourlySeries,
    horizon: int,
    *,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> list[Forecast | AlgorithmError]:
    """Each forecaster's forecast from ``history``, fitted side by side, or the
    ``AlgorithmError`` that stopped it; other exceptions propagate."""
    fit = partial(_forecast, history=history, horizon=horizon, levels=tuple(levels))
    return run_all(fit, forecasters)


def _forecast(
    forecaster: Forecaster,
    *,
    history: HalfHourlySeries,
    horizon: int,
    levels: tuple[float, ...],
) -> Forecast | AlgorithmError:
    try:
        return forecaster.forecast(history, horizon, levels=levels)
    except AlgorithmError as exc:
        return exc


def _cpus() -> int:
    if hasattr(os, "sched_getaffinity"):  # Linux: honours the container's CPU set
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _context() -> multiprocessing.context.BaseContext:
    """A fork server where there is one, else spawn: never a bare fork, which would copy the
    caller's database connection and threads into each worker."""
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")
        context.set_forkserver_preload(["gridpulse.analysis.algorithms.forecasting"])
        return context
    return multiprocessing.get_context("spawn")
