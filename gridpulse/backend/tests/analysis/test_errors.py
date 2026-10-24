import copy
import pickle
from collections.abc import Callable

import pytest

from gridpulse.analysis.algorithms.errors import (
    AlgorithmError,
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    MissingDataError,
)


@pytest.mark.parametrize(
    "make",
    [
        lambda: MissingDataError(range(12)),
        lambda: InsufficientDataError(100, 7),
        lambda: InsufficientDataError(31, 4, what="residuals"),
        lambda: DegenerateDataError("constant"),
        lambda: CalculationError("failed"),
    ],
    ids=["missing", "insufficient", "insufficient-what", "degenerate", "calculation"],
)
def test_errors_survive_pickling_and_copying(make: Callable[[], AlgorithmError]) -> None:
    # Worker processes pickle the errors they raise, e.g. in a process-pool batch job.
    error = make()
    error.add_note("fold 3")
    for clone in (pickle.loads(pickle.dumps(error)), copy.copy(error)):
        assert type(clone) is type(error)
        assert str(clone) == str(error)
        assert vars(clone) == vars(error)
