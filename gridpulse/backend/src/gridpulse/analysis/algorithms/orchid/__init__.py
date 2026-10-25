"""Orchid modules vendored verbatim; call them only through the GridPulse adapters.

Do not edit the vendored files: fix Orchid upstream and re-vendor. Their upstream test
suites live in ``tests/analysis/orchid`` with only the import lines changed.
"""

from collections.abc import Mapping
from typing import Final

UPSTREAM_REPOSITORY: Final = "https://github.com/Orthant-Research-Group/orchid"
UPSTREAM_COMMIT: Final = "ca8fc9290722fb45ab27e022b76365beba1306c8"

# Vendored file -> (upstream path, SHA-256 of its LF-normalised UTF-8 content).
VENDORED_FILES: Final[Mapping[str, tuple[str, str]]] = {
    "autoregressive.py": (
        "algorithms/core/forecasting/autoregressive/autoregressive_template.py",
        "06c9f8ca6930e9e7bdf9f5e5830afa36bfbc6ad877e85209ab3374d7d5ffb1cd",
    ),
    "forecasting_baseline.py": (
        "algorithms/core/forecasting/baseline/forecasting_baseline_template.py",
        "46035fd31b454c5b829d332602bf0501050d8560d5e345b87c304e8c19961c9b",
    ),
    "smoothing.py": (
        "algorithms/core/forecasting/smoothing/smoothing_template.py",
        "8063a0800aeddda0e81fcff30058c0d6796da6930bab424b4a498a9ea7f4056a",
    ),
    "spectral.py": (
        "algorithms/core/spectral/spectral_template.py",
        "b3310282d9fc0a03c5cbdeba96e8b2ee80ed5f5589a01ed65b4089120ec8b5fd",
    ),
    "state_space.py": (
        "algorithms/core/forecasting/state_space/state_space_template.py",
        "541d66eba3f66eb2650a73ea8d9088487bcb63e8b032d21d629730407156f891",
    ),
}
