import hashlib
from importlib import resources

import pytest

from gridpulse.analysis.algorithms import orchid


def test_every_vendored_file_is_recorded() -> None:
    package = resources.files(orchid)
    vendored = {entry.name for entry in package.iterdir() if entry.name.endswith(".py")}
    assert vendored - {"__init__.py"} == set(orchid.VENDORED_FILES)


@pytest.mark.parametrize("filename", sorted(orchid.VENDORED_FILES))
def test_vendored_file_matches_upstream(filename: str) -> None:
    # read_text normalises CRLF checkouts back to upstream's LF bytes.
    text = resources.files(orchid).joinpath(filename).read_text(encoding="utf-8")
    _, digest = orchid.VENDORED_FILES[filename]
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest
