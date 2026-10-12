"""Every script with a shebang is executable, and nothing else claims to be.

The README and the tutorials invoke `examples/serve.sh`, `examples/run_sweep.sh`
and `examples/request.sh` directly, so a lost executable bit makes a documented
command fail with "Permission denied". It has been lost once already, silently,
to an in-place rewrite of the files.

The check is the invariant rather than a list, so a new script is covered the
day it is added: a file that starts with `#!` is meant to be run directly and
must be executable; a file without one must not advertise that it can be.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SEARCH_DIRS = ("examples", "benchmark", "thetalker")


def scripts() -> list[Path]:
    found = []
    for directory in SEARCH_DIRS:
        for path in sorted((REPO / directory).rglob("*")):
            if path.is_file() and path.suffix in (".sh", ".py"):
                found.append(path)
    return found


def has_shebang(path: Path) -> bool:
    with open(path, "rb") as handle:
        return handle.read(2) == b"#!"


@pytest.mark.parametrize("path", scripts(), ids=lambda p: str(p.relative_to(REPO)))
def test_shebang_matches_executable_bit(path):
    name = path.relative_to(REPO)
    executable = os.access(path, os.X_OK)
    if has_shebang(path):
        assert executable, (
            f"{name} starts with a shebang but is not executable; the docs invoke "
            f"it directly. Fix with: chmod +x {name} && git update-index --chmod=+x {name}"
        )
    else:
        assert not executable, (
            f"{name} is executable but has no shebang; either add one or "
            f"chmod -x {name} && git update-index --chmod=-x {name}"
        )


def test_the_documented_shell_examples_are_executable():
    """The three the README and tutorials call by path, named explicitly."""
    for name in ("serve.sh", "run_sweep.sh", "request.sh"):
        path = REPO / "examples" / name
        assert os.access(path, os.X_OK), f"examples/{name} is not executable"
