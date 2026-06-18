"""Shared pytest fixtures for the molrs Python bindings.

Test data lives in a single, binding-neutral ``tests-data/`` directory at the
workspace root (gitignored, fetched via ``bash scripts/fetch-test-data.sh``).
The same copy is shared by the Rust suite (resolved through the
``molrs_testutil`` crate) and these Python tests, so there is no duplicated
fixture data inside ``molrs-python/``.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import molrs


def _resolve_tests_data_dir() -> Path:
    """Locate the shared ``tests-data/`` directory.

    Resolution order:
      1. ``MOLRS_TESTS_DATA`` environment variable, if set.
      2. The first ancestor of this file that contains a ``tests-data/`` child.
    """
    env = os.environ.get("MOLRS_TESTS_DATA")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_dir():
            return candidate
        raise RuntimeError(
            f"MOLRS_TESTS_DATA points to '{candidate}', which is not a directory. "
            "Fetch the shared data with `bash scripts/fetch-test-data.sh`."
        )

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "tests-data"
        if candidate.is_dir():
            return candidate

    raise RuntimeError(
        "Could not locate the shared 'tests-data/' directory by walking up from "
        f"{here}. Fetch it with `bash scripts/fetch-test-data.sh` (it is cloned "
        "to the workspace root), or set MOLRS_TESTS_DATA."
    )


@pytest.fixture(scope="session")
def tests_data_dir() -> Path:
    """Session-scoped path to the shared, binding-neutral ``tests-data/`` dir."""
    return _resolve_tests_data_dir()


@pytest.fixture(scope="session")
def pdb_dir(tests_data_dir: Path) -> Path:
    return tests_data_dir / "pdb"


@pytest.fixture(scope="session")
def xyz_dir(tests_data_dir: Path) -> Path:
    return tests_data_dir / "xyz"


@pytest.fixture(scope="session")
def gro_dir(tests_data_dir: Path) -> Path:
    return tests_data_dir / "gro"


@pytest.fixture(scope="session")
def dcd_dir(tests_data_dir: Path) -> Path:
    return tests_data_dir / "dcd"


@pytest.fixture(scope="session")
def trr_dir(tests_data_dir: Path) -> Path:
    return tests_data_dir / "trr"


@pytest.fixture(scope="session")
def xtc_dir(tests_data_dir: Path) -> Path:
    return tests_data_dir / "xtc"


@pytest.fixture(scope="session")
def lammps_dir(tests_data_dir: Path) -> Path:
    return tests_data_dir / "lammps"


@pytest.fixture
def cubic_box():
    """A 10x10x10 cubic box."""
    return molrs.Box.cube(10.0)


@pytest.fixture
def ortho_box():
    """A 5x10x15 orthorhombic box."""
    return molrs.Box.ortho(np.array([5.0, 10.0, 15.0], dtype=np.float64))


@pytest.fixture
def sample_points():
    """5 random points inside a 10x10x10 box."""
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
            [9.0, 8.0, 7.0],
            [0.1, 0.1, 0.1],
            [9.9, 9.9, 9.9],
        ],
        dtype=np.float64,
    )
