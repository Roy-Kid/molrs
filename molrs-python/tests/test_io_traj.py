"""Tests for the molpy-compatible trajectory readers in ``molrs.io``.

These cover the drop-in ``BaseTrajectoryReader``-shaped surface:
reader-object return type, multi-file concatenation, canonical field names,
slicing, and context-manager use.
"""

import glob

import pytest

import molrs


@pytest.fixture
def a_dcd(dcd_dir):
    # A plain, writer-friendly multi-frame file.
    return str(dcd_dir / "water.dcd")


@pytest.fixture
def a_lammps(lammps_dir):
    files = sorted(glob.glob(str(lammps_dir / "*.lammpstrj")))
    if not files:
        pytest.skip(f"no LAMMPS dump test data in {lammps_dir}")
    return files[0]


@pytest.fixture
def an_xyz_traj(xyz_dir):
    # Pick an XYZ file with at least one frame.
    for p in sorted(glob.glob(str(xyz_dir / "*.xyz"))):
        try:
            if len(molrs.read_xyz_trajectory(p)) >= 1:
                return p
        except OSError:
            continue
    pytest.skip("no readable XYZ trajectory test data")


class TestReturnsReaderNotList:
    def test_dcd_returns_reader(self, a_dcd):
        reader = molrs.io.read_dcd_trajectory(a_dcd)
        assert isinstance(reader, molrs.io.TrajectoryReader)
        assert reader.n_frames == len(reader) > 0

    def test_lammps_returns_reader(self, a_lammps):
        reader = molrs.io.read_lammps_trajectory(a_lammps)
        assert isinstance(reader, molrs.io.TrajectoryReader)
        assert reader.n_frames > 0

    def test_xyz_facade_returns_reader_but_toplevel_returns_list(self, an_xyz_traj):
        # The molpy-compatible facade returns a reader object...
        reader = molrs.io.read_xyz_trajectory(an_xyz_traj)
        assert isinstance(reader, molrs.io.TrajectoryReader)
        # ...while the top-level molrs function still returns a list.
        eager = molrs.read_xyz_trajectory(an_xyz_traj)
        assert isinstance(eager, list)
        assert reader.n_frames == len(eager)


class TestTrajectoryReaderSurface:
    def test_read_frame_and_negative_index(self, a_dcd):
        reader = molrs.io.read_dcd_trajectory(a_dcd)
        n = reader.n_frames
        assert reader.read_frame(0) is not None
        assert reader.read_frame(-1)["atoms"].nrows == reader.read_frame(n - 1)[
            "atoms"
        ].nrows

    def test_out_of_range_raises(self, a_dcd):
        reader = molrs.io.read_dcd_trajectory(a_dcd)
        with pytest.raises(IndexError):
            reader.read_frame(10_000_000)

    def test_read_all_and_read_range(self, a_dcd):
        reader = molrs.io.read_dcd_trajectory(a_dcd)
        n = reader.n_frames
        assert len(reader.read_all()) == n
        assert len(reader.read_range()) == n
        assert len(reader.read_range(0, n)) == n

    def test_read_range_step_zero_raises(self, a_dcd):
        reader = molrs.io.read_dcd_trajectory(a_dcd)
        with pytest.raises(ValueError):
            reader.read_range(0, 1, 0)

    def test_slicing(self, a_dcd):
        reader = molrs.io.read_dcd_trajectory(a_dcd)
        n = reader.n_frames
        assert isinstance(reader[0:1], list)
        assert len(reader[:]) == n
        assert len(reader[::-1]) == n

    def test_iteration(self, a_dcd):
        reader = molrs.io.read_dcd_trajectory(a_dcd)
        assert sum(1 for _ in reader) == reader.n_frames
        # Re-iteration resets the cursor.
        assert sum(1 for _ in reader) == reader.n_frames

    def test_context_manager_closes(self, a_dcd):
        with molrs.io.read_dcd_trajectory(a_dcd) as reader:
            assert reader.n_frames > 0
        with pytest.raises(ValueError):
            reader.read_frame(0)


class TestMultiFile:
    def test_concatenates_frame_counts(self, a_dcd):
        single = molrs.io.read_dcd_trajectory(a_dcd).n_frames
        doubled = molrs.io.read_dcd_trajectory([a_dcd, a_dcd])
        assert doubled.n_frames == 2 * single

    def test_multi_file_indexing_crosses_boundary(self, a_dcd):
        single = molrs.io.read_dcd_trajectory(a_dcd).n_frames
        reader = molrs.io.read_dcd_trajectory([a_dcd, a_dcd])
        # First frame of the second file mirrors the first frame of the first.
        a = reader.read_frame(0)["atoms"].nrows
        b = reader.read_frame(single)["atoms"].nrows
        assert a == b
        assert len(reader.read_all()) == 2 * single


class TestCanonicalFields:
    def test_lammps_canonical_columns(self, a_lammps):
        reader = molrs.io.read_lammps_trajectory(a_lammps)
        atoms = reader.read_frame(0)["atoms"]
        # LammpsFieldFormatter canonicalises e.g. id/mol; format-native keys
        # like "q" must not leak through the facade.
        assert "q" not in atoms
