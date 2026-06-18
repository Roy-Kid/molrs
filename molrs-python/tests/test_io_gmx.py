"""Tests for the GROMACS TRR/XTC bindings exposed through ``molrs``.

Covers the top-level eager readers (``molrs.read_trr`` / ``molrs.read_xtc``),
the molpy-compatible lazy facade (``molrs.io.read_trr_trajectory`` /
``read_xtc_trajectory``), random access, write round-trips, and the
cross-format consistency that anchors decoded coordinates to ground truth.
"""

import numpy as np
import pytest

import molrs
import molrs.io as mio


@pytest.fixture
def cell_trr(trr_dir):
    p = trr_dir / "cell_shapes.trr"
    if not p.exists():
        pytest.skip("no TRR test data")
    return str(p)


@pytest.fixture
def cell_xtc(xtc_dir):
    p = xtc_dir / "cell_shapes.xtc"
    if not p.exists():
        pytest.skip("no XTC test data")
    return str(p)


class TestTopLevelEagerReaders:
    def test_read_trr_returns_list_of_frames(self, cell_trr):
        frames = molrs.read_trr(cell_trr)
        assert isinstance(frames, list) and len(frames) >= 1
        atoms = frames[0]["atoms"]
        assert atoms.nrows > 0
        for axis in ("x", "y", "z"):
            assert np.all(np.isfinite(atoms.view(axis)))

    def test_read_xtc_returns_list_of_frames(self, cell_xtc):
        frames = molrs.read_xtc(cell_xtc)
        assert isinstance(frames, list) and len(frames) >= 1
        assert frames[0]["atoms"].nrows > 0


class TestLazyFacadeReaders:
    def test_trr_returns_reader(self, cell_trr):
        reader = mio.read_trr_trajectory(cell_trr)
        assert isinstance(reader, mio.TrajectoryReader)
        assert reader.n_frames == len(reader) > 0

    def test_xtc_returns_reader(self, cell_xtc):
        reader = mio.read_xtc_trajectory(cell_xtc)
        assert isinstance(reader, mio.TrajectoryReader)
        assert reader.n_frames > 0

    def test_random_access_matches_sequential(self, cell_trr):
        reader = mio.read_trr_trajectory(cell_trr)
        eager = molrs.read_trr(cell_trr)
        assert reader.n_frames == len(eager)
        last = reader.read_frame(-1)["atoms"].view("x")
        assert np.allclose(last, eager[-1]["atoms"].view("x"))

    def test_out_of_range_raises(self, cell_xtc):
        reader = mio.read_xtc_trajectory(cell_xtc)
        with pytest.raises(IndexError):
            reader.read_frame(10_000_000)

    def test_multi_file_concatenates(self, cell_trr):
        single = mio.read_trr_trajectory(cell_trr).n_frames
        doubled = mio.read_trr_trajectory([cell_trr, cell_trr])
        assert doubled.n_frames == 2 * single


class TestWriteRoundTrip:
    def test_trr_roundtrip_exact(self, cell_trr, tmp_path):
        frames = molrs.read_trr(cell_trr)
        out = tmp_path / "out.trr"
        mio.write_trr(str(out), frames)
        back = molrs.read_trr(str(out))
        assert len(back) == len(frames)
        for a, b in zip(frames, back):
            assert np.allclose(a["atoms"].view("x"), b["atoms"].view("x"), atol=1e-5)

    def test_xtc_roundtrip_within_precision(self, cell_xtc, tmp_path):
        frames = molrs.read_xtc(cell_xtc)
        out = tmp_path / "out.xtc"
        mio.write_xtc(str(out), frames)
        back = molrs.read_xtc(str(out))
        assert len(back) == len(frames)
        for a, b in zip(frames, back):
            # XTC is lossy at 1/precision (default 1000 → 1e-3 nm).
            assert np.allclose(a["atoms"].view("x"), b["atoms"].view("x"), atol=2e-3)


class TestCrossFormatGroundTruth:
    def test_cell_shapes_trr_matches_xtc(self, cell_trr, cell_xtc):
        trr0 = molrs.read_trr(cell_trr)[0]["atoms"]
        xtc0 = molrs.read_xtc(cell_xtc)[0]["atoms"]
        assert trr0.nrows == xtc0.nrows
        for axis in ("x", "y", "z"):
            assert np.allclose(trr0.view(axis), xtc0.view(axis), atol=2e-3)
