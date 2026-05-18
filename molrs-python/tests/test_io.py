import os
import pytest
import numpy as np
import molrs

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GRO_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "molrs-core", "target", "tests-data", "gro")


class TestReadPdb:
    def test_basic(self):
        frame = molrs.read_pdb(os.path.join(DATA_DIR, "water.pdb"))
        assert "atoms" in frame
        assert frame["atoms"].nrows == 3

    def test_has_coordinates(self):
        frame = molrs.read_pdb(os.path.join(DATA_DIR, "water.pdb"))
        atoms = frame["atoms"]
        x = atoms.view("x")
        y = atoms.view("y")
        z = atoms.view("z")
        assert x is not None
        assert y is not None
        assert z is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.read_pdb("/nonexistent/path.pdb")


class TestReadGro:
    def test_native_basic(self):
        frames = molrs.read_gro(os.path.join(GRO_DATA_DIR, "ubiquitin.gro"))
        assert len(frames) == 1
        f0 = frames[0]
        assert "atoms" in f0
        assert f0["atoms"].nrows > 0
        assert f0.simbox is not None

    def test_native_columns(self):
        frames = molrs.read_gro(os.path.join(GRO_DATA_DIR, "ubiquitin.gro"))
        atoms = frames[0]["atoms"]
        for col in ["resid", "resname", "atom_name", "atom_id", "x", "y", "z"]:
            assert col in atoms, f"missing column: {col}"

    def test_facade_canonical_columns(self):
        frames = molrs.io.read_gro(os.path.join(GRO_DATA_DIR, "ubiquitin.gro"))
        atoms = frames[0]["atoms"]
        for col in ["res_id", "res_name", "name", "id", "x", "y", "z"]:
            assert col in atoms, f"missing canonical column: {col}"

    def test_facade_no_format_native_columns(self):
        frames = molrs.io.read_gro(os.path.join(GRO_DATA_DIR, "ubiquitin.gro"))
        atoms = frames[0]["atoms"]
        for col in ["resid", "atom_name", "atom_id"]:
            assert col not in atoms, f"format-native column leaked: {col}"

    def test_round_trip(self):
        frames = molrs.io.read_gro(os.path.join(GRO_DATA_DIR, "ubiquitin.gro"))
        f0 = frames[0]
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".gro", delete=False) as tmp:
            tmpname = tmp.name
        try:
            molrs.io.write_gro(tmpname, f0)
            frames2 = molrs.io.read_gro(tmpname)
            f1 = frames2[0]
            assert f0["atoms"].nrows == f1["atoms"].nrows
        finally:
            os.unlink(tmpname)

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.read_gro("/nonexistent/path.gro")

    def test_triclinic_box(self):
        frames = molrs.io.read_gro(os.path.join(GRO_DATA_DIR, "1vln-triclinic.gro"))
        assert frames[0].simbox is not None


class TestReadXyz:
    def test_basic(self):
        frame = molrs.read_xyz(os.path.join(DATA_DIR, "water.xyz"))
        assert "atoms" in frame
        assert frame["atoms"].nrows == 3

    def test_has_coordinates(self):
        frame = molrs.read_xyz(os.path.join(DATA_DIR, "water.xyz"))
        atoms = frame["atoms"]
        x = atoms.view("x")
        assert x is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.read_xyz("/nonexistent/path.xyz")
