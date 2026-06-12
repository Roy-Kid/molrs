import numpy as np
import pytest
import molrs
from molrs.molrs import Block, Frame  # bare PyO3 cores (Block/Frame are the rich shadow)


class TestFrameConstruction:
    def test_empty(self):
        f = Frame()
        assert len(f) == 0
        assert f.keys() == []
        assert f.simbox is None

    def test_from_dict_blocks_envelope(self):
        f = Frame.from_dict(
            {
                "blocks": {
                    "atoms": {
                        "symbol": ["C", "H"],
                        "x": np.array([0.0, 1.0], dtype=np.float64),
                    }
                },
                "metadata": {"source": "pytest"},
            }
        )

        assert sorted(f.keys()) == ["atoms"]
        assert f["atoms"].nrows == 2
        assert list(f["atoms"].view("symbol")) == ["C", "H"]
        np.testing.assert_allclose(f["atoms"].view("x"), [0.0, 1.0])
        assert f.meta["source"] == "pytest"

    def test_from_dict_direct_block_mapping(self):
        f = Frame.from_dict(
            {
                "atoms": {
                    "id": np.array([1, 2], dtype=np.int64),
                    "selected": np.array([True, False], dtype=np.bool_),
                }
            }
        )

        assert sorted(f.keys()) == ["atoms"]
        np.testing.assert_array_equal(f["atoms"].view("id"), [1, 2])
        np.testing.assert_array_equal(f["atoms"].view("selected"), [True, False])

    def test_repr_empty(self):
        r = repr(Frame())
        assert "Frame" in r
        assert "no" in r  # simbox=no


class TestFrameBlockAccess:
    def test_setitem_getitem(self):
        f = Frame()
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b
        assert "atoms" in f
        assert len(f) == 1

        atoms = f["atoms"]
        assert atoms.nrows == 2

    def test_getitem_returns_live_block_handle(self):
        f = Frame()
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b

        atoms = f["atoms"]
        atoms.insert("y", np.array([3.0, 4.0], dtype=np.float64))

        np.testing.assert_allclose(f["atoms"].view("y"), [3.0, 4.0])

    def test_getitem_missing_raises_key_error(self):
        f = Frame()
        with pytest.raises(KeyError):
            _ = f["missing"]

    def test_delitem(self):
        f = Frame()
        f["atoms"] = Block()
        del f["atoms"]
        assert "atoms" not in f

    def test_delitem_missing_raises_key_error(self):
        f = Frame()
        with pytest.raises(KeyError):
            del f["missing"]

    def test_contains(self):
        f = Frame()
        f["atoms"] = Block()
        assert "atoms" in f
        assert "bonds" not in f

    def test_keys(self):
        f = Frame()
        f["atoms"] = Block()
        f["bonds"] = Block()
        assert sorted(f.keys()) == ["atoms", "bonds"]

    def test_overwrite_block(self):
        f = Frame()
        b1 = Block()
        b1.insert("x", np.array([1.0], dtype=np.float64))
        f["atoms"] = b1
        assert f["atoms"].nrows == 1

        b2 = Block()
        b2.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b2
        assert f["atoms"].nrows == 2


class TestFrameSimbox:
    def test_default_none(self):
        assert Frame().simbox is None

    def test_set_simbox(self):
        f = Frame()
        box_ = molrs.Box.cube(10.0)
        f.simbox = box_
        assert f.simbox is not None
        assert pytest.approx(f.simbox.volume(), abs=1) == 1000.0

    def test_clear_simbox(self):
        f = Frame()
        f.simbox = molrs.Box.cube(10.0)
        f.simbox = None
        assert f.simbox is None

    def test_repr_with_simbox(self):
        f = Frame()
        f.simbox = molrs.Box.cube(10.0)
        assert "yes" in repr(f)


class TestFrameMeta:
    def test_set_and_get(self):
        f = Frame()
        f.meta = {"title": "test", "source": "pytest"}
        meta = f.meta
        assert meta["title"] == "test"
        assert meta["source"] == "pytest"

    def test_empty_meta(self):
        f = Frame()
        assert len(f.meta) == 0

    def test_overwrite_meta(self):
        f = Frame()
        f.meta = {"a": "1"}
        f.meta = {"b": "2"}
        assert "b" in f.meta
        assert "a" not in f.meta


class TestFrameValidation:
    def test_validate_empty(self):
        Frame().validate()

    def test_validate_consistent(self):
        f = Frame()
        b = Block()
        b.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b.insert("y", np.array([0.0, 1.0, 2.0], dtype=np.float64))
        f["atoms"] = b
        f.validate()
