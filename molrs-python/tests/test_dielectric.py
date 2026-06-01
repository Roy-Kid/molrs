"""Integration tests for molrs.dielectric module."""

import numpy as np
import pytest


class TestDipoleMoment:
    def test_import(self):
        from molrs.dielectric import Dielectric

        assert callable(Dielectric.compute_dipole_moment)

    def test_two_charges(self):
        from molrs.dielectric import Dielectric

        charges = np.array([1.0, -1.0])
        positions = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        m = Dielectric.compute_dipole_moment(charges, positions)
        assert m.shape == (3,)
        assert np.allclose(m[0], 2.0, atol=1e-10)
        assert np.allclose(m[1], 0.0, atol=1e-10)

    def test_wrong_shape_raises(self):
        from molrs.dielectric import Dielectric

        charges = np.array([1.0, 2.0])
        positions = np.zeros((3, 3))
        with pytest.raises(ValueError):
            Dielectric.compute_dipole_moment(charges, positions)


class TestCurrentDensity:
    def test_import(self):
        from molrs.dielectric import Dielectric

        assert callable(Dielectric.compute_current_density)

    def test_linear(self):
        from molrs.dielectric import Dielectric

        dm = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        j = Dielectric.compute_current_density(dm, dt=1.0, volume=1.0)
        assert j.shape == (3, 3)
        assert np.isnan(j[0, 0])
        assert np.allclose(j[1, 0], 1.0, atol=1e-10)
        assert np.allclose(j[2, 0], 1.0, atol=1e-10)


class TestStaticDielectric:
    def test_zero_fluctuation(self):
        from molrs.dielectric import Dielectric

        dm = np.zeros((10, 3))
        eps = Dielectric.static_dielectric_constant(
            dm, volume=1000.0, temperature=300.0, epsilon_inf=1.0
        )
        assert np.allclose(eps, 1.0, atol=1e-10)


class TestEHSpectrum:
    def test_import(self):
        from molrs.dielectric import Dielectric

        assert callable(Dielectric.einstein_helfand_spectrum)

    def test_shape(self):
        from molrs.dielectric import Dielectric

        dm = np.ones((100, 3)) * 0.1
        s = Dielectric.einstein_helfand_spectrum(
            dm,
            dt=0.001,
            volume=1000.0,
            temperature=300.0,
            epsilon_inf=1.0,
            max_correlation_time=10,
            window_type="hann",
        )
        assert len(s["frequencies"]) > 0
        assert len(s["frequencies"]) == len(s["epsilon_real"]) == len(s["epsilon_imag"])


class TestGKSpectrum:
    def test_import(self):
        from molrs.dielectric import Dielectric

        assert callable(Dielectric.green_kubo_spectrum)

    def test_shape(self):
        from molrs.dielectric import Dielectric

        j = np.ones((100, 3)) * 0.001
        s = Dielectric.green_kubo_spectrum(
            j,
            dt=0.001,
            volume=1000.0,
            temperature=300.0,
            epsilon_inf=1.0,
            max_correlation_time=10,
            window_type="hann",
        )
        assert s["frequencies"] is not None


class TestDecomposeCurrent:
    def test_conservation(self):
        from molrs.dielectric import Dielectric

        current = np.zeros((4, 5, 3))
        for p in range(4):
            for t in range(5):
                current[p, t, 0] = p + t
                current[p, t, 1] = p * 2.0
        mask = np.array([True, True, False, False])
        j_w, j_i = Dielectric.decompose_current(current, mask)
        assert j_w.shape == (5, 3)
        assert j_i.shape == (5, 3)
        total = current.sum(axis=0)
        assert np.allclose(j_w + j_i, total, atol=1e-12)


class TestImmutability:
    def test_all_functions_immutable(self):
        from molrs.dielectric import Dielectric

        charges = np.array([1.0, -1.0])
        positions = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        p_copy = positions.copy()
        Dielectric.compute_dipole_moment(charges, positions)
        assert np.array_equal(positions, p_copy)

        dm = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        dm_copy = dm.copy()
        Dielectric.compute_current_density(dm, dt=1.0, volume=1.0)
        assert np.array_equal(dm, dm_copy)

        dm2 = np.zeros((10, 3))
        dm2_copy = dm2.copy()
        Dielectric.static_dielectric_constant(
            dm2, volume=1000.0, temperature=300.0, epsilon_inf=1.0
        )
        assert np.array_equal(dm2, dm2_copy)
