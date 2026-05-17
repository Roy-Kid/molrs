"""Integration tests for molrs.signal module.

Tests the Python bindings for the molrs-signal Rust crate:
- acf_fft: raw un-normalized FFT-based autocorrelation
- apply_window: Hann/Blackman window application
- frequency_grid: angular frequency grid generation
"""

import numpy as np
import pytest


class TestACFFFT:
    """Tests for molrs.signal.acf_fft."""

    def test_import(self):
        from molrs.signal import acf_fft

        assert callable(acf_fft)

    def test_constant_signal_unnormalized(self):
        from molrs.signal import acf_fft

        data = np.array([2.0, 2.0, 2.0, 2.0])
        result = acf_fft(data, max_lag=3)
        assert result.shape == (4,)
        assert result.dtype == np.float64
        expected_lag0 = 4.0 * 2.0**2  # N * c^2 = 16
        assert np.allclose(result[0], expected_lag0, atol=1e-10)

    def test_max_lag_zero(self):
        from molrs.signal import acf_fft

        data = np.array([1.0, 2.0, 3.0])
        result = acf_fft(data, max_lag=0)
        assert result.shape == (1,)
        expected = 1.0**2 + 2.0**2 + 3.0**2
        assert np.allclose(result[0], expected, atol=1e-10)

    def test_max_lag_too_large_raises(self):
        from molrs.signal import acf_fft

        data = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="max_lag"):
            acf_fft(data, max_lag=2)

    def test_empty_input_raises(self):
        from molrs.signal import acf_fft

        data = np.array([], dtype=np.float64)
        with pytest.raises(ValueError):
            acf_fft(data, max_lag=0)

    def test_does_not_mutate_input(self):
        from molrs.signal import acf_fft

        data = np.random.default_rng(42).random(100)
        original = data.copy()
        acf_fft(data, max_lag=10)
        assert np.array_equal(data, original)


class TestApplyWindow:
    """Tests for molrs.signal.apply_window."""

    def test_import(self):
        from molrs.signal import apply_window

        assert callable(apply_window)

    def test_hann_1d(self):
        from molrs.signal import apply_window

        data = np.ones(5)
        result = apply_window(data, "hann", axis=0)
        assert result.shape == (5,)
        assert result.dtype == np.float64
        # Boundary values should be ~0
        assert np.allclose(result[0], 0.0, atol=1e-10)
        assert np.allclose(result[-1], 0.0, atol=1e-10)

    def test_blackman_1d(self):
        from molrs.signal import apply_window

        data = np.ones(5)
        result = apply_window(data, "blackman", axis=0)
        assert result.shape == (5,)
        expected_first = 0.42 - 0.5 + 0.08  # = 0.0
        assert np.allclose(result[0], expected_first, atol=1e-10)

    def test_hann_2d_axis0(self):
        from molrs.signal import apply_window

        data = np.ones((5, 3))
        result = apply_window(data, "hann", axis=0)
        assert result.shape == (5, 3)
        # Each column should be the Hann window
        col0 = result[:, 0]
        assert np.allclose(col0[0], 0.0, atol=1e-10)
        assert np.allclose(col0[-1], 0.0, atol=1e-10)

    def test_unknown_window_raises(self):
        from molrs.signal import apply_window

        data = np.ones(5)
        with pytest.raises(ValueError, match="window"):
            apply_window(data, "hamming", axis=0)

    def test_does_not_mutate_input(self):
        from molrs.signal import apply_window

        data = np.ones((10, 4))
        original = data.copy()
        apply_window(data, "hann", axis=0)
        assert np.array_equal(data, original)


class TestFrequencyGrid:
    """Tests for molrs.signal.frequency_grid."""

    def test_import(self):
        from molrs.signal import frequency_grid

        assert callable(frequency_grid)

    def test_basic(self):
        from molrs.signal import frequency_grid

        grid = frequency_grid(8, 0.5)
        assert grid.shape == (5,)
        assert grid.dtype == np.float64
        assert grid[0] == 0.0
        assert np.allclose(grid[-1], np.pi / 0.5)  # Nyquist

    def test_minimal(self):
        from molrs.signal import frequency_grid

        grid = frequency_grid(2, 1.0)
        assert grid.shape == (2,)
        assert grid[0] == 0.0
        assert np.allclose(grid[1], np.pi)

    def test_uniform_spacing(self):
        from molrs.signal import frequency_grid

        grid = frequency_grid(1024, 0.001)
        assert len(grid) == 513
        diffs = np.diff(grid)
        expected_spacing = 2.0 * np.pi / (1024.0 * 0.001)
        assert np.allclose(diffs, expected_spacing, atol=1e-10)
