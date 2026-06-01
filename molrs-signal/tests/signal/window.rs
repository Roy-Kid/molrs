//! End-to-end tests for window functions (`apply_window`, `WindowType`).
//!
//! Analytical references used:
//! * Hann: w[n] = 0.5·(1 − cos(2π·n/(N−1))); symmetric, zero at both ends,
//!   sum Σ w = N/2 for the symmetric form's interior.
//! * Blackman three-term (0.42, 0.5, 0.08): zero at both ends.
//! * CosineSq: w[n] = cos²(π·n/(2(N−1))); w[0] = 1, w[N−1] = 0, midpoint 0.5.
//! All windows are symmetric: w[n] = w[N−1−n].

use molrs_signal::{WindowType, apply_window};
use ndarray::{ArrayD, IxDyn};
use std::f64::consts::PI;

fn ones(n: usize) -> ArrayD<f64> {
    ArrayD::from_elem(IxDyn(&[n]), 1.0)
}

/// Hann window values match the analytical formula exactly.
#[test]
fn hann_matches_formula() {
    let n = 9;
    let w = apply_window(&ones(n), WindowType::Hann, 0).unwrap();
    for i in 0..n {
        let expected = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
        assert!(
            (w[i] - expected).abs() < 1e-12,
            "Hann idx {i}: {} != {expected}",
            w[i]
        );
    }
}

/// Hann and Blackman are symmetric: w[n] = w[N−1−n]. (CosineSq is deliberately
/// one-sided — w[0]=1, w[N−1]=0 — so it is excluded here.)
#[test]
fn windows_are_symmetric() {
    let n = 17;
    for wt in [WindowType::Hann, WindowType::Blackman] {
        let w = apply_window(&ones(n), wt, 0).unwrap();
        for i in 0..n {
            assert!(
                (w[i] - w[n - 1 - i]).abs() < 1e-12,
                "{wt:?} not symmetric at idx {i}"
            );
        }
    }
}

/// Hann and Blackman taper to zero at both endpoints.
#[test]
fn hann_blackman_zero_at_ends() {
    let n = 64;
    for wt in [WindowType::Hann, WindowType::Blackman] {
        let w = apply_window(&ones(n), wt, 0).unwrap();
        assert!(w[0].abs() < 1e-10, "{wt:?} first sample should be ~0");
        assert!(w[n - 1].abs() < 1e-10, "{wt:?} last sample should be ~0");
    }
}

/// CosineSq starts at 1, ends at 0, and is 0.5 at the midpoint.
#[test]
fn cosine_sq_known_values() {
    let n = 9; // midpoint at index 4 -> cos²(π/4) = 0.5
    let w = apply_window(&ones(n), WindowType::CosineSq, 0).unwrap();
    assert!((w[0] - 1.0).abs() < 1e-12, "w[0] should be 1");
    assert!((w[n - 1] - 0.0).abs() < 1e-10, "w[N-1] should be 0");
    assert!(
        (w[(n - 1) / 2] - 0.5).abs() < 1e-12,
        "midpoint should be 0.5"
    );
}

/// The Hann window's sum equals (N−1)/2 — the integral of 0.5(1−cos) over a
/// full period, since the cosine terms cancel.
#[test]
fn hann_sum_equals_half_n_minus_one() {
    let n = 100;
    let w = apply_window(&ones(n), WindowType::Hann, 0).unwrap();
    let sum: f64 = w.iter().sum();
    let expected = (n - 1) as f64 / 2.0;
    assert!(
        (sum - expected).abs() < 1e-9,
        "Hann sum {sum} should equal (N-1)/2 = {expected}"
    );
}

/// apply_window multiplies element-wise: applying to a non-unit signal scales
/// it by the window, leaving the original untouched (immutability).
#[test]
fn apply_window_scales_and_does_not_mutate() {
    let n = 8;
    let signal =
        ArrayD::from_shape_vec(IxDyn(&[n]), (0..n).map(|i| i as f64 + 1.0).collect()).unwrap();
    let windowed = apply_window(&signal, WindowType::Hann, 0).unwrap();
    for i in 0..n {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
        assert!((windowed[i] - signal[i] * w).abs() < 1e-12);
    }
    // Original unchanged.
    for i in 0..n {
        assert!((signal[i] - (i as f64 + 1.0)).abs() < 1e-12);
    }
}

/// A 2-D array is windowed independently along the chosen axis; every column
/// (axis 0) equals the 1-D window.
#[test]
fn window_along_axis0_each_column() {
    let (rows, cols) = (6usize, 4usize);
    let data = ArrayD::from_elem(IxDyn(&[rows, cols]), 1.0);
    let w = apply_window(&data, WindowType::Hann, 0).unwrap();
    assert_eq!(w.shape(), &[rows, cols]);
    for r in 0..rows {
        let expected = 0.5 * (1.0 - (2.0 * PI * r as f64 / (rows - 1) as f64).cos());
        for c in 0..cols {
            assert!((w[[r, c]] - expected).abs() < 1e-12);
        }
    }
}

/// A 2-D array windowed along axis 1: every row equals the 1-D window.
#[test]
fn window_along_axis1_each_row() {
    let (rows, cols) = (3usize, 7usize);
    let data = ArrayD::from_elem(IxDyn(&[rows, cols]), 1.0);
    let w = apply_window(&data, WindowType::CosineSq, 1).unwrap();
    for c in 0..cols {
        let angle = PI * c as f64 / (2.0 * (cols - 1) as f64);
        let expected = angle.cos().powi(2);
        for r in 0..rows {
            assert!((w[[r, c]] - expected).abs() < 1e-12);
        }
    }
}

/// Edge case: single-sample window is 1 for every type.
#[test]
fn single_sample_window_is_unity() {
    for wt in [WindowType::Hann, WindowType::Blackman, WindowType::CosineSq] {
        let w = apply_window(&ones(1), wt, 0).unwrap();
        assert!(
            (w[0] - 1.0).abs() < 1e-12,
            "{wt:?} single sample should be 1"
        );
    }
}

/// Edge case: an out-of-bounds axis errors instead of panicking.
#[test]
fn out_of_bounds_axis_errors() {
    let data = ArrayD::from_elem(IxDyn(&[5, 3]), 1.0);
    assert!(
        apply_window(&data, WindowType::Hann, 2).is_err(),
        "axis 2 on a 2-D array must error"
    );
}
