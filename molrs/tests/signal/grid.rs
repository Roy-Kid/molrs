//! End-to-end tests for the angular-frequency grid (`frequency_grid`).
//!
//! Analytical references used:
//! * Length is n_fft/2 + 1 (one-sided real-FFT grid).
//! * Bin 0 is DC (ω = 0).
//! * Spacing Δω = 2π/(n_fft·dt) — the rotational analogue of Δf = 1/(N·dt).
//! * Nyquist (last bin) ω = π/dt.

use molrs::signal::frequency_grid;
use std::f64::consts::PI;

/// Output length is the one-sided count n_fft/2 + 1.
#[test]
fn grid_length_is_half_plus_one() {
    for n in [2usize, 8, 16, 1024] {
        let g = frequency_grid(n, 0.5);
        assert_eq!(g.len(), n / 2 + 1, "wrong length for n_fft={n}");
    }
}

/// Bin 0 is DC and spacing equals 2π/(n_fft·dt) (the angular form of 1/(N·dt)).
#[test]
fn grid_dc_and_spacing() {
    let n = 256;
    let dt = 0.002;
    let g = frequency_grid(n, dt);
    assert!((g[0] - 0.0).abs() < 1e-12, "bin 0 must be DC");

    let expected_spacing = 2.0 * PI / (n as f64 * dt);
    for i in 1..g.len() {
        let diff = g[i] - g[i - 1];
        assert!(
            (diff - expected_spacing).abs() < 1e-9,
            "spacing at {i}: {diff} != {expected_spacing}"
        );
    }
}

/// Every bin equals k·Δω exactly.
#[test]
fn grid_bins_are_k_times_delta() {
    let n = 64;
    let dt = 1.5;
    let d_omega = 2.0 * PI / (n as f64 * dt);
    let g = frequency_grid(n, dt);
    for (k, &w) in g.iter().enumerate() {
        let expected = k as f64 * d_omega;
        assert!((w - expected).abs() < 1e-9, "bin {k}: {w} != {expected}");
    }
}

/// The last bin is the Nyquist angular frequency π/dt.
#[test]
fn grid_last_bin_is_nyquist() {
    let n = 512;
    let dt = 0.01;
    let g = frequency_grid(n, dt);
    let nyquist = PI / dt;
    assert!(
        (g[g.len() - 1] - nyquist).abs() < 1e-7,
        "last bin {} should equal Nyquist {nyquist}",
        g[g.len() - 1]
    );
}

/// Halving dt doubles every frequency (grid scales as 1/dt).
#[test]
fn grid_scales_inversely_with_dt() {
    let n = 128;
    let coarse = frequency_grid(n, 0.02);
    let fine = frequency_grid(n, 0.01);
    assert_eq!(coarse.len(), fine.len());
    for i in 1..coarse.len() {
        assert!(
            (fine[i] - 2.0 * coarse[i]).abs() < 1e-9,
            "halving dt should double bin {i}"
        );
    }
}

/// Minimal even grid: n_fft = 2 gives [0, π/dt].
#[test]
fn grid_minimal_even() {
    let g = frequency_grid(2, 1.0);
    assert_eq!(g.len(), 2);
    assert!((g[0] - 0.0).abs() < 1e-12);
    assert!((g[1] - PI).abs() < 1e-12);
}
