//! End-to-end tests for the FFT-based linear autocorrelation (`acf_fft`).
//!
//! Analytical references used:
//! * Linear ACF at lag 0 equals the signal power Σ x[τ]² (= "energy").
//! * Linear ACF of a constant c over n samples is r[t] = (n − t)·c².
//! * The linear (not circular) ACF of a pure cosine, divided by the number of
//!   overlapping samples (n − t), is itself a cosine of the same frequency:
//!   r̂[t] ≈ (A²/2)·cos(ω·t). This is the discrete analogue of the
//!   Wiener–Khinchin result that the ACF of a sinusoid is a cosine.

use molrs::signal::{acf_fft, acf_fft_with_planner};
use ndarray::Array1;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// ACF at lag 0 of any real signal equals Σ x[τ]² (the signal energy).
#[test]
fn acf_lag0_equals_signal_power() {
    let data = Array1::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0]);
    let energy: f64 = data.iter().map(|x| x * x).sum();
    let r = acf_fft(&data, 4).unwrap();
    assert!(
        (r[0] - energy).abs() < 1e-9,
        "lag-0 ACF {} should equal signal energy {energy}",
        r[0]
    );
}

/// Linear ACF of a constant c over n samples: r[t] = (n − t)·c².
#[test]
fn acf_constant_signal_linear_decay() {
    let n = 16;
    let c = 2.5;
    let data = Array1::from_elem(n, c);
    let max_lag = n - 1;
    let r = acf_fft(&data, max_lag).unwrap();
    for t in 0..=max_lag {
        let expected = (n - t) as f64 * c * c;
        assert!(
            (r[t] - expected).abs() < 1e-8,
            "lag {t}: got {}, expected {expected}",
            r[t]
        );
    }
}

/// The normalized linear ACF of a pure cosine is a cosine of the same
/// frequency: r̂[t] = r[t]/(n − t) ≈ (A²/2)·cos(ω·t).
#[test]
fn acf_of_cosine_is_a_cosine() {
    let n = 2048;
    let period = 32.0; // samples per cycle
    let omega = 2.0 * PI / period;
    let amplitude = 1.0;
    let data: Array1<f64> = Array1::from_iter((0..n).map(|i| amplitude * (omega * i as f64).cos()));

    // Only test lags well inside the signal so the unbiased (n − t) estimator
    // remains stable (few-cycle support gives a noisy ratio).
    let max_lag = 256;
    let r = acf_fft(&data, max_lag).unwrap();

    for t in 0..=max_lag {
        let normalized = r[t] / (n - t) as f64;
        let expected = (amplitude * amplitude / 2.0) * (omega * t as f64).cos();
        assert!(
            (normalized - expected).abs() < 1e-2,
            "lag {t}: normalized ACF {normalized} should match cosine {expected}"
        );
    }
}

/// A cosine's ACF is itself maximal at lag 0 and returns to (near) its peak
/// after exactly one period — periodicity of the autocorrelation.
#[test]
fn acf_of_cosine_is_periodic() {
    let n = 1024;
    let period = 64usize;
    let omega = 2.0 * PI / period as f64;
    let data: Array1<f64> = Array1::from_iter((0..n).map(|i| (omega * i as f64).cos()));

    let max_lag = 2 * period;
    let r = acf_fft(&data, max_lag).unwrap();
    let r0 = r[0] / n as f64;
    let r_period = r[period] / (n - period) as f64;
    let r_half = r[period / 2] / (n - period / 2) as f64;

    // At a full period the (normalized) ACF returns near its peak value.
    assert!(
        (r_period - r0).abs() < 5e-3,
        "ACF at one period {r_period} should approach lag-0 value {r0}"
    );
    // At half a period the cosine is in anti-phase: ACF is negative.
    assert!(
        r_half < 0.0,
        "ACF at half period {r_half} should be negative (anti-phase)"
    );
}

/// White-noise ACF peaks at lag 0 and is small elsewhere.
#[test]
fn acf_white_noise_peaks_at_zero() {
    use rand::RngExt;
    let mut rng = rand::rng();
    let n = 4096;
    let data: Array1<f64> = Array1::from_iter((0..n).map(|_| rng.random::<f64>() * 2.0 - 1.0));
    let r = acf_fft(&data, 32).unwrap();
    for t in 1..r.len() {
        assert!(
            r[t].abs() < r[0],
            "white-noise ACF at lag {t} ({}) should be below the lag-0 peak ({})",
            r[t],
            r[0]
        );
    }
}

/// The planner-reusing entry point yields bit-identical results.
#[test]
fn acf_planner_matches_plain() {
    let data = Array1::from_vec(vec![0.3, 1.7, -0.5, 2.2, -1.1, 0.9, 4.0, -3.3]);
    let plain = acf_fft(&data, 5).unwrap();
    let mut planner = FftPlanner::<f64>::new();
    let with = acf_fft_with_planner(&mut planner, &data, 5).unwrap();
    assert_eq!(plain.len(), with.len());
    for t in 0..plain.len() {
        assert!((plain[t] - with[t]).abs() < 1e-12, "mismatch at lag {t}");
    }
}

/// Edge case: single-sample signal.
#[test]
fn acf_single_sample() {
    let data = Array1::from_vec(vec![7.0]);
    let r = acf_fft(&data, 0).unwrap();
    assert_eq!(r.len(), 1);
    assert!((r[0] - 49.0).abs() < 1e-12);
}

/// Edge case: empty and over-large lag produce the documented errors.
#[test]
fn acf_invalid_inputs_error() {
    let empty = Array1::<f64>::zeros(0);
    assert!(acf_fft(&empty, 0).is_err(), "empty input must error");

    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(
        acf_fft(&data, 3).is_err(),
        "max_lag == len must error (max_lag must be < len)"
    );
}
