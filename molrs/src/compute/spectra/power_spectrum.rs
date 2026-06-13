//! Velocity power spectrum / vibrational density of states (VDOS).
//!
//! Computes `G(ω) ∝ FT[⟨v(0)·v(t)⟩]` from a velocity time series.
//!
//! Reference: Dickey & Paskin, Phys. Rev. **188**, 1407 (1969).

use molrs::signal as sig;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

use super::{Spectrum, validate_input, window_and_fft};
use crate::compute::error::ComputeError;

/// Compute the vibrational power spectrum (VDOS) from velocity time series.
///
/// # Arguments
/// * `velocities` — `(n_frames, n_dof)`. Each column is one degree of
///   freedom (e.g. a Cartesian velocity component of one atom). Rows
///   advance in time with step `dt_fs`.
/// * `dt_fs` — timestep between consecutive frames, in femtoseconds.
/// * `resolution` — number of ACF lags retained. Clamped to
///   `n_frames - 1`. Practical choice: ≤ 1/10 of `n_frames`.
///
/// # Returns
/// [`Spectrum`] with `frequencies_cm1` and `intensities` (arbitrary units).
///
/// # Errors
/// * `EmptyInput` if `n_frames < 2`.
/// * `OutOfRange` if `dt_fs ≤ 0`.
pub fn power_spectrum(
    velocities: &Array2<f64>,
    dt_fs: f64,
    resolution: usize,
) -> Result<Spectrum, ComputeError> {
    let n_frames = velocities.shape()[0];
    let n_dof = velocities.shape()[1];
    let max_lag = validate_input(n_frames, dt_fs, resolution)?;

    let inv_n_frames = 1.0 / n_frames as f64;
    let mut planner = FftPlanner::new();
    let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);

    for d in 0..n_dof {
        let mut col: Array1<f64> = (0..n_frames).map(|t| velocities[[t, d]]).collect();
        let mean: f64 = col.iter().sum::<f64>() * inv_n_frames;
        for v in col.iter_mut() {
            *v -= mean;
        }
        let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).map_err(|e| {
            ComputeError::OutOfRange {
                field: "acf_fft",
                value: e.to_string(),
            }
        })?;
        for k in 0..=max_lag {
            acf_sum[k] += acf[k];
        }
    }

    let inv_n_dof = 1.0 / n_dof as f64;
    for k in 0..=max_lag {
        acf_sum[k] *= inv_n_dof;
    }

    let (frequencies_cm1, intensities) = window_and_fft(&mut planner, &acf_sum, dt_fs)?;

    Ok(Spectrum {
        frequencies_cm1,
        intensities,
        resolution: max_lag,
        n_frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_power_spectrum_shapes() {
        let n_frames = 100;
        let n_dof = 15;
        let velocities = Array2::from_elem((n_frames, n_dof), 0.1);
        let result = power_spectrum(&velocities, 1.0, 10).unwrap();
        let n_pad = (4usize * 11).next_power_of_two();
        let expected_len = n_pad / 2 + 1;
        assert_eq!(result.frequencies_cm1.len(), expected_len);
        assert_eq!(result.intensities.len(), expected_len);
        assert_eq!(result.n_frames, n_frames);
        assert_eq!(result.resolution, 10);
    }

    #[test]
    fn test_power_spectrum_all_zero_signal() {
        let velocities = Array2::zeros((50, 1));
        let result = power_spectrum(&velocities, 0.5, 8).unwrap();
        for (j, v) in result.intensities.iter().enumerate() {
            assert!(v.abs() < 1e-10, "non-zero at bin {j}: {v}");
        }
    }

    #[test]
    fn test_power_spectrum_sine_wave_peak() {
        // v_x(t) = sin(2π * 10 THz * t), dt = 0.5 fs, 4096 frames
        // 10 THz = 0.01 fs⁻¹ → 333.56 cm⁻¹.
        let n = 4096;
        let dt_fs = 0.5;
        let freq_thz = 10.0;
        let mut velocities = Array2::zeros((n, 3));
        for t in 0..n {
            let time_fs = t as f64 * dt_fs;
            velocities[[t, 0]] = (2.0 * std::f64::consts::PI * freq_thz * 1e-3 * time_fs).sin();
        }
        let result = power_spectrum(&velocities, dt_fs, 200).unwrap();

        let n_bins = result.intensities.len();
        let search_end = n_bins.saturating_sub(3);
        let max_idx = result
            .intensities
            .iter()
            .enumerate()
            .skip(1)
            .take(search_end.saturating_sub(1))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let peak_cm1 = result.frequencies_cm1[max_idx];
        let expected_cm1 = 333.56;
        assert!(
            (peak_cm1 - expected_cm1).abs() < 20.0,
            "peak at {peak_cm1} cm⁻¹, expected ~{expected_cm1} cm⁻¹"
        );
    }

    #[test]
    fn test_power_spectrum_rejects_single_frame() {
        let velocities = Array2::zeros((1, 3));
        assert!(power_spectrum(&velocities, 0.5, 10).is_err());
    }

    #[test]
    fn test_power_spectrum_rejects_nonpositive_dt() {
        let velocities = Array2::zeros((10, 3));
        assert!(power_spectrum(&velocities, 0.0, 5).is_err());
        assert!(power_spectrum(&velocities, -1.0, 5).is_err());
    }

    #[test]
    fn test_power_spectrum_clamps_resolution() {
        let velocities = Array2::zeros((5, 2));
        let result = power_spectrum(&velocities, 1.0, 100).unwrap();
        assert_eq!(result.resolution, 4);
    }

    #[test]
    fn test_immutability_power_spectrum() {
        let velocities = Array2::from_elem((20, 3), 1.0);
        let copy = velocities.clone();
        power_spectrum(&velocities, 0.5, 5).unwrap();
        assert_eq!(velocities, copy);
    }
}
