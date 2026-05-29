//! Infrared (IR) absorption spectrum from dipole-moment time series.
//!
//! Computes `I(ω) ∝ FT[⟨Ṁ(0)·Ṁ(t)⟩]` where `Ṁ` is the dipole flux
//! obtained via central finite difference.
//!
//! Reference: Gaigeot & Sprik, J. Phys. Chem. B **107**, 10344 (2003).

use molrs_signal as sig;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

use super::{Spectrum, window_and_fft};
use crate::error::ComputeError;

/// Compute the IR absorption spectrum from the total dipole trajectory.
///
/// The dipole flux is computed via central finite difference:
/// `Ṁ(t) = (M(t+Δt) − M(t−Δt)) / (2·Δt)`. This eliminates the first
/// and last frames, so the effective trajectory length is `n_frames − 2`.
///
/// # Arguments
/// * `dipole_moments` — `(n_frames, 3)`, total system dipole vector
///   (units arbitrary; only the time variation matters).
/// * `dt_fs` — timestep between consecutive frames, femtoseconds.
/// * `resolution` — number of ACF lags retained. Clamped to
///   `n_frames − 3` (accounting for the two lost frames).
///
/// # Returns
/// [`Spectrum`] with `frequencies_cm1` and `intensities` (arbitrary units).
///
/// # Errors
/// * `EmptyInput` if `n_frames < 3` (need ≥3 for central difference).
/// * `DimensionMismatch` if the second dimension is not 3.
/// * `OutOfRange` if `dt_fs ≤ 0`.
pub fn ir_spectrum(
    dipole_moments: &Array2<f64>,
    dt_fs: f64,
    resolution: usize,
) -> Result<Spectrum, ComputeError> {
    let shape = dipole_moments.shape();
    let n_frames = shape[0];
    if shape[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[1],
            what: "dipole_moments (expected (n_frames, 3))",
        });
    }
    if n_frames < 3 {
        return Err(ComputeError::EmptyInput);
    }
    if dt_fs <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt_fs",
            value: dt_fs.to_string(),
        });
    }

    let flux_len = n_frames - 2;
    let max_lag = resolution.min(flux_len.saturating_sub(1));
    let inv_2dt = 0.5 / dt_fs;

    let mut planner = FftPlanner::new();
    let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);

    for d in 0..3 {
        let flux: Array1<f64> = (1..n_frames - 1)
            .map(|t| (dipole_moments[[t + 1, d]] - dipole_moments[[t - 1, d]]) * inv_2dt)
            .collect();

        let acf = sig::acf_fft_with_planner(&mut planner, &flux, max_lag).map_err(|e| {
            ComputeError::OutOfRange {
                field: "acf_fft",
                value: e.to_string(),
            }
        })?;
        for k in 0..=max_lag {
            acf_sum[k] += acf[k];
        }
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
    fn test_ir_spectrum_shapes() {
        let n = 50;
        let dm = Array2::from_elem((n, 3), 0.0);
        let result = ir_spectrum(&dm, 1.0, 10).unwrap();
        assert!(!result.frequencies_cm1.is_empty());
        assert_eq!(result.frequencies_cm1.len(), result.intensities.len());
        assert_eq!(result.n_frames, n);
    }

    #[test]
    fn test_ir_spectrum_constant_dipole() {
        let dm = Array2::from_elem((20, 3), 2.5);
        let result = ir_spectrum(&dm, 0.5, 5).unwrap();
        for j in 0..result.intensities.len() {
            assert!(
                result.intensities[j].abs() < 1e-10,
                "non-zero at bin {j}: {}",
                result.intensities[j]
            );
        }
    }

    #[test]
    fn test_ir_spectrum_oscillating_dipole() {
        // M_z(t) = sin(2π * 10 THz * t), dt = 0.5 fs, 4096 frames.
        // 10 THz = 0.01 fs⁻¹ → 333.56 cm⁻¹.
        let n = 4096;
        let dt_fs = 0.5;
        let freq_thz = 10.0;
        let mut dm = Array2::zeros((n, 3));
        for t in 0..n {
            let time_fs = t as f64 * dt_fs;
            dm[[t, 2]] = (2.0 * std::f64::consts::PI * freq_thz * 1e-3 * time_fs).sin();
        }
        let result = ir_spectrum(&dm, dt_fs, 200).unwrap();

        let max_idx = result
            .intensities
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let peak_cm1 = result.frequencies_cm1[max_idx];
        let expected_cm1 = 333.56;
        assert!(
            (peak_cm1 - expected_cm1).abs() < 100.0,
            "IR peak at {peak_cm1} cm⁻¹, expected ~{expected_cm1} cm⁻¹"
        );
    }

    #[test]
    fn test_ir_spectrum_rejects_wrong_shape() {
        let dm = Array2::from_elem((10, 4), 0.0);
        assert!(ir_spectrum(&dm, 1.0, 5).is_err());
    }

    #[test]
    fn test_ir_spectrum_rejects_too_few_frames() {
        let dm = Array2::zeros((2, 3));
        assert!(ir_spectrum(&dm, 0.5, 5).is_err());
    }

    #[test]
    fn test_ir_spectrum_rejects_nonpositive_dt() {
        let dm = Array2::zeros((10, 3));
        assert!(ir_spectrum(&dm, 0.0, 5).is_err());
        assert!(ir_spectrum(&dm, -1.0, 5).is_err());
    }

    #[test]
    fn test_immutability_ir_spectrum() {
        let dm = Array2::from_elem((20, 3), 1.0);
        let copy = dm.clone();
        ir_spectrum(&dm, 0.5, 5).unwrap();
        assert_eq!(dm, copy);
    }
}
