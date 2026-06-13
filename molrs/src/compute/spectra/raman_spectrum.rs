//! Raman spectrum from polarizability-tensor time series.
//!
//! Computes isotropic and anisotropic Raman spectra via autocorrelation
//! of the polarizability time derivative, with optional cross-section
//! correction and Bose factor.
//!
//! Reference: Long, D. A. *The Raman Effect*, 2nd ed. Wiley (2002).

use molrs::signal as sig;
use ndarray::Array1;
use rustfft::FftPlanner;

use super::{RamanSpectrum, acf_to_intensities, acf_to_spectrum, cosine_sq_window};
use crate::compute::error::ComputeError;

/// Weight for diagonal anisotropy components in the Raman ACF.
const DIAG_ANISO_WEIGHT: f64 = 0.5;
/// Weight for off-diagonal anisotropy components in the Raman ACF.
const OFFDIAG_ANISO_WEIGHT: f64 = 3.0;
/// Number of anisotropy components (3 diagonal diffs + 3 off-diagonals).
const N_ANISO_COMPS: usize = 6;
/// Parallel polarization: `I_∥ = I_iso + (4/45)·I_aniso`.
const PARALLEL_ANISO_COEFF: f64 = 4.0 / 45.0;
/// Perpendicular polarization denominator: `I_⊥ = I_aniso / 15`.
const PERPENDICULAR_DENOM: f64 = 15.0;

/// Compute the Raman spectrum from polarizability time series.
///
/// The polarizability derivative is computed via central finite
/// difference, then decomposed into isotropic and anisotropic
/// components.
///
/// # Arguments
/// * `polarizabilities` — `(n_frames, 6)`, Voigt notation:
///   `[α_xx, α_yy, α_zz, α_xy, α_xz, α_yz]`.
/// * `dt_fs` — timestep, femtoseconds.
/// * `resolution` — number of ACF lags.
/// * `incident_frequency_cm1` — laser frequency in cm⁻¹. The
///   cross-section correction scales as `(ν₀ − ν)⁴ / ν`.
///   Set to `0.0` to skip (arbitrary units).
/// * `temperature_k` — temperature in Kelvin for the Bose factor
///   `1 / (1 − exp(−h·c·ν / (k_B·T)))`. Set to `0.0` to skip.
/// * `averaged` — if `true`, compute parallel / perpendicular
///   polarization-resolved spectra.
///
/// # Returns
/// [`RamanSpectrum`] with `isotropic` and `anisotropic` components.
/// When `averaged = true`, `parallel` and `perpendicular` are `Some`.
///
/// # Errors
/// * `EmptyInput` if `n_frames < 3`.
/// * `DimensionMismatch` if the second dimension is not 6.
/// * `OutOfRange` if `dt_fs ≤ 0`.
pub fn raman_spectrum(
    polarizabilities: &ndarray::Array2<f64>,
    dt_fs: f64,
    resolution: usize,
    incident_frequency_cm1: f64,
    temperature_k: f64,
    averaged: bool,
) -> Result<RamanSpectrum, ComputeError> {
    let shape = polarizabilities.shape();
    let n_frames = shape[0];
    if shape[1] != 6 {
        return Err(ComputeError::DimensionMismatch {
            expected: 6,
            got: shape[1],
            what: "polarizabilities (expected (n_frames, 6) Voigt)",
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

    // Build flux component vectors directly (no intermediate 2D array).
    let mut iso = Vec::with_capacity(flux_len);
    let mut aniso_comps: [Vec<f64>; N_ANISO_COMPS] = [
        Vec::with_capacity(flux_len), // α_xx − α_yy
        Vec::with_capacity(flux_len), // α_yy − α_zz
        Vec::with_capacity(flux_len), // α_zz − α_xx
        Vec::with_capacity(flux_len), // α_xy
        Vec::with_capacity(flux_len), // α_xz
        Vec::with_capacity(flux_len), // α_yz
    ];

    for t in 1..n_frames - 1 {
        let a_prev = polarizabilities.row(t - 1);
        let a_next = polarizabilities.row(t + 1);

        let xx_dot = (a_next[0] - a_prev[0]) * inv_2dt;
        let yy_dot = (a_next[1] - a_prev[1]) * inv_2dt;
        let zz_dot = (a_next[2] - a_prev[2]) * inv_2dt;

        iso.push((xx_dot + yy_dot + zz_dot) / 3.0);
        aniso_comps[0].push(xx_dot - yy_dot);
        aniso_comps[1].push(yy_dot - zz_dot);
        aniso_comps[2].push(zz_dot - xx_dot);
        aniso_comps[3].push((a_next[3] - a_prev[3]) * inv_2dt); // xy
        aniso_comps[4].push((a_next[4] - a_prev[4]) * inv_2dt); // xz
        aniso_comps[5].push((a_next[5] - a_prev[5]) * inv_2dt); // yz
    }

    let mut planner = FftPlanner::new();

    // Isotropic ACF.
    let iso_series = Array1::from_vec(iso);
    let acf_iso = sig::acf_fft_with_planner(&mut planner, &iso_series, max_lag).map_err(|e| {
        ComputeError::OutOfRange {
            field: "acf_fft",
            value: e.to_string(),
        }
    })?;

    // Anisotropic ACF: weighted sum of component ACFs.
    let mut acf_aniso = Array1::<f64>::zeros(max_lag + 1);
    for (c, comp) in aniso_comps.iter_mut().enumerate() {
        let weight = if c < 3 {
            DIAG_ANISO_WEIGHT
        } else {
            OFFDIAG_ANISO_WEIGHT
        };
        let col = Array1::from_vec(std::mem::take(comp));
        let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).map_err(|e| {
            ComputeError::OutOfRange {
                field: "acf_fft",
                value: e.to_string(),
            }
        })?;
        for k in 0..=max_lag {
            acf_aniso[k] += weight * acf[k];
        }
    }

    // Pre-compute CosineSq window once and apply to both ACFs.
    let window = cosine_sq_window(max_lag + 1);
    let win_iso: Array1<f64> = acf_iso.iter().zip(&window).map(|(a, w)| a * w).collect();
    let win_aniso: Array1<f64> = acf_aniso.iter().zip(&window).map(|(a, w)| a * w).collect();

    // FFT both; reuse frequencies from the isotropic call.
    let n_pad = (4 * (max_lag + 1)).next_power_of_two();
    let (frequencies_cm1, raw_iso) = acf_to_spectrum(&mut planner, &win_iso, dt_fs, n_pad);
    let raw_aniso = acf_to_intensities(&mut planner, &win_aniso, n_pad);

    let mut iso_int = raw_iso;
    let mut aniso_int = raw_aniso;

    // Cross-section correction and Bose factor.
    let apply_cross_section = incident_frequency_cm1 > 0.0;
    let apply_bose = temperature_k > 0.0;

    for j in 0..frequencies_cm1.len() {
        let nu = frequencies_cm1[j];
        let mut correction = 1.0;

        if apply_cross_section && nu > 0.0 {
            let dnu = incident_frequency_cm1 - nu;
            correction *= dnu.powi(4) / nu;
        }
        if apply_bose {
            correction *= super::bose_factor(nu, temperature_k);
        }

        iso_int[j] *= correction;
        aniso_int[j] *= correction;
    }

    let (parallel, perpendicular) = if averaged {
        let n = frequencies_cm1.len();
        let mut par = Array1::zeros(n);
        let mut perp = Array1::zeros(n);
        for j in 0..n {
            par[j] = iso_int[j] + PARALLEL_ANISO_COEFF * aniso_int[j];
            perp[j] = aniso_int[j] / PERPENDICULAR_DENOM;
        }
        (Some(par), Some(perp))
    } else {
        (None, None)
    };

    Ok(RamanSpectrum {
        frequencies_cm1,
        isotropic: iso_int,
        anisotropic: aniso_int,
        parallel,
        perpendicular,
        resolution: max_lag,
        n_frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_raman_spectrum_not_averaged() {
        let n = 100;
        let pol = Array2::from_elem((n, 6), 0.1);
        let result = raman_spectrum(&pol, 1.0, 10, 0.0, 300.0, false).unwrap();
        assert_eq!(result.frequencies_cm1.len(), result.isotropic.len());
        assert_eq!(result.frequencies_cm1.len(), result.anisotropic.len());
        assert!(result.parallel.is_none());
        assert!(result.perpendicular.is_none());
        assert_eq!(result.n_frames, n);
    }

    #[test]
    fn test_raman_spectrum_averaged() {
        let n = 100;
        let pol = Array2::from_elem((n, 6), 0.1);
        let result = raman_spectrum(&pol, 1.0, 10, 0.0, 0.0, true).unwrap();
        assert!(result.parallel.is_some());
        assert!(result.perpendicular.is_some());
        let par = result.parallel.as_ref().unwrap();
        let perp = result.perpendicular.as_ref().unwrap();
        assert_eq!(par.len(), result.frequencies_cm1.len());
        assert_eq!(perp.len(), result.frequencies_cm1.len());
    }

    #[test]
    fn test_raman_spectrum_varying_signal() {
        // Pure isotropic: diagonal equal, off-diagonals zero.
        let n = 256;
        let dt_fs = 0.5;
        let freq_thz = 30.0;
        let mut pol = Array2::zeros((n, 6));
        for t in 0..n {
            let time_fs = t as f64 * dt_fs;
            let val = (2.0 * std::f64::consts::PI * freq_thz * 1e-3 * time_fs).sin();
            pol[[t, 0]] = val;
            pol[[t, 1]] = val;
            pol[[t, 2]] = val;
        }
        let result = raman_spectrum(&pol, dt_fs, 60, 0.0, 0.0, false).unwrap();

        let iso_rms: f64 = (result.isotropic.iter().map(|x| x * x).sum::<f64>()
            / result.isotropic.len() as f64)
            .sqrt();
        assert!(iso_rms > 0.01, "isotropic RMS too low: {iso_rms}");

        let aniso_max = result
            .anisotropic
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        assert!(
            aniso_max < 1e-8,
            "anisotropic unexpectedly non-zero: {aniso_max}"
        );
    }

    #[test]
    fn test_raman_spectrum_cross_section_factor() {
        let n = 100;
        let dt_fs = 0.5;
        let mut pol = Array2::zeros((n, 6));
        for t in 0..n {
            let time_fs = t as f64 * dt_fs;
            let val = (2.0 * std::f64::consts::PI * 30.0 * 1e-3 * time_fs).sin();
            for c in 0..6 {
                pol[[t, c]] = val;
            }
        }
        let r1 = raman_spectrum(&pol, dt_fs, 20, 0.0, 0.0, false).unwrap();
        let r2 = raman_spectrum(&pol, dt_fs, 20, 10000.0, 0.0, false).unwrap();
        let diff: f64 = r1
            .isotropic
            .iter()
            .zip(r2.isotropic.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "cross-section had no effect");
    }

    #[test]
    fn test_raman_spectrum_rejects_wrong_shape() {
        let pol = Array2::from_elem((10, 9), 0.0);
        assert!(raman_spectrum(&pol, 1.0, 5, 0.0, 0.0, false).is_err());
    }

    #[test]
    fn test_raman_spectrum_rejects_too_few_frames() {
        let pol = Array2::zeros((2, 6));
        assert!(raman_spectrum(&pol, 0.5, 5, 0.0, 0.0, false).is_err());
    }

    #[test]
    fn test_raman_spectrum_rejects_nonpositive_dt() {
        let pol = Array2::zeros((10, 6));
        assert!(raman_spectrum(&pol, 0.0, 5, 0.0, 0.0, false).is_err());
    }

    #[test]
    fn test_immutability_raman_spectrum() {
        let pol = Array2::from_elem((20, 6), 1.0);
        let copy = pol.clone();
        raman_spectrum(&pol, 0.5, 5, 0.0, 0.0, false).unwrap();
        assert_eq!(pol, copy);
    }

    #[test]
    fn test_raman_bose_factor() {
        let n = 100;
        let dt_fs = 0.5;
        let mut pol = Array2::zeros((n, 6));
        for t in 0..n {
            let time_fs = t as f64 * dt_fs;
            let val = (2.0 * std::f64::consts::PI * 30.0 * 1e-3 * time_fs).sin();
            for c in 0..6 {
                pol[[t, c]] = val;
            }
        }
        let r_cold = raman_spectrum(&pol, dt_fs, 20, 0.0, 0.0, false).unwrap();
        let r_hot = raman_spectrum(&pol, dt_fs, 20, 0.0, 300.0, false).unwrap();
        let diff: f64 = r_cold
            .isotropic
            .iter()
            .zip(r_hot.isotropic.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "Bose factor had no effect");
    }
}
