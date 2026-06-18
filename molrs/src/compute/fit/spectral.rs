//! Spectral-transform [`Fit`] impls: [`PowerSpectrum`], [`IRSpectrum`],
//! [`RamanSpectrum`].
//!
//! Each consumes a **raw autocorrelation function** (and `dt`, plus any
//! physical params) and applies window + FFT (+ prefactors) to produce a
//! frequency-domain spectrum. They compose [`molrs::signal`] window primitives
//! and the shared FFT core via the legacy
//! [`window_and_fft`](crate::compute::spectra::window_and_fft) /
//! [`acf_to_spectrum`](crate::compute::spectra::acf_to_spectrum) helpers —
//! **never** reimplementing window coefficients — so each `Fit` reproduces the
//! corresponding legacy free function (`power_spectrum` / `ir_spectrum` /
//! `raman_spectrum`) bit-for-bit when handed the same raw ACF those functions
//! build internally.
//!
//! # Units
//!
//! | quantity   | unit |
//! |------------|------|
//! | time / dt  | fs   |
//! | frequency  | cm⁻¹ |
//! | intensity  | arb. |

use ndarray::Array1;
use rustfft::FftPlanner;

use crate::compute::error::ComputeError;
use crate::compute::spectra::{
    RamanSpectrumResult, SpectrumResult, acf_to_intensities, acf_to_spectrum, bose_factor,
    cosine_sq_window, window_and_fft,
};
use crate::compute::traits::Fit;

/// Parallel polarization: `I_∥ = I_iso + (4/45)·I_aniso`.
const PARALLEL_ANISO_COEFF: f64 = 4.0 / 45.0;
/// Perpendicular polarization denominator: `I_⊥ = I_aniso / 15`.
const PERPENDICULAR_DENOM: f64 = 15.0;

/// Velocity power spectrum (VDOS) transform of a **raw velocity ACF**.
///
/// Applies the CosineSq window + zero-padded forward FFT (the
/// [`window_and_fft`](crate::compute::spectra::window_and_fft) pipeline) to a
/// raw, unnormalized velocity ACF — e.g. the `acf_sum` that the legacy
/// `power_spectrum` builds before windowing, or the
/// [`VacfResult`](super::VacfResult) of the [`VACF`](super::VACF) compute.
#[derive(Debug, Clone, Copy, Default)]
pub struct PowerSpectrum;

impl Fit for PowerSpectrum {
    /// `(acf, dt_fs)` — the raw velocity ACF (1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = SpectrumResult;

    /// Window + FFT the raw ACF into a VDOS spectrum.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF is empty.
    /// * [`ComputeError::OutOfRange`] if `dt_fs <= 0`.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (acf, dt_fs) = input;
        let n = acf.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if dt_fs <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt_fs",
                value: dt_fs.to_string(),
            });
        }
        let mut planner = FftPlanner::new();
        let (frequencies_cm1, intensities) = window_and_fft(&mut planner, acf, dt_fs)?;
        Ok(SpectrumResult {
            frequencies_cm1,
            intensities,
            resolution: n - 1,
            n_frames: n,
        })
    }
}

/// Infrared absorption spectrum transform of a **raw dipole-flux ACF**.
///
/// Identical window+FFT pipeline as [`PowerSpectrum`]; the difference between
/// IR and the power spectrum is entirely in *which* ACF is supplied (dipole
/// flux vs velocity), computed upstream. Reproduces the legacy `ir_spectrum`
/// bit-for-bit on the flux ACF that function builds internally.
#[derive(Debug, Clone, Copy, Default)]
pub struct IRSpectrum;

impl Fit for IRSpectrum {
    /// `(acf, dt_fs)` — the raw dipole-flux ACF (1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = SpectrumResult;

    /// Window + FFT the raw flux ACF into an IR spectrum.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF is empty.
    /// * [`ComputeError::OutOfRange`] if `dt_fs <= 0`.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (acf, dt_fs) = input;
        let n = acf.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if dt_fs <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt_fs",
                value: dt_fs.to_string(),
            });
        }
        let mut planner = FftPlanner::new();
        let (frequencies_cm1, intensities) = window_and_fft(&mut planner, acf, dt_fs)?;
        Ok(SpectrumResult {
            frequencies_cm1,
            intensities,
            resolution: n - 1,
            n_frames: n,
        })
    }
}

/// Raman spectrum transform of **raw isotropic + anisotropic ACFs**.
///
/// Applies the one CosineSq window (`cosine_sq_window`) to each ACF, FFTs both
/// (reusing the isotropic frequency grid), and applies the cross-section
/// (`(ν₀ − ν)⁴ / ν`) and Bose (`1/(1 − exp(−hcν/kT))`) prefactors — the exact
/// tail of the legacy `raman_spectrum`. Reproduces that function bit-for-bit on
/// the iso/aniso ACFs it builds internally.
#[derive(Debug, Clone, Copy)]
pub struct RamanSpectrum {
    /// Laser frequency, cm⁻¹. The cross-section correction scales as
    /// `(ν₀ − ν)⁴ / ν`. Set to `0.0` to skip.
    pub incident_frequency_cm1: f64,
    /// Temperature, K, for the Bose factor. Set to `0.0` to skip.
    pub temperature_k: f64,
    /// If `true`, also emit parallel / perpendicular polarization spectra.
    pub averaged: bool,
}

impl Fit for RamanSpectrum {
    /// `(acf_iso, acf_aniso, dt_fs)` — the raw isotropic and (weighted)
    /// anisotropic ACFs (equal-length 1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>, f64);
    type Output = RamanSpectrumResult;

    /// Window + FFT + physical prefactors on the iso/aniso ACFs.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if either ACF is empty.
    /// * [`ComputeError::DimensionMismatch`] if the two ACFs differ in length.
    /// * [`ComputeError::OutOfRange`] if `dt_fs <= 0`.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (acf_iso, acf_aniso, dt_fs) = input;
        let n = acf_iso.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if acf_aniso.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: acf_aniso.len(),
                what: "Raman (acf_iso, acf_aniso) lengths",
            });
        }
        if dt_fs <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt_fs",
                value: dt_fs.to_string(),
            });
        }

        let max_lag = n - 1;
        // Pre-compute CosineSq window once and apply to both ACFs — identical to
        // the legacy raman_spectrum.
        let window = cosine_sq_window(max_lag + 1);
        let win_iso: Array1<f64> = acf_iso.iter().zip(&window).map(|(a, w)| a * w).collect();
        let win_aniso: Array1<f64> = acf_aniso.iter().zip(&window).map(|(a, w)| a * w).collect();

        let mut planner = FftPlanner::new();
        let n_pad = (4 * (max_lag + 1)).next_power_of_two();
        let (frequencies_cm1, raw_iso) = acf_to_spectrum(&mut planner, &win_iso, dt_fs, n_pad);
        let raw_aniso = acf_to_intensities(&mut planner, &win_aniso, n_pad);

        let mut iso_int = raw_iso;
        let mut aniso_int = raw_aniso;

        let apply_cross_section = self.incident_frequency_cm1 > 0.0;
        let apply_bose = self.temperature_k > 0.0;

        for j in 0..frequencies_cm1.len() {
            let nu = frequencies_cm1[j];
            let mut correction = 1.0;
            if apply_cross_section && nu > 0.0 {
                let dnu = self.incident_frequency_cm1 - nu;
                correction *= dnu.powi(4) / nu;
            }
            if apply_bose {
                correction *= bose_factor(nu, self.temperature_k);
            }
            iso_int[j] *= correction;
            aniso_int[j] *= correction;
        }

        let (parallel, perpendicular) = if self.averaged {
            let m = frequencies_cm1.len();
            let mut par = Array1::zeros(m);
            let mut perp = Array1::zeros(m);
            for j in 0..m {
                par[j] = iso_int[j] + PARALLEL_ANISO_COEFF * aniso_int[j];
                perp[j] = aniso_int[j] / PERPENDICULAR_DENOM;
            }
            (Some(par), Some(perp))
        } else {
            (None, None)
        };

        Ok(RamanSpectrumResult {
            frequencies_cm1,
            isotropic: iso_int,
            anisotropic: aniso_int,
            parallel,
            perpendicular,
            resolution: max_lag,
            n_frames: n,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::spectra::{ir_spectrum, power_spectrum, raman_spectrum};
    use molrs::signal as sig;
    use ndarray::{Array1, Array2};

    /// Rebuild the raw velocity ACF that power_spectrum builds internally
    /// (per-dof mean-subtract, FFT-ACF, sum, *= 1/n_dof).
    fn power_acf(velocities: &Array2<f64>, max_lag: usize) -> Array1<f64> {
        let n_frames = velocities.shape()[0];
        let n_dof = velocities.shape()[1];
        let inv_n_frames = 1.0 / n_frames as f64;
        let mut planner = FftPlanner::new();
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..n_dof {
            let mut col: Array1<f64> = (0..n_frames).map(|t| velocities[[t, d]]).collect();
            let mean: f64 = col.iter().sum::<f64>() * inv_n_frames;
            for v in col.iter_mut() {
                *v -= mean;
            }
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        let inv_n_dof = 1.0 / n_dof as f64;
        for k in 0..=max_lag {
            acf_sum[k] *= inv_n_dof;
        }
        acf_sum
    }

    /// Rebuild the raw dipole-flux ACF that ir_spectrum builds internally.
    fn ir_acf(dm: &Array2<f64>, dt_fs: f64, max_lag: usize) -> Array1<f64> {
        let n_frames = dm.shape()[0];
        let inv_2dt = 0.5 / dt_fs;
        let mut planner = FftPlanner::new();
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let flux: Array1<f64> = (1..n_frames - 1)
                .map(|t| (dm[[t + 1, d]] - dm[[t - 1, d]]) * inv_2dt)
                .collect();
            let acf = sig::acf_fft_with_planner(&mut planner, &flux, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        acf_sum
    }

    /// Rebuild the raw iso/aniso ACFs that raman_spectrum builds internally.
    fn raman_acfs(pol: &Array2<f64>, dt_fs: f64, max_lag: usize) -> (Array1<f64>, Array1<f64>) {
        const DIAG_W: f64 = 0.5;
        const OFFDIAG_W: f64 = 3.0;
        let n_frames = pol.shape()[0];
        let inv_2dt = 0.5 / dt_fs;
        let flux_len = n_frames - 2;
        let mut iso = Vec::with_capacity(flux_len);
        let mut comps: [Vec<f64>; 6] = Default::default();
        for t in 1..n_frames - 1 {
            let p = pol.row(t - 1);
            let q = pol.row(t + 1);
            let xx = (q[0] - p[0]) * inv_2dt;
            let yy = (q[1] - p[1]) * inv_2dt;
            let zz = (q[2] - p[2]) * inv_2dt;
            iso.push((xx + yy + zz) / 3.0);
            comps[0].push(xx - yy);
            comps[1].push(yy - zz);
            comps[2].push(zz - xx);
            comps[3].push((q[3] - p[3]) * inv_2dt);
            comps[4].push((q[4] - p[4]) * inv_2dt);
            comps[5].push((q[5] - p[5]) * inv_2dt);
        }
        let mut planner = FftPlanner::new();
        let iso_series = Array1::from_vec(iso);
        let acf_iso = sig::acf_fft_with_planner(&mut planner, &iso_series, max_lag).unwrap();
        let mut acf_aniso = Array1::<f64>::zeros(max_lag + 1);
        for (c, comp) in comps.iter_mut().enumerate() {
            let w = if c < 3 { DIAG_W } else { OFFDIAG_W };
            let col = Array1::from_vec(std::mem::take(comp));
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_aniso[k] += w * acf[k];
            }
        }
        (acf_iso, acf_aniso)
    }

    fn sine_velocities(n: usize, dt_fs: f64, freq_thz: f64) -> Array2<f64> {
        let mut v = Array2::zeros((n, 3));
        for t in 0..n {
            let tf = t as f64 * dt_fs;
            v[[t, 0]] = (2.0 * std::f64::consts::PI * freq_thz * 1e-3 * tf).sin();
        }
        v
    }

    #[test]
    fn power_spectrum_fit_matches_legacy_bit_for_bit() {
        // ac-006.
        let n = 1024;
        let dt = 0.5;
        let res = 200;
        let v = sine_velocities(n, dt, 10.0);
        let legacy = power_spectrum(&v, dt, res).unwrap();
        let max_lag = res.min(n - 1);
        let acf = power_acf(&v, max_lag);
        let fitted = PowerSpectrum.fit((&acf, dt)).unwrap();
        assert_eq!(fitted.frequencies_cm1, legacy.frequencies_cm1);
        assert_eq!(fitted.intensities, legacy.intensities);
    }

    #[test]
    fn ir_spectrum_fit_matches_legacy_bit_for_bit() {
        // ac-007 (IR).
        let n = 1024;
        let dt = 0.5;
        let res = 200;
        let mut dm = Array2::zeros((n, 3));
        for t in 0..n {
            let tf = t as f64 * dt;
            dm[[t, 2]] = (2.0 * std::f64::consts::PI * 10.0 * 1e-3 * tf).sin();
        }
        let legacy = ir_spectrum(&dm, dt, res).unwrap();
        let flux_len = n - 2;
        let max_lag = res.min(flux_len - 1);
        let acf = ir_acf(&dm, dt, max_lag);
        let fitted = IRSpectrum.fit((&acf, dt)).unwrap();
        assert_eq!(fitted.frequencies_cm1, legacy.frequencies_cm1);
        assert_eq!(fitted.intensities, legacy.intensities);
    }

    #[test]
    fn raman_spectrum_fit_matches_legacy_bit_for_bit() {
        // ac-007 (Raman): iso/aniso/parallel/perpendicular.
        let n = 256;
        let dt = 0.5;
        let res = 60;
        let mut pol = Array2::zeros((n, 6));
        for t in 0..n {
            let tf = t as f64 * dt;
            let val = (2.0 * std::f64::consts::PI * 30.0 * 1e-3 * tf).sin();
            for c in 0..6 {
                pol[[t, c]] = val * (1.0 + 0.1 * c as f64);
            }
        }
        let incident = 10000.0;
        let temp = 300.0;
        let legacy = raman_spectrum(&pol, dt, res, incident, temp, true).unwrap();

        let flux_len = n - 2;
        let max_lag = res.min(flux_len - 1);
        let (acf_iso, acf_aniso) = raman_acfs(&pol, dt, max_lag);
        let fitted = RamanSpectrum {
            incident_frequency_cm1: incident,
            temperature_k: temp,
            averaged: true,
        }
        .fit((&acf_iso, &acf_aniso, dt))
        .unwrap();

        assert_eq!(fitted.frequencies_cm1, legacy.frequencies_cm1);
        assert_eq!(fitted.isotropic, legacy.isotropic);
        assert_eq!(fitted.anisotropic, legacy.anisotropic);
        assert_eq!(fitted.parallel, legacy.parallel);
        assert_eq!(fitted.perpendicular, legacy.perpendicular);
    }

    #[test]
    fn raman_rejects_mismatched_acf_lengths() {
        // ac-016: shape mismatch -> DimensionMismatch.
        let iso = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let aniso = Array1::from_vec(vec![1.0, 2.0]);
        let err = RamanSpectrum {
            incident_frequency_cm1: 0.0,
            temperature_k: 0.0,
            averaged: false,
        }
        .fit((&iso, &aniso, 0.5))
        .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    #[test]
    fn power_rejects_empty_and_bad_dt() {
        let empty: Array1<f64> = Array1::from_vec(vec![]);
        assert!(matches!(
            PowerSpectrum.fit((&empty, 0.5)),
            Err(ComputeError::EmptyInput)
        ));
        let acf = Array1::from_vec(vec![1.0, 0.5, 0.25]);
        assert!(matches!(
            PowerSpectrum.fit((&acf, 0.0)),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
