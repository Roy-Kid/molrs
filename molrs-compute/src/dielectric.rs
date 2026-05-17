//! Dielectric susceptibility computation.
//!
//! Six single-responsibility functions for computing dielectric response
//! from dipole moment and current density time series. Functions compose
//! `molrs_signal` primitives (acf_fft, apply_window, frequency_grid)
//! — they do NOT reimplement FFT or window generation.

use molrs_signal as sig;
use ndarray::{Array1, Array2, Array3};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use rustfft::num_traits::Zero;

use crate::error::ComputeError;

/// Result of a dielectric spectrum computation.
#[derive(Debug, Clone)]
pub struct DielectricSpectrum {
    pub frequencies: Array1<f64>,
    pub epsilon_real: Array1<f64>,
    pub epsilon_imag: Array1<f64>,
    pub n_frames: usize,
    pub n_correlation_steps: usize,
}

// ── Physical constants (MD real units: kcal, mol, Angstrom, e, K) ─────────

const KAPPA: f64 = 332.0637; // 1/(4*pi*epsilon_0) in kcal·Å·mol⁻¹·e⁻²
const K_B: f64 = 1.98720425864083e-3; // Boltzmann constant in kcal/(mol·K)
const FOUR_PI_OVER_3: f64 = 4.1887902047863905; // 4π/3

// ── Basic observables ─────────────────────────────────────────────────────

/// Total dipole moment: M = Σ q_i * r_i.
///
/// Returns a length-3 vector in e·Å.
pub fn compute_dipole_moment(
    charges: &Array1<f64>,
    positions: &Array2<f64>,
) -> Result<Array1<f64>, ComputeError> {
    let n = charges.len();
    if positions.shape() != [n, 3] {
        return Err(ComputeError::DimensionMismatch {
            expected: n * 3,
            got: positions.len(),
            what: "positions (expected (n_atoms, 3))",
        });
    }
    let mut m = Array1::zeros(3);
    for i in 0..n {
        let q = charges[i];
        if !q.is_finite() {
            return Err(ComputeError::NonFinite {
                where_: "charges",
                index: i,
            });
        }
        m[0] += q * positions[[i, 0]];
        m[1] += q * positions[[i, 1]];
        m[2] += q * positions[[i, 2]];
    }
    Ok(m)
}

/// Current density: J(t) = ΔM(t) / (V * Δt).
///
/// First row is NaN (no previous frame). Returns (n_frames, 3).
pub fn compute_current_density(
    dipole_moments: &Array2<f64>,
    dt: f64,
    volume: f64,
) -> Result<Array2<f64>, ComputeError> {
    let shape = dipole_moments.shape();
    if shape[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[1],
            what: "dipole_moments (expected (n_frames, 3))",
        });
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    if volume <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "volume",
            value: volume.to_string(),
        });
    }
    let n_frames = shape[0];
    let mut j = Array2::from_elem((n_frames, 3), f64::NAN);
    if n_frames < 2 {
        return Ok(j);
    }
    let scale = 1.0 / (volume * dt);
    for t in 1..n_frames {
        for d in 0..3 {
            j[[t, d]] = (dipole_moments[[t, d]] - dipole_moments[[t - 1, d]]) * scale;
        }
    }
    Ok(j)
}

// ── Static dielectric constant ─────────────────────────────────────────────

/// Static dielectric constant via Neumann fluctuation formula.
pub fn static_dielectric_constant(
    dipole_moments: &Array2<f64>,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
) -> Result<f64, ComputeError> {
    let shape = dipole_moments.shape();
    if shape[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[1],
            what: "dipole_moments (expected (n_frames, 3))",
        });
    }
    if shape[0] < 2 {
        return Err(ComputeError::EmptyInput);
    }
    if volume <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "volume",
            value: volume.to_string(),
        });
    }
    if temperature <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "temperature",
            value: temperature.to_string(),
        });
    }

    // Mean dipole moment
    let mut mean_m = Array1::<f64>::zeros(3);
    for t in 0..shape[0] {
        for d in 0..3 {
            mean_m[d] += dipole_moments[[t, d]];
        }
    }
    let n = shape[0] as f64;
    for d in 0..3 {
        mean_m[d] /= n;
    }

    // Mean squared dipole moment
    let mut m_sq = 0.0;
    for t in 0..shape[0] {
        for d in 0..3 {
            m_sq += dipole_moments[[t, d]].powi(2);
        }
    }
    m_sq /= n;

    // Squared mean
    let mean_sq = mean_m[0].powi(2) + mean_m[1].powi(2) + mean_m[2].powi(2);

    let variance = m_sq - mean_sq;
    let prefactor = FOUR_PI_OVER_3 * KAPPA / (volume * K_B * temperature);

    Ok(epsilon_inf + prefactor * variance)
}

// ── Frequency-domain spectra ───────────────────────────────────────────────

/// Convert windowed ACF to frequency-domain spectrum via one-sided FT.
fn acf_to_spectrum(
    acf: &Array1<f64>,
    dt: f64,
    n_pad: usize,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut planner = FftPlanner::new();
    let fwd = planner.plan_fft_forward(n_pad);

    let mut complex_data: Vec<Complex64> = acf.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    complex_data.resize(n_pad, Complex64::zero());
    fwd.process(&mut complex_data);

    let frequencies = sig::frequency_grid(n_pad, dt);
    let n_freq = frequencies.len();
    let mut eps_real = Array1::zeros(n_freq);
    let mut eps_imag = Array1::zeros(n_freq);

    for j in 0..n_freq {
        let z = complex_data[j];
        eps_real[j] = z.re / n_pad as f64;
        eps_imag[j] = -z.im / n_pad as f64;
    }

    (frequencies, eps_real, eps_imag)
}

/// Einstein-Helfand route: dipole ACF → window → FT → ε*(ω).
pub fn einstein_helfand_spectrum(
    dipole_moments: &Array2<f64>,
    dt: f64,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
    max_correlation_time: usize,
    window_type: &str,
) -> Result<DielectricSpectrum, ComputeError> {
    let shape = dipole_moments.shape();
    if shape[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[1],
            what: "dipole_moments",
        });
    }
    if shape[0] < 2 {
        return Err(ComputeError::EmptyInput);
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    if volume <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "volume",
            value: volume.to_string(),
        });
    }
    if temperature <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "temperature",
            value: temperature.to_string(),
        });
    }

    let n_frames = shape[0];
    let max_lag = max_correlation_time.min(n_frames - 1);

    // ACF per component then average
    let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
    for d in 0..3 {
        let col: Array1<f64> = dipole_moments.column(d).to_owned();
        let acf = sig::acf_fft(&col, max_lag).map_err(|e| ComputeError::OutOfRange {
            field: "acf_fft",
            value: e.to_string(),
        })?;
        for k in 0..=max_lag {
            acf_sum[k] += acf[k];
        }
    }

    // Apply window
    let acf_dyn = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[max_lag + 1]), acf_sum.to_vec())
        .map_err(|e| ComputeError::BadShape {
            expected: "1d".into(),
            got: e.to_string(),
        })?;
    let wt = match window_type {
        "hann" => sig::WindowType::Hann,
        "blackman" => sig::WindowType::Blackman,
        other => {
            return Err(ComputeError::OutOfRange {
                field: "window_type",
                value: other.into(),
            });
        }
    };
    let windowed = sig::apply_window(&acf_dyn, wt, 0).map_err(|e| ComputeError::OutOfRange {
        field: "apply_window",
        value: e.to_string(),
    })?;
    let windowed_1d: Array1<f64> = windowed.iter().copied().collect();

    // FT to frequency domain
    let n_pad = (2 * (max_lag + 1)).next_power_of_two();
    let (frequencies, raw_real, raw_imag) = acf_to_spectrum(&windowed_1d, dt, n_pad);

    // Normalize with dielectric prefactor
    let prefactor = FOUR_PI_OVER_3 * KAPPA / (volume * K_B * temperature);
    let n_freq = frequencies.len();
    let mut eps_real = Array1::zeros(n_freq);
    let mut eps_imag = Array1::zeros(n_freq);
    for j in 0..n_freq {
        eps_real[j] = epsilon_inf + prefactor * raw_real[j];
        eps_imag[j] = prefactor * raw_imag[j];
    }

    Ok(DielectricSpectrum {
        frequencies,
        epsilon_real: eps_real,
        epsilon_imag: eps_imag,
        n_frames,
        n_correlation_steps: max_lag + 1,
    })
}

/// Green-Kubo route: current ACF → window → FT → σ(ω) → ε*(ω).
pub fn green_kubo_spectrum(
    current_density: &Array2<f64>,
    dt: f64,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
    max_correlation_time: usize,
    window_type: &str,
) -> Result<DielectricSpectrum, ComputeError> {
    let shape = current_density.shape();
    if shape[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[1],
            what: "current_density",
        });
    }
    if shape[0] < 2 {
        return Err(ComputeError::EmptyInput);
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    if volume <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "volume",
            value: volume.to_string(),
        });
    }
    if temperature <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "temperature",
            value: temperature.to_string(),
        });
    }

    let n_frames = shape[0];
    let max_lag = max_correlation_time.min(n_frames - 1);

    // Skip NaN first row in J(t)
    let start = 1;

    // ACF per component
    let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
    for d in 0..3 {
        let col: Vec<f64> = (start..n_frames).map(|t| current_density[[t, d]]).collect();
        let col_arr = Array1::from_vec(col);
        let acf = sig::acf_fft(&col_arr, max_lag).map_err(|e| ComputeError::OutOfRange {
            field: "acf_fft",
            value: e.to_string(),
        })?;
        for k in 0..=max_lag {
            acf_sum[k] += acf[k];
        }
    }

    // Window
    let acf_dyn = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[max_lag + 1]), acf_sum.to_vec())
        .map_err(|e| ComputeError::BadShape {
            expected: "1d".into(),
            got: e.to_string(),
        })?;
    let wt = match window_type {
        "hann" => sig::WindowType::Hann,
        "blackman" => sig::WindowType::Blackman,
        other => {
            return Err(ComputeError::OutOfRange {
                field: "window_type",
                value: other.into(),
            });
        }
    };
    let windowed = sig::apply_window(&acf_dyn, wt, 0).map_err(|e| ComputeError::OutOfRange {
        field: "apply_window",
        value: e.to_string(),
    })?;
    let windowed_1d: Array1<f64> = windowed.iter().copied().collect();

    // FT
    let n_pad = (2 * (max_lag + 1)).next_power_of_two();
    let (frequencies, raw_real, raw_imag) = acf_to_spectrum(&windowed_1d, dt, n_pad);

    // Convert conductivity spectrum to dielectric
    // σ(ω) = (1/(3*V*k_B*T)) * FT[C_J]
    // ε*(ω) = ε_∞ + i*σ(ω)/(ε_0*ω)
    // In MD units: ε*(ω) = ε_∞ + i * σ * (4π*KAPPA) / ω
    let sigma_prefactor = 1.0 / (3.0 * volume * K_B * temperature);
    let n_freq = frequencies.len();
    let mut eps_real = Array1::zeros(n_freq);
    let mut eps_imag = Array1::zeros(n_freq);
    let eps0_factor = 4.0 * std::f64::consts::PI * KAPPA;
    for j in 0..n_freq {
        let sigma_re = sigma_prefactor * raw_real[j];
        let sigma_im = sigma_prefactor * raw_imag[j];
        let omega = frequencies[j];
        if omega < 1e-30 {
            // DC limit: ε_real = ε_∞, ε_imag = 0
            eps_real[j] = epsilon_inf;
            eps_imag[j] = 0.0;
        } else {
            eps_real[j] = epsilon_inf + eps0_factor * sigma_im / omega;
            eps_imag[j] = eps0_factor * sigma_re / omega;
        }
    }

    Ok(DielectricSpectrum {
        frequencies,
        epsilon_real: eps_real,
        epsilon_imag: eps_imag,
        n_frames,
        n_correlation_steps: max_lag + 1,
    })
}

// ── System decomposition ───────────────────────────────────────────────────

/// Decompose per-particle current into water and ion components.
///
/// Returns (J_water, J_ion) each of shape (n_frames, 3).
pub fn decompose_current(
    per_particle_current: &Array3<f64>,
    water_mask: &Array1<bool>,
) -> Result<(Array2<f64>, Array2<f64>), ComputeError> {
    let shape = per_particle_current.shape();
    let n_particles = shape[0];
    let n_frames = shape[1];
    if shape[2] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[2],
            what: "per_particle_current (expected (n_particles, n_frames, 3))",
        });
    }
    if water_mask.len() != n_particles {
        return Err(ComputeError::DimensionMismatch {
            expected: n_particles,
            got: water_mask.len(),
            what: "water_mask",
        });
    }

    let mut j_water = Array2::zeros((n_frames, 3));
    let mut j_ion = Array2::zeros((n_frames, 3));

    for p in 0..n_particles {
        let target = if water_mask[p] {
            &mut j_water
        } else {
            &mut j_ion
        };
        for t in 0..n_frames {
            for d in 0..3 {
                target[[t, d]] += per_particle_current[[p, t, d]];
            }
        }
    }

    Ok((j_water, j_ion))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Axis, arr1};

    #[test]
    fn test_dipole_moment_two_charges() {
        let charges = arr1(&[1.0, -1.0]);
        let positions = ndarray::arr2(&[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let m = compute_dipole_moment(&charges, &positions).unwrap();
        assert!((m[0] - 2.0).abs() < 1e-10);
        assert!((m[1] - 0.0).abs() < 1e-10);
        assert!((m[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dipole_moment_zero_charge() {
        let charges = arr1(&[0.0, 0.0, 0.0]);
        let positions = ndarray::Array2::zeros((3, 3));
        let m = compute_dipole_moment(&charges, &positions).unwrap();
        assert!((m[0].abs() + m[1].abs() + m[2].abs()) < 1e-10);
    }

    #[test]
    fn test_dipole_moment_wrong_shape() {
        let charges = arr1(&[1.0, 2.0]);
        let positions = ndarray::Array2::zeros((3, 3));
        assert!(compute_dipole_moment(&charges, &positions).is_err());
    }

    #[test]
    fn test_current_density_constant_dipole() {
        let dm = ndarray::Array2::from_elem((3, 3), 1.0);
        let j = compute_current_density(&dm, 1.0, 1.0).unwrap();
        assert_eq!(j.shape(), &[3, 3]);
        assert!(j[[0, 0]].is_nan());
        assert!((j[[1, 0]]).abs() < 1e-10);
        assert!((j[[2, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_current_density_linear() {
        let dm = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let j = compute_current_density(&dm, 1.0, 1.0).unwrap();
        assert!(j[[0, 0]].is_nan());
        assert!((j[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((j[[2, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_current_density_dt_scaling() {
        let dm = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let j1 = compute_current_density(&dm, 1.0, 1.0).unwrap();
        let j2 = compute_current_density(&dm, 2.0, 1.0).unwrap();
        assert!((j2[[1, 0]] * 2.0 - j1[[1, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_static_dielectric_zero_fluctuation() {
        let dm = ndarray::Array2::from_elem((10, 3), 0.0);
        let eps = static_dielectric_constant(&dm, 1000.0, 300.0, 1.0).unwrap();
        assert!((eps - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_static_dielectric_known_fluctuation() {
        let dm = ndarray::arr2(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        let eps = static_dielectric_constant(&dm, 1000.0, 300.0, 1.0).unwrap();
        // ⟨M⟩ = 0, ⟨M²⟩ = (1²+(-1)²)/2 = 1.0
        // ε(0) = 1.0 + (4π/3)*332.0637*1.0/(1000*1.9872e-3*300)
        let expected = 1.0 + FOUR_PI_OVER_3 * KAPPA * 1.0 / (1000.0 * K_B * 300.0);
        assert!((eps - expected).abs() < 1e-10);
    }

    #[test]
    fn test_static_dielectric_single_frame_rejected() {
        let dm = ndarray::Array2::zeros((1, 3));
        assert!(static_dielectric_constant(&dm, 1000.0, 300.0, 1.0).is_err());
    }

    #[test]
    fn test_einstein_helfand_shape() {
        let n = 100;
        let dm = ndarray::Array2::from_elem((n, 3), 0.1);
        let spectrum =
            einstein_helfand_spectrum(&dm, 0.001, 1000.0, 300.0, 1.0, 10, "hann").unwrap();
        assert_eq!(spectrum.n_frames, n);
        assert!(!spectrum.frequencies.is_empty());
        assert_eq!(spectrum.frequencies.len(), spectrum.epsilon_real.len());
        assert_eq!(spectrum.frequencies.len(), spectrum.epsilon_imag.len());
    }

    #[test]
    fn test_einstein_helfand_rejects_single_frame() {
        let dm = ndarray::Array2::zeros((1, 3));
        assert!(einstein_helfand_spectrum(&dm, 0.001, 1000.0, 300.0, 1.0, 10, "hann").is_err());
    }

    #[test]
    fn test_green_kubo_shape() {
        let n = 100;
        let j = ndarray::Array2::from_elem((n, 3), 0.001);
        let spectrum = green_kubo_spectrum(&j, 0.001, 1000.0, 300.0, 1.0, 10, "hann").unwrap();
        assert!(!spectrum.frequencies.is_empty());
        assert_eq!(spectrum.frequencies.len(), spectrum.epsilon_real.len());
    }

    #[test]
    fn test_decompose_current_conservation() {
        let n_particles = 4;
        let n_frames = 5;
        let mut current = Array3::zeros((n_particles, n_frames, 3));
        for p in 0..n_particles {
            for t in 0..n_frames {
                current[[p, t, 0]] = p as f64 + t as f64;
                current[[p, t, 1]] = (p as f64) * 2.0;
                current[[p, t, 2]] = t as f64 * 0.5;
            }
        }
        let mask = arr1(&[true, true, false, false]);
        let (j_w, j_i) = decompose_current(&current, &mask).unwrap();
        assert_eq!(j_w.shape(), &[n_frames, 3]);
        assert_eq!(j_i.shape(), &[n_frames, 3]);
        // Total should be sum of all particles
        let total: Array2<f64> = current.sum_axis(Axis(0));
        for t in 0..n_frames {
            for d in 0..3 {
                assert!(((j_w[[t, d]] + j_i[[t, d]]) - total[[t, d]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_decompose_current_mask_mismatch() {
        let current = Array3::zeros((2, 3, 3));
        let mask = arr1(&[true, false, true]);
        assert!(decompose_current(&current, &mask).is_err());
    }

    #[test]
    fn test_immutability_dipole_moment() {
        let charges = arr1(&[1.0, -1.0]);
        let positions = ndarray::arr2(&[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let pos_copy = positions.clone();
        compute_dipole_moment(&charges, &positions).unwrap();
        assert_eq!(positions, pos_copy);
    }

    #[test]
    fn test_immutability_current_density() {
        let dm = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let dm_copy = dm.clone();
        compute_current_density(&dm, 1.0, 1.0).unwrap();
        assert_eq!(dm, dm_copy);
    }
}
