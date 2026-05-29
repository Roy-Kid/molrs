//! Vibrational spectra from molecular-dynamics time-series data.
//!
//! Three spectrum types, adapted from SchNetPack's `md.data.spectra`:
//!
//! * [`power_spectrum`] — velocity autocorrelation → VDOS
//! * [`ir_spectrum`] — dipole-flux autocorrelation → IR absorption
//! * [`raman_spectrum`] — polarizability-derivative autocorrelation →
//!   isotropic / anisotropic Raman with optional cross-section correction
//!
//! All routines compose [`molrs_signal`] primitives (`acf_fft`,
//! `apply_window`, `frequency_grid`) plus shared one-sided FFT helpers.
//! Input is `ndarray::Array2<f64>` — no HDF5 dependency.
//!
//! # Units
//!
//! | quantity      | unit   |
//! |---------------|--------|
//! | time / dt     | fs     |
//! | frequency     | cm⁻¹   |
//! | intensity     | arb.   |
//! | temperature   | K      |

pub mod ir_spectrum;
pub mod power_spectrum;
pub mod raman_spectrum;

pub use ir_spectrum::ir_spectrum;
pub use power_spectrum::power_spectrum;
pub use raman_spectrum::raman_spectrum;

use molrs_signal as sig;
use ndarray::{Array1, ArrayD};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use rustfft::num_traits::Zero;

use crate::error::ComputeError;

// ── Physical constants ───────────────────────────────────────────────────────

/// Speed of light in m/s (exact).
const C_MS: f64 = 299_792_458.0;

/// Femtoseconds to seconds.
const FS_TO_S: f64 = 1e-15;
/// Metres to centimetres.
const M_TO_CM: f64 = 100.0;

/// Conversion from angular frequency (rad / fs) to wavenumber (cm⁻¹).
///
/// ν̃ = ω · (2π · c · 10⁻¹⁵ · 100)⁻¹ = ω / (2π · c · FS_TO_S · M_TO_CM)
pub(crate) const ANGULAR_FREQ_TO_CM1: f64 =
    1.0 / (2.0 * std::f64::consts::PI * C_MS * FS_TO_S * M_TO_CM);

/// Largest exponent such that `exp(x)` does not overflow f64.
const MAX_EXP_ARG: f64 = 700.0;

// ── Result types ─────────────────────────────────────────────────────────────

/// Single-spectrum result (VDOS, IR).
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Frequency grid in cm⁻¹, length `n_pad / 2 + 1`.
    pub frequencies_cm1: Array1<f64>,
    /// Spectral intensities (arbitrary units).
    pub intensities: Array1<f64>,
    /// Number of ACF lags retained before windowing.
    pub resolution: usize,
    /// Number of input frames.
    pub n_frames: usize,
}

/// Raman spectrum result with isotropic / anisotropic decomposition.
#[derive(Debug, Clone)]
pub struct RamanSpectrum {
    /// Frequency grid in cm⁻¹.
    pub frequencies_cm1: Array1<f64>,
    /// Isotropic (trace) contribution.
    pub isotropic: Array1<f64>,
    /// Anisotropic (deviatoric) contribution.
    pub anisotropic: Array1<f64>,
    /// Parallel polarization `I_∥ = I_iso + (4/45)·I_aniso`. Set when
    /// `averaged = true`.
    pub parallel: Option<Array1<f64>>,
    /// Perpendicular polarization `I_⊥ = I_aniso / 15`. Set when
    /// `averaged = true`.
    pub perpendicular: Option<Array1<f64>>,
    /// Number of ACF lags retained.
    pub resolution: usize,
    /// Number of input frames.
    pub n_frames: usize,
}

// ── Shared helpers ───────────────────────────────────────────────────────────

/// Validate common spectrum inputs.
///
/// Returns `resolution` clamped to `n_frames - 1` on success.
pub(crate) fn validate_input(
    n_frames: usize,
    dt_fs: f64,
    resolution: usize,
) -> Result<usize, ComputeError> {
    if n_frames < 2 {
        return Err(ComputeError::EmptyInput);
    }
    if dt_fs <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt_fs",
            value: dt_fs.to_string(),
        });
    }
    Ok(resolution.min(n_frames.saturating_sub(1)))
}

/// Apply CosineSq window to an ACF array, zero-pad, and FFT.
///
/// Wraps the `Array1` → `ArrayD` → window → `Array1` → FFT pipeline
/// that is shared by all three spectrum types.
fn window_and_fft(
    planner: &mut FftPlanner<f64>,
    acf: &Array1<f64>,
    dt_fs: f64,
) -> Result<(Array1<f64>, Array1<f64>), ComputeError> {
    let n = acf.len();
    let acf_dyn = ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), acf.to_vec()).map_err(|e| {
        ComputeError::BadShape {
            expected: "1d".into(),
            got: e.to_string(),
        }
    })?;
    let windowed = sig::apply_window(&acf_dyn, sig::WindowType::CosineSq, 0).map_err(|e| {
        ComputeError::OutOfRange {
            field: "apply_window",
            value: e.to_string(),
        }
    })?;
    let windowed_1d: Array1<f64> = windowed.iter().copied().collect();
    let n_pad = (4 * n).next_power_of_two();
    Ok(acf_to_spectrum(planner, &windowed_1d, dt_fs, n_pad))
}

/// Convert a windowed one-sided ACF to a frequency-domain spectrum.
///
/// Returns `(frequencies_cm1, intensities_raw)`. The caller is
/// responsible for applying physical prefactors (cross-section + Bose
/// for Raman).
fn acf_to_spectrum(
    planner: &mut FftPlanner<f64>,
    acf: &Array1<f64>,
    dt_fs: f64,
    n_pad: usize,
) -> (Array1<f64>, Array1<f64>) {
    let freqs_rad = sig::frequency_grid(n_pad, dt_fs);
    let intensities = acf_to_intensities(planner, acf, n_pad);
    let n_freq = intensities.len();
    let mut frequencies_cm1 = Array1::zeros(n_freq);
    for j in 0..n_freq {
        frequencies_cm1[j] = freqs_rad[j] * ANGULAR_FREQ_TO_CM1;
    }
    (frequencies_cm1, intensities)
}

/// FFT a windowed ACF and return only the intensity spectrum (no
/// frequency grid). Used for the second Raman component so we don't
/// allocate a second identical frequency array.
fn acf_to_intensities(
    planner: &mut FftPlanner<f64>,
    acf: &Array1<f64>,
    n_pad: usize,
) -> Array1<f64> {
    let fwd = planner.plan_fft_forward(n_pad);

    let mut complex_data: Vec<Complex64> = acf.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    complex_data.resize(n_pad, Complex64::zero());
    fwd.process(&mut complex_data);

    let n_freq = n_pad / 2 + 1;
    let mut intensities = Array1::zeros(n_freq);
    for j in 0..n_freq {
        intensities[j] = complex_data[j].re / n_pad as f64;
    }
    intensities
}

/// Pre-compute a CosineSq window of length `n`.
pub(crate) fn cosine_sq_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| {
            let angle = std::f64::consts::PI * i as f64 / (2.0 * (n - 1) as f64);
            angle.cos().powi(2)
        })
        .collect()
}

/// Evaluate the Bose-Einstein factor at frequency `nu` (cm⁻¹) and
/// temperature `T` (K). Returns 1.0 when `nu <= 0` or the exponent
/// would underflow.
pub(crate) fn bose_factor(nu: f64, temperature_k: f64) -> f64 {
    if nu <= 0.0 || temperature_k <= 0.0 {
        return 1.0;
    }
    // HC_KB = h·c / k_B ≈ 1.438777 cm·K
    let exponent = -1.438777 * nu / temperature_k;
    if exponent > -MAX_EXP_ARG {
        1.0 / (1.0 - exponent.exp())
    } else {
        1.0
    }
}
