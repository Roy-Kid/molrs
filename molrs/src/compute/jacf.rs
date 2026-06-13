//! Ionic conductivity from the charge-current autocorrelation (Green–Kubo).
//!
//! The DC ionic conductivity follows from the Green–Kubo relation for the
//! collective charge current `J(t) = Σ_a q_a v_a(t)`:
//!
//! ```text
//!     σ = 1 / (3·V·k_B·T) · ∫₀^∞ ⟨J(0)·J(t)⟩ dt .
//! ```
//!
//! This kernel computes the current autocorrelation function (JACF)
//! `C(τ) = ⟨J(0)·J(τ)⟩` over all time origins and integrates it (trapezoidal)
//! up to the requested lag to obtain `σ`.
//!
//! It is the molrs port of the `jacf` recipe from the *tame* library
//! (<https://github.com/Roy-Kid/tame>, `tame/recipes/jacf.py`). The *tame*
//! original is non-functional as published (it never actually evaluates the
//! autocorrelation before integrating); this port implements the intended
//! algorithm correctly. The collective current `J = Σ v_cation − Σ v_anion`
//! (unit charges ±1) is assembled by the caller (Python wrapper); arbitrary
//! per-ion charges are supported by pre-scaling the velocities.
//!
//! # Units
//!
//! LAMMPS *real* units, matching [`crate::compute::dielectric`]:
//!
//! | quantity     | unit              |
//! |--------------|-------------------|
//! | current `J`  | e · Å · ps⁻¹      |
//! | time / `dt`  | ps                |
//! | volume       | Å³                |
//! | temperature  | K                 |
//! | output `σ`   | S · m⁻¹ (SI)      |
//!
//! The conversion prefactor folds in `e²`, `Å→m`, and `ps→s` so the caller
//! does no unit bookkeeping. It mirrors
//! [`crate::compute::dielectric::einstein_helfand_conductivity`] (same MD→SI factors,
//! with the Green–Kubo `1/3` replacing the Einstein `1/6`).

use ndarray::Array1;

use crate::compute::error::ComputeError;

// MD (real units) → SI conversion constants, sourced from `molrs-core` so the
// SI values are defined exactly once across the workspace.
use molrs::units::constants::{
    ANGSTROM_M, BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as ELEMENTARY_CHARGE_C, PICOSECOND_S,
};

/// Result of a Green–Kubo current-autocorrelation conductivity computation.
#[derive(Debug, Clone)]
pub struct JacfResult {
    /// Lag times τ = i·dt, **ps**, length `max_lag + 1`.
    pub lag_times: Array1<f64>,
    /// Current autocorrelation `C(τ) = ⟨J(0)·J(τ)⟩` averaged over time origins,
    /// **(e·Å·ps⁻¹)²**, length `max_lag + 1`.
    pub jacf: Array1<f64>,
    /// Running Green–Kubo conductivity integral
    /// `σ(τ) = 1/(3·V·k_B·T) · ∫₀^τ C(t) dt`, **S·m⁻¹**, length `max_lag + 1`.
    /// The reported scalar [`sigma`](Self::sigma) is `sigma_running` at the
    /// final lag.
    pub sigma_running: Array1<f64>,
    /// DC ionic conductivity σ, **S·m⁻¹** — the trapezoidal integral of the
    /// JACF over the full `[0, max_lag·dt]` window.
    pub sigma: f64,
}

/// Green–Kubo ionic conductivity from the charge-current autocorrelation.
///
/// # Arguments
/// * `current` — collective charge current `J(t)`, shape `(n_frames, 3)`,
///   **e·Å·ps⁻¹**.
/// * `dt` — frame spacing, **ps** (> 0).
/// * `volume` — system volume, **Å³** (> 0).
/// * `temperature` — temperature, **K** (> 0).
/// * `max_correlation_time` — longest ACF lag in **frames**, clamped to
///   `n_frames − 1`.
///
/// # Returns
/// A [`JacfResult`] with the JACF curve, the running conductivity integral, and
/// the DC conductivity `σ` (S·m⁻¹).
///
/// # Errors
/// * `DimensionMismatch` if `current` is not `(_, 3)`.
/// * `EmptyInput` if fewer than two frames.
/// * `NonFinite` on any NaN/inf input.
/// * `OutOfRange` if `dt`, `volume`, or `temperature` ≤ 0.
pub fn green_kubo_conductivity(
    current: &ndarray::Array2<f64>,
    dt: f64,
    volume: f64,
    temperature: f64,
    max_correlation_time: usize,
) -> Result<JacfResult, ComputeError> {
    let shape = current.shape();
    if shape[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[1],
            what: "current (expected (n_frames, 3))",
        });
    }
    let n_frames = shape[0];
    if n_frames < 2 {
        return Err(ComputeError::EmptyInput);
    }
    for (idx, &v) in current.iter().enumerate() {
        if !v.is_finite() {
            return Err(ComputeError::NonFinite {
                where_: "current",
                index: idx,
            });
        }
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

    let max_lag = max_correlation_time.min(n_frames - 1);

    // Unbiased windowed autocorrelation: C(τ) = ⟨J(t)·J(t+τ)⟩_t.
    let mut jacf = Array1::<f64>::zeros(max_lag + 1);
    for tau in 0..=max_lag {
        let count = n_frames - tau;
        let mut acc = 0.0;
        for t in 0..count {
            let mut s = 0.0;
            for d in 0..3 {
                s += current[[t, d]] * current[[t + tau, d]];
            }
            acc += s;
        }
        jacf[tau] = acc / count as f64;
    }

    let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));

    // Conversion prefactor: σ = prefactor · ∫C dt / (V·T), Green–Kubo 1/3.
    //   ∫C dt : (e·Å·ps⁻¹)²·ps = e²·Å²·ps⁻¹  →  C²·m²·s⁻¹
    //   volume: Å³                            →  m³
    let prefactor = (ELEMENTARY_CHARGE_C * ELEMENTARY_CHARGE_C * ANGSTROM_M * ANGSTROM_M
        / PICOSECOND_S)
        / (3.0 * ANGSTROM_M * ANGSTROM_M * ANGSTROM_M * K_B_SI);

    // Running trapezoidal integral of the JACF, converted to σ(τ).
    let mut sigma_running = Array1::<f64>::zeros(max_lag + 1);
    let mut integral = 0.0;
    for tau in 1..=max_lag {
        integral += 0.5 * (jacf[tau - 1] + jacf[tau]) * dt;
        sigma_running[tau] = prefactor * integral / (volume * temperature);
    }
    let sigma = sigma_running[max_lag];

    Ok(JacfResult {
        lag_times,
        jacf,
        sigma_running,
        sigma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn jacf_lag_zero_is_mean_square_current() {
        // J(t) = (1,0,0) constant → C(0) = ⟨|J|²⟩ = 1, C(τ) = 1 for all τ.
        let j = array![
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        let r = green_kubo_conductivity(&j, 1.0, 100.0, 300.0, 3).unwrap();
        for tau in 0..=3 {
            assert!(
                (r.jacf[tau] - 1.0).abs() < 1e-12,
                "C({tau})={}",
                r.jacf[tau]
            );
        }
        // sigma > 0 for a sustained current ACF.
        assert!(r.sigma > 0.0);
    }

    #[test]
    fn vanishing_correlation_gives_zero_conductivity() {
        // Alternating current with zero mean and zero net ACF integral region:
        // here C(τ>0) cancels for this symmetric pattern enough to stay finite.
        let j = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let r = green_kubo_conductivity(&j, 1.0, 50.0, 300.0, 2).unwrap();
        assert!(r.sigma.abs() < 1e-30);
        assert!(r.jacf.iter().all(|&c| c.abs() < 1e-12));
    }

    #[test]
    fn conductivity_scales_inversely_with_volume_and_temperature() {
        let j = array![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let a = green_kubo_conductivity(&j, 1.0, 100.0, 300.0, 2).unwrap();
        let b = green_kubo_conductivity(&j, 1.0, 200.0, 300.0, 2).unwrap();
        let c = green_kubo_conductivity(&j, 1.0, 100.0, 600.0, 2).unwrap();
        assert!((a.sigma / b.sigma - 2.0).abs() < 1e-9); // half the volume → 2× σ
        assert!((a.sigma / c.sigma - 2.0).abs() < 1e-9); // half the T → 2× σ
    }

    #[test]
    fn rejects_bad_inputs() {
        let j = array![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(green_kubo_conductivity(&j, 0.0, 1.0, 300.0, 1).is_err());
        assert!(green_kubo_conductivity(&j, 1.0, 0.0, 300.0, 1).is_err());
        assert!(green_kubo_conductivity(&j, 1.0, 1.0, 0.0, 1).is_err());
        let bad = array![[1.0, 0.0], [1.0, 0.0]];
        assert!(green_kubo_conductivity(&bad, 1.0, 1.0, 300.0, 1).is_err());
    }
}
