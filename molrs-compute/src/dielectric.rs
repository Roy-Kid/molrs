//! Dielectric susceptibility computation.
//!
//! Computes the frequency-dependent dielectric permittivity ε*(ω) =
//! ε′(ω) − i·ε″(ω) of a polar fluid from molecular-dynamics dipole
//! trajectories, plus the static dielectric constant ε(0).
//!
//! # Routes
//!
//! - [`static_dielectric_constant`] — Neumann fluctuation formula
//!   (Neumann, *Mol. Phys.* **50**, 841 (1983); conducting/tin-foil
//!   Ewald boundary conditions assumed).
//! - [`einstein_helfand_spectrum`] — dipole-ACF → window → FT → ε*(ω)
//!   (Caillol, Levesque & Weis, *J. Chem. Phys.* **85**, 6645 (1986)).
//! - [`green_kubo_spectrum`] — current-ACF → σ(ω) → ε*(ω)
//!   (same reference; equivalent route for conducting systems).
//!
//! All routines compose [`molrs_signal`] primitives (`acf_fft`,
//! `apply_window`, `frequency_grid`) plus one local helper
//! `acf_to_spectrum` for the one-sided FT.
//!
//! # Units
//!
//! All inputs and outputs use LAMMPS *real* units throughout:
//!
//! | quantity        | unit                |
//! |-----------------|---------------------|
//! | length          | Å                   |
//! | charge          | e                   |
//! | energy          | kcal / mol          |
//! | time            | ps                  |
//! | temperature     | K                   |
//! | volume          | Å³                  |
//! | dipole moment   | e · Å               |
//! | current density | e · Å⁻² · ps⁻¹      |
//! | angular ω       | rad · ps⁻¹          |
//! | ε permittivity  | dimensionless       |

use molrs_signal as sig;
use ndarray::{Array1, Array2, Array3};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use rustfft::num_traits::Zero;

use crate::error::ComputeError;

/// Result of a dielectric spectrum computation.
///
/// The complex permittivity is `ε*(ω) = ε′(ω) − i·ε″(ω)`. `epsilon_imag`
/// stores `ε″(ω)` with the **positive-loss** convention (≥ 0 for stable
/// causal systems). FT convention throughout: `X(ω) = ∫₀^∞ f(t) e^{−iωt} dt`.
#[derive(Debug, Clone)]
pub struct DielectricSpectrum {
    /// Angular frequency grid, rad·ps⁻¹, length `n_pad/2 + 1` with
    /// `n_pad = (2·n_correlation_steps).next_power_of_two()`.
    /// Bin 0 is the DC bin; bin 1 is Δω = 2π/(n_pad·dt); the last bin
    /// is the Nyquist frequency π/dt.
    pub frequencies: Array1<f64>,
    /// Real part ε′(ω), dimensionless.
    pub epsilon_real: Array1<f64>,
    /// Loss spectrum ε″(ω), dimensionless (positive sign convention).
    pub epsilon_imag: Array1<f64>,
    /// Number of input frames the spectrum was computed from. For the
    /// Green-Kubo route this is the dipole-trajectory length; the
    /// underlying current series is one shorter (row 0 of J(t) is NaN
    /// by construction; see [`compute_current_density`]).
    pub n_frames: usize,
    /// Number of ACF lags retained before windowing and zero-padding,
    /// equal to `max_correlation_time + 1` (or clamped to the available
    /// post-NaN-skip length for Green-Kubo).
    pub n_correlation_steps: usize,
}

/// Per-axis static dielectric constant result (MDAnalysis-compatible).
///
/// Follows the MDAnalysis `DielectricConstant.results` layout:
/// per-axis dipole mean `M`, squared mean `M2`, fluctuation, and
/// directional ε.
#[derive(Debug, Clone)]
pub struct StaticDielectricResult {
    /// Per-axis mean dipole moment ⟨**M**⟩, **e · Å**, length 3.
    pub dipole_mean: Array1<f64>,
    /// Per-axis mean squared dipole ⟨**M**²⟩, **(e · Å)²**, length 3.
    pub dipole_sq_mean: Array1<f64>,
    /// Per-axis fluctuation ⟨M_d²⟩ − ⟨M_d⟩², **(e · Å)²**, length 3.
    pub fluctuation: Array1<f64>,
    /// Per-axis static dielectric constant ε_d (d = x, y, z), dimensionless.
    pub eps: Array1<f64>,
    /// Isotropic average (ε_x + ε_y + ε_z) / 3, dimensionless.
    pub eps_mean: f64,
    /// High-frequency / electronic permittivity used (ε_∞).
    pub epsilon_inf: f64,
    /// Number of frames analysed.
    pub n_frames: usize,
}

// ── Physical constants (MD real units: kcal, mol, Angstrom, e, K) ─────────

const KAPPA: f64 = 332.0637; // 1/(4*pi*epsilon_0) in kcal·Å·mol⁻¹·e⁻²
const K_B: f64 = 1.98720425864083e-3; // Boltzmann constant in kcal/(mol·K)
const FOUR_PI_OVER_3: f64 = 4.1887902047863905; // 4π/3

// ── Basic observables ─────────────────────────────────────────────────────

/// Total instantaneous dipole moment **M** = Σᵢ qᵢ · **rᵢ**.
///
/// # Arguments
/// * `charges` — partial charges (`n_atoms`), units **e**.
/// * `positions` — atomic coordinates `(n_atoms, 3)`, units **Å**.
///   Coordinates must already be unwrapped across periodic images;
///   otherwise the dipole jumps by `q · L` whenever an atom crosses a
///   box face. (The caller does the unwrapping.)
///
/// # Returns
/// Length-3 vector **M** = (Mₓ, M_y, M_z) in **e · Å**.
///
/// # Errors
/// * `DimensionMismatch` if `positions.shape() != (n_atoms, 3)`.
/// * `NonFinite` if any charge is NaN/inf.
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

/// System current density **J**(t) = (**M**(t) − **M**(t − Δt)) / (V · Δt).
///
/// # Arguments
/// * `dipole_moments` — total dipole trajectory `(n_frames, 3)`, **e · Å**.
/// * `dt` — timestep between consecutive frames, **ps**.
/// * `volume` — system volume, **Å³** (constant; assumes NVT/NVE).
///
/// # Returns
/// `(n_frames, 3)` array in **e · Å⁻² · ps⁻¹**. **Row 0 is filled with
/// `f64::NAN`** because no previous frame exists to form the finite
/// difference. All downstream consumers must skip row 0
/// (`green_kubo_spectrum` does this internally; bare `np.mean` will
/// poison its output).
///
/// # Errors
/// * `DimensionMismatch` if shape is not `(_, 3)`.
/// * `OutOfRange` if `dt ≤ 0` or `volume ≤ 0`.
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

/// Static dielectric constant ε(0) from the Neumann fluctuation formula.
///
/// `ε(0) = ε_∞ + (4π/3) · (1/(V·k_B·T)) · ⟨|**M** − ⟨**M**⟩|²⟩`
///
/// in Gaussian/MD units (in LAMMPS *real* units the prefactor includes
/// the Coulomb constant `KAPPA = 332.0637 kcal·Å·mol⁻¹·e⁻²` so that
/// the formula evaluates to a dimensionless number).
///
/// **Assumes conducting / tin-foil Ewald boundary conditions** (the
/// only case for which this prefactor is exact). For reaction-field
/// boundary conditions an additional `2(ε_RF − 1)/(2ε_RF + 1)` factor
/// is required (Neumann 1983 §IV).
///
/// # Arguments
/// * `dipole_moments` — `(n_frames, 3)`, **e · Å**, `n_frames ≥ 2`.
/// * `volume` — **Å³**.
/// * `temperature` — **K**.
/// * `epsilon_inf` — high-frequency / electronic permittivity ε_∞,
///   dimensionless (typically 1.0 for non-polarizable force fields,
///   1.5–2.5 for water with polarizable models).
///
/// # Reference
/// Neumann, M. *Mol. Phys.* **50** (4), 841 (1983),
/// "Dipole moment fluctuation formulas in computer simulations of
/// polar systems."
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

    // Two-pass centered variance: stable when |M| ≫ |δM| (avoids the
    // catastrophic cancellation of ⟨M²⟩ − ⟨M⟩² that bites at ~10⁶ frames).
    let n = shape[0] as f64;
    let mut mean_m = Array1::<f64>::zeros(3);
    for t in 0..shape[0] {
        for d in 0..3 {
            mean_m[d] += dipole_moments[[t, d]];
        }
    }
    for d in 0..3 {
        mean_m[d] /= n;
    }

    let mut variance = 0.0;
    for t in 0..shape[0] {
        for d in 0..3 {
            let dev = dipole_moments[[t, d]] - mean_m[d];
            variance += dev * dev;
        }
    }
    variance /= n;
    let prefactor = FOUR_PI_OVER_3 * KAPPA / (volume * K_B * temperature);

    Ok(epsilon_inf + prefactor * variance)
}

/// Per-axis static dielectric constant (MDAnalysis-compatible output).
///
/// Returns directional ε_x, ε_y, ε_z and their isotropic mean, plus the
/// per-axis dipole statistics. Uses the same Neumann (1983) fluctuation
/// formula as [`static_dielectric_constant`] but preserves the Cartesian
/// decomposition.
///
/// # Arguments
/// Same as [`static_dielectric_constant`].
///
/// # Reference
/// MDAnalysis `DielectricConstant`:
/// <https://docs.mdanalysis.org/stable/documentation_pages/analysis/dielectric.html>
pub fn static_dielectric_constant_components(
    dipole_moments: &Array2<f64>,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
) -> Result<StaticDielectricResult, ComputeError> {
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

    let n = shape[0] as f64;
    let n_frames = shape[0];

    let mut mean_m = Array1::<f64>::zeros(3);
    let mut mean_sq = Array1::<f64>::zeros(3);

    for t in 0..n_frames {
        for d in 0..3 {
            let m = dipole_moments[[t, d]];
            mean_m[d] += m;
            mean_sq[d] += m * m;
        }
    }
    for d in 0..3 {
        mean_m[d] /= n;
        mean_sq[d] /= n;
    }

    let mut fluctuation = Array1::<f64>::zeros(3);
    let mut eps = Array1::<f64>::zeros(3);
    // Per-axis prefactor: 4π·KAPPA / (V·k_B·T).  This is 3× the
    // isotropic prefactor because the diagonal dielectric-tensor
    // component integrates the full dipole in one direction, while the
    // isotropic ε averages over 3 directions.
    let per_axis_prefactor = 3.0 * FOUR_PI_OVER_3 * KAPPA / (volume * K_B * temperature);

    for d in 0..3 {
        fluctuation[d] = mean_sq[d] - mean_m[d] * mean_m[d];
        eps[d] = epsilon_inf + per_axis_prefactor * fluctuation[d];
    }

    let eps_mean = (eps[0] + eps[1] + eps[2]) / 3.0;

    Ok(StaticDielectricResult {
        dipole_mean: mean_m,
        dipole_sq_mean: mean_sq,
        fluctuation,
        eps,
        eps_mean,
        epsilon_inf,
        n_frames,
    })
}

// ── Shared helpers ─────────────────────────────────────────────────────────

/// Validate the shared input contract for the three spectrum routines:
/// (n_frames, 3) layout, n_frames ≥ 2, and positive dt / volume / temperature.
fn validate_thermo_series_3d(
    series: &Array2<f64>,
    dt: f64,
    volume: f64,
    temperature: f64,
    what: &'static str,
) -> Result<(), ComputeError> {
    let shape = series.shape();
    if shape[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[1],
            what,
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
    Ok(())
}

fn parse_window_type(s: &str) -> Result<sig::WindowType, ComputeError> {
    match s {
        "hann" => Ok(sig::WindowType::Hann),
        "blackman" => Ok(sig::WindowType::Blackman),
        // Preferred for one-sided ACFs (1 at t=0, 0 at t=max_lag).
        // Hann/Blackman zero out C(0), the static-ε signal.
        "cosine_sq" => Ok(sig::WindowType::CosineSq),
        other => Err(ComputeError::OutOfRange {
            field: "window_type",
            value: other.into(),
        }),
    }
}

/// `(frequencies, spec_re, spec_im)` triple returned by `windowed_acf_spectrum`.
///
/// `spec_re` and `spec_im` are the real and imaginary parts of the continuous
/// one-sided Fourier transform `X(ω) = ∫₀^∞ C(t) w(t) e^{−iωt} dt`,
/// where C(t) is the per-component-summed dipole/current ACF and w(t) is the
/// chosen window. No physical prefactor (β, V, k_B T, ε₀, …) is applied.
type RawSpectrum = (Array1<f64>, Array1<f64>, Array1<f64>);

/// Continuous-FT-of-ACF of `series[start.., :]`, summed over the 3 Cartesian
/// components and windowed.
///
/// Returns `(frequencies, spec_re, spec_im)` where `spec_re + i·spec_im` is
/// the continuous FT `X(ω)` evaluated on the rfft frequency grid.
///
/// The conversion from `rustfft`'s unscaled DFT to the continuous FT uses the
/// rectangle rule: `∫₀^T f(t) e^{−iωt} dt ≈ dt · DFT[f](ω_k)` (Press et al.,
/// *Numerical Recipes* §13.9).
///
/// `sig::acf_fft` returns the linear un-normalized ACF
/// `r[k] = Σ_{τ=0}^{N−1−k} x[τ]·x[τ+k] ≈ (N − k)·C(k)`. We convert to the
/// unbiased estimator `C(k) = r[k] / (N − k)` before windowing.
fn windowed_acf_spectrum(
    series: &Array2<f64>,
    start: usize,
    max_lag: usize,
    dt: f64,
    window_type: &str,
) -> Result<RawSpectrum, ComputeError> {
    let n_frames = series.shape()[0];
    let n_eff = n_frames - start;
    if n_eff <= max_lag {
        return Err(ComputeError::OutOfRange {
            field: "max_correlation_time",
            value: format!("{max_lag} >= effective_length {n_eff}"),
        });
    }
    let wt = parse_window_type(window_type)?;

    // One planner amortizes plan construction across the 3 component ACFs
    // plus the forward FFT in `acf_to_spectrum`.
    let mut planner = FftPlanner::new();

    let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
    for d in 0..3 {
        let col: Array1<f64> = (start..n_frames).map(|t| series[[t, d]]).collect();
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
    // Linear-ACF → unbiased ensemble estimator C(k·dt) = ⟨x(0)·x(k·dt)⟩.
    for k in 0..=max_lag {
        acf_sum[k] /= (n_eff - k) as f64;
    }

    let acf_dyn = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[max_lag + 1]), acf_sum.to_vec())
        .map_err(|e| ComputeError::BadShape {
            expected: "1d".into(),
            got: e.to_string(),
        })?;
    let windowed = sig::apply_window(&acf_dyn, wt, 0).map_err(|e| ComputeError::OutOfRange {
        field: "apply_window",
        value: e.to_string(),
    })?;
    let windowed_1d: Array1<f64> = windowed.iter().copied().collect();

    let n_pad = (2 * (max_lag + 1)).next_power_of_two();
    Ok(acf_to_spectrum(&mut planner, &windowed_1d, dt, n_pad))
}

// ── Frequency-domain spectra ───────────────────────────────────────────────

/// Convert windowed ACF to the continuous one-sided FT via DFT.
///
/// Returns `X.re` and `X.im` where `X(ω_k) = ∫₀^T C(t)·w(t) e^{−iωt} dt`
/// under the `e^{−iωt}` Fourier convention. Scaling uses the rectangle-rule
/// `dt` factor, not the FFT's internal `1/n_pad`.
fn acf_to_spectrum(
    planner: &mut FftPlanner<f64>,
    acf: &Array1<f64>,
    dt: f64,
    n_pad: usize,
) -> RawSpectrum {
    let fwd = planner.plan_fft_forward(n_pad);

    let mut complex_data: Vec<Complex64> = acf.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    complex_data.resize(n_pad, Complex64::zero());
    fwd.process(&mut complex_data);

    let frequencies = sig::frequency_grid(n_pad, dt);
    let n_freq = frequencies.len();
    let mut spec_re = Array1::zeros(n_freq);
    let mut spec_im = Array1::zeros(n_freq);

    for j in 0..n_freq {
        let z = complex_data[j];
        spec_re[j] = z.re * dt;
        spec_im[j] = z.im * dt;
    }

    (frequencies, spec_re, spec_im)
}

/// Dipole-ACF route to the frequency-dependent permittivity (Caillol-Levesque-
/// Weis Eq. (30); a.k.a. the Kubo / fluctuation-dissipation route).
///
/// Implements
///
/// ```text
///     χ(ω) = (β / (3V)) · [ ⟨M·M⟩ − iω · X(ω) ]
///     ε*(ω) − ε_∞ = χ(ω) / ε₀
/// ```
///
/// where `X(ω) = ∫₀^∞ ⟨M(0)·M(t)⟩ e^{−iωt} dt` is the one-sided FT of the
/// dipole-moment autocorrelation. With `X = X_re + i·X_im` and prefactor
/// `A = (4π·KAPPA)/(3·V·k_B·T) = β/(3V·ε₀)`, separating real/imaginary parts
/// under the positive-loss convention `ε* = ε′ − i·ε″`:
///
/// ```text
///     ε′(ω) − ε_∞ = A · ( ⟨M·M⟩ + ω · X_im )
///     ε″(ω)       = A · ω · X_re
/// ```
///
/// At `ω = 0` this contracts to the Neumann static formula
/// `ε(0) − ε_∞ = A · ⟨|δM|²⟩`, exactly matching
/// [`static_dielectric_constant`]; at large `ω` the spectrum tends to `ε_∞`.
///
/// # Algorithm
/// 1. Compute `⟨|δM|²⟩` via centered variance (numerically stable when
///    `|⟨M⟩| ≫ |δM|`; same form as [`static_dielectric_constant`]).
/// 2. Compute the un-normalized linear ACF of each component via
///    [`molrs_signal::acf_fft`], sum to a scalar `r[k]`, then divide by
///    `(N − k)` to obtain the unbiased estimator `C(k)`.
/// 3. Apply the requested window. For one-sided ACFs `"cosine_sq"` is
///    preferred — `"hann"` and `"blackman"` zero out `C(0)`, biasing the
///    spectrum at high ω; the static `⟨M·M⟩` term still recovers correctly.
/// 4. Zero-pad to `n_pad = (2·(max_lag+1)).next_power_of_two()` and take a
///    forward FFT scaled by `dt` (rectangle-rule discrete-to-continuous FT).
/// 5. Combine `⟨M·M⟩` and `X(ω)` per the formulas above.
///
/// # Arguments
/// * `dipole_moments` — `(n_frames, 3)`, **e · Å**, unwrapped across PBC.
/// * `dt` — frame spacing, **ps**.
/// * `volume`, `temperature` — **Å³**, **K**.
/// * `epsilon_inf` — high-frequency / electronic permittivity (dimensionless).
/// * `max_correlation_time` — longest ACF lag, in **frames**;
///   clamped to `n_frames − 1`. Practical choice: ≤ 1/10 of `n_frames`
///   to keep statistical noise low. Sets the frequency resolution
///   `Δω = 2π / (n_pad · dt)`.
/// * `window_type` — `"cosine_sq"` (recommended), `"hann"`, or `"blackman"`.
///
/// # References
/// - Caillol, Levesque & Weis, *J. Chem. Phys.* **85**, 6645 (1986),
///   Kubo-relation derivation (their Eq. (30)).
/// - Neumann, *Mol. Phys.* **50**, 841 (1983), static-limit identity used to
///   verify `ε(ω = 0) = ε_static`.
pub fn einstein_helfand_spectrum(
    dipole_moments: &Array2<f64>,
    dt: f64,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
    max_correlation_time: usize,
    window_type: &str,
) -> Result<DielectricSpectrum, ComputeError> {
    validate_thermo_series_3d(dipole_moments, dt, volume, temperature, "dipole_moments")?;

    let n_frames = dipole_moments.shape()[0];
    let max_lag = max_correlation_time.min(n_frames - 1);

    // ⟨|δM|²⟩ — the static term in Eq. (30). Centered form for FP stability.
    let n = n_frames as f64;
    let mut mean_m = [0.0_f64; 3];
    for t in 0..n_frames {
        for d in 0..3 {
            mean_m[d] += dipole_moments[[t, d]];
        }
    }
    for m in mean_m.iter_mut() {
        *m /= n;
    }
    let mut m_sq = 0.0;
    for t in 0..n_frames {
        for d in 0..3 {
            let dev = dipole_moments[[t, d]] - mean_m[d];
            m_sq += dev * dev;
        }
    }
    m_sq /= n;

    let (frequencies, spec_re, spec_im) =
        windowed_acf_spectrum(dipole_moments, 0, max_lag, dt, window_type)?;

    let prefactor = FOUR_PI_OVER_3 * KAPPA / (volume * K_B * temperature);
    let n_freq = frequencies.len();
    let mut eps_real = Array1::zeros(n_freq);
    let mut eps_imag = Array1::zeros(n_freq);
    for j in 0..n_freq {
        let omega = frequencies[j];
        eps_real[j] = epsilon_inf + prefactor * (m_sq + omega * spec_im[j]);
        eps_imag[j] = prefactor * omega * spec_re[j];
    }

    Ok(DielectricSpectrum {
        frequencies,
        epsilon_real: eps_real,
        epsilon_imag: eps_imag,
        n_frames,
        n_correlation_steps: max_lag + 1,
    })
}

/// Green–Kubo route: current-ACF → conductivity → permittivity.
///
/// Implements
///
/// ```text
///     σ(ω) = (V / (3·k_B·T)) · ∫₀^∞ ⟨J(0)·J(t)⟩ e^{−iωt} dt = σ′ + i·σ″
///     ε*(ω) − ε_∞ = −i · σ(ω) / (ω · ε₀)
/// ```
///
/// (Hansen & McDonald, *Theory of Simple Liquids*, Eq. 7.7.20; equivalent to
/// Eq. (39) of Caillol-Levesque-Weis after substituting `J = Ṁ/V`). The
/// prefactor uses `V/(3·k_B·T)` because the input is current density
/// `J = Ṁ/V`; the textbook formula `1/(3·V·k_B·T)` applies to total `Ṁ`.
/// Separating real/imaginary parts under `ε* = ε′ − i·ε″`:
///
/// ```text
///     ε′(ω) − ε_∞ = σ″(ω) / (ω · ε₀)
///     ε″(ω)       = σ′(ω) / (ω · ε₀)
/// ```
///
/// with `1/ε₀ = 4π·KAPPA` in LAMMPS real units. In the limit of a Debye
/// relaxation this is equivalent to the [`einstein_helfand_spectrum`] result
/// (verified by `test_einstein_helfand_recovers_debye`).
///
/// Row 0 of `current_density` is skipped automatically (it is `NaN` by
/// construction; see [`compute_current_density`]). The effective ACF is
/// therefore over `n_frames − 1` samples.
///
/// The DC bin (`ω = 0`) is regularized to `(ε_∞, 0)`; the physical static
/// limit must be taken from [`static_dielectric_constant`] for insulators or
/// extrapolated from low-ω bins for conductors.
///
/// # Arguments
/// * `current_density` — `(n_frames, 3)` as returned by
///   [`compute_current_density`], **e · Å⁻² · ps⁻¹**.
/// * `dt`, `volume`, `temperature`, `epsilon_inf`,
///   `max_correlation_time`, `window_type` — see
///   [`einstein_helfand_spectrum`].
///
/// # References
/// - Caillol, Levesque & Weis, *J. Chem. Phys.* **85**, 6645 (1986), Eqs.
///   (36)–(39).
/// - Hansen & McDonald, *Theory of Simple Liquids*, Eq. 7.7.20.
pub fn green_kubo_spectrum(
    current_density: &Array2<f64>,
    dt: f64,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
    max_correlation_time: usize,
    window_type: &str,
) -> Result<DielectricSpectrum, ComputeError> {
    validate_thermo_series_3d(current_density, dt, volume, temperature, "current_density")?;

    let n_frames = current_density.shape()[0];
    // Skip NaN row 0 (no previous frame; see compute_current_density). The
    // effective series has n_frames - 1 samples; clamp max_lag accordingly so
    // acf_fft never sees max_lag ≥ len.
    let start = 1;
    let effective_len = n_frames - start;
    let max_lag = max_correlation_time.min(effective_len.saturating_sub(1));

    let (frequencies, spec_re, spec_im) =
        windowed_acf_spectrum(current_density, start, max_lag, dt, window_type)?;

    // σ(ω) = sigma_prefactor · (spec_re + i·spec_im) under e^{−iωt}.
    // ε*(ω) − ε_∞ = −i·σ(ω)/(ω·ε₀):
    //   ε′(ω) − ε_∞ = σ″/(ω·ε₀) = (1/ε₀)·sigma_prefactor·spec_im/ω
    //   ε″(ω)       = σ′/(ω·ε₀) = (1/ε₀)·sigma_prefactor·spec_re/ω
    // With J(t) = Ṁ(t)/V: ⟨Ṁ·Ṁ⟩ = V²·⟨J·J⟩, so the textbook
    // 1/(3·V·k_B·T) prefactor for Ṁ becomes V/(3·k_B·T) for J.
    // 1/ε₀ = 4π·KAPPA in MD real units.
    let sigma_prefactor = volume / (3.0 * K_B * temperature);
    let n_freq = frequencies.len();
    let mut eps_real = Array1::zeros(n_freq);
    let mut eps_imag = Array1::zeros(n_freq);
    let eps0_factor = 4.0 * std::f64::consts::PI * KAPPA;
    for j in 0..n_freq {
        let omega = frequencies[j];
        if omega == 0.0 {
            // DC bin: σ/ω is 0/0. The true static limit lives in
            // `static_dielectric_constant`; this routine regularizes the bin
            // to (ε_∞, 0) and the caller must consult the static helper.
            eps_real[j] = epsilon_inf;
            eps_imag[j] = 0.0;
        } else {
            let sigma_re = sigma_prefactor * spec_re[j];
            let sigma_im = sigma_prefactor * spec_im[j];
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

/// Partition per-particle current into two disjoint groups by mask.
///
/// The "water/ion" naming is suggestive only — the function is a pure
/// boolean partition: particles with `mask[p] == true` are summed
/// into the first output, the rest into the second. The total
/// `J_a + J_b` equals the system current to floating-point precision.
///
/// # Arguments
/// * `per_particle_current` — `(n_particles, n_frames, 3)`,
///   **e · Å⁻² · ps⁻¹**.
/// * `water_mask` — boolean array of length `n_particles`; `true`
///   → first bucket.
///
/// # Returns
/// `(J_a, J_b)`, each `(n_frames, 3)`, same units as the input.
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

    #[test]
    fn test_static_dielectric_components() {
        let dm = ndarray::arr2(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        let result = static_dielectric_constant_components(&dm, 1000.0, 300.0, 1.0).unwrap();
        assert_eq!(result.dipole_mean.len(), 3);
        assert_eq!(result.eps.len(), 3);
        assert_eq!(result.n_frames, 2);
        // M fluctuates in x only → ε_x > ε_y, ε_z
        assert!(result.eps[0] > result.eps[1]);
        assert!((result.eps[1] - 1.0).abs() < 1e-10);
        assert!((result.eps[2] - 1.0).abs() < 1e-10);
        // Isotropic mean matches scalar version.
        let scalar = static_dielectric_constant(&dm, 1000.0, 300.0, 1.0).unwrap();
        assert!((result.eps_mean - scalar).abs() < 1e-10);
    }

    #[test]
    fn test_einstein_helfand_recovers_static_limit() {
        // Eq. (30) of Caillol-Levesque-Weis contracts at ω=0 to
        // χ(0) = (β/(3V))·⟨M²⟩, i.e. ε(0) - ε_∞ = (4π·KAPPA/(3·V·k_B·T))·⟨|δM|²⟩.
        // The spectrum's DC bin must therefore equal `static_dielectric_constant`
        // exactly (up to FP roundoff).
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);
        let n = 256;
        let mut dm = Array2::<f64>::zeros((n, 3));
        for t in 0..n {
            for d in 0..3 {
                dm[[t, d]] = rng.random::<f64>() - 0.5;
            }
        }
        let vol = 1000.0;
        let temp = 300.0;
        let eps_inf = 1.5;
        let static_eps = static_dielectric_constant(&dm, vol, temp, eps_inf).unwrap();
        for window in ["hann", "blackman", "cosine_sq"] {
            let spectrum =
                einstein_helfand_spectrum(&dm, 0.001, vol, temp, eps_inf, 50, window).unwrap();
            let dc = spectrum.epsilon_real[0];
            assert!(
                (dc - static_eps).abs() < 1e-10,
                "window={window}: ε(ω=0)={dc} should equal static_dielectric_constant={static_eps}",
            );
            // ε″(ω=0) = 0 by the (β/(3V·ε₀))·ω·X_re structure of Eq. (30).
            assert!(
                spectrum.epsilon_imag[0].abs() < 1e-12,
                "window={window}: ε″(0)={} should be 0",
                spectrum.epsilon_imag[0],
            );
        }
    }

    #[test]
    fn test_einstein_helfand_loss_positive() {
        // For a relaxing polar system the dissipative spectrum ε″(ω) must be
        // ≥ 0 at every frequency (causality / passivity). Eq. (30) gives
        // ε″(ω) = (β/(3V·ε₀))·ω·X_re(ω); X_re(ω) = ∫C(t)cos(ωt)dt ≥ 0 only on
        // average, so individual bins of a finite-sample spectrum may dip
        // slightly negative — but the *macroscopic* envelope must be non-
        // negative within statistical noise.
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(11);
        let n = 2048;
        let mut dm = Array2::<f64>::zeros((n, 3));
        // AR(1) with exponential ACF; supplies positive C(t) over [0, ∞).
        let a: f64 = (-0.01_f64).exp();
        for d in 0..3 {
            let mut prev = 0.0;
            for t in 0..n {
                let noise: f64 =
                    (0..12).map(|_| rng.random::<f64>() - 0.5).sum::<f64>() * (1.0 - a * a).sqrt();
                prev = a * prev + noise;
                dm[[t, d]] = prev;
            }
        }
        let spectrum =
            einstein_helfand_spectrum(&dm, 1.0, 1000.0, 300.0, 1.0, 100, "cosine_sq").unwrap();
        // ε″(0) is exactly zero by Eq. (30) at ω=0; non-DC bins should be
        // mostly positive.
        assert_eq!(spectrum.epsilon_imag[0], 0.0);
        let mean_loss: f64 = spectrum.epsilon_imag.iter().skip(1).sum::<f64>()
            / (spectrum.epsilon_imag.len() - 1) as f64;
        assert!(
            mean_loss > 0.0,
            "mean ε″ over ω > 0 should be positive, got {mean_loss}",
        );
    }

    #[test]
    fn test_static_dielectric_components_isotropic() {
        // Isotropic dipole (equal in all directions) → ε_x = ε_y = ε_z = ε_mean.
        let dm = ndarray::arr2(&[[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]);
        let result = static_dielectric_constant_components(&dm, 1000.0, 300.0, 1.0).unwrap();
        let eps_x = result.eps[0];
        assert!((result.eps[1] - eps_x).abs() < 1e-12);
        assert!((result.eps[2] - eps_x).abs() < 1e-12);
        assert!((result.eps_mean - eps_x).abs() < 1e-12);
    }
}
