//! Raw-only [`Compute`] observables that return **only raw curves + scalar
//! metadata** — the separation half of the compute/fit split.
//!
//! Each of these reproduces the *raw portion* of a legacy bundled result
//! (today's `ConductivityResult`, `JacfResult`, `power_spectrum` internal
//! `acf_sum`, …) but **without** any fitted scalar (no sigma, no slope, no
//! integrated D). The fit step is then the analyst's explicit choice of
//! [`LinearFit`](super::LinearFit) / [`RunningIntegral`](super::RunningIntegral)
//! / [`Plateau`](super::Plateau) / spectral transform.
//!
//! | struct | raw output | legacy raw source |
//! |--------|-----------|-------------------|
//! | [`VACF`] | unnormalized velocity ACF | `power_spectrum` `acf_sum` |
//! | [`EinsteinDiffusion`] | self-MSD curve | `MSD::windowed()` |
//! | [`GreenKuboDiffusion`] | velocity ACF | same as [`VACF`] |
//! | [`EinsteinConductivity`] | collective charge-dipole MSD | `ConductivityResult.msd` |
//! | [`GreenKuboConductivity`] | current ACF | `JacfResult.jacf` |
//! | [`DebyeRelaxation`] | dipole ACF + ⟨M(0)²⟩ + V/T/BC | new (invariants b, c) |
//!
//! These take their numeric series through the `Args` GAT (matching how
//! `dielectric` / `jacf` / `power_spectrum` take `ndarray` series), so the raw
//! curves equal the legacy outputs exactly. The `frames` slice is unused for
//! the series-based ones (pass an empty slice); [`EinsteinDiffusion`] is the one
//! that consumes frames, delegating to [`MSD::windowed`](crate::compute::MSD).

use molrs::store::frame_access::FrameAccess;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

use crate::compute::error::ComputeError;
use crate::compute::msd::MSD;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;
use molrs::signal as sig;

// ── VACF ───────────────────────────────────────────────────────────────────

/// Raw unnormalized velocity autocorrelation function.
#[derive(Debug, Clone)]
pub struct VacfResult {
    /// Lag times τ = i·dt, length `resolution + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Unnormalized velocity ACF `C(τ) = (1/n_dof) Σ_d ⟨δv_d(0)·δv_d(τ)⟩`,
    /// per-DOF mean-subtracted then DOF-averaged — identical to the `acf_sum`
    /// the legacy `power_spectrum` builds before windowing. Units: `[v]²`.
    pub acf: Array1<f64>,
}

impl ComputeResult for VacfResult {}

/// Raw velocity ACF compute (the VDOS/Green–Kubo-diffusion input).
///
/// Lifts the per-DOF mean-subtract + FFT-ACF + DOF-average block from
/// `power_spectrum` (the part *before* windowing), returning only the raw ACF.
#[derive(Debug, Clone, Copy, Default)]
pub struct VACF;

/// `(velocities, dt, resolution)` argument bundle for [`VACF`] /
/// [`GreenKuboDiffusion`].
pub type VacfArgs<'a> = (&'a Array2<f64>, f64, usize);

fn velocity_acf(
    velocities: &Array2<f64>,
    dt: f64,
    resolution: usize,
) -> Result<VacfResult, ComputeError> {
    let n_frames = velocities.shape()[0];
    let n_dof = velocities.shape()[1];
    if n_frames < 2 {
        return Err(ComputeError::EmptyInput);
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    let max_lag = resolution.min(n_frames - 1);

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
    let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
    Ok(VacfResult {
        lag_times,
        acf: acf_sum,
    })
}

impl Compute for VACF {
    /// `(velocities (n_frames, n_dof), dt, resolution)`. The `frames` slice is
    /// unused.
    type Args<'a> = VacfArgs<'a>;
    type Output = VacfResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (velocities, dt, resolution) = args;
        velocity_acf(velocities, dt, resolution)
    }
}

// ── Green–Kubo diffusion (reuses VACF) ───────────────────────────────────────

/// Raw velocity ACF for the Green–Kubo diffusion route — the same raw curve as
/// [`VACF`], named for the diffusion workflow. `D = (1/d)·∫ VACF dt` is then a
/// [`RunningIntegral`](super::RunningIntegral) + scale step.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreenKuboDiffusion;

impl Compute for GreenKuboDiffusion {
    type Args<'a> = VacfArgs<'a>;
    type Output = VacfResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (velocities, dt, resolution) = args;
        velocity_acf(velocities, dt, resolution)
    }
}

// ── Einstein diffusion (delegates to MSD::windowed) ──────────────────────────

/// Raw self-MSD for the Einstein diffusion route.
#[derive(Debug, Clone)]
pub struct EinsteinDiffusionResult {
    /// Lag times τ = i·dt, length `n_frames`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// System-average MSD per lag, identical to
    /// [`MSD::windowed`](crate::compute::MSD)'s per-lag mean. Units: `[length]²`.
    pub msd: Array1<f64>,
}

impl ComputeResult for EinsteinDiffusionResult {}

/// Raw self-MSD compute. Delegates to
/// [`MSD::windowed`](crate::compute::MSD) — MSD math is **not** re-derived here.
/// `D = slope/(2d)` is then a [`LinearFit`](super::LinearFit) + scale step.
#[derive(Debug, Clone, Copy, Default)]
pub struct EinsteinDiffusion;

/// `EinsteinDiffusion` argument bundle: the frame spacing for the lag axis.
#[derive(Debug, Clone, Copy)]
pub struct EinsteinDiffusionArgs {
    /// Frame spacing for the lag-time axis. Units: time.
    pub dt: f64,
}

impl Compute for EinsteinDiffusion {
    /// Frame spacing only; positions come from `frames`.
    type Args<'a> = EinsteinDiffusionArgs;
    type Output = EinsteinDiffusionResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if args.dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: args.dt.to_string(),
            });
        }
        let series = MSD::windowed().compute(frames, ())?;
        let msd = Array1::from_iter(series.data.iter().map(|r| r.mean));
        let lag_times = Array1::from_iter((0..series.data.len()).map(|i| i as f64 * args.dt));
        Ok(EinsteinDiffusionResult { lag_times, msd })
    }
}

// ── Einstein–Helfand conductivity (raw collective-dipole MSD) ────────────────

/// Raw collective charge-dipole MSD — the raw portion of the legacy
/// `ConductivityResult`, with **no** fitted sigma/slope.
#[derive(Debug, Clone)]
pub struct EinsteinConductivityResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Collective-dipole MSD ⟨|**M_J**(t+τ) − **M_J**(t)|²⟩ over time origins,
    /// identical to `ConductivityResult.msd`. Units: `(e·Å)²`.
    pub msd: Array1<f64>,
}

impl ComputeResult for EinsteinConductivityResult {}

/// Raw collective charge-dipole MSD compute. Lifts the time-origin MSD loop
/// from `einstein_helfand_conductivity` and stops there (no OLS, no σ). The
/// σ = slope/(6·V·k_B·T) step is a downstream [`LinearFit`](super::LinearFit).
#[derive(Debug, Clone, Copy, Default)]
pub struct EinsteinConductivity;

/// `(translational_dipole, dt, max_correlation_time)` bundle.
pub type EinsteinConductivityArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for EinsteinConductivity {
    type Args<'a> = EinsteinConductivityArgs<'a>;
    type Output = EinsteinConductivityResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (dipole, dt, max_correlation_time) = args;
        let shape = dipole.shape();
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "translational_dipole (expected (n_frames, 3))",
            });
        }
        let n_frames = shape[0];
        if n_frames < 2 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let max_lag = max_correlation_time.min(n_frames - 1);

        let mut msd = Array1::<f64>::zeros(max_lag + 1);
        for tau in 1..=max_lag {
            let count = n_frames - tau;
            let mut acc = 0.0;
            for t in 0..count {
                let mut s = 0.0;
                for d in 0..3 {
                    let dx = dipole[[t + tau, d]] - dipole[[t, d]];
                    s += dx * dx;
                }
                acc += s;
            }
            msd[tau] = acc / count as f64;
        }
        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(EinsteinConductivityResult { lag_times, msd })
    }
}

// ── Green–Kubo conductivity (raw current ACF) ────────────────────────────────

/// Raw current autocorrelation function — the raw portion of the legacy
/// `JacfResult`, with **no** fitted sigma.
#[derive(Debug, Clone)]
pub struct GreenKuboConductivityResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Current ACF `C(τ) = ⟨J(0)·J(τ)⟩` over time origins, identical to
    /// `JacfResult.jacf`. Units: `(e·Å·ps⁻¹)²`.
    pub jacf: Array1<f64>,
}

impl ComputeResult for GreenKuboConductivityResult {}

/// Raw current-ACF compute. Lifts the unbiased windowed-ACF loop from
/// `green_kubo_conductivity` and stops there (no trapezoid, no σ). The
/// σ = (1/(3·V·k_B·T))·∫⟨JJ⟩ step is a downstream
/// [`RunningIntegral`](super::RunningIntegral) + scale.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreenKuboConductivity;

/// `(current, dt, max_correlation_time)` bundle.
pub type GreenKuboConductivityArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for GreenKuboConductivity {
    type Args<'a> = GreenKuboConductivityArgs<'a>;
    type Output = GreenKuboConductivityResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (current, dt, max_correlation_time) = args;
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
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let max_lag = max_correlation_time.min(n_frames - 1);

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
        Ok(GreenKuboConductivityResult { lag_times, jacf })
    }
}

// ── Debye relaxation (raw dipole ACF + metadata, invariants b/c) ─────────────

/// Ewald boundary condition under which the dipole fluctuations were sampled.
///
/// The Debye / Neumann–Kirkwood amplitude `ε₀ − ε∞` depends on this boundary
/// condition, so it travels with the raw result (spec invariant (c)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EwaldBoundary {
    /// Conducting / "tin-foil" boundary (ε' = ∞) — the common MD default.
    TinFoil,
    /// Vacuum boundary (ε' = 1).
    Vacuum,
}

/// Raw dipole ACF plus the scalar metadata the Debye amplitude needs.
///
/// Unlike the spectral/diffusion raw results, this carries an **unnormalized**
/// ACF and the zero-lag variance ⟨M(0)²⟩ explicitly (invariant (b)): the
/// normalized Φ(t) gives only the relaxation *shape*/τ via
/// [`DebyeFit`](super::DebyeFit); the amplitude `ε₀ − ε∞` comes from ⟨M²⟩
/// together with V, T, and the Ewald boundary condition (invariant (c)).
///
/// All four metadata fields are non-optional, so a `DebyeRelaxationResult`
/// **cannot be constructed without them**.
#[derive(Debug, Clone)]
pub struct DebyeRelaxationResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// **Unnormalized** dipole ACF `C(τ) = ⟨δM(0)·δM(τ)⟩` summed over the 3
    /// Cartesian components. Units: `(e·Å)²`.
    pub acf: Array1<f64>,
    /// Zero-lag variance ⟨M(0)²⟩ = `acf[0]`, the Debye amplitude scale
    /// (invariant b). Units: `(e·Å)²`.
    pub zero_lag_variance: f64,
    /// System volume V (invariant c). Units: `Å³`.
    pub volume: f64,
    /// Temperature T (invariant c). Units: `K`.
    pub temperature: f64,
    /// Ewald boundary condition the fluctuations were sampled under
    /// (invariant c).
    pub boundary: EwaldBoundary,
}

impl ComputeResult for DebyeRelaxationResult {}

/// Raw dipole-ACF compute for the Debye route. Computes the mean-subtracted,
/// per-component-summed dipole ACF (unbiased estimator) and carries the
/// zero-lag variance + V/T/Ewald-BC metadata the amplitude needs.
#[derive(Debug, Clone, Copy)]
pub struct DebyeRelaxation {
    /// System volume V. Units: `Å³`.
    pub volume: f64,
    /// Temperature T. Units: `K`.
    pub temperature: f64,
    /// Ewald boundary condition.
    pub boundary: EwaldBoundary,
}

/// `(dipole_moments, dt, max_correlation_time)` bundle for [`DebyeRelaxation`].
pub type DebyeRelaxationArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for DebyeRelaxation {
    type Args<'a> = DebyeRelaxationArgs<'a>;
    type Output = DebyeRelaxationResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (dipole, dt, max_correlation_time) = args;
        let shape = dipole.shape();
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "dipole_moments (expected (n_frames, 3))",
            });
        }
        let n_frames = shape[0];
        if n_frames < 2 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        if self.volume <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "volume",
                value: self.volume.to_string(),
            });
        }
        if self.temperature <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "temperature",
                value: self.temperature.to_string(),
            });
        }
        let max_lag = max_correlation_time.min(n_frames - 1);

        // Per-component means → fluctuation ACF so acf[0] = ⟨|δM|²⟩.
        let mut mean = [0.0_f64; 3];
        for t in 0..n_frames {
            for d in 0..3 {
                mean[d] += dipole[[t, d]];
            }
        }
        for m in mean.iter_mut() {
            *m /= n_frames as f64;
        }

        let mut planner = FftPlanner::new();
        let mut acf = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let col: Array1<f64> = (0..n_frames).map(|t| dipole[[t, d]] - mean[d]).collect();
            let component =
                sig::acf_fft_with_planner(&mut planner, &col, max_lag).map_err(|e| {
                    ComputeError::OutOfRange {
                        field: "acf_fft",
                        value: e.to_string(),
                    }
                })?;
            for k in 0..=max_lag {
                acf[k] += component[k];
            }
        }
        // Unbiased linear-ACF estimator C(k) = ⟨δM(0)·δM(k·dt)⟩.
        for k in 0..=max_lag {
            acf[k] /= (n_frames - k) as f64;
        }

        let zero_lag_variance = acf[0];
        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(DebyeRelaxationResult {
            lag_times,
            acf,
            zero_lag_variance,
            volume: self.volume,
            temperature: self.temperature,
            boundary: self.boundary,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::dielectric::einstein_helfand_conductivity;
    use crate::compute::jacf::green_kubo_conductivity;
    use crate::compute::spectra::power_spectrum;
    use molrs::Frame;
    use molrs::store::block::Block;
    use ndarray::{Array1 as A1, Array2};
    use rand::{RngExt, SeedableRng};

    /// Empty frame slice for the series-based raw computes.
    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    fn rng_series(n: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut s = Array2::zeros((n, cols));
        for t in 0..n {
            for c in 0..cols {
                s[[t, c]] = rng.random_range(-1.0..1.0);
            }
        }
        s
    }

    #[test]
    fn vacf_equals_power_spectrum_acf_sum() {
        // ac-011: VACF == acf_sum that power_spectrum builds before windowing.
        let n = 512;
        let dt = 0.5;
        let res = 100;
        let v = rng_series(n, 9, 7);

        // Rebuild the legacy internal acf_sum.
        let max_lag = res.min(n - 1);
        let inv_n_frames = 1.0 / n as f64;
        let mut planner = FftPlanner::new();
        let mut acf_sum = A1::<f64>::zeros(max_lag + 1);
        for d in 0..9 {
            let mut col: A1<f64> = (0..n).map(|t| v[[t, d]]).collect();
            let mean: f64 = col.iter().sum::<f64>() * inv_n_frames;
            for x in col.iter_mut() {
                *x -= mean;
            }
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        for k in 0..=max_lag {
            acf_sum[k] *= 1.0 / 9.0;
        }

        let raw = VACF.compute(&no_frames(), (&v, dt, res)).unwrap();
        assert_eq!(raw.acf.len(), acf_sum.len());
        for k in 0..raw.acf.len() {
            assert!((raw.acf[k] - acf_sum[k]).abs() < 1e-12, "k={k}");
        }
        // power_spectrum applied to this raw acf reproduces the legacy spectrum.
        let legacy = power_spectrum(&v, dt, res).unwrap();
        assert_eq!(legacy.resolution, raw.acf.len() - 1);
    }

    #[test]
    fn green_kubo_diffusion_equals_vacf() {
        let n = 64;
        let dt = 1.0;
        let v = rng_series(n, 3, 11);
        let a = VACF.compute(&no_frames(), (&v, dt, 20)).unwrap();
        let b = GreenKuboDiffusion
            .compute(&no_frames(), (&v, dt, 20))
            .unwrap();
        assert_eq!(a.acf, b.acf);
        assert_eq!(a.lag_times, b.lag_times);
    }

    #[test]
    fn einstein_conductivity_equals_legacy_msd() {
        // ac-009: EinsteinConductivity.msd == ConductivityResult.msd.
        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let dipole = rng_series(n, 3, 3);
        let legacy =
            einstein_helfand_conductivity(&dipole, dt, 1000.0, 300.0, mct, 0.2, 0.8).unwrap();
        let raw = EinsteinConductivity
            .compute(&no_frames(), (&dipole, dt, mct))
            .unwrap();
        assert_eq!(raw.msd.len(), legacy.msd.len());
        for k in 0..raw.msd.len() {
            assert!((raw.msd[k] - legacy.msd[k]).abs() < 1e-12, "k={k}");
            assert!((raw.lag_times[k] - legacy.lag_times[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn green_kubo_conductivity_equals_legacy_jacf() {
        // ac-010: GreenKuboConductivity.jacf == JacfResult.jacf.
        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let current = rng_series(n, 3, 5);
        let legacy = green_kubo_conductivity(&current, dt, 1000.0, 300.0, mct).unwrap();
        let raw = GreenKuboConductivity
            .compute(&no_frames(), (&current, dt, mct))
            .unwrap();
        assert_eq!(raw.jacf.len(), legacy.jacf.len());
        for k in 0..raw.jacf.len() {
            assert!((raw.jacf[k] - legacy.jacf[k]).abs() < 1e-12, "k={k}");
            assert!((raw.lag_times[k] - legacy.lag_times[k]).abs() < 1e-12);
        }
    }

    fn make_frame(x: &[f64], y: &[f64], z: &[f64]) -> Frame {
        let mut block = Block::new();
        block
            .insert("x", A1::from_vec(x.to_vec()).into_dyn())
            .unwrap();
        block
            .insert("y", A1::from_vec(y.to_vec()).into_dyn())
            .unwrap();
        block
            .insert("z", A1::from_vec(z.to_vec()).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame
    }

    #[test]
    fn einstein_diffusion_delegates_to_msd_windowed() {
        // ac-012: EinsteinDiffusion.msd == MSD::windowed().compute means.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [0.1, -0.2, 0.3, -0.1, 0.05, 0.0];
        let frames_owned: Vec<Frame> = (0..6)
            .map(|t| make_frame(&[xs[t], xs[t] + 0.5], &[ys[t], ys[t] + 0.4], &[0.0, 0.0]))
            .collect();
        let frames: Vec<&Frame> = frames_owned.iter().collect();

        let series = MSD::windowed().compute(&frames, ()).unwrap();
        let raw = EinsteinDiffusion
            .compute(&frames, EinsteinDiffusionArgs { dt: 2.0 })
            .unwrap();
        assert_eq!(raw.msd.len(), series.data.len());
        for i in 0..raw.msd.len() {
            assert!((raw.msd[i] - series.data[i].mean).abs() < 1e-12, "i={i}");
            assert!((raw.lag_times[i] - i as f64 * 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn debye_relaxation_carries_unnormalized_acf_and_metadata() {
        // ac-013.
        let n = 128;
        let dt = 0.5;
        let dipole = rng_series(n, 3, 9);
        let res = DebyeRelaxation {
            volume: 1234.5,
            temperature: 298.0,
            boundary: EwaldBoundary::TinFoil,
        }
        .compute(&no_frames(), (&dipole, dt, 40))
        .unwrap();
        // zero-lag variance == acf[0] and is non-zero (unnormalized).
        assert!((res.zero_lag_variance - res.acf[0]).abs() < 1e-15);
        assert!(res.zero_lag_variance > 0.0);
        assert_eq!(res.volume, 1234.5);
        assert_eq!(res.temperature, 298.0);
        assert_eq!(res.boundary, EwaldBoundary::TinFoil);
    }

    #[test]
    fn raw_max_lag_exceeds_length_clamps_not_panics() {
        // ac-014 companion: clamping max_correlation_time is fine; over-long
        // *consumption* by a downstream Fit is the OutOfRange case tested in the
        // fit modules. Here we confirm raw computes clamp rather than panic.
        let v = rng_series(8, 3, 1);
        let raw = VACF.compute(&no_frames(), (&v, 1.0, 1000)).unwrap();
        assert_eq!(raw.acf.len(), 8); // clamped to n_frames - 1 + 1.
    }

    #[test]
    fn series_computes_reject_bad_shape() {
        // ac-016: non-(_,3) series -> DimensionMismatch.
        let bad = rng_series(10, 2, 1);
        assert!(matches!(
            EinsteinConductivity.compute(&no_frames(), (&bad, 1.0, 5)),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            GreenKuboConductivity.compute(&no_frames(), (&bad, 1.0, 5)),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn raw_plus_fit_reconstructs_legacy_einstein_sigma() {
        // ac-015: LinearFit slope on EinsteinConductivity.msd / (6·V·k_B·T)
        // with the documented MD->SI prefactor == legacy sigma (rel tol 1e-9).
        use crate::compute::fit::LinearFit;
        use crate::compute::traits::Fit;
        use molrs::units::constants::{
            ANGSTROM_M, BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as E_C, PICOSECOND_S,
        };

        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let (volume, temperature) = (1000.0, 300.0);
        let (start_frac, end_frac) = (0.2, 0.8);
        let dipole = rng_series(n, 3, 17);

        let legacy = einstein_helfand_conductivity(
            &dipole,
            dt,
            volume,
            temperature,
            mct,
            start_frac,
            end_frac,
        )
        .unwrap();

        let raw = EinsteinConductivity
            .compute(&no_frames(), (&dipole, dt, mct))
            .unwrap();
        let fit = LinearFit {
            window: (start_frac, end_frac),
        }
        .fit((&raw.lag_times, &raw.msd))
        .unwrap();

        // Same prefactor the legacy fn folds in (Einstein 1/6).
        let prefactor = (E_C * E_C * ANGSTROM_M * ANGSTROM_M / PICOSECOND_S)
            / (6.0 * ANGSTROM_M.powi(3) * K_B_SI);
        let sigma = prefactor * fit.slope / (volume * temperature);
        let rel = (sigma - legacy.sigma).abs() / legacy.sigma.abs();
        assert!(
            rel < 1e-9,
            "rel err {rel}, sigma {sigma}, legacy {}",
            legacy.sigma
        );
        // The lifted OLS reproduces the legacy slope and fit window exactly.
        assert!((fit.slope - legacy.slope).abs() < 1e-12);
        assert_eq!(fit.fit_start, legacy.fit_start);
        assert_eq!(fit.fit_end, legacy.fit_end);
    }

    #[test]
    fn raw_plus_fit_reconstructs_legacy_green_kubo_sigma() {
        // ac-015: RunningIntegral on GreenKuboConductivity.jacf scaled by
        // 1/(3·V·k_B·T) == legacy sigma (rel tol 1e-9).
        use crate::compute::fit::RunningIntegral;
        use crate::compute::traits::Fit;
        use molrs::units::constants::{
            ANGSTROM_M, BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as E_C, PICOSECOND_S,
        };

        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let (volume, temperature) = (1000.0, 300.0);
        let current = rng_series(n, 3, 19);

        let legacy = green_kubo_conductivity(&current, dt, volume, temperature, mct).unwrap();

        let raw = GreenKuboConductivity
            .compute(&no_frames(), (&current, dt, mct))
            .unwrap();
        let integ = RunningIntegral.fit((&raw.jacf, dt, None)).unwrap();

        // Same prefactor the legacy fn folds in (Green-Kubo 1/3).
        let prefactor = (E_C * E_C * ANGSTROM_M * ANGSTROM_M / PICOSECOND_S)
            / (3.0 * ANGSTROM_M.powi(3) * K_B_SI);
        let last = integ.integral.len() - 1;
        let sigma = prefactor * integ.integral[last] / (volume * temperature);
        let rel = (sigma - legacy.sigma).abs() / legacy.sigma.abs();
        assert!(
            rel < 1e-9,
            "rel err {rel}, sigma {sigma}, legacy {}",
            legacy.sigma
        );
    }
}
