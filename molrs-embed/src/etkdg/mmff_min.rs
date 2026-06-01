//! Second-stage MMFF94 cleanup minimizer (L-BFGS).
//!
//! RDKit's `MMFFOptimizeMolecule` runs a full energy minimization (BFGS in
//! `$RDBASE/Code/Numerics/Optimizer/BFGSOpt.h`) to a gradient-norm tolerance,
//! relaxing the freshly-embedded geometry to the MMFF94 minimum. The original
//! cleanup here used a plain steepest-descent loop with a step-doubling line
//! search; on stiff terms (aromatic-ring bond/angle coupling in benzene) that
//! path stalls far above the true minimum (benzene final ≈ 21.7 vs the RDKit
//! MMFF-optimized 16.2, ≈ 34 % high).
//!
//! This module provides a limited-memory BFGS (L-BFGS) minimizer with a
//! backtracking line search satisfying the Armijo sufficient-decrease and weak
//! Wolfe curvature conditions, mirroring the convergence behaviour of RDKit's
//! BFGS so cleanup reaches the MMFF minimum. It consumes the same
//! `(energy, forces = -grad)` evaluator that `molrs_ff::potential::Potential`
//! exposes — the force field itself is untouched.
//!
//! Reference:
//!   Nocedal & Wright, *Numerical Optimization* (2nd ed.), Algorithm 7.4
//!   (L-BFGS two-loop recursion) + Algorithm 3.1 (backtracking line search).
//!   RDKit `BFGSOpt::minimize`, BSD-3, Copyright (C) Greg Landrum / RDKit.

/// Outcome of an L-BFGS minimization: `(energy, grad_rms, steps, converged)`.
/// `grad_rms` is the root-mean-square gradient at the returned point
/// (kcal/mol/Å), the quantity compared against `grad_rms_tol`.
pub type MinResult = (f64, f64, usize, bool);

/// Number of correction pairs retained by the L-BFGS history.
const HISTORY: usize = 8;
/// Armijo sufficient-decrease parameter (`c1`).
const ARMIJO_C1: f64 = 1e-4;
/// Weak-Wolfe curvature parameter (`c2`).
const WOLFE_C2: f64 = 0.9;
/// Line-search step contraction factor on Armijo failure.
const BACKTRACK: f64 = 0.5;
/// Line-search step expansion factor while curvature is still unmet.
const EXPAND: f64 = 2.1;
/// Maximum line-search trial evaluations per outer iteration.
const MAX_LS_TRIALS: usize = 40;

/// Minimize a `(energy, forces = -grad)` evaluator with L-BFGS.
///
/// Convergence is declared when the RMS gradient
/// (`‖grad‖ / sqrt(n)`) drops below `grad_rms_tol` (kcal/mol/Å). The search
/// also stops — reporting `converged = true` at a stationary point — when the
/// line search can make no further progress, matching RDKit's BFGS exit on a
/// vanishing step.
///
/// Arguments:
///   * `coords` — flat `3·n_atoms` buffer, updated in place to the minimizer.
///   * `max_iters` — outer-iteration cap.
///   * `grad_rms_tol` — RMS-gradient convergence threshold (kcal/mol/Å).
///   * `eval` — returns `(energy, forces)` with `forces = -grad`.
pub fn minimize_lbfgs<F>(
    coords: &mut [f64],
    max_iters: usize,
    grad_rms_tol: f64,
    mut eval: F,
) -> MinResult
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let n = coords.len();
    if n == 0 {
        return (0.0, 0.0, 0, true);
    }

    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let (mut energy, forces) = eval(coords);
    // grad = -forces.
    let mut grad: Vec<f64> = forces.iter().map(|f| -f).collect();

    // L-BFGS correction history: s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k.
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(HISTORY);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(HISTORY);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(HISTORY);

    let mut converged = false;
    let mut iters = 0;

    for it in 0..max_iters {
        iters = it + 1;

        let gnorm2 = dot(&grad, &grad);
        let grad_rms = gnorm2.sqrt() * inv_sqrt_n;
        if grad_rms < grad_rms_tol {
            converged = true;
            break;
        }

        // --- L-BFGS two-loop recursion: direction = -H·grad ---------------
        let mut q = grad.clone();
        let m = s_hist.len();
        let mut alpha = vec![0.0; m];
        // Newest correction first.
        for i in (0..m).rev() {
            let a = rho_hist[i] * dot(&s_hist[i], &q);
            alpha[i] = a;
            axpy(&mut q, -a, &y_hist[i]);
        }
        // Initial Hessian scaling H0 = (s·y)/(y·y)·I (Nocedal & Wright 7.20).
        let gamma = if m > 0 {
            let last = m - 1;
            let sy = dot(&s_hist[last], &y_hist[last]);
            let yy = dot(&y_hist[last], &y_hist[last]);
            if yy > 1e-30 { sy / yy } else { 1.0 }
        } else {
            // First step: scale by 1/‖g‖ so the initial displacement is ~1 Å.
            let gn = gnorm2.sqrt();
            if gn > 1e-12 { (1.0 / gn).min(0.1) } else { 1.0 }
        };
        scale(&mut q, gamma);
        for i in 0..m {
            let beta = rho_hist[i] * dot(&y_hist[i], &q);
            axpy(&mut q, alpha[i] - beta, &s_hist[i]);
        }
        // Search direction p = -H·grad.
        let mut dir = q;
        scale(&mut dir, -1.0);

        let g_dot_dir = dot(&grad, &dir);
        // Guard against a non-descent direction (numerical breakdown): fall
        // back to steepest descent.
        let dir = if g_dot_dir >= 0.0 {
            let mut d = grad.clone();
            scale(&mut d, -1.0);
            d
        } else {
            dir
        };
        let g_dot_dir = dot(&grad, &dir);

        // --- Backtracking + curvature line search -------------------------
        let x0 = coords.to_vec();
        let e0 = energy;
        let mut step = 1.0;
        let mut lo = 0.0;
        let mut hi = f64::INFINITY;
        let mut accepted = false;
        let mut new_grad = grad.clone();

        for _ in 0..MAX_LS_TRIALS {
            let mut trial = x0.clone();
            axpy(&mut trial, step, &dir);
            let (e_trial, f_trial) = eval(&trial);

            if !e_trial.is_finite() {
                hi = step;
                step = 0.5 * (lo + hi);
                continue;
            }

            // Armijo sufficient decrease.
            if e_trial > e0 + ARMIJO_C1 * step * g_dot_dir {
                hi = step;
                step = 0.5 * (lo + hi);
                continue;
            }

            // Weak-Wolfe curvature on the new gradient along dir.
            let g_new: Vec<f64> = f_trial.iter().map(|f| -f).collect();
            let g_new_dot_dir = dot(&g_new, &dir);
            if g_new_dot_dir < WOLFE_C2 * g_dot_dir {
                // Curvature unmet: step too short, expand.
                lo = step;
                if hi.is_finite() {
                    step = 0.5 * (lo + hi);
                } else {
                    step *= EXPAND;
                }
                continue;
            }

            // Both conditions satisfied — accept.
            coords.copy_from_slice(&trial);
            energy = e_trial;
            new_grad = g_new;
            accepted = true;
            break;
        }

        if !accepted {
            // Armijo-only fallback: take the best sufficient-decrease point we
            // can find by pure backtracking, so a too-strict curvature test
            // never blocks progress.
            let mut s = step.max(BACKTRACK);
            let mut made = false;
            for _ in 0..MAX_LS_TRIALS {
                let mut trial = x0.clone();
                axpy(&mut trial, s, &dir);
                let (e_trial, f_trial) = eval(&trial);
                if e_trial.is_finite() && e_trial < e0 + ARMIJO_C1 * s * g_dot_dir {
                    coords.copy_from_slice(&trial);
                    energy = e_trial;
                    new_grad = f_trial.iter().map(|f| -f).collect();
                    made = true;
                    break;
                }
                s *= BACKTRACK;
            }
            if !made {
                // Stationary point — no decrease possible.
                converged = true;
                break;
            }
        }

        // --- Update L-BFGS history ---------------------------------------
        let mut s_k = coords.to_vec();
        axpy(&mut s_k, -1.0, &x0);
        let mut y_k = new_grad.clone();
        axpy(&mut y_k, -1.0, &grad);
        let sy = dot(&s_k, &y_k);
        if sy > 1e-12 {
            if s_hist.len() == HISTORY {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s_k);
            y_hist.push(y_k);
        }
        grad = new_grad;
    }

    let grad_rms = dot(&grad, &grad).sqrt() * inv_sqrt_n;
    (energy, grad_rms, iters, converged)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `a += alpha · b`.
#[inline]
fn axpy(a: &mut [f64], alpha: f64, b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += alpha * y;
    }
}

#[inline]
fn scale(a: &mut [f64], factor: f64) {
    for x in a.iter_mut() {
        *x *= factor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple anisotropic quadratic bowl: minimum at the origin, stiff in
    /// one direction (condition number 100) — a stress test that plain
    /// steepest descent handles poorly but L-BFGS solves in a few steps.
    #[test]
    fn lbfgs_solves_stiff_quadratic() {
        let k = [1.0, 10.0, 100.0, 1.0, 10.0, 100.0];
        let eval = |x: &[f64]| -> (f64, Vec<f64>) {
            let mut e = 0.0;
            let mut forces = vec![0.0; x.len()];
            for i in 0..x.len() {
                e += 0.5 * k[i] * x[i] * x[i];
                forces[i] = -k[i] * x[i]; // force = -grad
            }
            (e, forces)
        };
        let mut x = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        let (e, grms, steps, conv) = minimize_lbfgs(&mut x, 200, 1e-6, eval);
        assert!(conv, "should converge");
        assert!(e < 1e-8, "energy should reach ~0, got {e}");
        assert!(grms < 1e-6, "grad RMS should be tiny, got {grms}");
        assert!(steps < 100, "should converge quickly, took {steps}");
        for xi in &x {
            assert!(xi.abs() < 1e-4, "coord should reach 0, got {xi}");
        }
    }
}
