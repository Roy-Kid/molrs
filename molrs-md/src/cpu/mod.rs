pub mod dynamics;

use molrs::core::frame::Frame;
use molrs::core::potential::{PotentialSet, Potentials, extract_coords};
use molrs::core::types::F;

use crate::backend::{Backend, CPU, DynamicsBackend, MinimizerConfig};
use crate::error::MDError;
use crate::md::MinState;
use crate::run::dump::Dump;
use crate::run::fix::Fix;

/// CPU minimizer — wraps a PotentialSet + steepest descent.
pub struct CPUMinimizer {
    potential_set: PotentialSet,
    config: MinimizerConfig,
}

impl Backend for CPU {
    type Device = ();
    type Minimizer = CPUMinimizer;

    fn compile(
        _device: (),
        potential_set: PotentialSet,
        config: MinimizerConfig,
    ) -> Result<CPUMinimizer, MDError> {
        Ok(CPUMinimizer {
            potential_set,
            config,
        })
    }
}

impl CPUMinimizer {
    /// Run minimization on a Frame. Returns the updated Frame and a typed
    /// `MinState` with energy, n_steps, n_evals, and converged.
    ///
    /// Topology is bound once from the Frame at the start (pre-bound indices for
    /// hot loop performance). Only coordinates change during iteration.
    pub fn run(&mut self, frame: &Frame) -> Result<(Frame, MinState), MDError> {
        // Bind topology once → pre-bound Potentials
        let potentials = self
            .potential_set
            .bind(frame)
            .map_err(MDError::ForcefieldError)?;

        // Extract coordinates
        let mut coords = extract_coords(frame).map_err(MDError::ForcefieldError)?;

        // Run minimization
        let result = self.minimize_loop(&potentials, coords.as_mut_slice())?;

        // Build output frame
        let mut out = frame.clone();
        let n_atoms = coords.len() / 3;
        write_coords_to_frame(&mut out, &coords, n_atoms);

        Ok((out, result))
    }

    /// Steepest descent with Armijo backtracking line search.
    ///
    /// Convergence criteria (LAMMPS-style, stops when ANY is satisfied):
    ///   - force_tol:  ||force||_2 / sqrt(3N) < ftol   (RMS force per component)
    ///   - energy_tol: |dE| < etol * 0.5*(|E_new|+|E_old|+eps)  (relative energy change)
    ///   - max_iter:   iteration count exceeded
    ///   - max_eval:   energy/force evaluation count exceeded
    ///
    /// dmax caps the maximum per-atom displacement in a single step.
    fn minimize_loop(
        &self,
        potentials: &Potentials,
        coords: &mut [F],
    ) -> Result<MinState, MDError> {
        use std::time::Instant;

        let max_iter = self.config.max_iter;
        let max_eval = self.config.max_eval;
        let etol = self.config.energy_tol;
        let ftol = self.config.force_tol;
        let dmax = self.config.dmax;
        let n = coords.len();
        let n_atoms = n / 3;
        let ndof = n as F; // 3N degrees of freedom

        // Resolve log_every: 0 = auto
        let log_every = if self.config.log_every > 0 {
            self.config.log_every
        } else {
            (max_iter / 10).clamp(10, 500)
        };

        // ── header ──
        log::info!("Minimization: {} atoms, method=sd/armijo", n_atoms);
        log::info!(
            "  ftol={:.1e}, etol={:.1e}, maxiter={}, maxeval={}, dmax={:.3}",
            ftol,
            etol,
            max_iter,
            if max_eval == usize::MAX {
                "inf".to_string()
            } else {
                max_eval.to_string()
            },
            dmax,
        );
        log::info!(
            "{:>10} {:>16} {:>14} {:>14}",
            "Step",
            "Energy",
            "ForceRMS",
            "Alpha"
        );

        let t0 = Instant::now();

        let mut trial = vec![0.0 as F; n];
        let mut converged = false;
        let mut n_steps: usize = 0;
        let mut n_evals: usize = 0;
        let mut prev_alpha: F = 0.1;
        let mut energy_prev: F = F::MAX;
        let mut last_frms: F = F::NAN;

        let eps_energy: F = 1e-16;

        for _ in 0..max_iter {
            if n_evals >= max_eval {
                break;
            }

            n_steps += 1;

            // ── forces (= -gradient) ──
            let forces = potentials.forces(coords);
            n_evals += 1;

            // ── force tolerance: RMS force per component ──
            let force_sq: F = forces.iter().map(|f| f * f).sum();
            let frms = (force_sq / ndof).sqrt();
            last_frms = frms;

            if ftol > 0.0 && frms < ftol {
                converged = true;
                // Print final step before breaking
                let energy_current = potentials.energy(coords);
                log::info!(
                    "{:>10} {:>16.6} {:>14.4e} {:>14.4e}",
                    n_steps,
                    energy_current,
                    frms,
                    prev_alpha
                );
                break;
            }

            // ── current energy ──
            let energy_current = potentials.energy(coords);
            n_evals += 1;

            // ── progress line ──
            if n_steps == 1 || n_steps.is_multiple_of(log_every) {
                log::info!(
                    "{:>10} {:>16.6} {:>14.4e} {:>14.4e}",
                    n_steps,
                    energy_current,
                    frms,
                    prev_alpha
                );
            }

            // ── energy tolerance: LAMMPS relative criterion ──
            if etol > 0.0 && energy_prev < F::MAX {
                let de = (energy_current - energy_prev).abs();
                let eref = 0.5 * (energy_current.abs() + energy_prev.abs() + eps_energy);
                if de < etol * eref {
                    converged = true;
                    break;
                }
            }
            energy_prev = energy_current;

            // ── steepest-descent direction (along forces = -gradient) ──
            // max per-atom force magnitude (for dmax clamping)
            let max_atom_force = (0..n_atoms)
                .map(|i| {
                    let fx = forces[3 * i];
                    let fy = forces[3 * i + 1];
                    let fz = forces[3 * i + 2];
                    (fx * fx + fy * fy + fz * fz).sqrt()
                })
                .fold(0.0 as F, F::max);

            if max_atom_force == 0.0 {
                converged = true;
                break;
            }

            // Initial step size (adaptive from previous accepted alpha)
            let mut alpha = (2.0 * prev_alpha).min(1.0 / max_atom_force);

            // Clamp so no atom moves more than dmax
            if dmax > 0.0 {
                let alpha_max = dmax / max_atom_force;
                alpha = alpha.min(alpha_max);
            }

            // ── Armijo backtracking line search ──
            // Step along force direction: x_trial = x + alpha * force
            // Armijo condition: E(trial) <= E(current) - c * alpha * ||force||^2
            let c: F = 1e-4;
            let rho: F = 0.5;
            let mut accepted = false;

            for _ in 0..40 {
                if n_evals >= max_eval {
                    break;
                }

                trial
                    .iter_mut()
                    .zip(coords.iter().zip(forces.iter()))
                    .for_each(|(t, (x, f))| *t = x + alpha * f);

                let energy_trial = potentials.energy(&trial);
                n_evals += 1;

                if energy_trial <= energy_current - c * alpha * force_sq {
                    accepted = true;
                    break;
                }
                alpha *= rho;
            }

            // Only update coordinates when line search succeeds
            if accepted {
                prev_alpha = alpha;
                coords
                    .iter_mut()
                    .zip(forces.iter())
                    .for_each(|(x, f)| *x += alpha * f);
            }
        }

        let energy = potentials.energy(coords);
        let elapsed = t0.elapsed();

        // ── summary ──
        let status = if converged {
            "CONVERGED"
        } else {
            "NOT CONVERGED"
        };
        log::info!(
            "Minimization {}: E={:.6}, steps={}, evals={}, frms={:.4e}",
            status,
            energy,
            n_steps,
            n_evals,
            last_frms
        );
        log::info!("Wall time: {:.3}s", elapsed.as_secs_f64());

        Ok(MinState {
            energy,
            n_steps,
            n_evals,
            converged,
        })
    }
}

impl DynamicsBackend for CPU {
    type Device = ();
    type Dynamics = dynamics::CPUDynamics;

    fn compile_dynamics(
        _device: (),
        potential_set: PotentialSet,
        fixes: Vec<Box<dyn Fix>>,
        dumps: Vec<Box<dyn Dump>>,
        dt: F,
    ) -> Result<dynamics::CPUDynamics, MDError> {
        Ok(dynamics::CPUDynamics::new(potential_set, fixes, dumps, dt))
    }
}

/// Write flat coordinates back into Frame's atoms block as F x/y/z columns.
fn write_coords_to_frame(frame: &mut Frame, coords: &[F], n_atoms: usize) {
    use ndarray::Array1;

    let mut xs = Vec::with_capacity(n_atoms);
    let mut ys = Vec::with_capacity(n_atoms);
    let mut zs = Vec::with_capacity(n_atoms);
    for i in 0..n_atoms {
        xs.push(coords[i * 3]);
        ys.push(coords[i * 3 + 1]);
        zs.push(coords[i * 3 + 2]);
    }

    if let Some(atoms) = frame.get_mut("atoms") {
        atoms.insert("x", Array1::from_vec(xs).into_dyn()).ok();
        atoms.insert("y", Array1::from_vec(ys).into_dyn()).ok();
        atoms.insert("z", Array1::from_vec(zs).into_dyn()).ok();
    }
}
