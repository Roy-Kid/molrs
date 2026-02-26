use std::time::Instant;

use molrs::core::frame::Frame;
use molrs::core::potential::{PotentialSet, Potentials};
use molrs::core::types::F;

use crate::backend::DynamicsEngine;
use crate::error::MDError;
use crate::run::dump::Dump;
use crate::run::fix::Fix;
use crate::run::stage::StageMask;
use crate::run::state::MDState;

/// CPU dynamics engine — orchestrates the LAMMPS-style time-step loop.
///
/// The engine is responsible for:
/// 1. Managing the execution stages (initial_integrate → force → final_integrate → ...)
/// 2. Computing forces
/// 3. Dispatching fix callbacks at each stage
///
/// The integrator itself is a fix (e.g. FixNVE), not built into the engine.
pub struct CPUDynamics {
    potential_set: PotentialSet,
    fixes: Vec<Box<dyn Fix>>,
    dumps: Vec<Box<dyn Dump>>,
    dt: F,
    /// Bound potentials (set during init)
    potentials: Option<Potentials>,
    /// Cumulative wall time across all run() calls
    total_steps: usize,
    total_wall_secs: f64,
}

impl CPUDynamics {
    pub(crate) fn new(
        potential_set: PotentialSet,
        fixes: Vec<Box<dyn Fix>>,
        dumps: Vec<Box<dyn Dump>>,
        dt: F,
    ) -> Self {
        CPUDynamics {
            potential_set,
            fixes,
            dumps,
            dt,
            potentials: None,
            total_steps: 0,
            total_wall_secs: 0.0,
        }
    }

    /// Compute forces and store in state.f. Also sets state.pe.
    fn compute_forces(&mut self, state: &mut MDState) -> Result<(), MDError> {
        let potentials = self
            .potentials
            .as_ref()
            .ok_or_else(|| MDError::ConfigError("potentials not bound (call init first)".into()))?;

        // Compute energy
        state.pe = potentials.energy(&state.x);

        // Compute forces directly (already = -gradient)
        let forces = potentials.forces(&state.x);
        state.f.copy_from_slice(&forces);

        Ok(())
    }

    /// Dispatch dumps: check frequency and call write() with immutable state.
    fn dispatch_dumps(&mut self, state: &MDState) -> Result<(), MDError> {
        for dump in &mut self.dumps {
            if state.step.is_multiple_of(dump.every()) {
                dump.write(state)?;
            }
        }
        Ok(())
    }

    /// Dispatch a stage callback to all fixes that participate in it.
    fn dispatch<Cb>(
        &mut self,
        stage: StageMask,
        state: &mut MDState,
        callback: Cb,
    ) -> Result<(), MDError>
    where
        Cb: Fn(&mut Box<dyn Fix>, &mut MDState) -> Result<(), MDError>,
    {
        for fix in &mut self.fixes {
            if fix.stages().contains(stage) {
                callback(fix, state)?;
            }
        }
        Ok(())
    }
}

impl DynamicsEngine for CPUDynamics {
    /// Initialize simulation state from a Frame.
    ///
    /// Binds the force field to the frame topology and creates an MDState.
    /// Call `run` afterwards for stepping.
    fn init(&mut self, frame: &Frame) -> Result<MDState, MDError> {
        // Bind topology → pre-bound Potentials
        let potentials = self
            .potential_set
            .bind(frame)
            .map_err(MDError::ForcefieldError)?;
        self.potentials = Some(potentials);

        let mut state = MDState::from_frame(frame, self.dt)?;

        // Compute initial forces
        self.compute_forces(&mut state)?;

        // Call setup on all fixes
        for fix in &mut self.fixes {
            fix.setup_with_frame(frame, &mut state)?;
            fix.setup(&mut state)?;
        }

        // Call setup on all dumps (read-only access to state)
        for dump in &mut self.dumps {
            dump.setup(frame, &state)?;
        }

        // ── system info ──
        let fix_names: Vec<&str> = self.fixes.iter().map(|f| f.name()).collect();
        log::info!(
            "Dynamics: {} atoms, dt={:.4}, ndof={}, fixes=[{}], dumps={}",
            state.n_atoms,
            self.dt,
            state.n_dof,
            fix_names.join(", "),
            self.dumps.len(),
        );
        log::info!(
            "  Initial: PE={:.6}, KE={:.6}, T={:.4}",
            state.pe,
            state.ke,
            state.temperature(),
        );

        self.total_steps = 0;
        self.total_wall_secs = 0.0;

        Ok(state)
    }

    /// Run N steps, consuming and returning the MDState (functional style).
    fn run(&mut self, n_steps: usize, mut state: MDState) -> Result<MDState, MDError> {
        let step_start = state.step;
        log::info!(
            "Run: {} steps (step {} -> {})",
            n_steps,
            step_start,
            step_start + n_steps,
        );

        let t0 = Instant::now();

        for _ in 0..n_steps {
            state.step += 1;

            // 1. initial_integrate
            self.dispatch(StageMask::INITIAL_INTEGRATE, &mut state, |f, s| {
                f.initial_integrate(s)
            })?;

            // 2. post_integrate
            self.dispatch(StageMask::POST_INTEGRATE, &mut state, |f, s| {
                f.post_integrate(s)
            })?;

            // 3. pre_force
            self.dispatch(StageMask::PRE_FORCE, &mut state, |f, s| f.pre_force(s))?;

            // 4. force computation (engine responsibility)
            self.compute_forces(&mut state)?;

            // 5. post_force
            self.dispatch(StageMask::POST_FORCE, &mut state, |f, s| f.post_force(s))?;

            // 6. final_integrate
            self.dispatch(StageMask::FINAL_INTEGRATE, &mut state, |f, s| {
                f.final_integrate(s)
            })?;

            // 7. end_of_step
            self.dispatch(StageMask::END_OF_STEP, &mut state, |f, s| f.end_of_step(s))?;

            // 8. dispatch dumps (read-only, after all fixes)
            self.dispatch_dumps(&state)?;
        }

        let elapsed = t0.elapsed().as_secs_f64();
        self.total_steps += n_steps;
        self.total_wall_secs += elapsed;

        let steps_per_sec = if elapsed > 0.0 {
            n_steps as f64 / elapsed
        } else {
            f64::INFINITY
        };
        log::info!(
            "  Done: {:.3}s ({:.0} steps/s) | PE={:.6}, KE={:.6}, T={:.4}",
            elapsed,
            steps_per_sec,
            state.pe,
            state.ke,
            state.temperature(),
        );

        Ok(state)
    }

    /// Cleanup all fixes and dumps. Call after the last `run` segment.
    fn finish(&mut self) -> Result<(), MDError> {
        for dump in &mut self.dumps {
            dump.cleanup()?;
        }

        let avg_steps_per_sec = if self.total_wall_secs > 0.0 {
            self.total_steps as f64 / self.total_wall_secs
        } else {
            f64::INFINITY
        };
        log::info!(
            "Total: {} steps in {:.3}s ({:.0} steps/s)",
            self.total_steps,
            self.total_wall_secs,
            avg_steps_per_sec,
        );
        Ok(())
    }
}
