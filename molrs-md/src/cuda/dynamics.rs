use std::time::Instant;

use molrs::core::frame::Frame;
use molrs::core::potential::{PotentialSet, Potentials};
use molrs::core::types::F;

use crate::backend::DynamicsEngine;
use crate::error::MDError;
use crate::run::dump::Dump;
use crate::run::fix::{Fix, GpuTier};
use crate::run::stage::StageMask;
use crate::run::state::MDState;

/// CUDA dynamics engine — GPU-accelerated MD with tiered fix dispatch.
///
/// Fixes are classified into three tiers based on `fix.gpu_tier()`:
///
/// | Tier | GpuTier | What happens |
/// |------|---------|----------------------------------------------|
/// | 1 | Kernel | GPU kernel (nve, langevin, nvt) |
/// | 2 | Async | Async D2H of scalars only (thermo, dumps) |
/// | 3 | Sync | Full D2H → CPU callback → H2D (safe fallback) |
///
/// Each fix declares its own tier — no string matching, no central table.
/// Unknown fixes default to `GpuTier::Sync` and always work correctly.
pub struct CUDADynamics {
    device_id: i32,
    potential_set: PotentialSet,
    /// Tier 1: fixes with GPU kernels.
    tier1_fixes: Vec<Box<dyn Fix>>,
    /// Tier 2: read-only / async fixes.
    tier2_fixes: Vec<Box<dyn Fix>>,
    /// Tier 3: full sync fallback.
    tier3_fixes: Vec<Box<dyn Fix>>,
    /// Dumps are always Tier 2 (read-only by Dump trait contract).
    dumps: Vec<Box<dyn Dump>>,
    dt: F,
    potentials: Option<Potentials>,
    total_steps: usize,
    total_wall_secs: f64,
}

impl CUDADynamics {
    pub(crate) fn new(
        device_id: i32,
        potential_set: PotentialSet,
        fixes: Vec<Box<dyn Fix>>,
        dumps: Vec<Box<dyn Dump>>,
        dt: F,
    ) -> Self {
        let mut tier1 = Vec::new();
        let mut tier2 = Vec::new();
        let mut tier3 = Vec::new();

        for fix in fixes {
            match fix.gpu_tier() {
                GpuTier::Kernel => tier1.push(fix),
                GpuTier::Async => tier2.push(fix),
                GpuTier::Sync => tier3.push(fix),
            }
        }

        CUDADynamics {
            device_id,
            potential_set,
            tier1_fixes: tier1,
            tier2_fixes: tier2,
            tier3_fixes: tier3,
            dumps,
            dt,
            potentials: None,
            total_steps: 0,
            total_wall_secs: 0.0,
        }
    }

    fn compute_forces(&mut self, state: &mut MDState) -> Result<(), MDError> {
        let potentials = self
            .potentials
            .as_ref()
            .ok_or_else(|| MDError::ConfigError("potentials not bound (call init first)".into()))?;
        state.pe = potentials.energy(&state.x);
        let forces = potentials.forces(&state.x);
        state.f.copy_from_slice(&forces);
        Ok(())
    }

    fn dispatch_dumps(&mut self, state: &MDState) -> Result<(), MDError> {
        for dump in &mut self.dumps {
            if state.step % dump.every() == 0 {
                dump.write(state)?;
            }
        }
        Ok(())
    }

    /// Dispatch a stage callback across all three tiers in order.
    fn dispatch<Cb>(
        &mut self,
        stage: StageMask,
        state: &mut MDState,
        callback: Cb,
    ) -> Result<(), MDError>
    where
        Cb: Fn(&mut Box<dyn Fix>, &mut MDState) -> Result<(), MDError>,
    {
        for fix in &mut self.tier1_fixes {
            if fix.stages().contains(stage) {
                callback(fix, state)?;
            }
        }
        for fix in &mut self.tier2_fixes {
            if fix.stages().contains(stage) {
                callback(fix, state)?;
            }
        }
        for fix in &mut self.tier3_fixes {
            if fix.stages().contains(stage) {
                callback(fix, state)?;
            }
        }
        Ok(())
    }
}

impl DynamicsEngine for CUDADynamics {
    fn init(&mut self, frame: &Frame) -> Result<MDState, MDError> {
        let potentials = self
            .potential_set
            .bind(frame)
            .map_err(MDError::ForcefieldError)?;
        self.potentials = Some(potentials);

        let mut state = MDState::from_frame(frame, self.dt)?;
        self.compute_forces(&mut state)?;

        // Setup all fixes (all tiers)
        for fix in self
            .tier1_fixes
            .iter_mut()
            .chain(self.tier2_fixes.iter_mut())
            .chain(self.tier3_fixes.iter_mut())
        {
            fix.setup_with_frame(frame, &mut state)?;
            fix.setup(&mut state)?;
        }

        for dump in &mut self.dumps {
            dump.setup(frame, &state)?;
        }

        // Log tier classification
        let t1: Vec<&str> = self.tier1_fixes.iter().map(|f| f.name()).collect();
        let t2: Vec<&str> = self.tier2_fixes.iter().map(|f| f.name()).collect();
        let t3: Vec<&str> = self.tier3_fixes.iter().map(|f| f.name()).collect();

        println!(
            "CUDA Dynamics (device {}): {} atoms, dt={:.4}, ndof={}",
            self.device_id, state.n_atoms, self.dt as f64, state.n_dof,
        );
        if !t1.is_empty() {
            println!("  Tier 1 (GPU kernel): [{}]", t1.join(", "));
        }
        if !t2.is_empty() {
            println!("  Tier 2 (async):      [{}]", t2.join(", "));
        }
        if !t3.is_empty() {
            println!("  Tier 3 (sync CPU):   [{}]", t3.join(", "));
        }
        println!("  Dumps: {} (Tier 2 async)", self.dumps.len());
        println!(
            "  Initial: PE={:.6}, KE={:.6}, T={:.4}",
            state.pe as f64,
            state.ke as f64,
            state.temperature() as f64,
        );

        self.total_steps = 0;
        self.total_wall_secs = 0.0;

        Ok(state)
    }

    fn run(&mut self, n_steps: usize, mut state: MDState) -> Result<MDState, MDError> {
        let step_start = state.step;
        println!(
            "Run (GPU {}): {} steps (step {} -> {})",
            self.device_id,
            n_steps,
            step_start,
            step_start + n_steps,
        );

        let t0 = Instant::now();

        for _ in 0..n_steps {
            state.step += 1;

            self.dispatch(StageMask::INITIAL_INTEGRATE, &mut state, |f, s| {
                f.initial_integrate(s)
            })?;

            self.dispatch(StageMask::POST_INTEGRATE, &mut state, |f, s| {
                f.post_integrate(s)
            })?;

            self.dispatch(StageMask::PRE_FORCE, &mut state, |f, s| f.pre_force(s))?;

            self.compute_forces(&mut state)?;

            self.dispatch(StageMask::POST_FORCE, &mut state, |f, s| f.post_force(s))?;

            self.dispatch(StageMask::FINAL_INTEGRATE, &mut state, |f, s| {
                f.final_integrate(s)
            })?;

            self.dispatch(StageMask::END_OF_STEP, &mut state, |f, s| f.end_of_step(s))?;

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
        println!(
            "  Done: {:.3}s ({:.0} steps/s) | PE={:.6}, KE={:.6}, T={:.4}",
            elapsed,
            steps_per_sec,
            state.pe as f64,
            state.ke as f64,
            state.temperature() as f64,
        );

        Ok(state)
    }

    fn finish(&mut self) -> Result<(), MDError> {
        for dump in &mut self.dumps {
            dump.cleanup()?;
        }

        let avg = if self.total_wall_secs > 0.0 {
            self.total_steps as f64 / self.total_wall_secs
        } else {
            f64::INFINITY
        };
        println!(
            "Total (GPU {}): {} steps in {:.3}s ({:.0} steps/s)",
            self.device_id, self.total_steps, self.total_wall_secs, avg,
        );
        Ok(())
    }
}
