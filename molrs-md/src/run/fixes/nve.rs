use crate::error::MDError;
use crate::run::fix::{Fix, GpuTier};
use crate::run::stage::StageMask;
use crate::run::state::MDState;

/// Velocity Verlet integrator (NVE ensemble).
///
/// Equivalent to LAMMPS `fix nve`. Splits the Verlet step into:
/// - `initial_integrate`: x += v*dt + 0.5*f/m*dt^2; v += 0.5*f/m*dt
/// - `final_integrate`:   v += 0.5*f/m*dt  (using updated forces)
pub struct FixNVE;

impl FixNVE {
    pub fn new() -> Self {
        FixNVE
    }
}

impl Default for FixNVE {
    fn default() -> Self {
        Self::new()
    }
}

impl Fix for FixNVE {
    fn name(&self) -> &str {
        "nve"
    }

    fn stages(&self) -> StageMask {
        StageMask::INITIAL_INTEGRATE | StageMask::FINAL_INTEGRATE
    }

    fn gpu_tier(&self) -> GpuTier {
        GpuTier::Kernel
    }

    fn initial_integrate(&mut self, s: &mut MDState) -> Result<(), MDError> {
        let dt = s.dt;
        let dt_half = 0.5 * dt;
        for i in 0..s.n_atoms {
            let inv_m = s.inv_mass[i];
            for d in 0..3 {
                let idx = 3 * i + d;
                // v(t+dt/2) = v(t) + 0.5*dt*f(t)/m
                s.v[idx] += dt_half * s.f[idx] * inv_m;
                // x(t+dt) = x(t) + dt*v(t+dt/2)
                s.x[idx] += dt * s.v[idx];
            }
        }
        Ok(())
    }

    fn final_integrate(&mut self, s: &mut MDState) -> Result<(), MDError> {
        let dt_half = 0.5 * s.dt;
        for i in 0..s.n_atoms {
            let inv_m = s.inv_mass[i];
            for d in 0..3 {
                let idx = 3 * i + d;
                // v(t+dt) = v(t+dt/2) + 0.5*dt*f(t+dt)/m
                s.v[idx] += dt_half * s.f[idx] * inv_m;
            }
        }
        s.compute_ke();
        Ok(())
    }
}
