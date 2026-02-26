use molrs::core::types::F;

use crate::error::MDError;
use crate::run::fix::{Fix, GpuTier};
use crate::run::stage::StageMask;
use crate::run::state::MDState;

/// Nosé-Hoover thermostat (NVT ensemble).
///
/// Equivalent to LAMMPS `fix nvt`. Implements velocity Verlet with a
/// Nosé-Hoover chain (single thermostat variable xi).
///
/// The equations of motion are:
///   dxi/dt = (KE_current - KE_target) / Q
///   dv/dt  = f/m - xi*v
///
/// where Q = n_dof * kB * T * tau^2 is the thermostat mass.
pub struct FixNVT {
    target_temp: F,
    tau: F,
    xi: F, // thermostat variable
    q: F,  // thermostat mass (set in setup)
}

impl FixNVT {
    /// Create a Nosé-Hoover thermostat.
    ///
    /// - `target_temp`: target temperature (kB=1)
    /// - `tau`: coupling time constant
    pub fn new(target_temp: F, tau: F) -> Self {
        FixNVT {
            target_temp,
            tau,
            xi: 0.0,
            q: 0.0,
        }
    }
}

impl Fix for FixNVT {
    fn name(&self) -> &str {
        "nvt"
    }

    fn stages(&self) -> StageMask {
        StageMask::INITIAL_INTEGRATE | StageMask::FINAL_INTEGRATE
    }

    fn gpu_tier(&self) -> GpuTier {
        GpuTier::Kernel
    }

    fn setup(&mut self, s: &mut MDState) -> Result<(), MDError> {
        // Q = n_dof * kB * T * tau^2, kB=1
        self.q = s.n_dof as F * self.target_temp * self.tau * self.tau;
        if self.q <= 0.0 {
            return Err(MDError::ConfigError(
                "NVT thermostat mass Q must be positive".into(),
            ));
        }
        Ok(())
    }

    fn initial_integrate(&mut self, s: &mut MDState) -> Result<(), MDError> {
        let dt = s.dt;
        let dt_half = 0.5 * dt;
        let ke_target = 0.5 * s.n_dof as F * self.target_temp;

        // Update xi at half step
        s.compute_ke();
        self.xi += dt_half * (s.ke - ke_target) / self.q;

        // Update velocities and positions
        for i in 0..s.n_atoms {
            let inv_m = s.inv_mass[i];
            for d in 0..3 {
                let idx = 3 * i + d;
                s.v[idx] += dt_half * (s.f[idx] * inv_m - self.xi * s.v[idx]);
                s.x[idx] += dt * s.v[idx];
            }
        }
        Ok(())
    }

    fn final_integrate(&mut self, s: &mut MDState) -> Result<(), MDError> {
        let dt_half = 0.5 * s.dt;
        let ke_target = 0.5 * s.n_dof as F * self.target_temp;

        // Update velocities
        for i in 0..s.n_atoms {
            let inv_m = s.inv_mass[i];
            for d in 0..3 {
                let idx = 3 * i + d;
                // Implicit: v += dt/2 * (f/m - xi*v) => v*(1+dt/2*xi) = v_old + dt/2*f/m
                let denom = 1.0 + dt_half * self.xi;
                s.v[idx] = (s.v[idx] + dt_half * s.f[idx] * inv_m) / denom;
            }
        }

        // Update xi at half step
        s.compute_ke();
        self.xi += dt_half * (s.ke - ke_target) / self.q;

        Ok(())
    }
}
