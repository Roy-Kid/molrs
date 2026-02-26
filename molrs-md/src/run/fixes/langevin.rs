use molrs::core::types::F;

use crate::error::MDError;
use crate::run::fix::{Fix, GpuTier};
use crate::run::stage::StageMask;
use crate::run::state::MDState;

const PI: F = std::f64::consts::PI as F;

/// Langevin thermostat — adds friction and random forces in post_force.
///
/// Equivalent to LAMMPS `fix langevin T_start T_stop damp seed`.
/// Uses the simple Langevin equation: f_i += -gamma*v_i + sqrt(2*gamma*kB*T/dt) * eta
/// where eta ~ N(0,1). Here kB=1 (reduced units).
pub struct FixLangevin {
    target_temp: F,
    damp: F,
    rng_state: u64,
}

impl FixLangevin {
    /// Create a Langevin thermostat.
    ///
    /// - `target_temp`: target temperature (reduced units, kB=1)
    /// - `damp`: damping time (larger = weaker coupling)
    /// - `seed`: random seed
    pub fn new(target_temp: F, damp: F, seed: u64) -> Self {
        FixLangevin {
            target_temp,
            damp,
            rng_state: seed,
        }
    }

    /// Simple xorshift64 RNG for reproducibility without external deps.
    fn rand_f(&mut self) -> F {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as F) / (u64::MAX as F)
    }

    /// Box-Muller transform: two uniform -> one normal.
    fn rand_normal(&mut self) -> F {
        let u1 = self.rand_f().max(1e-30); // avoid log(0)
        let u2 = self.rand_f();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

impl Fix for FixLangevin {
    fn name(&self) -> &str {
        "langevin"
    }

    fn stages(&self) -> StageMask {
        StageMask::POST_FORCE
    }

    fn gpu_tier(&self) -> GpuTier {
        GpuTier::Kernel
    }

    fn post_force(&mut self, s: &mut MDState) -> Result<(), MDError> {
        let dt = s.dt;
        let gamma = 1.0 / self.damp; // friction coefficient
        let t = self.target_temp;

        for i in 0..s.n_atoms {
            let m = s.mass[i];
            // Noise amplitude: sqrt(2 * gamma * m * kB * T / dt)
            let noise_amp = (2.0 * gamma * m * t / dt).sqrt();
            let drag_coeff = gamma * m;

            for d in 0..3 {
                let idx = 3 * i + d;
                let eta = self.rand_normal();
                s.f[idx] += -drag_coeff * s.v[idx] + noise_amp * eta;
            }
        }
        Ok(())
    }
}
