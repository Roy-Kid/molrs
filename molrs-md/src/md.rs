use molrs::core::forcefield::ForceField;
use molrs::core::types::F;

use crate::backend::{Backend, MinimizerConfig};
use crate::error::MDError;
use crate::run::builder::DynamicsBuilder;

/// Result of a minimization run.
#[derive(Debug, Clone)]
pub struct MinState {
    pub energy: F,
    pub n_steps: usize,
    pub n_evals: usize,
    pub converged: bool,
}

/// Entry point for building MD simulations.
pub struct MD;

impl MD {
    pub fn minimizer() -> MinimizerBuilder {
        MinimizerBuilder::new()
    }

    pub fn dynamics() -> DynamicsBuilder {
        DynamicsBuilder::new()
    }
}

/// Builder for constructing a minimizer.
///
/// API mirrors LAMMPS:
/// ```text
/// minimize  etol ftol maxiter maxeval
/// min_modify dmax <value>
/// ```
pub struct MinimizerBuilder {
    potential_set: Option<molrs::core::potential::PotentialSet>,
    config: MinimizerConfig,
}

impl MinimizerBuilder {
    pub fn new() -> Self {
        Self {
            potential_set: None,
            config: MinimizerConfig::default(),
        }
    }

    /// Set the force field. Internally compiles it into a PotentialSet.
    pub fn forcefield(mut self, ff: &ForceField) -> Self {
        self.potential_set = Some(ff.to_potentials());
        self
    }

    /// Energy tolerance (relative).  Set 0.0 to disable (default).
    pub fn energy_tol(mut self, etol: F) -> Self {
        self.config.energy_tol = etol;
        self
    }

    /// Force tolerance — RMS force per component (default 1e-6).
    pub fn force_tol(mut self, ftol: F) -> Self {
        self.config.force_tol = ftol;
        self
    }

    /// Maximum minimization iterations (default 1000).
    pub fn max_iter(mut self, n: usize) -> Self {
        self.config.max_iter = n;
        self
    }

    /// Maximum energy/force evaluations (default unlimited).
    pub fn max_eval(mut self, n: usize) -> Self {
        self.config.max_eval = n;
        self
    }

    /// Maximum displacement of any atom per step (default 0.1).
    pub fn dmax(mut self, d: F) -> Self {
        self.config.dmax = d;
        self
    }

    /// Print progress every N steps (default 0 = auto).
    pub fn log_every(mut self, n: usize) -> Self {
        self.config.log_every = n;
        self
    }

    pub fn compile<B: Backend>(self, device: B::Device) -> Result<B::Minimizer, MDError> {
        let potential_set = self
            .potential_set
            .ok_or_else(|| MDError::ConfigError("forcefield is required".into()))?;

        B::compile(device, potential_set, self.config)
    }
}

impl Default for MinimizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
