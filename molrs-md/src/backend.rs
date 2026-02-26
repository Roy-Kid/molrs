use molrs::core::frame::Frame;
use molrs::core::potential::PotentialSet;
use molrs::core::types::F;

use crate::error::MDError;
use crate::run::state::MDState;

/// Configuration for the minimizer.
///
/// Parameters follow LAMMPS conventions:
///   minimize etol ftol maxiter maxeval
///   min_modify dmax <value>
#[derive(Debug, Clone)]
pub struct MinimizerConfig {
    /// Maximum number of minimization iterations.
    pub max_iter: usize,
    /// Maximum number of energy/force evaluations.
    pub max_eval: usize,
    /// Energy tolerance (relative). Convergence when
    /// |dE| < etol * 0.5 * (|E_new| + |E_old| + eps).
    /// Set to 0.0 to disable.
    pub energy_tol: F,
    /// Force tolerance. Convergence when
    /// ||grad||_2 / sqrt(3N) < ftol  (i.e. RMS force per component).
    pub force_tol: F,
    /// Maximum displacement of any single atom in one step (LAMMPS min_modify dmax).
    pub dmax: F,
    /// Print progress every N steps (0 = auto: max_iter/10 clamped to [10, 500]).
    pub log_every: usize,
}

impl Default for MinimizerConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            max_eval: usize::MAX,
            energy_tol: 0.0,
            force_tol: 1e-6,
            dmax: 0.1,
            log_every: 0,
        }
    }
}

/// Backend trait — implemented by CPU and CUDA.
pub trait Backend {
    type Device;
    type Minimizer;

    fn compile(
        device: Self::Device,
        potential_set: PotentialSet,
        config: MinimizerConfig,
    ) -> Result<Self::Minimizer, MDError>;
}

/// Common interface for all dynamics engines (CPU, CUDA, ...).
///
/// Guarantees that every backend exposes the same init/run/finish API.
/// Also object-safe, so you can use `Box<dyn DynamicsEngine>` for
/// runtime backend selection.
pub trait DynamicsEngine {
    fn init(&mut self, frame: &Frame) -> Result<MDState, MDError>;
    fn run(&mut self, n_steps: usize, state: MDState) -> Result<MDState, MDError>;
    fn finish(&mut self) -> Result<(), MDError>;
}

/// Backend trait for dynamics — implemented by CPU and CUDA.
pub trait DynamicsBackend {
    type Device;
    type Dynamics: DynamicsEngine;

    fn compile_dynamics(
        device: Self::Device,
        potential_set: PotentialSet,
        fixes: Vec<Box<dyn crate::run::fix::Fix>>,
        dumps: Vec<Box<dyn crate::run::dump::Dump>>,
        dt: F,
    ) -> Result<Self::Dynamics, MDError>;
}

/// CPU backend marker.
pub struct CPU;

/// CUDA backend marker.
#[cfg(feature = "cuda")]
pub struct CUDA;
