// CUDA backend — only compiled with "cuda" feature.
// Infrastructure (buffer, device, potential, kernels, neighborlist, registry) lives in molrs::cuda.
// This module contains the MD-specific minimizer, dynamics engine, and Backend impls.

pub mod dynamics;
pub mod minimizer;

use molrs::core::potential::PotentialSet;
use molrs::core::types::F;
use molrs::cuda::device::CUDADevice;

use crate::backend::{Backend, CUDA, DynamicsBackend, MinimizerConfig};
use crate::error::MDError;
use crate::run::dump::Dump;
use crate::run::fix::Fix;

use self::dynamics::CUDADynamics;
use self::minimizer::CUDAMinimizer;

impl Backend for CUDA {
    type Device = CUDADevice;
    type Minimizer = CUDAMinimizer;

    fn compile(
        _device: CUDADevice,
        _potential_set: PotentialSet,
        _config: MinimizerConfig,
    ) -> Result<CUDAMinimizer, MDError> {
        // TODO: implement CUDA backend with PotentialSet
        Err(MDError::ConfigError(
            "CUDA backend not yet updated for PotentialSet API".into(),
        ))
    }
}

impl DynamicsBackend for CUDA {
    type Device = i32;
    type Dynamics = CUDADynamics;

    fn compile_dynamics(
        device: i32,
        potential_set: PotentialSet,
        fixes: Vec<Box<dyn Fix>>,
        dumps: Vec<Box<dyn Dump>>,
        dt: F,
    ) -> Result<CUDADynamics, MDError> {
        Ok(CUDADynamics::new(device, potential_set, fixes, dumps, dt))
    }
}
