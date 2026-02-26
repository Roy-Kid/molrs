//! Molecular dynamics engine for molrs.
//!
//! Provides CPU-based minimization and dynamics with a LAMMPS-style fix/dump
//! plugin architecture. The default build is pure Rust with zero external
//! dependencies; CUDA acceleration is available behind the `cuda` feature.
//!
//! # Quick start
//!
//! ```ignore
//! use molrs_md::{CPU, DynamicsEngine, FixNVE, MD};
//!
//! let mut dynamics = MD::dynamics()
//!     .forcefield(&ff)
//!     .dt(0.001)
//!     .fix(FixNVE::new())
//!     .compile::<CPU>(())
//!     .unwrap();
//!
//! let state = dynamics.init(&frame).unwrap();
//! let state = dynamics.run(1000, state).unwrap();
//! dynamics.finish().unwrap();
//! ```

pub mod backend;
pub mod cpu;
pub mod error;
pub mod md;
pub mod run;

#[cfg(feature = "cuda")]
pub mod cuda;

// Public re-exports
pub use backend::{Backend, CPU, DynamicsBackend, DynamicsEngine, MinimizerConfig};
pub use cpu::CPUMinimizer;
pub use cpu::dynamics::CPUDynamics;
pub use error::MDError;
pub use md::{MD, MinState, MinimizerBuilder};
#[cfg(feature = "zarr")]
pub use run::DumpZarr;
pub use run::{
    Dump, DynamicsBuilder, Fix, FixLangevin, FixNVE, FixNVT, FixThermo, GpuTier, MDState, StageMask,
};

#[cfg(feature = "cuda")]
pub use backend::CUDA;
#[cfg(feature = "cuda")]
pub use cuda::dynamics::CUDADynamics;
#[cfg(feature = "cuda")]
pub use cuda::minimizer::CUDAMinimizer;
