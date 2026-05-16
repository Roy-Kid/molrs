//! Analysis compute modules for molrs molecular simulation.
//!
//! Trajectory analysis (RDF, MSD, clustering, gyration/inertia, PCA,
//! k-means) built around a single unified [`Compute`] trait.
//! Every analysis is stateless — orchestrate from the caller.
//!
//! # Stateless `Compute` — orchestrate from the caller
//!
//! Each [`Compute`] impl is a pure function: `&self` is an immutable
//! parameter bag. Two `compute` calls with identical `frames` + `args`
//! always produce identical output. There is no hidden mutable state,
//! no DAG, no store — just the trait and per-analysis modules.
//!
//! For DAG orchestration (topological order, diamond reuse, external
//! input validation), use `molpy.compute.Workflow` on the Python side.
//! It composes `Compute` nodes via Python's stdlib `graphlib` and
//! calls each Rust kernel directly.
//!
//! # Unified trait
//!
//! Every analysis implements [`Compute`]. A single frame, a trajectory, and a
//! whole dataset are the same kind of input — a slice `&[&F]` — and each
//! impl decides how to interpret it:
//!
//! - **Whole-sequence accumulators** (RDF) iterate every frame.
//! - **Time-series analyses** (MSD) use `frames[0]` as a reference.
//! - **Per-frame analyses** (Cluster, COM) return one result per frame.
//! - **Matrix consumers** (PCA, k-means) take upstream per-frame outputs as
//!   rows (via [`DescriptorRow`]).
//!
//! # Results
//!
//! Every Compute output implements [`ComputeResult`]. Accumulating outputs
//! (RDF) override [`finalize`](ComputeResult::finalize) to normalize; other
//! outputs use the default no-op.
//!
//! # Available analyses
//!
//! | Module | Args | Output |
//! |--------|------|--------|
//! | [`rdf`] | `&Vec<NeighborList>` | [`RDFResult`] |
//! | [`msd`] | `()` | [`MSDTimeSeries`] |
//! | [`cluster`] | `&Vec<NeighborList>` | `Vec<`[`ClusterResult`]`>` |
//! | [`cluster_centers`] | `&Vec<ClusterResult>` | `Vec<`[`ClusterCentersResult`]`>` |
//! | [`center_of_mass`] | `&Vec<ClusterResult>` | `Vec<`[`COMResult`]`>` |
//! | [`gyration_tensor`] | `(&Vec<ClusterResult>, &Vec<ClusterCentersResult>)` | `Vec<`[`GyrationTensorResult`]`>` |
//! | [`inertia_tensor`] | `(&Vec<ClusterResult>, &Vec<COMResult>)` | `Vec<`[`InertiaTensorResult`]`>` |
//! | [`radius_of_gyration`] | `(&Vec<ClusterResult>, &Vec<COMResult>)` | `Vec<`[`RgResult`]`>` |
//! | [`pca`] | `&Vec<T: DescriptorRow>` | [`PcaResult`] |
//! | [`kmeans`] | `&PcaResult` | [`KMeansResult`] |

pub mod center_of_mass;
pub mod cluster;
pub mod cluster_centers;
pub mod error;
pub mod gyration_tensor;
pub mod inertia_tensor;
pub mod kmeans;
pub mod msd;
pub mod order;
pub mod pca;
pub mod radius_of_gyration;
pub mod rdf;
pub mod result;
pub mod traits;
pub mod util;

// Re-exports
pub use center_of_mass::{COMResult, CenterOfMass};
pub use cluster::{Cluster, ClusterProperties, ClusterPropertiesResult, ClusterResult};
pub use cluster_centers::{ClusterCenters, ClusterCentersResult};
pub use error::ComputeError;
pub use gyration_tensor::{GyrationTensor, GyrationTensorResult};
pub use inertia_tensor::{InertiaTensor, InertiaTensorResult};
pub use kmeans::{KMeans, KMeansResult};
pub use msd::{MSD, MSDResult, MSDTimeSeries, MsdMode};
pub use order::{
    Hexatic, HexaticResult, Nematic, NematicResult, SolidLiquid, SolidLiquidResult, Steinhardt,
    SteinhardtResult,
};
pub use pca::{Pca2, PcaResult};
pub use radius_of_gyration::{RadiusOfGyration, RgResult};
pub use rdf::{RDF, RDFResult};
pub use result::{ComputeResult, DescriptorRow};
pub use traits::Compute;
