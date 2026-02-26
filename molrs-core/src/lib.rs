//! # molrs
//!
//! A Rust library providing core molecular modeling functionality.
//!
//! ## Features
//!
//! - Element data and lookup by atomic number or symbol
//! - Core data structures, geometry, IO, and neighbor-list algorithms
//! - Type-safe molecular representations
//!
//! ## Examples
//!
//! ### Element lookup
//!
//! ```
//! use molrs::core::Element;
//!
//! // Look up elements by atomic number
//! let hydrogen = Element::by_number(1).unwrap();
//! assert_eq!(hydrogen.symbol(), "H");
//!
//! // Or by symbol (case-insensitive)
//! let h = Element::by_symbol("h").unwrap();
//! assert_eq!(h.name(), "Hydrogen");
//! ```
//!
//! ### Packing (moved to `molrs-pack`)
//!
//! ```ignore
//! use molrs_pack::Molpack;
//!
//! let mut packer = Molpack::new(None).with_precision(1e-3);
//! let _ = &mut packer;
//! ```

#![allow(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod core;
pub mod error;
pub mod io;
pub mod math;
pub mod neighbors;

// CUDA FFI bindings (only with "cuda" feature)
#[cfg(feature = "cuda")]
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// CUDA infrastructure (only with "cuda" feature)
#[cfg(feature = "cuda")]
pub mod cuda;

// Public re-exports for common types
pub use core::block::Block;
pub use core::element::Element;
pub use core::frame::Frame;
pub use core::gasteiger::{GasteigerCharges, compute_gasteiger_charges};
pub use core::hydrogens::{add_hydrogens, implicit_h_count};
pub use core::molgraph::{Atom, AtomId, Bead, Bond, BondId, MolGraph, PropValue};
pub use core::rings::{RingInfo, find_rings};
pub use core::stereo::{
    BondStereo, TetrahedralStereo, assign_bond_stereo_from_3d, assign_stereo_from_3d,
    chiral_volume, find_chiral_centers,
};
pub use core::topology::Topology;
pub use error::MolRsError;

// Re-export IO functions for convenience
pub use io::lammps_data::read_lammps_data;
pub use io::pdb::{read_pdb_frame, write_pdb_frame};
pub use io::xyz::{read_xyz_frame, read_xyz_traj, write_xyz_frame};

// Zarr I/O re-exports (only with "zarr" feature)
#[cfg(feature = "zarr")]
pub use io::zarr::{StoreData, ZarrFrame, ZarrStoreReader, ZarrStoreWriter};
