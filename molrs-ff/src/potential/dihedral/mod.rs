//! Dihedral potential kernels.

pub mod mmff;
pub mod opls;

pub use mmff::{MMFFTorsion, mmff_torsion_ctor};
pub use opls::{DihedralOPLS, dihedral_opls_ctor};
