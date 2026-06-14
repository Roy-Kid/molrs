//! MMFF94 force field support.
//!
//! In addition to the parameter [`tables`] (ported from RDKit
//! `Code/ForceField/MMFF/Params.cpp`), this module ports the MMFF94 atom
//! typing, aromaticity perception, and partial-charge model from RDKit
//! `Code/GraphMol/ForceFieldHelpers/MMFF/AtomTyper.cpp` and
//! `Code/GraphMol/Aromaticity.cpp` (BSD-3, Paolo Tosco / RDKit
//! contributors).
//!
//! Pipeline (mirrors `MMFFMolProperties`'s constructor):
//! `aromaticity::set_mmff_aromaticity` → `atomtype::assign_atom_types`
//! → `charges::compute_partial_charges`.
//!
//! ```no_run
//! use molrs::ff::mmff::{MmffMolProperties, MmffVariant};
//! # fn run(mol: &molrs::Atomistic) -> Result<(), molrs::error::MolRsError> {
//! let props = MmffMolProperties::compute(mol, MmffVariant::Mmff94)?;
//! let t = props.atom_type(0);
//! let q = props.partial_charge(0);
//! # let _ = (t, q); Ok(())
//! # }
//! ```

pub(crate) mod aromaticity;
pub(crate) mod atomtype;
pub(crate) mod charges;
pub mod energy;
mod hybrid;
pub mod tables;
pub(crate) mod topo;

pub use energy::{MmffEnergyBreakdown, MmffForceField};

use molrs::Atomistic;
use molrs::error::MolRsError;

use topo::Topo;

/// MMFF parameterization variant.
///
/// `Mmff94s` is the "static" variant (Halgren 1999); for atom typing and
/// charges it differs from `Mmff94` only in the planarity treatment of a
/// few delocalized nitrogens, which is governed by the Oop/Tor tables not
/// yet present in [`tables`]. The typing / charge paths here are identical
/// for both variants; the field is retained so callers can branch and so
/// the energy layer can pick the right Oop/Tor set later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmffVariant {
    Mmff94,
    Mmff94s,
}

/// Per-atom MMFF properties (numeric atom types + partial charges) for a
/// molecule, computed once.
#[derive(Debug, Clone)]
pub struct MmffMolProperties {
    variant: MmffVariant,
    atom_types: Vec<u8>,
    partial_charges: Vec<f64>,
    valid: bool,
}

impl MmffMolProperties {
    /// Run the full MMFF setup (aromaticity → typing → charges).
    ///
    /// Returns `Err` if any atom could not be assigned an MMFF type
    /// (e.g. an unsupported element / transition metal with no MMFF type).
    pub fn compute(mol: &Atomistic, variant: MmffVariant) -> Result<Self, MolRsError> {
        let base = Topo::build(mol).map_err(|sym| {
            MolRsError::validation(format!("MMFF: unsupported element symbol '{sym}'"))
        })?;
        let topo = aromaticity::set_mmff_aromaticity(&base);
        let atom_types = atomtype::assign_atom_types(&topo);

        // Locate the first untyped atom for a useful error message.
        if let Some(bad) = atom_types.iter().position(|&t| t == 0) {
            let z = topo.atno[bad];
            return Err(MolRsError::validation(format!(
                "MMFF: could not assign an atom type to atom index {bad} (Z={z})"
            )));
        }

        let partial_charges = charges::compute_partial_charges(&topo, &atom_types);

        Ok(Self {
            variant,
            atom_types,
            partial_charges,
            valid: true,
        })
    }

    /// The variant this was computed for.
    pub fn variant(&self) -> MmffVariant {
        self.variant
    }

    /// MMFF numeric atom type (1..=99) for atom index `i`
    /// (the index is the molecule's atom iteration order).
    pub fn atom_type(&self, i: usize) -> u8 {
        self.atom_types[i]
    }

    /// MMFF partial charge for atom index `i`.
    pub fn partial_charge(&self, i: usize) -> f64 {
        self.partial_charges[i]
    }

    /// Whether every atom received a valid MMFF type.
    pub fn is_setup_complete(&self) -> bool {
        self.valid
    }

    /// Number of atoms.
    pub fn len(&self) -> usize {
        self.atom_types.len()
    }

    /// Whether the molecule was empty.
    pub fn is_empty(&self) -> bool {
        self.atom_types.is_empty()
    }
}
