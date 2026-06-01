//! MMFF94 force field support.
//!
//! Currently exposes the parameter tables ([`tables`]) ported faithfully from
//! RDKit. Atom typing, charge assignment, aromaticity perception, and energy
//! terms are intentionally **not** here — they are separate layers built on top
//! of these pure data tables.

pub mod tables;
