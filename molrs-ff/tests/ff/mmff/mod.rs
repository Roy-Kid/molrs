//! Integration tests for the `mmff` module tree (typing + charges),
//! validated per-atom against RDKit fixtures.

#[path = "typing.rs"]
mod typing;

#[path = "energy.rs"]
mod energy;
