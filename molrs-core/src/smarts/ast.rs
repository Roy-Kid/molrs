//! SMARTS query AST and primitive evaluation against a [`MolGraph`].
//!
//! Ported (semantics only) from RDKit's query primitives under BSD-3:
//! - `Code/GraphMol/QueryAtom.h`, `Code/GraphMol/QueryBond.h`
//! - `Code/Query/` (query trees)
//!
//! The AST mirrors RDKit's atom/bond query trees: leaf *primitives* (element,
//! charge, H-count, ...) combined by AND / OR / NOT nodes. Evaluation is a
//! pure read against a [`MolGraph`] plus a precomputed [`MolContext`] that
//! carries ring info and the project-convention aromaticity perception.

use std::collections::HashMap;

use crate::molgraph::{AtomId, MolGraph, PropValue};
use crate::rings::{RingInfo, find_rings};

/// Precomputed, read-only context shared by every primitive evaluation.
///
/// Holds ring info and the per-atom / per-bond aromatic perception. The
/// project convention (documented in `CLAUDE.md`) is that aromatic atoms are
/// those in a ring whose bonds carry order `1.5`; loaders that already know
/// the truth (e.g. transplanted from RDKit) may instead set an `is_aromatic`
/// atom prop, which takes precedence when present.
pub struct MolContext<'m> {
    pub mol: &'m MolGraph,
    pub rings: RingInfo,
    /// atom → is-aromatic (perceived once, up front).
    aromatic_atom: HashMap<AtomId, bool>,
    /// atom → total H count (explicit H neighbours; see module note).
    h_count: HashMap<AtomId, u32>,
    /// atom → explicit degree (number of bonded neighbours).
    degree: HashMap<AtomId, u32>,
}

impl<'m> MolContext<'m> {
    /// Build the context for `mol` (runs ring perception once).
    pub fn new(mol: &'m MolGraph) -> Self {
        let rings = find_rings(mol);
        let mut aromatic_atom = HashMap::new();
        let mut h_count = HashMap::new();
        let mut degree = HashMap::new();

        for (id, atom) in mol.atoms() {
            // Aromaticity: explicit `is_aromatic` prop wins; otherwise infer
            // from any incident aromatic (order ~= 1.5) bond.
            let explicit = match atom.get("is_aromatic") {
                Some(PropValue::Int(v)) => Some(*v != 0),
                Some(PropValue::F64(v)) => Some(*v != 0.0),
                _ => None,
            };
            let arom = explicit.unwrap_or_else(|| {
                mol.neighbor_bonds(id)
                    .any(|(_, order)| (order - 1.5).abs() < 1e-6)
            });
            aromatic_atom.insert(id, arom);

            // Total H count = number of neighbour atoms whose element is "H".
            let h = mol
                .neighbors(id)
                .filter(|&nb| {
                    mol.get_atom(nb)
                        .ok()
                        .and_then(|a| a.get_str("element"))
                        .map(element_is_hydrogen)
                        .unwrap_or(false)
                })
                .count() as u32;
            h_count.insert(id, h);

            degree.insert(id, mol.neighbors(id).count() as u32);
        }

        Self {
            mol,
            rings,
            aromatic_atom,
            h_count,
            degree,
        }
    }

    fn is_aromatic(&self, id: AtomId) -> bool {
        self.aromatic_atom.get(&id).copied().unwrap_or(false)
    }

    fn h_count(&self, id: AtomId) -> u32 {
        self.h_count.get(&id).copied().unwrap_or(0)
    }

    fn degree(&self, id: AtomId) -> u32 {
        self.degree.get(&id).copied().unwrap_or(0)
    }
}

fn element_is_hydrogen(sym: &str) -> bool {
    sym.eq_ignore_ascii_case("h")
}

/// Read an atom's element symbol, defaulting to `""` when absent.
fn atom_symbol(mol: &MolGraph, id: AtomId) -> &str {
    mol.get_atom(id)
        .ok()
        .and_then(|a| a.get_str("element"))
        .unwrap_or("")
}

/// Read an atom's formal charge as an integer (`PropValue::Int` or `F64`).
fn atom_charge(mol: &MolGraph, id: AtomId) -> i32 {
    match mol.get_atom(id).ok().and_then(|a| a.get("formal_charge")) {
        Some(PropValue::Int(v)) => *v,
        Some(PropValue::F64(v)) => v.round() as i32,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Atom query
// ---------------------------------------------------------------------------

/// A leaf atom primitive.
#[derive(Debug, Clone)]
pub enum AtomPrimitive {
    /// `*` — matches any atom.
    Any,
    /// `a` — any aromatic atom.
    AnyAromatic,
    /// `A` — any aliphatic atom.
    AnyAliphatic,
    /// Aliphatic element by atomic number (`C`, `N`, `Cl`, ...).
    AliphaticElement(u8),
    /// Aromatic element by atomic number (`c`, `n`, `o`, ...).
    AromaticElement(u8),
    /// `#<n>` — atomic number, either aromatic or aliphatic.
    AtomicNum(u8),
    /// `H<n>` — total hydrogen count.
    TotalH(u32),
    /// `X<n>` — total connections (degree incl. implicit H; here = degree).
    TotalConnections(u32),
    /// `D<n>` — explicit degree.
    Degree(u32),
    /// `R` — in any ring (`R0` = in no ring).
    RingMembership(Option<u32>),
    /// `r<n>` — in a ring of size exactly `n` (`r` alone = in any ring).
    RingSize(Option<u32>),
    /// `+`/`-`/`+n`/`-n` — formal charge.
    Charge(i32),
}

/// An atom query tree (logical combination of primitives).
#[derive(Debug, Clone)]
pub enum AtomQuery {
    Prim(AtomPrimitive),
    /// `$(...)` — recursive SMARTS rooted at the candidate atom.
    /// The boxed pattern is stored separately to break the type cycle.
    Recursive(usize),
    Not(Box<AtomQuery>),
    And(Vec<AtomQuery>),
    Or(Vec<AtomQuery>),
}

impl AtomPrimitive {
    fn eval(&self, ctx: &MolContext, id: AtomId) -> bool {
        let mol = ctx.mol;
        match self {
            AtomPrimitive::Any => true,
            AtomPrimitive::AnyAromatic => ctx.is_aromatic(id),
            AtomPrimitive::AnyAliphatic => !ctx.is_aromatic(id),
            AtomPrimitive::AliphaticElement(z) => {
                !ctx.is_aromatic(id) && symbol_matches_z(atom_symbol(mol, id), *z)
            }
            AtomPrimitive::AromaticElement(z) => {
                ctx.is_aromatic(id) && symbol_matches_z(atom_symbol(mol, id), *z)
            }
            AtomPrimitive::AtomicNum(z) => symbol_matches_z(atom_symbol(mol, id), *z),
            AtomPrimitive::TotalH(n) => ctx.h_count(id) == *n,
            AtomPrimitive::TotalConnections(n) => ctx.degree(id) == *n,
            AtomPrimitive::Degree(n) => ctx.degree(id) == *n,
            AtomPrimitive::RingMembership(None) => ctx.rings.is_atom_in_ring(id),
            AtomPrimitive::RingMembership(Some(n)) => ctx.rings.num_atom_rings(id) as u32 == *n,
            AtomPrimitive::RingSize(None) => ctx.rings.is_atom_in_ring(id),
            // RDKit's `r<n>` matches when the atom's *smallest* ring has size
            // exactly `n` (not "is in any ring of size n"). E.g. a fused-ring
            // atom shared by a 5- and a 6-ring matches `r5`, never `r6`.
            AtomPrimitive::RingSize(Some(n)) => {
                ctx.rings.smallest_ring_containing_atom(id) == Some(*n as usize)
            }
            AtomPrimitive::Charge(c) => atom_charge(mol, id) == *c,
        }
    }
}

/// Whether the element symbol corresponds to atomic number `z`.
fn symbol_matches_z(sym: &str, z: u8) -> bool {
    crate::element::Element::by_symbol(sym).map(|e| e.z()) == Some(z)
}

// ---------------------------------------------------------------------------
// Bond query
// ---------------------------------------------------------------------------

/// A leaf bond primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondPrimitive {
    /// `-` single.
    Single,
    /// `=` double.
    Double,
    /// `#` triple.
    Triple,
    /// `:` aromatic.
    Aromatic,
    /// `~` any bond.
    Any,
    /// `@` ring bond.
    InRing,
    /// Default bond (no symbol) — single OR aromatic.
    SingleOrAromatic,
}

/// A bond query tree.
#[derive(Debug, Clone)]
pub enum BondQuery {
    Prim(BondPrimitive),
    Not(Box<BondQuery>),
    And(Vec<BondQuery>),
    Or(Vec<BondQuery>),
}

/// Resolved facts about a concrete molecule bond, for primitive evaluation.
pub struct BondFacts {
    pub order: f64,
    pub aromatic: bool,
    pub in_ring: bool,
}

impl BondPrimitive {
    fn eval(&self, f: &BondFacts) -> bool {
        let is_single = !f.aromatic && (f.order - 1.0).abs() < 1e-6;
        let is_double = !f.aromatic && (f.order - 2.0).abs() < 1e-6;
        let is_triple = !f.aromatic && (f.order - 3.0).abs() < 1e-6;
        match self {
            BondPrimitive::Single => is_single,
            BondPrimitive::Double => is_double,
            BondPrimitive::Triple => is_triple,
            BondPrimitive::Aromatic => f.aromatic,
            BondPrimitive::Any => true,
            BondPrimitive::InRing => f.in_ring,
            BondPrimitive::SingleOrAromatic => is_single || f.aromatic,
        }
    }
}

impl BondQuery {
    /// Evaluate the bond query against resolved facts.
    pub fn eval(&self, f: &BondFacts) -> bool {
        match self {
            BondQuery::Prim(p) => p.eval(f),
            BondQuery::Not(inner) => !inner.eval(f),
            BondQuery::And(items) => items.iter().all(|q| q.eval(f)),
            BondQuery::Or(items) => items.iter().any(|q| q.eval(f)),
        }
    }
}

// ---------------------------------------------------------------------------
// Atom-query evaluation with recursive-SMARTS support
// ---------------------------------------------------------------------------

/// Callback used to evaluate a recursive `$(...)` subpattern rooted at `id`.
///
/// Implemented by the matcher (which owns the compiled subpatterns) to avoid
/// a hard type cycle between `ast` and `matcher`.
pub trait RecursiveEval {
    fn eval_recursive(&self, sub_index: usize, ctx: &MolContext, id: AtomId) -> bool;
}

impl AtomQuery {
    /// Evaluate this atom query against atom `id`.
    pub fn eval(&self, ctx: &MolContext, id: AtomId, rec: &dyn RecursiveEval) -> bool {
        match self {
            AtomQuery::Prim(p) => p.eval(ctx, id),
            AtomQuery::Recursive(idx) => rec.eval_recursive(*idx, ctx, id),
            AtomQuery::Not(inner) => !inner.eval(ctx, id, rec),
            AtomQuery::And(items) => items.iter().all(|q| q.eval(ctx, id, rec)),
            AtomQuery::Or(items) => items.iter().any(|q| q.eval(ctx, id, rec)),
        }
    }
}
