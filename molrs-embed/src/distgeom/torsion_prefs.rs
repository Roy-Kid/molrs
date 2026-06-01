//! ETKDGv3 experimental torsion-angle preferences (CrystalFF) + SMARTS matcher.
//!
//! Data and selection semantics ported from RDKit (BSD-3, Copyright (C)
//! 2017-2023 Sereina Riniker and other RDKit contributors):
//!   * `$RDBASE/Code/GraphMol/ForceFieldHelpers/CrystalFF/`
//!     `torsionPreferences_{v2,smallrings,macrocycles}.in` — the full
//!     ETKDGv2/v3 SMARTS → `(signs, V)` parameter tables (embedded verbatim in
//!     [`super::torsion_tables`], extracted by `gen_torsion_tables.py`),
//!   * `.../TorsionPreferences.cpp` (`getExperimentalTorsions`) — pattern
//!     matching + first-match-wins dedup per rotatable bond,
//!   * `$RDBASE/Code/ForceField/CrystalFF/TorsionAngleM6.h` — the M6 potential
//!     `V = Σ_m V_m·(1 + s_m·cos(m·x))`.
//!
//! ## Faithful port (this replaces the former representative subset)
//!
//! Every torsion assignment now flows through the **full** three-table data
//! set and the project SMARTS engine ([`molrs::smarts::SmartsPattern`]). For
//! each rotatable bond we reproduce RDKit's exact selection:
//!
//! 1. The three tables are matched in RDKit concatenation order — v2, then
//!    small-rings, then macrocycles ([`super::torsion_tables`]). The first
//!    pattern (in that global order) that matches a bond wins; a bond receives
//!    at most one experimental torsion.
//! 2. Ring-size gating is *intrinsic to the SMARTS* (macrocycle patterns carry
//!    `r{9-}`, small-ring patterns `r{-8}` / `r{5-6}` / …), so there is no
//!    separate ring-membership dispatch: the right table simply fails to match
//!    bonds of the wrong ring class. This is exactly how RDKit layers them.
//!
//! ### SMARTS compatibility shim
//!
//! The core SMARTS engine covers the primitives in these tables *except* two
//! that it has no AST node for: RDKit's **ring-size range** `r{lo-hi}` (and the
//! open forms `r{-hi}` / `r{lo-}`) and **ring connectivity** `x<n>` (number of
//! ring bonds on the atom). Rather than reimplement matching, we *normalise*
//! each pattern before parsing: the `r{…}` / `x<n>` tokens are stripped out and
//! recorded per atom-map label, then re-checked against the molecule's ring
//! info **after** the engine returns its matches. The residual constraints are
//! pure local ring facts, so this post-filter is equivalent to evaluating them
//! inline (validated against RDKit `GetExperimentalTorsions` in
//! `tests/embed/torsions.rs`).

use std::collections::HashMap;

use molrs::molgraph::{AtomId, MolGraph, PropValue};
use molrs::smarts::SmartsPattern;

use super::perceive::Perceived;
use super::torsion_tables::{self, TorsionRow};

/// One assigned experimental torsion: four atoms + the M6 `(signs, V)` set.
#[derive(Clone, Debug)]
pub struct TorsionConstraint {
    /// Ordered atom indices `i-j-k-l` (the rotatable bond is `j-k`).
    pub atoms: [usize; 4],
    /// Per-order signs `s1..s6`.
    pub signs: [i8; 6],
    /// Per-order force constants `V1..V6`.
    pub force_constants: [f64; 6],
    /// The originating pattern SMARTS (for diagnostics / spec traceability).
    pub pattern: &'static str,
}

/// Which source table a pattern came from (for diagnostics + `tests`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TorsionTable {
    /// `torsionPreferences_v2` — acyclic / general bonds.
    V2,
    /// `torsionPreferences_smallrings` — small-ring bonds.
    SmallRings,
    /// `torsionPreferences_macrocycles` — ring bonds of size ≥ 9.
    Macrocycles,
}

// ---------------------------------------------------------------------------
// Pattern compilation (with the `r{…}` / `x<n>` shim)
// ---------------------------------------------------------------------------

/// A residual ring constraint that the core engine cannot express, recorded
/// against an atom-map label and re-checked post-match.
#[derive(Clone, Copy, Debug)]
enum RingConstraint {
    /// `r{lo-hi}` — atom's smallest ring size must lie in `[lo, hi]` inclusive
    /// (`hi == None` means unbounded above).
    RingSizeRange { lo: usize, hi: Option<usize> },
    /// `x<n>` — atom must have exactly `n` ring bonds.
    RingConnectivity(usize),
}

/// A compiled table entry: the engine pattern, the residual ring constraints
/// keyed by atom-map label (1..=4), the M6 parameters, and provenance.
struct CompiledPattern {
    smarts: &'static str,
    pattern: SmartsPattern,
    residual: Vec<(u32, RingConstraint)>,
    signs: [i8; 6],
    v: [f64; 6],
    table: TorsionTable,
}

/// Strip the unsupported `r{…}` / `x<n>` primitives out of `smarts`, returning
/// the engine-parseable SMARTS plus the residual constraints keyed by the
/// atom-map label of the bracketed atom they belonged to.
///
/// The scan is bracket-atom aware: it walks each `[...]` group, finds the
/// atom-map `:n`, and pulls out any `r{…}` / `x<digits>` token (cleaning up the
/// adjacent `;`/`&` separator so the residue still parses).
fn strip_ring_primitives(smarts: &str) -> (String, Vec<(u32, RingConstraint)>) {
    let bytes: Vec<char> = smarts.chars().collect();
    let mut out = String::with_capacity(smarts.len());
    let mut residual: Vec<(u32, RingConstraint)> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == '[' {
            // Find the matching ']'.
            let start = i;
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != ']' {
                j += 1;
            }
            let inner: String = bytes[start + 1..j].iter().collect();
            let map = parse_map_label(&inner);
            let (cleaned, cons) = strip_atom_inner(&inner);
            for c in cons {
                if let Some(m) = map {
                    residual.push((m, c));
                }
            }
            out.push('[');
            out.push_str(&cleaned);
            out.push(']');
            i = j + 1;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    (out, residual)
}

/// The atom-map label `:n` inside a bracket-atom body, if any.
fn parse_map_label(inner: &str) -> Option<u32> {
    let pos = inner.rfind(':')?;
    inner[pos + 1..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

/// Remove `r{…}` / `x<digits>` tokens from a single bracket-atom body, cleaning
/// up the separators they leave behind. Returns the cleaned body and the
/// residual constraints (un-keyed; the caller attaches the map label).
fn strip_atom_inner(inner: &str) -> (String, Vec<RingConstraint>) {
    let chars: Vec<char> = inner.chars().collect();
    let mut out = String::with_capacity(inner.len());
    let mut cons: Vec<RingConstraint> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // `r{...}` range (only when followed by '{'; a plain `r<digit>` /
        // bare `r` stays for the engine).
        if c == 'r' && i + 1 < chars.len() && chars[i + 1] == '{' {
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '}' {
                j += 1;
            }
            let body: String = chars[i + 2..j].iter().collect();
            if let Some(rc) = parse_ring_range(&body) {
                cons.push(rc);
            }
            i = j + 1;
            i = skip_trailing_sep(&chars, i);
            continue;
        }
        // `x<digits>` ring connectivity.
        if c == 'x' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            let mut j = i + 1;
            let mut num = String::new();
            while j < chars.len() && chars[j].is_ascii_digit() {
                num.push(chars[j]);
                j += 1;
            }
            if let Ok(n) = num.parse::<usize>() {
                cons.push(RingConstraint::RingConnectivity(n));
            }
            i = j;
            i = skip_trailing_sep(&chars, i);
            continue;
        }
        out.push(c);
        i += 1;
    }
    let cleaned = tidy_separators(&out);
    (cleaned, cons)
}

/// After removing a token, drop one immediately following `;`/`&` separator so
/// we do not leave a dangling logical operator. Returns the new cursor.
fn skip_trailing_sep(chars: &[char], i: usize) -> usize {
    if i < chars.len() && (chars[i] == ';' || chars[i] == '&') {
        // Only skip if there is real content before, else leave for tidy pass.
        return i + 1;
    }
    i
}

/// Collapse `;;`, `&&`, leading/trailing/`[`-adjacent separators left behind by
/// token removal, so the residue parses cleanly.
fn tidy_separators(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == ';' || c == '&' {
            // Skip if at the very start, at the end, or duplicated / before ':'.
            let at_start = out.is_empty();
            let next = chars.get(i + 1).copied();
            let dup_or_end = matches!(next, None | Some(';') | Some('&') | Some(':'));
            if at_start || dup_or_end {
                i += 1;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Parse the body of `r{...}` into a [`RingConstraint::RingSizeRange`].
///
/// Forms: `lo-hi`, `-hi` (lo defaults to the smallest chemical ring, 3),
/// `lo-` (hi unbounded), or a bare `n` (treated as `n-n`).
fn parse_ring_range(body: &str) -> Option<RingConstraint> {
    if let Some((a, b)) = body.split_once('-') {
        let lo = if a.is_empty() {
            3
        } else {
            a.trim().parse().ok()?
        };
        let hi = if b.is_empty() {
            None
        } else {
            Some(b.trim().parse().ok()?)
        };
        Some(RingConstraint::RingSizeRange { lo, hi })
    } else {
        let n: usize = body.trim().parse().ok()?;
        Some(RingConstraint::RingSizeRange { lo: n, hi: Some(n) })
    }
}

/// Compile one `(smarts, signs, V)` row into a [`CompiledPattern`].
fn compile_row(row: &TorsionRow, table: TorsionTable) -> Option<CompiledPattern> {
    let (smarts, signs, v) = (row.0, row.1, row.2);
    let (cleaned, residual) = strip_ring_primitives(smarts);
    let pattern = SmartsPattern::parse(&cleaned).ok()?;
    Some(CompiledPattern {
        smarts,
        pattern,
        residual,
        signs,
        v,
        table,
    })
}

/// Compile the tables for ETKDGv3 in RDKit concatenation order.
///
/// RDKit's `ETKDGv3()` sets `useSmallRingTorsions = false` and
/// `useMacrocycleTorsions = true`, so the concatenated parameter set is
/// **v2 ++ macrocycles** (the small-ring table is *not* included). We mirror
/// that exactly: matching v2 first, then macrocycles, first-match-wins. Rows
/// whose residue fails to parse are skipped (none in the validated set).
fn compile_all() -> Vec<CompiledPattern> {
    let mut out = Vec::new();
    for row in torsion_tables::V2 {
        if let Some(p) = compile_row(row, TorsionTable::V2) {
            out.push(p);
        }
    }
    for row in torsion_tables::MACROCYCLES {
        if let Some(p) = compile_row(row, TorsionTable::Macrocycles) {
            out.push(p);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Matching molecule (aromaticity transplanted from perception)
// ---------------------------------------------------------------------------

/// Build a working copy of `mol` whose atoms / bonds carry the `is_aromatic`
/// flag from the project perception, so the SMARTS engine's `a` / `c` / `:`
/// queries agree with RDKit (the engine reads `is_aromatic`, see
/// `molrs::smarts` aromaticity convention).
fn aromatic_working_copy(mol: &MolGraph, p: &Perceived) -> MolGraph {
    let mut g = mol.clone();
    for (i, &aid) in p.atom_ids.iter().enumerate() {
        if p.atoms[i].aromatic {
            if let Ok(a) = g.get_atom_mut(aid) {
                a.set("is_aromatic", 1_i32);
            }
        }
    }
    // Flag aromatic bonds so `:` and `BondFacts.aromatic` agree.
    let bond_ids: Vec<_> = g.bonds().map(|(bid, b)| (bid, b.atoms)).collect();
    let idx_of: HashMap<AtomId, usize> = p
        .atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    for (bid, [a, b]) in bond_ids {
        let (Some(&ia), Some(&ib)) = (idx_of.get(&a), idx_of.get(&b)) else {
            continue;
        };
        if p.is_aromatic_bond(ia, ib) {
            if let Ok(bond) = g.get_bond_mut(bid) {
                bond.props
                    .insert("is_aromatic".to_string(), PropValue::Int(1));
            }
        }
    }
    g
}

/// Smallest ring size containing the perceived atom index `i`, or `None`.
fn smallest_ring_size(p: &Perceived, i: usize) -> Option<usize> {
    p.ring_idx
        .iter()
        .filter(|ring| ring.contains(&i))
        .map(|ring| ring.len())
        .min()
}

/// Number of ring bonds incident on perceived atom index `i` (RDKit `x`).
fn ring_connectivity(p: &Perceived, i: usize) -> usize {
    let mut count = 0;
    for &j in &p.adj[i] {
        let in_ring = p.ring_idx.iter().any(|ring| {
            let rs = ring.len();
            (0..rs).any(|w| {
                let a = ring[w];
                let b = ring[(w + 1) % rs];
                (a == i && b == j) || (a == j && b == i)
            })
        });
        if in_ring {
            count += 1;
        }
    }
    count
}

/// Check one residual ring constraint against perceived atom index `i`.
fn residual_holds(p: &Perceived, i: usize, c: RingConstraint) -> bool {
    match c {
        RingConstraint::RingSizeRange { lo, hi } => match smallest_ring_size(p, i) {
            Some(sz) => sz >= lo && hi.is_none_or(|h| sz <= h),
            None => false,
        },
        RingConstraint::RingConnectivity(n) => ring_connectivity(p, i) == n,
    }
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

/// A single assigned torsion with its provenance, for diagnostics / tests.
#[derive(Clone, Debug)]
pub struct AssignedTorsion {
    pub constraint: TorsionConstraint,
    pub table: TorsionTable,
}

/// Assign experimental torsions to `mol` by matching the full ETKDGv3 tables
/// (v2 ++ small-rings ++ macrocycles) through the SMARTS engine, reproducing
/// RDKit `getExperimentalTorsions`: the first matching pattern (global table
/// order) wins per rotatable bond; one torsion per bond.
///
/// `p` is the perception of `mol` (aromaticity / hybridization / rings); it is
/// reused to transplant aromatic flags onto the matching copy and to evaluate
/// the `r{…}` / `x<n>` residual constraints.
pub fn assign_with_provenance(mol: &MolGraph, p: &Perceived) -> Vec<AssignedTorsion> {
    let work = aromatic_working_copy(mol, p);
    let patterns = compile_all();

    let idx_of: HashMap<AtomId, usize> = p
        .atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // RDKit keys "done" by central-bond index; we key by the unordered central
    // atom-index pair, which is equivalent for a simple molecular graph.
    let mut done: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let mut out: Vec<AssignedTorsion> = Vec::new();

    for cp in &patterns {
        for m in cp.pattern.find_matches(&work) {
            // Resolve atom-map labels :1..:4 to mol-atom indices.
            let mut idx = [usize::MAX; 4];
            let mut ok = true;
            for (qi, &aid) in m.iter().enumerate() {
                if let Some(lbl) = cp.pattern.map_label(qi) {
                    if (1..=4).contains(&lbl) {
                        match idx_of.get(&aid) {
                            Some(&ix) => idx[(lbl - 1) as usize] = ix,
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                }
            }
            if !ok || idx.contains(&usize::MAX) {
                continue;
            }

            // Residual `r{…}` / `x<n>` constraints (keyed by atom-map label).
            let residual_ok = cp.residual.iter().all(|&(lbl, c)| {
                let ai = idx[(lbl - 1) as usize];
                residual_holds(p, ai, c)
            });
            if !residual_ok {
                continue;
            }

            let (j, k) = (idx[1], idx[2]);
            let key = if j < k { (j, k) } else { (k, j) };
            if done.contains(&key) {
                continue;
            }
            // The central pair must be a real bond.
            if !p.adj[j].contains(&k) {
                continue;
            }
            done.insert(key);
            out.push(AssignedTorsion {
                constraint: TorsionConstraint {
                    atoms: idx,
                    signs: cp.signs,
                    force_constants: cp.v,
                    pattern: cp.smarts,
                },
                table: cp.table,
            });
        }
    }
    out
}

/// Public entry point used by [`super::build_constraints`]: the bare
/// [`TorsionConstraint`] list (provenance dropped).
pub fn assign_experimental_torsions(mol: &MolGraph, p: &Perceived) -> Vec<TorsionConstraint> {
    assign_with_provenance(mol, p)
        .into_iter()
        .map(|a| a.constraint)
        .collect()
}
