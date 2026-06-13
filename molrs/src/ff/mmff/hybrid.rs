//! Hybridization + conjugation perception, ported from RDKit
//! `Code/GraphMol/ConjugHybrid.cpp` (BSD-3, RDKit contributors).
//!
//! MMFF aromaticity perception (`setMMFFAromaticity`) reads
//! `atom->getHybridization()` to reject non-sp2 carbon / nitrogen from
//! aromatic rings. RDKit computes hybridization from
//! `numBondsPlusLonePairs`, dropping a `norbs==4` atom to SP2 when it
//! carries a conjugated bond (so pyrrole-type N is sp2). We reproduce
//! exactly that decision here.

use super::topo::{BondOrder, Topo};

/// Coarse hybridization labels (only the ones MMFF aromaticity needs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hyb {
    S,
    Sp,
    Sp2,
    Sp3,
    Other,
}

/// Number of outer-shell (valence) electrons — the periodic group count
/// RDKit reads via `getNouterElecs`. Provided for the elements that occur
/// in MMFF-typable organic molecules.
fn nouter_elecs(atno: u8) -> Option<i32> {
    Some(match atno {
        1 => 1,
        2 => 2,
        3 | 11 | 19 => 1,
        4 | 12 | 20 => 2,
        5 | 13 => 3,
        6 | 14 => 4,
        7 | 15 => 5,
        8 | 16 => 6,
        9 | 17 | 35 | 53 => 7,
        10 | 18 => 8,
        _ => return None,
    })
}

/// First (minimal) standard valence, RDKit `getValenceList().front()`.
fn min_valence(atno: u8) -> i32 {
    match atno {
        1 => 1,
        5 => 3,
        6 | 14 => 4,
        7 | 15 => 3,
        8 | 16 => 2,
        9 | 17 | 35 | 53 => 1,
        3 | 11 | 19 => 1,
        4 | 12 | 20 => 2,
        _ => -1,
    }
}

/// RDKit `PeriodicTable::getDefaultValence`. For the organic subset this is
/// the first standard valence; we list the ones MMFF can type.
fn default_valence(atno: u8) -> i32 {
    match atno {
        1 => 1,
        5 => 3,
        6 | 14 => 4,
        7 | 15 => 3,
        8 | 16 => 2,
        9 | 17 | 35 | 53 => 1,
        3 | 11 | 19 => 1,
        4 | 12 | 20 => 2,
        _ => -1,
    }
}

/// Faithful port of `RDKit::MolOps::countAtomElec` (`Aromaticity.cpp`):
/// number of pi-donatable electrons (`-1` if univalent or >3-coordinate).
fn count_atom_elec(topo: &Topo, i: usize) -> i32 {
    let atno = topo.atno[i];
    let dv = default_valence(atno);
    if dv <= 1 {
        return -1;
    }
    let degree = topo.total_degree(i) as i32; // degree + Hs (Hs explicit here)
    if degree > 3 {
        return -1;
    }
    let nouter = match nouter_elecs(atno) {
        Some(v) => v,
        None => return -1,
    };
    let mut nlp = nouter - dv;
    nlp = (nlp - topo.formal_charge[i]).max(0);
    let mut res = (dv - degree) + nlp; // no radicals in MMFF-sanitized mols
    if res > 1 {
        let n_unsat = topo.total_bond_order(i) as i32 - topo.degree(i) as i32;
        if n_unsat > 1 {
            res = 1;
        }
    }
    res
}

/// RDKit `isAtomConjugCand`.
fn is_atom_conjug_cand(topo: &Topo, i: usize) -> bool {
    let atno = topo.atno[i];
    let minv = min_valence(atno);
    if topo.formal_charge[i] == 0 && minv >= 0 && (topo.total_bond_order(i) as i32) > minv {
        return false;
    }
    let nouter = match nouter_elecs(atno) {
        Some(v) => v,
        None => return false,
    };
    ((atno <= 10) || (nouter != 5 && nouter != 6) || (nouter == 6 && topo.total_degree(i) < 2))
        && count_atom_elec(topo, i) > 0
}

fn valence_contrib(o: BondOrder) -> f64 {
    match o {
        BondOrder::Single => 1.0,
        BondOrder::Aromatic => 1.5,
        BondOrder::Double => 2.0,
        BondOrder::Triple => 3.0,
    }
}

/// RDKit `markConjAtomBonds(at)` restricted to "does it mark the bond
/// `at`–`other` conjugated?". `at` must be a candidate with 2–3 substituents;
/// then a bond pair `(bnd1, bnd2)` is marked when `bnd1` has valence contrib
/// ≥ 1.5 to a candidate and `bnd2` (the other) reaches a candidate of sbo ≤ 3.
/// Both `bnd1` and `bnd2` get flagged, so the target bond qualifies if it is
/// either of the two.
fn marks_bond_conjugated(topo: &Topo, at: usize, other: usize) -> bool {
    if !is_atom_conjug_cand(topo, at) {
        return false;
    }
    let sbo = topo.degree(at);
    if !(2..=3).contains(&sbo) {
        return false;
    }
    for (p1, &n1) in topo.nbrs[at].iter().enumerate() {
        if valence_contrib(topo.nbr_order[at][p1]) < 1.5 || !is_atom_conjug_cand(topo, n1) {
            continue;
        }
        for (p2, &n2) in topo.nbrs[at].iter().enumerate() {
            if p1 == p2 || topo.degree(n2) > 3 || !is_atom_conjug_cand(topo, n2) {
                continue;
            }
            // bnd1 = at-n1, bnd2 = at-n2 both become conjugated.
            if n1 == other || n2 == other {
                return true;
            }
        }
    }
    false
}

/// Does atom `i` participate in at least one conjugated bond, per RDKit
/// `setConjugation` (aromatic bonds are conjugated; otherwise a bond is
/// conjugated if `markConjAtomBonds` from *either* endpoint flags it)?
fn atom_has_conjugated_bond(topo: &Topo, i: usize) -> bool {
    for (p, &j) in topo.nbrs[i].iter().enumerate() {
        if topo.nbr_order[i][p] == BondOrder::Aromatic {
            return true;
        }
        if marks_bond_conjugated(topo, i, j) || marks_bond_conjugated(topo, j, i) {
            return true;
        }
    }
    false
}

/// RDKit `numBondsPlusLonePairs` (no ZERO/dative bonds in our graphs).
fn num_bonds_plus_lone_pairs(topo: &Topo, i: usize) -> i32 {
    let deg = topo.total_degree(i) as i32;
    let atno = topo.atno[i];
    if atno <= 1 {
        return deg;
    }
    let nouter = match nouter_elecs(atno) {
        Some(v) => v,
        None => return deg,
    };
    let total_valence = topo.total_bond_order(i) as i32;
    let chg = topo.formal_charge[i];
    let num_free = nouter - (total_valence + chg);
    // No radicals in MMFF-sanitized molecules, so the octet branch and the
    // sub-octet branch reduce to the same lone-pair count here.
    let num_lone = num_free / 2;
    deg + num_lone
}

/// Port of RDKit `MolOps::setHybridization` for a single atom (no explicit
/// chirality tags in embedded molecules → falls through to the orbital count).
pub fn hybridization(topo: &Topo, i: usize) -> Hyb {
    if topo.atno[i] == 0 {
        return Hyb::Other;
    }
    let norbs = if topo.atno[i] < 89 {
        num_bonds_plus_lone_pairs(topo, i)
    } else {
        topo.total_degree(i) as i32
    };
    match norbs {
        0 | 1 => Hyb::S,
        2 => Hyb::Sp,
        3 => Hyb::Sp2,
        4 => {
            if topo.total_degree(i) > 3 || !atom_has_conjugated_bond(topo, i) {
                Hyb::Sp3
            } else {
                Hyb::Sp2
            }
        }
        5 | 6 => Hyb::Other,
        _ => Hyb::Other,
    }
}
