//! MMFF94 / MMFF94s energy assembly: the [`MmffForceField`] potential.
//!
//! This is the real MMFF94 functional form (seven energy contributions with
//! analytical gradients), ported from RDKit
//! `Code/ForceField/MMFF/*.cpp` (the term kernels) and
//! `Code/GraphMol/ForceFieldHelpers/MMFF/Builder.cpp` (term enumeration,
//! type codes, 1-4 masking, empirical-rule fallbacks)
//! (BSD-3, Paolo Tosco / RDKit contributors).
//!
//! Reference: Halgren, T. A. *J. Comput. Chem.* 1996, 17, 490-641 (MMFF.I-V).
//!
//! ```no_run
//! use molrs_ff::mmff::{MmffForceField, MmffMolProperties, MmffVariant};
//! use molrs_ff::potential::Potential;
//! # fn run(mol: &molrs::molgraph::MolGraph) -> Result<(), molrs::error::MolRsError> {
//! let props = MmffMolProperties::compute(mol, MmffVariant::Mmff94)?;
//! let ff = MmffForceField::build(mol, &props)?;
//! let coords = vec![0.0; 3 * props.len()];
//! let (energy, forces) = ff.eval(&coords);
//! # let _ = (energy, forces); Ok(())
//! # }
//! ```

mod angle;
mod bond;
mod geom;
mod nonbonded;
mod oop;
mod params;
mod stretchbend;
mod torsion;

use molrs::error::MolRsError;
use molrs::molgraph::MolGraph;
use molrs::types::F;

use super::aromaticity::set_mmff_aromaticity;
use super::hybrid::{Hyb, hybridization};
use super::topo::{BondOrder, Topo};
use super::{MmffMolProperties, MmffVariant};
use crate::potential::Potential;

use angle::AngleTerm;
use bond::BondTerm;
use nonbonded::NonbondedTerm;
use oop::OopTerm;
use params as p;
use stretchbend::StretchBendTerm;
use torsion::TorsionTerm;

/// Per-term energy decomposition (kcal/mol), for cross-checking against RDKit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MmffEnergyBreakdown {
    pub bond: f64,
    pub angle: f64,
    pub stretch_bend: f64,
    pub oop: f64,
    pub torsion: f64,
    pub vdw: f64,
    pub electrostatic: f64,
    pub total: f64,
}

/// An assembled MMFF94/MMFF94s force field bound to one molecule's topology
/// and [`MmffMolProperties`]. Implements [`Potential`].
#[derive(Debug)]
pub struct MmffForceField {
    variant: MmffVariant,
    bonds: Vec<BondTerm>,
    angles: Vec<AngleTerm>,
    stretch_bends: Vec<StretchBendTerm>,
    oops: Vec<OopTerm>,
    torsions: Vec<TorsionTerm>,
    nonbonded: Vec<NonbondedTerm>,
}

impl MmffForceField {
    /// Enumerate every MMFF94 energy term (including the empirical-rule
    /// fallbacks) for `mol`, using the atom types / partial charges in `props`.
    pub fn build(mol: &MolGraph, props: &MmffMolProperties) -> Result<Self, MolRsError> {
        let base = Topo::build(mol).map_err(|sym| {
            MolRsError::validation(format!("MMFF: unsupported element symbol '{sym}'"))
        })?;
        let topo = set_mmff_aromaticity(&base);
        let n = topo.n_atoms();
        if props.len() != n {
            return Err(MolRsError::validation(
                "MMFF energy: properties / molecule atom-count mismatch",
            ));
        }
        let types: Vec<u8> = (0..n).map(|i| props.atom_type(i)).collect();
        let charges: Vec<f64> = (0..n).map(|i| props.partial_charge(i)).collect();

        let bonds = enumerate_bonds(&topo, &types);
        let angles = enumerate_angles(&topo, &types);
        let stretch_bends = enumerate_stretch_bends(&topo, &types);
        let oops = enumerate_oops(&topo, &types);
        let torsions = enumerate_torsions(&topo, &types);
        let nonbonded = enumerate_nonbonded(&topo, &types, &charges);

        Ok(Self {
            variant: props.variant(),
            bonds,
            angles,
            stretch_bends,
            oops,
            torsions,
            nonbonded,
        })
    }

    /// The MMFF variant this force field was assembled for.
    pub fn variant(&self) -> MmffVariant {
        self.variant
    }

    /// Per-term energy decomposition at `coords` (flat `[x0,y0,z0,...]`).
    pub fn energy_terms(&self, coords: &[f64]) -> MmffEnergyBreakdown {
        let mut scratch = vec![0.0f64; coords.len()];
        let bond = self
            .bonds
            .iter()
            .map(|t| t.energy_grad(coords, &mut scratch))
            .sum();
        let angle = self
            .angles
            .iter()
            .map(|t| t.energy_grad(coords, &mut scratch))
            .sum();
        let stretch_bend = self
            .stretch_bends
            .iter()
            .map(|t| t.energy_grad(coords, &mut scratch))
            .sum();
        let oop = self
            .oops
            .iter()
            .map(|t| t.energy_grad(coords, &mut scratch))
            .sum();
        let torsion = self
            .torsions
            .iter()
            .map(|t| t.energy_grad(coords, &mut scratch))
            .sum();

        // vdW and electrostatic share NonbondedTerm; evaluate separately so the
        // breakdown can report them in isolation.
        let mut vdw = 0.0;
        let mut electrostatic = 0.0;
        for t in &self.nonbonded {
            if let Some(vp) = t.vdw {
                vdw += NonbondedTerm {
                    charge: None,
                    vdw: Some(vp),
                    ..*t
                }
                .energy_grad(coords, &mut scratch);
            }
            if let Some(c) = t.charge {
                electrostatic += NonbondedTerm {
                    vdw: None,
                    charge: Some(c),
                    ..*t
                }
                .energy_grad(coords, &mut scratch);
            }
        }
        let total = bond + angle + stretch_bend + oop + torsion + vdw + electrostatic;
        MmffEnergyBreakdown {
            bond,
            angle,
            stretch_bend,
            oop,
            torsion,
            vdw,
            electrostatic,
            total,
        }
    }
}

impl Potential for MmffForceField {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let mut grad = vec![0.0f64; coords.len()];
        let mut energy = 0.0;
        for t in &self.bonds {
            energy += t.energy_grad(coords, &mut grad);
        }
        for t in &self.angles {
            energy += t.energy_grad(coords, &mut grad);
        }
        for t in &self.stretch_bends {
            energy += t.energy_grad(coords, &mut grad);
        }
        for t in &self.oops {
            energy += t.energy_grad(coords, &mut grad);
        }
        for t in &self.torsions {
            energy += t.energy_grad(coords, &mut grad);
        }
        for t in &self.nonbonded {
            energy += t.energy_grad(coords, &mut grad);
        }
        // Potential returns forces = -gradient.
        for g in &mut grad {
            *g = -*g;
        }
        (energy, grad)
    }
}

// --- term enumeration (Builder.cpp) --------------------------------------

fn enumerate_bonds(topo: &Topo, types: &[u8]) -> Vec<BondTerm> {
    let mut out = Vec::new();
    let n = topo.n_atoms();
    for i in 0..n {
        for &j in &topo.nbrs[i] {
            if j <= i {
                continue;
            }
            if let Some(bp) = p::bond_params(topo, types, i, j) {
                out.push(BondTerm {
                    i,
                    j,
                    r0: bp.r0,
                    kb: bp.kb,
                });
            }
        }
    }
    out
}

fn enumerate_angles(topo: &Topo, types: &[u8]) -> Vec<AngleTerm> {
    let mut out = Vec::new();
    let n = topo.n_atoms();
    for j in 0..n {
        if topo.degree(j) < 2 {
            continue;
        }
        let linear = p::central_prop(types, j)
            .map(|pp| pp.linh != 0)
            .unwrap_or(false);
        let nbrs = &topo.nbrs[j];
        for a in 0..nbrs.len() {
            for b in (a + 1)..nbrs.len() {
                let (i, k) = (nbrs[a], nbrs[b]);
                if let Some(ap) = p::angle_params(topo, types, i, j, k) {
                    out.push(AngleTerm {
                        i,
                        j,
                        k,
                        theta0: ap.theta0,
                        ka: ap.ka,
                        linear,
                    });
                }
            }
        }
    }
    out
}

fn enumerate_stretch_bends(topo: &Topo, types: &[u8]) -> Vec<StretchBendTerm> {
    let mut out = Vec::new();
    let n = topo.n_atoms();
    for j in 0..n {
        if topo.degree(j) < 2 {
            continue;
        }
        if p::central_prop(types, j)
            .map(|pp| pp.linh != 0)
            .unwrap_or(false)
        {
            continue;
        }
        let nbrs = &topo.nbrs[j];
        for a in 0..nbrs.len() {
            for b in (a + 1)..nbrs.len() {
                let (i, k) = (nbrs[a], nbrs[b]);
                if let Some((sp, r1, r2, theta0)) = p::stretch_bend_params(topo, types, i, j, k) {
                    out.push(StretchBendTerm {
                        i,
                        j,
                        k,
                        rest1: r1,
                        rest2: r2,
                        theta0,
                        fc1: sp.kba_ijk,
                        fc2: sp.kba_kji,
                    });
                }
            }
        }
    }
    out
}

fn enumerate_oops(topo: &Topo, types: &[u8]) -> Vec<OopTerm> {
    let mut out = Vec::new();
    let n = topo.n_atoms();
    for j in 0..n {
        if topo.degree(j) != 3 {
            continue;
        }
        let nbrs = &topo.nbrs[j];
        let (a, b, c) = (nbrs[0], nbrs[1], nbrs[2]);
        let koop = match p::oop_koop(types, a, j, b, c) {
            Some(k) => k,
            None => continue,
        };
        // three Wilson permutations (RDKit addOop)
        for &(i, k, l) in &[(a, b, c), (a, c, b), (b, c, a)] {
            out.push(OopTerm { i, j, k, l, koop });
        }
    }
    out
}

/// Is atom `a` part of any triple bond? (the `!$(*#*)` half of the torsion SMARTS)
fn in_triple_bond(topo: &Topo, a: usize) -> bool {
    topo.nbr_kekule[a].contains(&BondOrder::Triple)
}

fn enumerate_torsions(topo: &Topo, types: &[u8]) -> Vec<TorsionTerm> {
    let mut out = Vec::new();
    let n = topo.n_atoms();
    let sp2_or_sp3 = |x: usize| matches!(hybridization(topo, x), Hyb::Sp2 | Hyb::Sp3);
    // central bond j-k: both ends degree>1, not in a triple bond, SP2/SP3.
    for j in 0..n {
        for &k in &topo.nbrs[j] {
            if k <= j {
                continue;
            }
            if topo.degree(j) < 2 || topo.degree(k) < 2 {
                continue;
            }
            if in_triple_bond(topo, j) || in_triple_bond(topo, k) {
                continue;
            }
            if !(sp2_or_sp3(j) && sp2_or_sp3(k)) {
                continue;
            }
            for &i in &topo.nbrs[j] {
                if i == k {
                    continue;
                }
                for &l in &topo.nbrs[k] {
                    if l == j || l == i {
                        continue;
                    }
                    if let Some(tp) = p::torsion_params(topo, types, i, j, k, l) {
                        out.push(TorsionTerm {
                            i,
                            j,
                            k,
                            l,
                            v1: tp.v1,
                            v2: tp.v2,
                            v3: tp.v3,
                        });
                    }
                }
            }
        }
    }
    out
}

fn enumerate_nonbonded(topo: &Topo, types: &[u8], charges: &[f64]) -> Vec<NonbondedTerm> {
    let mut out = Vec::new();
    let n = topo.n_atoms();
    for i in 0..n {
        for j in (i + 1)..n {
            let rel = p::relation(topo, i, j);
            if rel < 3 {
                continue; // 1-2 / 1-3 excluded
            }
            let is_1_4 = rel == 3;
            let vdw = p::vdw_params(types, i, j).map(|v| (v.r_star, v.epsilon));
            let charge = if charges[i].abs() < 1.0e-8 || charges[j].abs() < 1.0e-8 {
                None
            } else {
                Some(charges[i] * charges[j]) // dielectric constant D = 1
            };
            if vdw.is_none() && charge.is_none() {
                continue;
            }
            out.push(NonbondedTerm {
                i,
                j,
                vdw,
                charge,
                is_1_4,
            });
        }
    }
    out
}
