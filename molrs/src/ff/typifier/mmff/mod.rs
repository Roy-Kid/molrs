//! MMFF94 atom/bond/angle/torsion/improper typifier.
//!
//! Annotates an [`Atomistic`](molrs::Atomistic) with MMFF94 type labels and
//! partial charges (the typifier's job). MMFF carries no bespoke energy path —
//! it is a parameter set plus a topology labeler — so the
//! [`build`](MMFFTypifier::build) convenience just materializes the labeled graph
//! to a [`Frame`](molrs::store::frame::Frame) and routes it through the generic
//! [`ForceField::to_potentials`](crate::ff::potential) compile path.
//!
//! # Example
//!
//! ```ignore
//! use molrs::ff::typifier::mmff::MMFFTypifier;
//!
//! let typifier = MMFFTypifier::mmff94()?;
//! let potentials = typifier.build(&mol)?;       // typify → to_frame → to_potentials
//! let (energy, forces) = potentials.calc_energy_forces(&coords);
//! ```

#![allow(clippy::type_complexity)]

use std::collections::HashMap;

use ndarray::Array1;

use crate::ff::forcefield::ForceField;
use crate::ff::potential::{Potentials, intramolecular_pairs};
use molrs::Atomistic;
use molrs::store::frame::Frame;
use molrs::types::F;

use super::Typifier;

pub(crate) mod classify;
pub(crate) mod frame_builder;
pub mod params;

#[cfg(test)]
mod tests;

// Re-exports
pub use params::{MMFFAtomProp, MMFFParams};

/// MMFF94 typifier — owns typing metadata and force-field parameters.
///
/// Primary constructor: [`from_xml_str`](Self::from_xml_str) parses both
/// typing metadata (`MMFFParams`) and potential parameters (`ForceField`)
/// from a single XML string. Users never need to manage these separately.
pub struct MMFFTypifier {
    params: MMFFParams,
    ff: ForceField,
}

impl MMFFTypifier {
    /// Create a typifier from an MMFF94 XML string.
    ///
    /// Parses both typing metadata and force-field parameters in one call.
    pub fn from_xml_str(xml: &str) -> Result<Self, String> {
        let params = crate::ff::forcefield::xml::read_mmff_params_xml_str(xml)?;
        let ff = crate::ff::forcefield::xml::read_forcefield_xml_str(xml)?;
        Ok(Self { params, ff })
    }

    /// Create a typifier from the embedded MMFF94 parameter set.
    pub fn mmff94() -> Result<Self, String> {
        Self::from_xml_str(molrs::data::MMFF94_XML)
    }

    /// Access the underlying MMFF typing parameters.
    pub fn params(&self) -> &MMFFParams {
        &self.params
    }

    /// Access the underlying force field.
    pub fn ff(&self) -> &ForceField {
        &self.ff
    }

    /// Typify a molecule and compile potentials in one step.
    ///
    /// `mol → Frame → Potentials`. The intermediate `Frame` is not retained.
    ///
    /// Requires [`Atomistic`](molrs::system::atomistic::Atomistic) because MMFF94
    /// typing depends on element symbols, bond orders, and ring membership.
    pub fn build(&self, mol: &Atomistic) -> Result<Potentials, String> {
        // typify → labeled Atomistic → Frame (the generic `to_potentials` input).
        let mut frame = self.typify(mol)?.to_frame();
        // The stretch-bend kernel needs each angle's two *reference* bond lengths
        // (r0_ij, r0_kj) — per-bond params, not per-stbn-type — so merge them onto
        // the angles block before compiling.
        merge_stbn_r0(&mut frame, &self.ff)?;
        // The neighbour list is the consumer's concern: build it here.
        let pairs = intramolecular_pairs(&frame);
        frame.insert("pairs", pairs);
        self.ff.to_potentials(&frame)
    }

    /// Classify MMFF bond type: 0=normal, 1=delocalized/aromatic.
    pub fn classify_bond_type(&self, t1: u32, t2: u32, bond_order: f64) -> u32 {
        classify::classify_bond_type(t1, t2, bond_order, &self.params)
    }

    /// Classify MMFF angle type from bond types of the two bonds forming the angle.
    pub fn classify_angle_type(&self, bt_ij: u32, bt_jk: u32) -> u32 {
        classify::classify_angle_type(bt_ij, bt_jk)
    }

    /// Classify MMFF torsion type from the three bond types in the dihedral.
    pub fn classify_torsion_type(&self, bt_ij: u32, bt_jk: u32, bt_kl: u32) -> u32 {
        classify::classify_torsion_type(bt_ij, bt_jk, bt_kl)
    }
}

impl Typifier for MMFFTypifier {
    fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        frame_builder::annotate_mmff(mol, &self.params)
    }
}

/// Write each angle's two reference bond lengths (`r0_ij`, `r0_kj`) onto the
/// `angles` block as per-angle columns the `mmff_stbn` kernel reads.
///
/// The stretch-bend coupling references the *equilibrium* lengths of the i-j and
/// k-j bonds. Those are `mmff_bond` per-bond params (keyed by bond-type label),
/// not per-stretch-bend-type values, so the kernel cannot read them from its own
/// type params — this merges them in from the force field's bond style. No-op if
/// the frame has no `bonds`/`angles` block.
fn merge_stbn_r0(frame: &mut Frame, ff: &ForceField) -> Result<(), String> {
    // r0 per bond-type label, from the mmff_bond style.
    let Some(bond_style) = ff.get_style("bond", "mmff_bond") else {
        return Ok(());
    };
    let r0_by_type: HashMap<String, f64> = bond_style
        .defs
        .collect_type_params()
        .into_iter()
        .filter_map(|(name, p)| p.get("r0").map(|r0| (name, r0)))
        .collect();

    // theta0 (reference angle, degrees) per angle-type label, from mmff_angle —
    // the stretch-bend term references the same equilibrium angle as the bend.
    let theta0_by_type: HashMap<String, f64> = ff
        .get_style("angle", "mmff_angle")
        .map(|s| {
            s.defs
                .collect_type_params()
                .into_iter()
                .filter_map(|(name, p)| p.get("theta0").map(|t| (name, t)))
                .collect()
        })
        .unwrap_or_default();

    // r0 per atom pair (canonical (lo, hi)), via each bond's type label. Collect
    // owned so the frame borrow is released before the mutable insert below.
    let r0_by_pair: HashMap<(u32, u32), f64> = {
        let Some(bonds) = frame.get("bonds") else {
            return Ok(());
        };
        let (Some(bi), Some(bj), Some(bt)) = (
            bonds.get_uint("atomi"),
            bonds.get_uint("atomj"),
            bonds.get_string("type"),
        ) else {
            return Ok(());
        };
        (0..bi.len())
            .filter_map(|idx| {
                let (a, b) = (bi[idx], bj[idx]);
                let key = if a < b { (a, b) } else { (b, a) };
                r0_by_type.get(&bt[idx]).map(|&r0| (key, r0))
            })
            .collect()
    };

    let angle_rows: Vec<(u32, u32, u32, String)> = {
        let Some(angles) = frame.get("angles") else {
            return Ok(());
        };
        let (Some(ai), Some(aj), Some(ak), Some(at)) = (
            angles.get_uint("atomi"),
            angles.get_uint("atomj"),
            angles.get_uint("atomk"),
            angles.get_string("type"),
        ) else {
            return Ok(());
        };
        (0..ai.len())
            .map(|i| (ai[i], aj[i], ak[i], at[i].clone()))
            .collect()
    };

    let pair = |x: u32, y: u32| if x < y { (x, y) } else { (y, x) };
    let r0_ij: Vec<F> = angle_rows
        .iter()
        .map(|(i, j, _, _)| *r0_by_pair.get(&pair(*i, *j)).unwrap_or(&0.0) as F)
        .collect();
    let r0_kj: Vec<F> = angle_rows
        .iter()
        .map(|(_, j, k, _)| *r0_by_pair.get(&pair(*k, *j)).unwrap_or(&0.0) as F)
        .collect();
    let theta0: Vec<F> = angle_rows
        .iter()
        .map(|(_, _, _, ty)| *theta0_by_type.get(ty).unwrap_or(&0.0) as F)
        .collect();

    if let Some(angles) = frame.get_mut("angles") {
        angles
            .insert("r0_ij", Array1::from_vec(r0_ij).into_dyn())
            .map_err(|e| e.to_string())?;
        angles
            .insert("r0_kj", Array1::from_vec(r0_kj).into_dyn())
            .map_err(|e| e.to_string())?;
        angles
            .insert("theta0", Array1::from_vec(theta0).into_dyn())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
