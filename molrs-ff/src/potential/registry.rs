//! Kernel registry: maps `(category, style_name)` → [`KernelConstructor`].
//!
//! `ForceField::to_potentials` resolves each style's kernel through this
//! registry instead of a hard-coded match, so a new potential is added by
//! *registering* its constructor rather than editing core dispatch. The
//! built-ins are seeded on first use; [`register_kernel`] adds or overrides
//! entries at runtime (the advertised extension point).

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::forcefield::Params;
use crate::potential::Potential;
use molrs::frame::Frame;

use super::{angle, bond, dihedral, improper, kspace, pair};

/// Builds a molecule-bound [`Potential`] from a style's params, its per-type
/// params (`(type_label, params)`), and a typed [`Frame`]. Every kernel
/// constructor in the crate matches this signature.
pub type KernelConstructor =
    fn(&Params, &[(&str, &Params)], &Frame) -> Result<Box<dyn Potential>, String>;

/// Maps `(category, style_name)` to the constructor that builds its potential.
#[derive(Default)]
pub struct KernelRegistry {
    ctors: HashMap<(String, String), KernelConstructor>,
}

impl KernelRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or override) the kernel for `(category, name)`.
    pub fn register(&mut self, category: &str, name: &str, ctor: KernelConstructor) {
        self.ctors
            .insert((category.to_owned(), name.to_owned()), ctor);
    }

    /// The constructor registered for `(category, name)`, if any.
    pub fn get(&self, category: &str, name: &str) -> Option<KernelConstructor> {
        self.ctors
            .get(&(category.to_owned(), name.to_owned()))
            .copied()
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.ctors.len()
    }

    /// Whether the registry has no kernels.
    pub fn is_empty(&self) -> bool {
        self.ctors.is_empty()
    }

    /// A registry seeded with every built-in kernel.
    pub fn builtin() -> Self {
        let mut r = Self::new();
        // bonded
        r.register("bond", "harmonic", bond::harmonic::bond_harmonic_ctor);
        r.register("bond", "class2", bond::class2::bond_class2_ctor);
        r.register("bond", "morse", bond::morse::bond_morse_ctor);
        r.register("angle", "harmonic", angle::harmonic::angle_harmonic_ctor);
        r.register("angle", "class2", angle::class2::angle_class2_ctor);
        r.register("dihedral", "opls", dihedral::opls::dihedral_opls_ctor);
        // pair / nonbonded
        r.register("pair", "lj/cut", pair::lj_cut::pair_lj_cut_ctor);
        r.register("pair", "lj/class2", pair::lj_class2::pair_lj_class2_ctor);
        r.register("pair", "buck", pair::buck::pair_buck_ctor);
        r.register("pair", "morse", pair::morse::pair_morse_ctor);
        r.register("pair", "thole", pair::thole::pair_thole_ctor);
        r.register(
            "pair",
            "coul/tt",
            pair::tang_toennies::pair_tang_toennies_ctor,
        );
        r.register("pair", "coul/cut", pair::coul_cut::pair_coul_cut_ctor);
        // MMFF94
        r.register("bond", "mmff_bond", bond::mmff::mmff_bond_ctor);
        r.register("angle", "mmff_angle", angle::mmff::mmff_angle_ctor);
        r.register("angle", "mmff_stbn", angle::mmff::mmff_stbn_ctor);
        r.register(
            "dihedral",
            "mmff_torsion",
            dihedral::mmff::mmff_torsion_ctor,
        );
        r.register("improper", "mmff_oop", improper::mmff::mmff_oop_ctor);
        r.register("pair", "mmff_vdw", pair::mmff::mmff_vdw_ctor);
        r.register("pair", "mmff_ele", pair::mmff::mmff_ele_ctor);
        // k-space
        r.register("kspace", "pme", kspace::pme::pme_ctor);
        r
    }
}

/// The process-wide kernel registry, initialized with the built-ins on first use.
fn global() -> &'static RwLock<KernelRegistry> {
    static REGISTRY: OnceLock<RwLock<KernelRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(KernelRegistry::builtin()))
}

/// Register (or override) a kernel in the global registry. The extension point
/// for new potentials — no core dispatch edit required.
pub fn register_kernel(category: &str, name: &str, ctor: KernelConstructor) {
    global().write().unwrap().register(category, name, ctor);
}

/// Look up a kernel constructor in the global registry.
pub fn lookup_kernel(category: &str, name: &str) -> Option<KernelConstructor> {
    global().read().unwrap().get(category, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_has_core_kernels() {
        let r = KernelRegistry::builtin();
        assert!(r.get("bond", "harmonic").is_some());
        assert!(r.get("pair", "lj/cut").is_some());
        assert!(r.get("pair", "buck").is_some());
        assert!(r.get("kspace", "pme").is_some());
        assert!(r.get("bond", "does-not-exist").is_none());
    }

    #[test]
    fn register_overrides_and_adds() {
        let mut r = KernelRegistry::new();
        assert!(r.is_empty());
        r.register("pair", "lj/cut", pair::lj_cut::pair_lj_cut_ctor);
        assert_eq!(r.len(), 1);
        assert!(r.get("pair", "lj/cut").is_some());
        // re-registering the same key overrides, not duplicates
        r.register("pair", "lj/cut", pair::buck::pair_buck_ctor);
        assert_eq!(r.len(), 1);
    }
}
