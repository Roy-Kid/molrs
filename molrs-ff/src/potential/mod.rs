//! Potential energy evaluation traits and kernel registry.
//!
//! A [`Potential`] stores pre-resolved topology indices and parameters.
//! Callers pass only flat coordinates — no [`Frame`] in the hot loop.
//! Construction from a [`Frame`] happens once via [`ForceField::to_potentials`].

pub mod geometry;

pub mod angle;
pub mod bond;
pub mod dihedral;
pub mod improper;
pub mod kspace;
pub mod pair;
pub mod registry;

pub use registry::{KernelConstructor, KernelRegistry, lookup_kernel, register_kernel};

/// Backward-compatible re-exports for existing consumers.
pub mod kernels {
    pub use super::angle::harmonic::{AngleHarmonic, angle_harmonic_ctor};
    pub use super::bond::harmonic::{BondHarmonic, bond_harmonic_ctor};
    pub use super::pair::lj_cut::{PairLJCut as PairLJ126, pair_lj_cut_ctor as pair_lj126_ctor};
}

use crate::forcefield::{ForceField, Params};
use molrs::store::frame::Frame;
use molrs::types::F;

// ---------------------------------------------------------------------------
// Potential trait
// ---------------------------------------------------------------------------

/// Interface for computing potential energy and forces.
///
/// A `Potential` is **molecule-bound**: its per-element parameters are expanded
/// against the molecule's topology once at [`ForceField::to_potentials`] (string
/// type labels resolved to per-bond/angle/… arrays). Evaluation therefore takes
/// only coordinates — there is no per-call topology resolution.
///
/// Implementors provide [`calc_energy_forces`](Potential::calc_energy_forces)
/// (both in one pass, avoiding redundant geometry); [`calc_energy`] and
/// [`calc_forces`] default to it.
///
/// [`calc_energy`]: Potential::calc_energy
/// [`calc_forces`]: Potential::calc_forces
pub trait Potential: Send + Sync {
    /// Compute energy and forces (= -gradient) in one pass.
    /// Returns `(energy, forces)` where forces has length `coords.len()`.
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>);

    /// Compute total potential energy (kcal/mol).
    fn calc_energy(&self, coords: &[F]) -> F {
        self.calc_energy_forces(coords).0
    }

    /// Compute forces (= -gradient), a length-3N vector.
    fn calc_forces(&self, coords: &[F]) -> Vec<F> {
        self.calc_energy_forces(coords).1
    }
}

// ---------------------------------------------------------------------------
// Potentials collection
// ---------------------------------------------------------------------------

/// Aggregates multiple potentials; energy/forces are summed.
pub struct Potentials {
    inner: Vec<Box<dyn Potential>>,
    /// Number of atoms the kernels were compiled against (`coords.len() / 3`).
    /// `0` when unknown (e.g. built incrementally via [`Potentials::push`]).
    n_atoms: usize,
}

impl std::fmt::Debug for Potentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Potentials")
            .field("len", &self.inner.len())
            .finish()
    }
}

impl Potentials {
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            n_atoms: 0,
        }
    }

    pub fn push(&mut self, pot: Box<dyn Potential>) {
        self.inner.push(pot);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Number of atoms the kernels were compiled against, i.e. the expected
    /// `coords.len() / 3`. Returns `0` if unknown (built via [`push`]).
    ///
    /// [`push`]: Potentials::push
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Record the compiled atom count (used by [`ForceField::to_potentials`]).
    pub fn set_n_atoms(&mut self, n_atoms: usize) {
        self.n_atoms = n_atoms;
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Compute total energy and forces in one pass over all potentials.
    pub fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n = coords.len();
        let mut total_e: F = 0.0;
        let mut total_f = vec![0.0; n];

        for p in &self.inner {
            let (e, f) = p.calc_energy_forces(coords);
            total_e += e;
            for (t, fi) in total_f.iter_mut().zip(f.iter()) {
                *t += fi;
            }
        }

        (total_e, total_f)
    }

    /// Total potential energy (kcal/mol).
    pub fn calc_energy(&self, coords: &[F]) -> F {
        self.calc_energy_forces(coords).0
    }

    /// Total forces (= -gradient), length 3N.
    pub fn calc_forces(&self, coords: &[F]) -> Vec<F> {
        self.calc_energy_forces(coords).1
    }
}

impl Default for Potentials {
    fn default() -> Self {
        Self::new()
    }
}

/// Make the aggregate usable wherever a single [`Potential`] is expected (e.g.
/// the geometry optimizer in [`crate::optimize`]), forwarding to the summed
/// evaluation over all kernels.
impl Potential for Potentials {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        Potentials::calc_energy_forces(self, coords)
    }
}

// ---------------------------------------------------------------------------
// Style -> Potential construction (OOP — replaces the kernel-registry free-fn map)
// ---------------------------------------------------------------------------

impl crate::forcefield::Style {
    /// Build this style's molecule-bound [`Potential`] by **expanding** its type
    /// parameters against `frame`'s topology — each bond/angle/… row's string
    /// type label is resolved to its parameters and stored as per-element
    /// arrays, so the resulting potential evaluates from coordinates alone.
    ///
    /// Returns `Ok(None)` for a style that carries no pairwise kernel (an atom
    /// style — types/charges only), `Err` for an unknown `(category, name)`.
    ///
    /// The `(category, name)` → constructor mapping lives in the
    /// [`registry`](crate::potential::registry); a new potential is added by
    /// registering its kernel, not by editing this dispatch.
    pub fn to_potential(&self, frame: &Frame) -> Result<Option<Box<dyn Potential>>, String> {
        let category = self.category();
        if category == "atom" {
            return Ok(None);
        }
        let type_params = self.defs.collect_type_params();
        if type_params.is_empty() {
            return Err(format!(
                "Style '{}' ({}) has no type definitions",
                self.name, category
            ));
        }
        let type_refs: Vec<(&str, &Params)> = type_params
            .iter()
            .map(|(name, params)| (name.as_str(), params))
            .collect();
        let ctor = registry::lookup_kernel(category, &self.name).ok_or_else(|| {
            format!(
                "no kernel for style category '{}' name '{}'",
                category, self.name
            )
        })?;
        let pot = ctor(&self.params, &type_refs, frame)?;
        Ok(Some(pot))
    }
}

// ---------------------------------------------------------------------------
// Frame helpers
// ---------------------------------------------------------------------------

/// Extract flat coordinate vector from Frame's `"atoms"` block.
///
/// Reads `"x"`, `"y"`, `"z"` float columns.
/// Returns `[x0,y0,z0, x1,y1,z1, ...]` as `Vec<F>`.
pub fn extract_coords(frame: &Frame) -> Result<Vec<F>, String> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "Frame has no \"atoms\" block".to_string())?;

    let (x, y, z) = (
        atoms.get_float("x"),
        atoms.get_float("y"),
        atoms.get_float("z"),
    );
    let (Some(x), Some(y), Some(z)) = (x, y, z) else {
        return Err("atoms block missing x/y/z float columns".into());
    };

    let xs: Vec<F> = x.iter().copied().collect();
    let ys: Vec<F> = y.iter().copied().collect();
    let zs: Vec<F> = z.iter().copied().collect();

    let n = xs.len();
    if ys.len() != n || zs.len() != n {
        return Err("atoms x/y/z columns have mismatched lengths".into());
    }

    let mut coords = Vec::with_capacity(n * 3);
    for i in 0..n {
        coords.push(xs[i]);
        coords.push(ys[i]);
        coords.push(zs[i]);
    }
    Ok(coords)
}

impl ForceField {
    /// Build evaluable [`Potentials`] by expanding every style against a
    /// typed [`Frame`].
    ///
    /// Each style's `to_potential` resolves its string type labels to per-element
    /// parameter arrays (see [`Style::to_potential`](crate::forcefield::Style::to_potential)),
    /// so the resulting potentials are **molecule-bound**: they retain no Frame
    /// and evaluate from coordinates alone. Styles with no kernel (atom styles)
    /// are skipped. This is the molpy-style `ForceField → Potentials` conversion;
    /// there is no separate "compile" step.
    pub fn to_potentials(&self, frame: &Frame) -> Result<Potentials, String> {
        let mut pots = Potentials::new();
        for style in self.styles() {
            // A style whose topology block is entirely absent contributes nothing
            // (the molecule simply has no bonds/angles/… of that kind) — skip it,
            // rather than error. A *present* block with an unknown type label is a
            // real error and still propagates from the kernel constructor.
            let block = match style.category() {
                "bond" => Some("bonds"),
                "angle" => Some("angles"),
                "dihedral" => Some("dihedrals"),
                "improper" => Some("impropers"),
                "pair" => Some("pairs"),
                _ => None,
            };
            if let Some(b) = block
                && frame.get(b).is_none()
            {
                continue;
            }
            if let Some(pot) = style.to_potential(frame)? {
                pots.push(pot);
            }
        }
        // Record the atom count so callers (e.g. the geometry optimizer's batch
        // path) can validate coordinate shapes against this topology.
        pots.set_n_atoms(frame.get("atoms").and_then(|b| b.nrows()).unwrap_or(0));
        Ok(pots)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::store::block::Block;
    use molrs::types::U;
    use ndarray::Array1;

    struct DummyPotential {
        value: F,
    }

    impl Potential for DummyPotential {
        fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
            (self.value, vec![self.value; coords.len()])
        }
    }

    fn make_atoms_only_frame() -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0 as F, 2.0 as F]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);
        frame
    }

    fn make_bond_frame() -> Frame {
        let mut frame = make_atoms_only_frame();
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as U]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1 as U]).into_dyn())
            .unwrap();
        bonds
            .insert("type", Array1::from_vec(vec!["A-A".to_string()]).into_dyn())
            .unwrap();
        frame.insert("bonds", bonds);
        frame
    }

    fn make_lj_frame() -> Frame {
        let mut frame = make_atoms_only_frame();

        let mut pairs = Block::new();
        pairs
            .insert("atomi", Array1::from_vec(vec![0 as U]).into_dyn())
            .unwrap();
        pairs
            .insert("atomj", Array1::from_vec(vec![1 as U]).into_dyn())
            .unwrap();
        pairs
            .insert("type", Array1::from_vec(vec!["A".to_string()]).into_dyn())
            .unwrap();
        frame.insert("pairs", pairs);

        frame
    }

    #[test]
    fn test_potentials_collection() {
        let mut pots = Potentials::new();
        pots.push(Box::new(DummyPotential { value: 1.0 }));
        pots.push(Box::new(DummyPotential { value: 2.0 }));

        assert_eq!(pots.len(), 2);

        let coords: Vec<F> = vec![0.0; 6];
        assert!((pots.calc_energy(&coords) - 3.0).abs() < 1e-5);

        let forces = pots.calc_forces(&coords);
        for f in &forces {
            assert!((*f - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn unknown_style_kernel_is_error() {
        // A style whose (category, name) has no kernel -> Err from to_potential.
        let mut ff = ForceField::new("test");
        ff.def_bondstyle("nonexistent")
            .def_type("A-A", &[("k", 1.0)]);
        let frame = make_bond_frame();
        let err = ff.to_potentials(&frame).unwrap_err();
        assert!(err.contains("no kernel"), "{err}");
    }

    #[test]
    fn register_kernel_extends_dispatch() {
        // A custom (category, name) with no built-in kernel becomes usable by
        // registering its constructor — no edit to to_potential required.
        fn my_ctor(
            _sp: &Params,
            _tp: &[(&str, &Params)],
            _f: &Frame,
        ) -> Result<Box<dyn Potential>, String> {
            Ok(Box::new(DummyPotential { value: 42.0 }))
        }
        register_kernel("pair", "test/custom", my_ctor);

        let mut ff = ForceField::new("test");
        ff.def_pairstyle("test/custom", &[]).def_type("A", &[]);
        let frame = make_lj_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();
        // the custom kernel ran: DummyPotential yields its constant value.
        assert!((pots.calc_energy(&coords) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn atom_style_is_skipped() {
        // Atom styles carry types/charges, not a pairwise kernel -> skipped.
        let ff = ForceField::new("test").with_atomstyle("full");
        let frame = make_atoms_only_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        assert_eq!(pots.len(), 0);
    }

    #[test]
    fn test_compile_requires_types() {
        let mut ff = ForceField::new("test");
        ff.def_bondstyle("harmonic");
        let frame = make_bond_frame();
        let err = ff
            .to_potentials(&frame)
            .expect_err("expected compile to fail");
        assert!(err.contains("has no type definitions"));
    }

    #[test]
    fn test_compile_energy() {
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let frame = make_lj_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();

        let (energy, _) = pots.calc_energy_forces(&coords);
        let expected: F = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
        assert!((energy - expected).abs() < 1e-5);
    }

    #[test]
    fn test_compile_forces() {
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let frame = make_lj_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();

        let (_, forces) = pots.calc_energy_forces(&coords);

        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-5);
        }
    }

    #[test]
    fn test_compile_empty_ff() {
        let ff = ForceField::new("test");
        let frame = make_lj_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();

        let (energy, forces) = pots.calc_energy_forces(&coords);
        assert!(energy.abs() < 1e-5);
        assert_eq!(forces.len(), 6);
        assert!(forces.iter().all(|x| x.abs() < 1e-5));
    }

    #[test]
    fn test_compile_skips_absent_topology() {
        // A style whose topology block is absent from the frame contributes
        // nothing (no rows of that kind) — skipped, not an error.
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let frame = make_atoms_only_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        assert_eq!(pots.len(), 0);
        let coords = extract_coords(&frame).unwrap();
        assert!(pots.calc_energy(&coords).abs() < 1e-9);
    }
}
