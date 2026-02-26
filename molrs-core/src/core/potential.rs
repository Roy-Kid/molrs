//! Potential energy evaluation traits and kernel registry.
//!
//! A [`Potential`] computes energy and gradient from atomic coordinates.
//! Indices are pre-bound into each kernel at construction time from the
//! [`Frame`]'s topology blocks. The [`KernelRegistry`] maps
//! `(category, style_name)` → constructor so that
//! [`ForceField::to_potentials`](super::forcefield::ForceField::to_potentials)
//! can build the right kernel from a declarative style definition.

use super::forcefield::{ForceField, Params, Style, StyleCategory};
use super::frame::Frame;
use crate::core::types::F;

// ---------------------------------------------------------------------------
// Potential trait
// ---------------------------------------------------------------------------

/// Interface for computing potential energy and forces.
///
/// Forces are returned as a new `Vec<F>` (= −∇E), making each kernel
/// independently parallelisable with no shared mutable state.
/// Indices are pre-bound into the kernel — no topology parameter needed.
pub trait Potential: Send + Sync {
    /// Compute total potential energy.
    fn energy(&self, coords: &[F]) -> F;

    /// Compute forces (= −gradient) and return a length-3N vector.
    fn forces(&self, coords: &[F]) -> Vec<F>;
}

// ---------------------------------------------------------------------------
// Potentials collection
// ---------------------------------------------------------------------------

/// Aggregates multiple potentials; energy/forces are summed.
pub struct Potentials {
    inner: Vec<Box<dyn Potential>>,
}

impl Potentials {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn push(&mut self, pot: Box<dyn Potential>) {
        self.inner.push(pot);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn energy(&self, coords: &[F]) -> F {
        self.inner.iter().map(|p| p.energy(coords)).sum()
    }

    /// Compute total forces from all potentials.
    /// Each potential returns an independent Vec, then they are summed.
    /// With the `rayon` feature, potentials are evaluated in parallel.
    pub fn forces(&self, coords: &[F]) -> Vec<F> {
        let n = coords.len();

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let partials: Vec<Vec<F>> = self.inner.par_iter().map(|p| p.forces(coords)).collect();

            let mut total = vec![0.0; n];
            for pf in &partials {
                for (t, f) in total.iter_mut().zip(pf.iter()) {
                    *t += f;
                }
            }
            total
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut total = vec![0.0; n];
            for p in &self.inner {
                let f = p.forces(coords);
                for (t, fi) in total.iter_mut().zip(f.iter()) {
                    *t += fi;
                }
            }
            total
        }
    }
}

impl Default for Potentials {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Kernel registry
// ---------------------------------------------------------------------------

/// Constructor function signature: takes style-level params, per-type params,
/// and the Frame (topology + atom data), returns a boxed Potential.
pub type KernelConstructor = fn(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String>;

/// Registry mapping `(category, style_name)` → [`KernelConstructor`].
///
/// Stored as a flat `Vec` rather than a `HashMap`; the registry is small
/// (typically < 20 entries) so a linear scan with `&str` comparison is
/// cache-friendly and avoids any heap allocation on lookup.
pub struct KernelRegistry {
    entries: Vec<(String, String, KernelConstructor)>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            entries: Vec::new(),
        };
        reg.register_builtins();
        reg
    }

    pub fn register(&mut self, category: &str, name: &str, ctor: KernelConstructor) {
        self.entries
            .push((category.to_owned(), name.to_owned(), ctor));
    }

    /// Look up a constructor by category and name. No heap allocation.
    pub fn get(&self, category: &str, name: &str) -> Option<&KernelConstructor> {
        self.entries
            .iter()
            .find(|(c, n, _)| c == category && n == name)
            .map(|(_, _, ctor)| ctor)
    }

    fn register_builtins(&mut self) {
        self.register(
            "bond",
            "harmonic",
            super::potential_kernels::bond_harmonic_ctor,
        );
        self.register(
            "angle",
            "harmonic",
            super::potential_kernels::angle_harmonic_ctor,
        );
        self.register("pair", "lj/cut", super::potential_kernels::pair_lj126_ctor);
        self.register("electrostatic", "pme", super::pme::pme_ctor);
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ForceField → Potentials conversion
// ---------------------------------------------------------------------------

impl ForceField {
    /// Convert all styles into computational [`Potential`] objects using the
    /// given registry and Frame. Styles without a matching kernel are skipped.
    pub fn to_potentials_with(
        &self,
        registry: &KernelRegistry,
        frame: &Frame,
    ) -> Result<Potentials, String> {
        let mut potentials = Potentials::new();

        for style in self.styles() {
            let category = style.category.as_str();
            if let Some(ctor) = registry.get(category, &style.name) {
                let type_params = collect_type_params(style);
                let type_refs: Vec<(&str, &Params)> = type_params
                    .iter()
                    .map(|(name, params)| (name.as_str(), params))
                    .collect();
                let pot = ctor(&style.params, &type_refs, frame)?;
                potentials.push(pot);
            }
            // Styles without a kernel are silently skipped (e.g. AtomStyle)
        }

        Ok(potentials)
    }

    /// Convert using the default registry (built-in kernels) with the given Frame.
    pub fn to_potentials_from_frame(&self, frame: &Frame) -> Result<Potentials, String> {
        self.to_potentials_with(&KernelRegistry::default(), frame)
    }
}

/// Extract flat coordinate vector from Frame's `"atoms"` block.
///
/// Reads `"x"`, `"y"`, `"z"` columns (f64 preferred, f32 converted).
/// Returns `[x0,y0,z0, x1,y1,z1, ...]` as `Vec<F>`.
pub fn extract_coords(frame: &Frame) -> Result<Vec<F>, String> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "Frame has no \"atoms\" block".to_string())?;

    let (xs, ys, zs): (Vec<F>, Vec<F>, Vec<F>);

    if let (Some(x), Some(y), Some(z)) =
        (atoms.get_f64("x"), atoms.get_f64("y"), atoms.get_f64("z"))
    {
        xs = x.iter().map(|&v| v as F).collect();
        ys = y.iter().map(|&v| v as F).collect();
        zs = z.iter().map(|&v| v as F).collect();
    } else if let (Some(x), Some(y), Some(z)) =
        (atoms.get_f32("x"), atoms.get_f32("y"), atoms.get_f32("z"))
    {
        xs = x.iter().map(|&v| v as F).collect();
        ys = y.iter().map(|&v| v as F).collect();
        zs = z.iter().map(|&v| v as F).collect();
    } else {
        return Err("atoms block missing x/y/z columns (f64 or f32)".into());
    }

    let n = xs.len();
    let mut coords = Vec::with_capacity(n * 3);
    for i in 0..n {
        coords.push(xs[i]);
        coords.push(ys[i]);
        coords.push(zs[i]);
    }
    Ok(coords)
}

// ---------------------------------------------------------------------------
// StyleRecipe — one style's recipe for building a Potential
// ---------------------------------------------------------------------------

/// A style's compiled recipe: kernel constructor + type parameters.
struct StyleRecipe {
    style_params: Params,
    type_params: Vec<(String, Params)>,
    ctor: KernelConstructor,
}

// ---------------------------------------------------------------------------
// PotentialSet — the "recipe" returned by ff.to_potentials()
// ---------------------------------------------------------------------------

/// A compiled set of potential recipes from a ForceField.
///
/// Created by [`ForceField::to_potentials()`]. Does not hold atom indices —
/// those are extracted from the Frame at evaluation time.
///
/// # Usage
///
/// ```no_run
/// # use molrs::core::forcefield::ForceField;
/// # use molrs::core::frame::Frame;
/// let ff = ForceField::new("test");
/// let pots = ff.to_potentials();
/// // pots.energy(&frame) extracts topology + coords from frame
/// ```
pub struct PotentialSet {
    recipes: Vec<StyleRecipe>,
}

impl PotentialSet {
    /// Bind the recipes to a Frame's topology, producing pre-bound [`Potentials`]
    /// for efficient repeated evaluation (e.g. in a minimizer hot loop).
    ///
    /// # Hot-loop pattern
    ///
    /// ```no_run
    /// # use molrs::core::forcefield::ForceField;
    /// # use molrs::core::frame::Frame;
    /// # use molrs::core::potential::extract_coords;
    /// # let ff = ForceField::new("test");
    /// # let frame = Frame::new();
    /// let pots = ff.to_potentials();
    /// // Bind once outside the loop:
    /// let bound = pots.bind(&frame).unwrap();
    /// let mut coords = extract_coords(&frame).unwrap();
    /// // Then iterate cheaply:
    /// // loop { let e = bound.energy(&coords); ... }
    /// ```
    pub fn bind(&self, frame: &Frame) -> Result<Potentials, String> {
        let mut potentials = Potentials::new();

        for recipe in &self.recipes {
            let type_refs: Vec<(&str, &Params)> = recipe
                .type_params
                .iter()
                .map(|(name, params)| (name.as_str(), params))
                .collect();
            let pot = (recipe.ctor)(&recipe.style_params, &type_refs, frame)?;
            potentials.push(pot);
        }

        Ok(potentials)
    }

    /// Compute energy **and** forces from a Frame in a single pass.
    ///
    /// Binds topology and extracts coordinates exactly once, then evaluates
    /// both quantities. Use this when you need both results; for hot loops
    /// prefer [`bind`](Self::bind) + [`extract_coords`] to avoid re-binding
    /// on every call.
    ///
    /// Returns `(energy, forces)` where `forces` is a length-3N vector (= −∇E).
    pub fn eval_frame(&self, frame: &Frame) -> Result<(F, Vec<F>), String> {
        let potentials = self.bind(frame)?;
        let coords = extract_coords(frame)?;
        Ok((potentials.energy(&coords), potentials.forces(&coords)))
    }
}

impl ForceField {
    /// Compile this ForceField into a [`PotentialSet`] — a set of potential
    /// recipes ready to evaluate on any Frame.
    ///
    /// Uses the default built-in kernel registry. Styles without a matching
    /// kernel are silently skipped (e.g. AtomStyle).
    pub fn to_potentials(&self) -> PotentialSet {
        let registry = KernelRegistry::default();
        let mut recipes = Vec::new();

        for style in self.styles() {
            let category = style.category.as_str();
            if let Some(&ctor) = registry.get(category, &style.name) {
                recipes.push(StyleRecipe {
                    style_params: style.params.clone(),
                    type_params: collect_type_params(style),
                    ctor,
                });
            }
        }

        PotentialSet { recipes }
    }
}

// ---------------------------------------------------------------------------
// ForceField → Potentials (low-level, pre-bound)
// ---------------------------------------------------------------------------

/// Collect `(type_name, params)` pairs from a style based on its category.
fn collect_type_params(style: &Style) -> Vec<(String, Params)> {
    match style.category {
        StyleCategory::Bond => style
            .bondtypes
            .iter()
            .map(|t| (t.name.clone(), t.params.clone()))
            .collect(),
        StyleCategory::Angle => style
            .angletypes
            .iter()
            .map(|t| (t.name.clone(), t.params.clone()))
            .collect(),
        StyleCategory::Pair => style
            .pairtypes
            .iter()
            .map(|t| (t.name.clone(), t.params.clone()))
            .collect(),
        StyleCategory::Dihedral => style
            .dihedraltypes
            .iter()
            .map(|t| (t.name.clone(), t.params.clone()))
            .collect(),
        StyleCategory::Improper => style
            .impropertypes
            .iter()
            .map(|t| (t.name.clone(), t.params.clone()))
            .collect(),
        StyleCategory::Atom => style
            .atomtypes
            .iter()
            .map(|t| (t.name.clone(), t.params.clone()))
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyPotential {
        value: F,
    }

    impl Potential for DummyPotential {
        fn energy(&self, _coords: &[F]) -> F {
            self.value
        }
        fn forces(&self, coords: &[F]) -> Vec<F> {
            vec![self.value; coords.len()]
        }
    }

    #[test]
    fn test_potentials_collection() {
        let mut pots = Potentials::new();
        pots.push(Box::new(DummyPotential { value: 1.0 }));
        pots.push(Box::new(DummyPotential { value: 2.0 }));

        assert_eq!(pots.len(), 2);

        let coords: Vec<F> = vec![0.0; 6];
        assert!((pots.energy(&coords) - 3.0).abs() < 1e-5);

        let forces = pots.forces(&coords);
        for f in &forces {
            assert!((*f - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_registry_builtins() {
        let reg = KernelRegistry::default();
        assert!(reg.get("bond", "harmonic").is_some());
        assert!(reg.get("angle", "harmonic").is_some());
        assert!(reg.get("pair", "lj/cut").is_some());
        assert!(reg.get("pair", "nonexistent").is_none());
    }

    #[test]
    fn test_to_potentials_skips_atom_style() {
        let ff = ForceField::new("test").with_atomstyle("full");
        let frame = Frame::new();
        let pots = ff.to_potentials_from_frame(&frame).unwrap();
        assert!(pots.is_empty());
    }

    // --- PotentialSet tests ---

    use super::super::block::Block;
    use ndarray::Array1;

    fn make_lj_frame() -> Frame {
        let mut frame = Frame::new();

        // atoms block with x, y, z
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0_f64, 2.0]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![0.0_f64, 0.0]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(vec![0.0_f64, 0.0]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);

        // pairs block
        let mut pairs = Block::new();
        pairs
            .insert("i", Array1::from_vec(vec![0_u32]).into_dyn())
            .unwrap();
        pairs
            .insert("j", Array1::from_vec(vec![1_u32]).into_dyn())
            .unwrap();
        pairs
            .insert("type", Array1::from_vec(vec!["A".to_string()]).into_dyn())
            .unwrap();
        frame.insert("pairs", pairs);

        frame
    }

    #[test]
    fn test_potential_set_energy() {
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let pots = ff.to_potentials();
        let frame = make_lj_frame();

        let (energy, _) = pots.eval_frame(&frame).unwrap();
        // At r=2.0: E = 4*1*((1/2)^12 - (1/2)^6) = 4*(1/4096 - 1/64)
        let expected: F = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
        assert!(
            (energy - expected).abs() < 1e-5,
            "energy={}, expected={}",
            energy,
            expected
        );
    }

    #[test]
    fn test_potential_set_forces() {
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let pots = ff.to_potentials();
        let frame = make_lj_frame();

        let (_, forces) = pots.eval_frame(&frame).unwrap();

        // Newton's third law: f_i + f_j = 0
        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-5);
        }
    }

    #[test]
    fn test_potential_set_empty_ff() {
        let ff = ForceField::new("test");
        let pots = ff.to_potentials();
        let frame = make_lj_frame();

        let (energy, _) = pots.eval_frame(&frame).unwrap();
        assert!(energy.abs() < 1e-5);
    }

    #[test]
    fn test_potential_set_missing_topology() {
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let pots = ff.to_potentials();

        // Frame with only atoms, no pairs block
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0_f64, 2.0]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![0.0_f64, 0.0]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(vec![0.0_f64, 0.0]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);

        let (energy, _) = pots.eval_frame(&frame).unwrap();
        assert!(energy.abs() < 1e-5);
    }

    #[test]
    fn test_potential_set_bind() {
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let pots = ff.to_potentials();
        let frame = make_lj_frame();

        // bind gives pre-bound Potentials for hot-loop usage
        let bound = pots.bind(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();
        let energy = bound.energy(&coords);

        let expected: F = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
        assert!(
            (energy - expected).abs() < 1e-5,
            "energy={}, expected={}",
            energy,
            expected
        );
    }
}
