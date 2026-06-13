//! Public options for 3D generation.

/// Stage-1 embedding algorithm selector.
///
/// Names are algorithm-based (not toolkit-based), so backends can evolve
/// without coupling public API names to a specific implementation source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformerAlgorithm {
    /// Rule- and fragment-based coordinate construction.
    ///
    /// This is the algorithm family currently implemented in this crate.
    FragmentRules,
    /// Distance-geometry based embedding.
    DistanceGeometry,
}

/// Force-field backend selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceFieldKind {
    /// Merck Molecular Force Field 94.
    MMFF94,
    /// Universal Force Field.
    Uff,
    /// Prefer MMFF94, fall back to UFF.
    Auto,
}

/// Preset quality/speed profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformerSpeed {
    /// Short minimization and fewer rotor trials.
    Fast,
    /// Balanced defaults.
    Medium,
    /// More rotor trials and longer minimization.
    Better,
}

/// Options for [`super::generate_3d`].
#[derive(Debug, Clone)]
pub struct ConformerOptions {
    /// Stage-1 embedding algorithm.
    pub algorithm: ConformerAlgorithm,
    /// Target force-field family (or auto selection).
    pub forcefield: ForceFieldKind,
    /// Throughput/quality preset.
    pub speed: ConformerSpeed,
    /// Add explicit hydrogens before generation.
    pub add_hydrogens: bool,
    /// Total optimization budget. `0` means "use speed preset default".
    pub max_steps: usize,
    /// Optional deterministic RNG seed.
    pub rng_seed: Option<u64>,
}

impl Default for ConformerOptions {
    fn default() -> Self {
        Self {
            algorithm: ConformerAlgorithm::FragmentRules,
            forcefield: ForceFieldKind::Auto,
            speed: ConformerSpeed::Medium,
            add_hydrogens: true,
            max_steps: 0,
            rng_seed: None,
        }
    }
}

impl ConformerOptions {
    /// Effective optimization step budget.
    pub(crate) fn effective_max_steps(&self) -> usize {
        if self.max_steps > 0 {
            return self.max_steps;
        }
        match self.speed {
            ConformerSpeed::Fast => 120,
            ConformerSpeed::Medium => 260,
            ConformerSpeed::Better => 520,
        }
    }

    /// Coarse minimization steps.
    pub(crate) fn coarse_steps(&self) -> usize {
        (self.effective_max_steps() * 35 / 100).max(20)
    }

    /// Final minimization steps.
    pub(crate) fn final_steps(&self) -> usize {
        (self.effective_max_steps() * 50 / 100).max(20)
    }

    /// Rotor search attempts.
    pub(crate) fn rotor_attempts(&self, n_rot_bonds: usize) -> usize {
        if n_rot_bonds == 0 {
            return 0;
        }
        match self.speed {
            ConformerSpeed::Fast => (n_rot_bonds * 4).max(8),
            ConformerSpeed::Medium => (n_rot_bonds * 8).max(20),
            ConformerSpeed::Better => (n_rot_bonds * 16).max(40),
        }
    }

    /// Maximum per-step rotor perturbation (radians).
    pub(crate) fn rotor_max_delta(&self) -> f64 {
        match self.speed {
            ConformerSpeed::Fast => std::f64::consts::PI / 5.0,
            ConformerSpeed::Medium => std::f64::consts::PI / 3.0,
            ConformerSpeed::Better => std::f64::consts::PI / 2.0,
        }
    }

    // --- ETKDG-internal knobs ------------------------------------------------
    //
    // The public `ConformerOptions` shape is intentionally frozen (constructed by
    // `molrs-python`); the ETKDG pipeline reinterprets the existing fields
    // internally and supplies ETKDG defaults for the rest. `max_steps` doubles
    // as the explicit `maxIterations` override (0 = RDKit's `10×n_atoms`
    // heuristic).

    /// ETKDG `maxIterations` override. `max_steps == 0` means "use the RDKit
    /// `10 × n_atoms` heuristic" (resolved in `etkdg::retry`).
    pub(crate) fn max_iterations_internal(&self) -> usize {
        self.max_steps
    }

    /// Whether the `useRandomCoords` fallback embedding is allowed. Always on
    /// internally (matches RDKit's default retry behavior on hard cases).
    pub(crate) fn use_random_coords_fallback_internal(&self) -> bool {
        true
    }

    /// Whether the second-stage MMFF94 cleanup minimization runs. Always on
    /// internally (the spec's stage-3 contract).
    pub(crate) fn mmff_cleanup_internal(&self) -> bool {
        true
    }
}
