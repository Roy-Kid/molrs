use std::collections::HashMap;

use crate::core::forcefield::{ForceField, Params, Style, StyleCategory};
use crate::core::frame::Frame;

use super::device::CUDADevice;
use super::kernels;
use super::potential::{GPUPotential, GPUPotentials, Topology};

/// Constructor function signature for GPU kernels.
pub type GPUKernelConstructor = fn(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
    device: &CUDADevice,
) -> Result<(Box<dyn GPUPotential>, Topology), String>;

/// Registry mapping `(category, style_name)` → [`GPUKernelConstructor`].
pub struct GPUKernelRegistry {
    map: HashMap<(String, String), GPUKernelConstructor>,
}

impl GPUKernelRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            map: HashMap::new(),
        };
        reg.register_builtins();
        reg
    }

    pub fn register(&mut self, category: &str, name: &str, ctor: GPUKernelConstructor) {
        self.map
            .insert((category.to_owned(), name.to_owned()), ctor);
    }

    pub fn get(&self, category: &str, name: &str) -> Option<&GPUKernelConstructor> {
        self.map.get(&(category.to_owned(), name.to_owned()))
    }

    fn register_builtins(&mut self) {
        self.register("bond", "harmonic", kernels::bond_harmonic_gpu_ctor);
        self.register("angle", "harmonic", kernels::angle_harmonic_gpu_ctor);
        self.register("pair", "lj/cut", kernels::pair_lj126_gpu_ctor);
    }
}

impl Default for GPUKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ForceField → GPUPotentials conversion
// ---------------------------------------------------------------------------

/// Convert a ForceField into GPUPotentials using the given registry and device.
pub fn forcefield_to_gpu_potentials(
    ff: &ForceField,
    registry: &GPUKernelRegistry,
    frame: &Frame,
    device: &CUDADevice,
) -> Result<GPUPotentials, String> {
    let mut potentials = GPUPotentials::new();

    for style in ff.styles() {
        let category = style.category.as_str();
        if let Some(ctor) = registry.get(category, &style.name) {
            let type_params = collect_type_params(style);
            let type_refs: Vec<(&str, &Params)> = type_params
                .iter()
                .map(|(name, params)| (name.as_str(), params))
                .collect();
            let (pot, topo) = ctor(&style.params, &type_refs, frame, device)?;
            potentials.push(pot, topo);
        }
    }

    Ok(potentials)
}

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
