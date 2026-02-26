use molrs::core::forcefield::ForceField;
use molrs::core::potential::PotentialSet;
use molrs::core::types::F;

use crate::backend::DynamicsBackend;
use crate::error::MDError;
use crate::run::dump::Dump;
use crate::run::fix::Fix;

/// Builder for constructing a dynamics engine.
///
/// Usage:
/// ```ignore
/// MD::dynamics()
///     .forcefield(&ff)
///     .dt(0.001)
///     .fix(FixNVE::new())
///     .fix(FixThermo::every(100))
///     .compile::<CPU>(())
/// ```
pub struct DynamicsBuilder {
    potential_set: Option<PotentialSet>,
    dt: F,
    fixes: Vec<Box<dyn Fix>>,
    dumps: Vec<Box<dyn Dump>>,
}

impl DynamicsBuilder {
    pub fn new() -> Self {
        Self {
            potential_set: None,
            dt: 0.001,
            fixes: Vec::new(),
            dumps: Vec::new(),
        }
    }

    /// Set the force field. Internally compiles it into a PotentialSet.
    pub fn forcefield(mut self, ff: &ForceField) -> Self {
        self.potential_set = Some(ff.to_potentials());
        self
    }

    /// Set the timestep (default 0.001).
    pub fn dt(mut self, dt: F) -> Self {
        self.dt = dt;
        self
    }

    /// Add a fix to the dynamics engine.
    pub fn fix(mut self, f: impl Fix + 'static) -> Self {
        self.fixes.push(Box::new(f));
        self
    }

    /// Add a dump to the dynamics engine.
    pub fn dump(mut self, d: impl Dump + 'static) -> Self {
        self.dumps.push(Box::new(d));
        self
    }

    /// Compile the dynamics engine for a specific backend.
    pub fn compile<B: DynamicsBackend>(self, device: B::Device) -> Result<B::Dynamics, MDError> {
        let potential_set = self
            .potential_set
            .ok_or_else(|| MDError::ConfigError("forcefield is required".into()))?;

        B::compile_dynamics(device, potential_set, self.fixes, self.dumps, self.dt)
    }
}

impl Default for DynamicsBuilder {
    fn default() -> Self {
        Self::new()
    }
}
