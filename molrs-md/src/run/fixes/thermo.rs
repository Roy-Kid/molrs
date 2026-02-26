use crate::error::MDError;
use crate::run::fix::{Fix, GpuTier};
use crate::run::stage::StageMask;
use crate::run::state::MDState;

/// Periodic thermodynamic output, equivalent to LAMMPS `thermo N`.
///
/// Prints step, PE, KE, total energy, and temperature every N steps.
pub struct FixThermo {
    every: usize,
}

impl FixThermo {
    pub fn every(n: usize) -> Self {
        FixThermo { every: n.max(1) }
    }
}

impl Fix for FixThermo {
    fn name(&self) -> &str {
        "thermo"
    }

    fn stages(&self) -> StageMask {
        StageMask::END_OF_STEP
    }

    fn gpu_tier(&self) -> GpuTier {
        GpuTier::Async
    }

    fn setup(&mut self, s: &mut MDState) -> Result<(), MDError> {
        // Print header
        println!(
            "{:>10} {:>14} {:>14} {:>14} {:>14}",
            "Step", "PE", "KE", "TotalE", "Temp"
        );
        // Print initial state
        let total = s.pe + s.ke;
        let temp = s.temperature();
        println!(
            "{:>10} {:>14.6} {:>14.6} {:>14.6} {:>14.6}",
            s.step, s.pe, s.ke, total, temp
        );
        Ok(())
    }

    fn end_of_step(&mut self, s: &mut MDState) -> Result<(), MDError> {
        if s.step.is_multiple_of(self.every) {
            let total = s.pe + s.ke;
            let temp = s.temperature();
            println!(
                "{:>10} {:>14.6} {:>14.6} {:>14.6} {:>14.6}",
                s.step, s.pe, s.ke, total, temp
            );
        }
        Ok(())
    }
}
