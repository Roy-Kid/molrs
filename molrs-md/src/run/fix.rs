use molrs::core::frame::Frame;

use crate::error::MDError;
use crate::run::stage::StageMask;
use crate::run::state::MDState;

/// GPU execution tier for a Fix.
///
/// Each fix declares its own tier. The CUDA dynamics engine reads this
/// to decide how to dispatch the fix — no string matching, no central
/// registry, just ask the fix itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTier {
    /// Has a corresponding GPU kernel. The engine calls the kernel
    /// directly instead of going through the Fix trait methods.
    Kernel,
    /// Read-only access to scalars (e.g. thermo output). Can run
    /// asynchronously after a lightweight D2H of scalars only.
    Async,
    /// Needs full sync: D2H all state → CPU callback → H2D.
    /// This is the safe default — every fix works, just slower.
    Sync,
}

/// A Fix is a plugin that can be inserted into the MD time-step loop.
///
/// Mirrors the LAMMPS fix model: each fix declares which stages it participates
/// in via `stages()`, and the engine calls the corresponding methods.
/// The integrator itself is a fix (e.g. FixNVE = velocity Verlet).
pub trait Fix: Send {
    /// Human-readable name for this fix.
    fn name(&self) -> &str;

    /// Which stages this fix participates in.
    fn stages(&self) -> StageMask;

    /// Declare how this fix should run on a GPU backend.
    /// Default: `Sync` — safe fallback, every fix works.
    fn gpu_tier(&self) -> GpuTier {
        GpuTier::Sync
    }

    /// Called during setup with access to the original Frame.
    /// Useful for fixes that need topology or static data.
    fn setup_with_frame(&mut self, _frame: &Frame, _state: &mut MDState) -> Result<(), MDError> {
        Ok(())
    }

    fn setup(&mut self, _state: &mut MDState) -> Result<(), MDError> {
        Ok(())
    }
    fn initial_integrate(&mut self, _state: &mut MDState) -> Result<(), MDError> {
        Ok(())
    }
    fn post_integrate(&mut self, _state: &mut MDState) -> Result<(), MDError> {
        Ok(())
    }
    fn pre_force(&mut self, _state: &mut MDState) -> Result<(), MDError> {
        Ok(())
    }
    fn post_force(&mut self, _state: &mut MDState) -> Result<(), MDError> {
        Ok(())
    }
    fn final_integrate(&mut self, _state: &mut MDState) -> Result<(), MDError> {
        Ok(())
    }
    fn end_of_step(&mut self, _state: &mut MDState) -> Result<(), MDError> {
        Ok(())
    }
    fn cleanup(&mut self, _state: &mut MDState) -> Result<(), MDError> {
        Ok(())
    }
}
