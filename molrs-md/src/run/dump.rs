use molrs::core::frame::Frame;

use crate::error::MDError;
use crate::run::state::MDState;

/// A Dump is a read-only output plugin for the MD engine.
///
/// Unlike Fix, Dump receives `&MDState` (immutable) — it cannot modify
/// the simulation state. The engine checks `every()` and only calls `write()`
/// on matching steps.
pub trait Dump: Send {
    /// Human-readable name for this dump.
    fn name(&self) -> &str;

    /// Write frequency: the engine calls `write()` every N steps.
    fn every(&self) -> usize;

    /// Called once during setup with access to the original Frame and state.
    fn setup(&mut self, frame: &Frame, state: &MDState) -> Result<(), MDError>;

    /// Write output data. Called by the engine when `step % every() == 0`.
    fn write(&mut self, state: &MDState) -> Result<(), MDError>;

    /// Called once at the end of a run for cleanup (close files, flush, etc.).
    fn cleanup(&mut self) -> Result<(), MDError>;
}
