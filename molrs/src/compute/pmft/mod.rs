//! Pair Mode Fourier Transform (PMFT) analyzers.
//!
//! Family of 2-D / 3-D pair distribution histograms in fixed reference
//! frames, ported from `freud.pmft`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/)).
//!
//! Currently implemented:
//! - [`PMFTXY`](xy::PMFTXY) — 2-D `(x, y)` PMF (lab frame).
//!
//! Other variants (R12, XYT, XYZ) and the orientation-rotated forms will
//! land in follow-up phases. The 2-D / 3-D thread-local accumulator
//! pattern needed for full PMFTXYZ-with-orientations is encapsulated by
//! this module's eventual `base.rs`.

pub mod r12;
pub mod xy;
pub mod xyt;
pub mod xyz;

pub use r12::{PMFTR12, PMFTR12Args, PMFTR12Result};
pub use xy::{PMFTXY, PMFTXYArgs, PMFTXYResult};
pub use xyt::{PMFTXYT, PMFTXYTArgs, PMFTXYTResult};
pub use xyz::{PMFTXYZ, PMFTXYZArgs, PMFTXYZResult};
