//! FFI layer for molrs with handle-based abstractions.
//!
//! This crate provides a stable, handle-based API for Python and WASM bindings.
//! It separates the Rust-idiomatic core API from cross-language FFI concerns.

mod error;
#[cfg(feature = "ff")]
mod forcefield;
mod handle;
mod shared;
mod store;

pub use error::FfiError;
#[cfg(feature = "ff")]
pub use forcefield::ForceFieldRef;
pub use handle::{BlockHandle, FrameId};
pub use shared::{BlockRef, FrameRef, SharedStore, new_shared};
pub use store::Store;
