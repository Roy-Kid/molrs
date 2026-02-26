//! WASM bindings for molrs using FFI handle-based architecture.
//!
//! This module provides an object-oriented API for JavaScript/TypeScript
//! using stable handles from molrs-ffi.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

// Module declarations
mod core;
mod io;

// Re-exports following molrs-core layout.
pub use core::{Block, Box, ColumnView, Frame, WasmArray};
pub use io::*;
