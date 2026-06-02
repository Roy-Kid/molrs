//! Façade wiring smoke tests.
//!
//! `molrs` re-exports the workspace sub-crates behind feature flags. These tests
//! assert only that the re-export wiring is reachable under each feature — the
//! sub-crates' real behavior is tested in their own suites. Each block is
//! `#[cfg]`-gated, so the target compiles (empty) when a feature is off.
//!
//! Run the full surface like CI does:
//! `cargo test -p molcrafts-molrs --tests --features "full filesystem"`.

/// Core types are re-exported at the top level regardless of features.
#[test]
fn core_types_reexported_at_top_level() {
    let frame = molrs::Frame::new();
    assert!(frame.get("atoms").is_none());
}

// For the feature-gated sub-crates, importing the re-exported module with
// `use … as _;` is the wiring smoke: it compiles iff the re-export resolves,
// and names no internal items (so it can't drift with their APIs).

#[cfg(feature = "io")]
#[test]
fn io_module_reexported() {
    use molrs::io as _;
}

#[cfg(feature = "smiles")]
#[test]
fn smiles_module_reexported() {
    // `smiles` is a module re-export (`pub use molrs_io::smiles`), not a crate
    // alias like the others, so `use … as _;` trips `unused_imports`. The unused
    // import IS the wiring smoke (it compiles iff the re-export resolves); allow it.
    #[allow(unused_imports)]
    use molrs::smiles as _;
}

#[cfg(feature = "compute")]
#[test]
fn compute_module_reexported() {
    use molrs::compute as _;
}

#[cfg(feature = "ff")]
#[test]
fn ff_module_reexported() {
    use molrs::ff as _;
}

#[cfg(feature = "embed")]
#[test]
fn embed_module_reexported() {
    use molrs::embed as _;
}

#[cfg(feature = "signal")]
#[test]
fn signal_module_reexported() {
    use molrs::signal as _;
}
