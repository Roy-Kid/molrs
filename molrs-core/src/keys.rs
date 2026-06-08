//! Canonical molecular field-name convention — the single source of truth for
//! component (property) keys.
//!
//! ECS components are addressed by name. To keep field names from being
//! hard-coded as scattered string literals, **all** code references the
//! constants defined here rather than writing `"x"` / `"element"` inline. A
//! field is renamed in exactly one place. These names are kept in sync with
//! molpy's `core/fields.py` canonical field set so the two layers agree at the
//! boundary.
//!
//! # Examples
//!
//! ```
//! use molrs_core::keys;
//!
//! assert_eq!(keys::X, "x");
//! assert_eq!(keys::COORDS, [keys::X, keys::Y, keys::Z]);
//! ```

/// Cartesian x-coordinate component.
pub const X: &str = "x";
/// Cartesian y-coordinate component.
pub const Y: &str = "y";
/// Cartesian z-coordinate component.
pub const Z: &str = "z";

/// The three Cartesian coordinate component keys, in axis order.
///
/// Geometry systems (`translate`, `rotate`, …) read coordinates through this
/// constant rather than literal `"x"`/`"y"`/`"z"`.
pub const COORDS: [&str; 3] = [X, Y, Z];

/// Element symbol of an atom (e.g. `"C"`). Written by the all-atom builder.
pub const ELEMENT: &str = "element";
/// Coarse-grained bead type (e.g. `"W"`). Written by the CG builder.
pub const BEAD_TYPE: &str = "bead_type";

/// Partial charge.
pub const CHARGE: &str = "charge";
/// Bond order (e.g. `1.0`, `2.0`). Default seeded by the bond builder.
pub const ORDER: &str = "order";
/// Atomic mass.
pub const MASS: &str = "mass";
/// Force-field / atom type label.
pub const TYPE: &str = "type";
/// Stable per-entity identifier (when assigned by a reader/writer).
pub const ID: &str = "id";
/// Molecule identifier (groups atoms into molecules).
pub const MOL_ID: &str = "mol_id";
/// Element/site symbol when keyed separately from `element` (e.g. crystal sites).
pub const SYMBOL: &str = "symbol";
/// Human-readable name.
pub const NAME: &str = "name";
