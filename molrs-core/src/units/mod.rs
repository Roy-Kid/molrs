//! Pint-inspired unit system: dimensions, units, registry, parser, quantities.
//!
//! A [`UnitRegistry`] holds [`UnitDef`] entries over the 7 SI base dimensions,
//! the [`parse`] module turns compound expressions (`"kcal/mol/angstrom"`) into
//! self-contained [`Unit`] values, and [`Quantity`] provides dimension-checked
//! arithmetic and conversion.

pub mod constants;
pub mod dimension;
pub mod error;
mod parse;
pub mod quantity;
pub mod registry;
pub mod unit;

pub use dimension::Dimension;
pub use error::UnitsError;
pub use quantity::Quantity;
pub use registry::{UnitDef, UnitRegistry};
pub use unit::Unit;
