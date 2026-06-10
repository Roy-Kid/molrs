//! Error type for the units subsystem.

use std::fmt;

use super::dimension::Dimension;

/// Errors arising from unit parsing, conversion, and registry operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitsError {
    /// A unit atom could not be resolved against the registry.
    UnknownUnit {
        /// The offending unit name.
        name: String,
    },
    /// Two quantities/units have incompatible dimensions.
    DimensionMismatch {
        /// Left-hand dimension.
        left: Dimension,
        /// Right-hand dimension.
        right: Dimension,
    },
    /// An affine (offset) unit was used in a forbidden operation.
    AffineUnit {
        /// The affine unit name.
        name: String,
        /// The operation that rejected it (e.g. "mul", "compound").
        operation: &'static str,
    },
    /// A unit expression failed to parse.
    Parse {
        /// The expression that failed.
        expr: String,
        /// Human-readable detail.
        message: String,
    },
    /// A unit name/alias was already defined in the registry.
    Redefinition {
        /// The conflicting name.
        name: String,
    },
}

impl fmt::Display for UnitsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitsError::UnknownUnit { name } => write!(f, "unknown unit: {}", name),
            UnitsError::DimensionMismatch { left, right } => {
                write!(f, "dimension mismatch: {:?} vs {:?}", left, right)
            }
            UnitsError::AffineUnit { name, operation } => {
                write!(f, "affine unit '{}' is invalid in {}", name, operation)
            }
            UnitsError::Parse { expr, message } => {
                write!(f, "failed to parse '{}': {}", expr, message)
            }
            UnitsError::Redefinition { name } => write!(f, "unit already defined: {}", name),
        }
    }
}

impl std::error::Error for UnitsError {}
