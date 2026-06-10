//! A resolved unit value, detached from the registry after parse.

use std::fmt;
use std::str::FromStr;

use crate::types::F;

use super::dimension::Dimension;
use super::error::UnitsError;

/// A resolved unit: conversion to SI base is `si = value * factor + offset`.
#[derive(Clone, PartialEq, Debug)]
pub struct Unit {
    pub(crate) factor: F,
    pub(crate) offset: F,
    pub(crate) dimension: Dimension,
    pub(crate) name: String,
}

impl Unit {
    /// Construct a unit from its parts. Primarily for internal/parser use.
    pub(crate) fn new(factor: F, offset: F, dimension: Dimension, name: String) -> Unit {
        Unit {
            factor,
            offset,
            dimension,
            name,
        }
    }

    /// The dimension of this unit.
    pub fn dimension(&self) -> Dimension {
        self.dimension
    }

    /// True when this unit carries a non-zero additive offset (e.g. °C).
    pub fn is_affine(&self) -> bool {
        self.offset != 0.0
    }

    /// Multiplicative factor converting a value in `self` to a value in `other`.
    ///
    /// `Err(DimensionMismatch)` if dimensions differ; `Err(AffineUnit)` if
    /// either side carries an offset.
    pub fn factor_to(&self, other: &Unit) -> Result<F, UnitsError> {
        if self.dimension != other.dimension {
            return Err(UnitsError::DimensionMismatch {
                left: self.dimension,
                right: other.dimension,
            });
        }
        if self.is_affine() {
            return Err(UnitsError::AffineUnit {
                name: self.name.clone(),
                operation: "factor_to",
            });
        }
        if other.is_affine() {
            return Err(UnitsError::AffineUnit {
                name: other.name.clone(),
                operation: "factor_to",
            });
        }
        Ok(self.factor / other.factor)
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

impl FromStr for Unit {
    type Err = UnitsError;
    fn from_str(s: &str) -> Result<Unit, UnitsError> {
        super::registry::UnitRegistry::global().parse(s)
    }
}
