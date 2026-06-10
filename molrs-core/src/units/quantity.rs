//! A magnitude paired with a unit, with dimension-checked operations.

use std::fmt;
use std::ops::{Div, Mul, Neg};

use crate::types::F;

use super::dimension::Dimension;
use super::error::UnitsError;
use super::registry::UnitRegistry;
use super::unit::Unit;

/// A value with an attached unit.
#[derive(Clone, PartialEq, Debug)]
pub struct Quantity {
    pub(crate) value: F,
    pub(crate) unit: Unit,
}

impl Quantity {
    /// Construct a quantity from a magnitude and a unit.
    pub fn new(value: F, unit: Unit) -> Quantity {
        Quantity { value, unit }
    }

    /// The magnitude expressed in `self.unit()`.
    pub fn value(&self) -> F {
        self.value
    }

    /// The unit of this quantity.
    pub fn unit(&self) -> &Unit {
        &self.unit
    }

    /// Convert to a target unit (handles affine °C↔K correctly).
    pub fn to(&self, unit: &Unit) -> Result<Quantity, UnitsError> {
        if self.unit.dimension != unit.dimension {
            return Err(UnitsError::DimensionMismatch {
                left: self.unit.dimension,
                right: unit.dimension,
            });
        }
        let si = self.value * self.unit.factor + self.unit.offset;
        let value = (si - unit.offset) / unit.factor;
        Ok(Quantity::new(value, unit.clone()))
    }

    /// Convert to a target unit named by a string (uses the default registry).
    pub fn to_parsed(&self, expr: &str) -> Result<Quantity, UnitsError> {
        self.to(&UnitRegistry::global().parse(expr)?)
    }

    /// Convert to the SI base representation of this quantity's dimension.
    pub fn to_base_units(&self) -> Quantity {
        let dim = self.unit.dimension;
        let base = Unit::new(1.0, 0.0, dim, si_base_name(dim));
        let si = self.value * self.unit.factor + self.unit.offset;
        Quantity::new(si, base)
    }

    /// Add `rhs` (converted to `self`'s unit). `Err` on dimension mismatch.
    pub fn try_add(&self, rhs: &Quantity) -> Result<Quantity, UnitsError> {
        let rhs = rhs.to(&self.unit)?;
        Ok(Quantity::new(self.value + rhs.value, self.unit.clone()))
    }

    /// Subtract `rhs` (converted to `self`'s unit). `Err` on dimension mismatch.
    pub fn try_sub(&self, rhs: &Quantity) -> Result<Quantity, UnitsError> {
        let rhs = rhs.to(&self.unit)?;
        Ok(Quantity::new(self.value - rhs.value, self.unit.clone()))
    }

    /// Multiply two quantities, composing dimensions. `Err` on affine operand.
    pub fn try_mul(&self, rhs: &Quantity) -> Result<Quantity, UnitsError> {
        let (lhs_unit, rhs_unit) = check_multiplicative(&self.unit, &rhs.unit, "mul")?;
        let unit = Unit::new(
            lhs_unit.factor * rhs_unit.factor,
            0.0,
            lhs_unit.dimension * rhs_unit.dimension,
            format!("{} * {}", lhs_unit.name, rhs_unit.name),
        );
        Ok(Quantity::new(self.value * rhs.value, unit))
    }

    /// Divide two quantities, composing dimensions. `Err` on affine operand.
    pub fn try_div(&self, rhs: &Quantity) -> Result<Quantity, UnitsError> {
        let (lhs_unit, rhs_unit) = check_multiplicative(&self.unit, &rhs.unit, "div")?;
        let unit = Unit::new(
            lhs_unit.factor / rhs_unit.factor,
            0.0,
            lhs_unit.dimension / rhs_unit.dimension,
            format!("{} / ({})", lhs_unit.name, rhs_unit.name),
        );
        Ok(Quantity::new(self.value / rhs.value, unit))
    }
}

/// Reject affine operands for multiplicative operations.
fn check_multiplicative<'a>(
    lhs: &'a Unit,
    rhs: &'a Unit,
    operation: &'static str,
) -> Result<(&'a Unit, &'a Unit), UnitsError> {
    for unit in [lhs, rhs] {
        if unit.is_affine() {
            return Err(UnitsError::AffineUnit {
                name: unit.name.clone(),
                operation,
            });
        }
    }
    Ok((lhs, rhs))
}

/// Canonical SI-base display name for a dimension, e.g. `m^2 * kg * s^-2`.
fn si_base_name(dim: Dimension) -> String {
    const BASE_SYMBOLS: [&str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];
    let parts: Vec<String> = BASE_SYMBOLS
        .iter()
        .zip(dim.exponents())
        .filter(|(_, exp)| *exp != 0)
        .map(|(sym, exp)| {
            if exp == 1 {
                (*sym).to_string()
            } else {
                format!("{}^{}", sym, exp)
            }
        })
        .collect();
    if parts.is_empty() {
        "1".to_string()
    } else {
        parts.join(" * ")
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.unit)
    }
}

impl Neg for Quantity {
    type Output = Quantity;
    fn neg(self) -> Quantity {
        Quantity::new(-self.value, self.unit)
    }
}

impl Mul<F> for Quantity {
    type Output = Quantity;
    fn mul(self, rhs: F) -> Quantity {
        Quantity::new(self.value * rhs, self.unit)
    }
}

impl Div<F> for Quantity {
    type Output = Quantity;
    fn div(self, rhs: F) -> Quantity {
        Quantity::new(self.value / rhs, self.unit)
    }
}
