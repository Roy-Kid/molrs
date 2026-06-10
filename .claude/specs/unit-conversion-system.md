# Spec: Unit Conversion System (pint-style)

## Summary

A pint-inspired unit system in `molrs-core`: a `UnitRegistry` holding unit
definitions over the 7 SI base dimensions, a parser for compound unit
expressions (`"kcal/mol/angstrom"`), and a `Quantity` type supporting
dimension-checked arithmetic and conversion (`to`, `to_base_units`). Targets
molecular-simulation units: energy (kcal/mol, kJ/mol, eV, hartree), length
(Å, nm, bohr), time (fs, ps, ns), force, pressure, and offset temperature
(°C ↔ K).

## Motivation

The workspace has no centralized unit handling. Conversion constants are
scattered and duplicated:

- `MDYNE_A_TO_KCAL = 143.9325` appears 3× — `molrs-ff/src/potential/bond/mmff.rs:12`, `angle/mmff.rs:14`, `improper/mmff.rs:12`
- `BOHR_TO_ANG = 0.529_177_210_67` — `molrs-io/src/cube.rs:63`
- PME `coulomb` constant is a free-floating `F` field with a comment — `molrs-ff/src/potential/kspace/pme.rs:52`

A single registry with CODATA-pinned factors removes drift between copies,
gives readers/writers (cube, LAMMPS unit styles) one source of truth, and
gives users a safe API for converting reported observables.

Reference design: pint (https://github.com/hgrecco/pint) — registry of
definitions + dimensional vector + affine (offset) units + string grammar.

## Scope

- **Crates affected**: `molrs-core` (new `units` module). Follow-up adoption
  by `molrs-ff` / `molrs-io` is **out of scope** for this spec (separate
  refactor spec) — except re-exporting via the `molrs` umbrella crate.
- **Traits extended**: `std::ops::{Add, Sub, Mul, Div, Neg}` for `Quantity`;
  `Display` for `Unit`, `Quantity`, `Dimension`; `FromStr` for `Unit`.
- **Traits created**: none (concrete types; no dyn dispatch needed).
- **Data structures**: `Dimension`, `Unit`, `UnitRegistry`, `Quantity`,
  `UnitsError` (new); `MolRsError::Units(UnitsError)` variant (modified).
- **Feature flags**: none. Pure-Rust, no new dependencies.

## Technical Design

Design choices vs pint, adapted to Rust:

1. **Dimension = fixed vector, not dict.** `[i32; 7]` exponents over SI base
   dimensions (length, mass, time, current, temperature, amount, luminous
   intensity). Integer exponents suffice for MD observables; `Copy + Eq`
   makes dimension checks branch-cheap. (pint uses a hash map with fractional
   exponents — not needed here.)
2. **Conversion is multiplicative + optional offset.** Every `Unit` reduces
   to `value_si = value * factor + offset` against SI base units. Offset
   (affine) units (°C) are restricted exactly as pint restricts them:
   convertible via `to`, forbidden in `*`/`/` and in compound expressions.
3. **Registry is data, units are values.** `UnitRegistry` owns definitions;
   `Unit` is a self-contained value (factor, offset, dimension, display
   name) detached from the registry after parse — no lifetimes/Arc in hot
   paths, registry is only needed at parse time.
4. **Array-scale path.** For bulk data (FNx3 coordinates, energy columns)
   conversion goes through `Unit::factor_to(&Unit) -> Result<F>` so callers
   scale ndarrays themselves with one multiply — no per-element Quantity.

### API Surface

```rust
// molrs-core/src/units/dimension.rs
/// Exponents over the 7 SI base dimensions, in order:
/// [length, mass, time, current, temperature, amount, luminous intensity].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Dimension([i32; 7]);

impl Dimension {
    pub const DIMENSIONLESS: Dimension;
    pub const LENGTH: Dimension;
    pub const MASS: Dimension;
    pub const TIME: Dimension;
    pub const TEMPERATURE: Dimension;
    pub const AMOUNT: Dimension;
    pub const ENERGY: Dimension;          // M L² T⁻²
    pub const FORCE: Dimension;           // M L T⁻²
    pub const PRESSURE: Dimension;        // M L⁻¹ T⁻²
    pub fn is_dimensionless(&self) -> bool;
    pub fn pow(self, n: i32) -> Dimension;          // exponent scale
    // Mul/Div by another Dimension = exponent add/sub (std::ops impls)
}

// molrs-core/src/units/unit.rs
/// A resolved unit: conversion to SI base is `si = value * factor + offset`.
#[derive(Clone, PartialEq, Debug)]
pub struct Unit {
    factor: F,
    offset: F,            // non-zero only for affine units (degC)
    dimension: Dimension,
    name: String,         // canonical display, e.g. "kcal/mol/angstrom"
}

// Display contract: Unit/Quantity Display prints the canonical name stored
// at construction; `parse(s).to_string()` need not equal `s`, but
// `parse(parse(s).to_string())` must yield an equal Unit (parse round-trip).
impl Unit {
    pub fn dimension(&self) -> Dimension;
    pub fn is_affine(&self) -> bool;
    /// Multiplicative factor converting self → other.
    /// Err(DimensionMismatch) if dimensions differ; Err(AffineUnit) if either has offset.
    pub fn factor_to(&self, other: &Unit) -> Result<F, UnitsError>;
}
impl FromStr for Unit { type Err = UnitsError; /* parses via default registry */ }

// molrs-core/src/units/registry.rs
pub struct UnitRegistry { /* name → UnitDef; prefix table */ }

impl UnitRegistry {
    /// Preloaded with SI + molecular-simulation units (see table below).
    /// Matches KernelRegistry precedent: new() = builtins included.
    pub fn new() -> UnitRegistry;                          // Default delegates here
    /// SI base units only — for users building a custom system from scratch.
    pub fn empty() -> UnitRegistry;
    /// Shared immutable preloaded registry (OnceLock singleton).
    pub fn global() -> &'static UnitRegistry;
    /// Define a unit: name, aliases, factor/offset to SI base, dimension.
    pub fn define(&mut self, def: UnitDef) -> Result<(), UnitsError>;
    /// Parse a compound unit expression ("kcal/mol/angstrom", "m s^-2").
    pub fn parse(&self, expr: &str) -> Result<Unit, UnitsError>;
    /// Convenience: registry.quantity(1.5, "kcal/mol")
    pub fn quantity(&self, value: F, expr: &str) -> Result<Quantity, UnitsError>;
    /// Iterate all registered definitions (test/introspection support).
    pub fn definitions(&self) -> impl Iterator<Item = &UnitDef>;
}

pub struct UnitDef {
    pub name: String,              // canonical: "calorie"
    pub aliases: Vec<String>,      // ["cal"]
    pub symbol: String,            // preferred display
    pub factor: F,
    pub offset: F,
    pub dimension: Dimension,
    pub prefixable: bool,          // accepts SI prefixes (kcal, nm, fs)
}

// molrs-core/src/units/quantity.rs
#[derive(Clone, PartialEq, Debug)]
pub struct Quantity { value: F, unit: Unit }

impl Quantity {
    pub fn new(value: F, unit: Unit) -> Quantity;
    pub fn value(&self) -> F;                  // magnitude in self.unit
    pub fn unit(&self) -> &Unit;
    /// Convert to target unit (handles affine °C↔K correctly).
    pub fn to(&self, unit: &Unit) -> Result<Quantity, UnitsError>;
    pub fn to_parsed(&self, expr: &str) -> Result<Quantity, UnitsError>; // default registry
    /// Convert to SI base representation of self's dimension.
    pub fn to_base_units(&self) -> Quantity;
    // Fallible arithmetic — named methods, NOT std::ops (Result-typed
    // operator Output is an anti-idiom; architecture review 2026-06-10):
    pub fn try_add(&self, rhs: &Quantity) -> Result<Quantity, UnitsError>; // rhs converted to lhs unit
    pub fn try_sub(&self, rhs: &Quantity) -> Result<Quantity, UnitsError>;
    pub fn try_mul(&self, rhs: &Quantity) -> Result<Quantity, UnitsError>; // Err on affine operand
    pub fn try_div(&self, rhs: &Quantity) -> Result<Quantity, UnitsError>; // Err on affine operand
}
// std::ops impls ONLY where total (never panic):
//   Neg                        → Quantity
//   Mul<F> / Div<F>            → Quantity   (scalar scale)
// No panicking Add/Sub/Mul/Div<Quantity> — one fallible spelling only.

// molrs-core/src/units/error.rs
#[derive(Debug, Clone, PartialEq, Eq)]   // mirrors BlockError derives
pub enum UnitsError {
    UnknownUnit { name: String },
    DimensionMismatch { left: Dimension, right: Dimension },
    AffineUnit { name: String, operation: &'static str },
    Parse { expr: String, message: String },
    Redefinition { name: String },
}
// + Display, std::error::Error, From<UnitsError> for MolRsError (new Units variant)
```

`lib.rs` additions: `pub mod units;` and
`pub use units::{Dimension, Quantity, Unit, UnitDef, UnitRegistry, UnitsError};`
(constants stay module-pathed as `units::constants`).

Module layout (architecture-review final): `units/{mod,dimension,unit,registry,parse,quantity,error,constants}.rs` — `constants.rs` (~40 lines) added so layout and API agree.

### Preloaded definitions (default registry)

Base: `meter (m)`, `kilogram (kg — gram is the prefixable atom)`,
`second (s)`, `ampere (A)`, `kelvin (K)`, `mole (mol)`, `candela (cd)`.

Derived/MD set (factor = value in SI base; CODATA 2018 / SI-2019 exact where
applicable):

| Unit | Aliases | Factor | Source |
|---|---|---|---|
| angstrom | Å, ang | 1e-10 m | exact |
| bohr | a0 | 5.291_772_109_03e-11 m | CODATA 2018 |
| joule | J | 1 (derived) | exact |
| calorie | cal | 4.184 J | thermochemical, exact |
| electron_volt | eV | 1.602_176_634e-19 J | SI-2019 exact |
| hartree | Eh | 4.359_744_722_2071e-18 J | CODATA 2018 |
| newton | N | 1 (derived) | exact |
| pascal | Pa | 1 (derived) | exact |
| bar | — | 1e5 Pa | exact |
| atmosphere | atm | 101_325 Pa | exact |
| minute/hour | min, h | 60 / 3600 s | exact |
| dalton | Da, amu | 1.660_539_066_60e-27 kg | CODATA 2018 |
| degC | celsius, °C | factor 1, offset 273.15 K | exact |
| degree (angle) | deg | π/180 (dimensionless) | exact |
| coulomb | C | 1 A·s (derived) | exact |
| elementary_charge | e | 1.602_176_634e-19 C | SI-2019 exact |
| debye | D | 3.335_640_951_98e-30 C·m | 1e-21/c, derived exact |

Charge units added per scientific review 2026-06-10: MD partial charges are
conventionally in multiples of *e* (LAMMPS real/metal); without a
charge-dimensioned atom, electrostatic quantities cannot be parsed.

SI prefixes (prefixable units only): `y..Y` full ladder including
`f (1e-15)`, `p`, `n`, `u/µ (1e-6)`, `m`, `c`, `k`, `M`, `G`.
So `fs`, `ps`, `ns`, `nm`, `kJ`, `kcal` parse with zero extra definitions.

Physical constants exposed as `pub mod units::constants`:
`AVOGADRO = 6.022_140_76e23` (exact), `BOLTZMANN = 1.380_649e-23 J/K`
(exact), `GAS_CONSTANT = AVOGADRO * BOLTZMANN`.

### Data Flow

```
"kcal/mol/angstrom"
  → tokenizer (ident / number / * / / / ^ / ( ))
  → recursive-descent parser → AST of (unit-atom, exponent)
  → per atom: registry lookup (exact name → alias → prefix split)
  → fold: factor = Π fᵢ^eᵢ,  dimension = Σ eᵢ·dᵢ   (offset must be 0 unless
    the expression is a single bare atom — pint's affine rule)
  → Unit { factor, offset, dimension, name: canonicalized }

Quantity::to(target):
  same dimension check → value' = ((value * f_self + off_self) - off_tgt) / f_tgt
```

### Algorithm

Parser grammar (recursive descent, O(len(expr))):

```
expr     := term (('*' | '/' | '·' | ' ') term)*      # ' ' = implicit multiply
term     := factor ('^' | '**')? signed_int?
factor   := IDENT | NUMBER | '(' expr ')'
```

Unit-atom resolution order (first hit wins):
1. exact canonical name or alias (`"mol"`, `"Å"`)
2. longest-prefix split: prefix table × prefixable atoms (`"kcal"` → `k`+`cal`,
   `"fs"` → `f`+`s`); reject ambiguity by requiring the remainder to be a
   known prefixable atom.

Conversion math is closed-form (no iteration); all factors are `F = f64`.

### Integration Points

- `MolRsError` gains a `Units(UnitsError)` variant + `From` impl
  (`molrs-core/src/error.rs`, following the existing `Block(BlockError)`
  nesting pattern).
- `molrs` umbrella crate re-exports `molrs_core::units` (no feature gate —
  core is always present).
- Future (separate specs): `molrs-ff` MMFF kernels and `molrs-io` cube reader
  replace local constants with `units::constants` / registry factors;
  LAMMPS `units real/metal` style mapping.

## Constraints & Invariants

- `F = f64` everywhere; no raw `f64` in signatures (`molrs-core/src/types.rs` alias).
- `UnitRegistry`, `Unit`, `Quantity`, `Dimension` are `Send + Sync` (no
  interior mutability; the default registry is an immutable `OnceLock`).
- Immutable API: `define` is the only mutation and only on a user-owned
  registry before sharing; `Quantity`/`Unit` ops always return new values.
- Affine-unit calculus rule (pint parity): offset units valid only as a
  whole single unit in `to`/`to_base_units`/`Add(delta)`; any `*`, `/`,
  `pow`, or compound-expression use → `UnitsError::AffineUnit`.
- Exact SI-2019 / CODATA-2018 values verbatim as listed above; every factor
  carries a rustdoc citation.
- No panics in public API; all fallible paths return `Result<_, UnitsError>`.
- Files 200–400 lines; module layout:
  `units/{mod,dimension,unit,registry,parse,quantity,error,constants}.rs`.

## Test Criteria

Pure-logic feature → inline `#[cfg(test)]` unit tests in `src/units/*` plus
integration tests in `molrs-core/tests/test_units.rs` (IO rules N/A — no file
formats touched).

### Unit Tests

- Dimension algebra: `ENERGY == MASS*LENGTH²/TIME²`, mul/div/pow, hash/eq.
- Parser: atoms, aliases, prefixes (`fs`, `kcal`, `nm`, `µs`), `^`/`**`
  exponents, negative exponents, parentheses, implicit multiply (`"m s^-2"`),
  `·` separator, error cases (unknown unit, bad exponent, affine in compound,
  empty string, `"kkg"` double-prefix rejection).
- Registry: define/redefine error, alias collision error, custom registry
  independent of default.

### Integration Tests (`tests/test_units.rs`)

- Energy chain: `1 kcal/mol → 4.184 kJ/mol` (exact);
  `1 hartree → 27.211 386 245 988 eV`. Note: `eV → kJ/mol` is a
  DimensionMismatch as a direct `to` (ENERGY vs ENERGY/AMOUNT); the correct
  path is `eV.try_mul(N_A [mol⁻¹]).to("kJ/mol") → 96.485 332 12 kJ/mol`,
  which doubles as a try_mul dimension-composition test.
- Charge: `1 e → 1.602 176 634e-19 C` (exact); `e` composes in expressions
  (`e^2/angstrom` parses with dimension CHARGE²/LENGTH).
- Length: `1 bohr → 0.529 177 210 903 Å`; `10 Å → 1 nm` (exact).
- Force: `1 kcal/mol/Å → 4.184 kJ/mol/Å`; dimension equals ENERGY/AMOUNT/LENGTH.
- Time: `1 ps → 1000 fs` (exact).
- Pressure: `1 atm → 101 325 Pa → 1.013 25 bar`.
- Temperature: `0 °C → 273.15 K`, `298.15 K → 25 °C`, round-trip identity;
  `°C * m` → `AffineUnit` error.
- Quantity ops: add same-dim different-unit (auto-convert), add
  mismatched-dim → `DimensionMismatch`, mul/div compose dimensions,
  `to_base_units` idempotent.
- `factor_to` array path: scale an `Array1<F>` of energies kcal/mol → kJ/mol,
  compare against per-element `Quantity::to`.

### Numerical Validation

- Round-trip `a.to(u)?.to(a.unit())` recovers value to `rel ≤ 1e-14` across
  the full preloaded table (property-style loop over all registered units of
  equal dimension).
- Conversion factors checked against CODATA 2018 published values to printed
  precision (citations in test comments).

## Performance Requirements

- `Unit::factor_to` / `Quantity::to`: O(1), no allocation.
- Parse: O(n) in expression length; allowed to allocate (cold path).
- Bulk conversion = single scalar multiply per element via `factor_to`; no
  benchmark target needed (memory-bound trivially).

## Migration & Compatibility

- Purely additive: new module + one new `MolRsError` variant. Decision
  (architecture review 2026-06-10): `MolRsError` is **not**
  `#[non_exhaustive]`, so adding `Units(UnitsError)` is technically
  semver-breaking for exhaustive matches — accepted as a pre-1.0 enum
  extension, workspace-internal matches will be fixed in the same change.
- No behavior change to existing crates; constant deduplication in
  `molrs-ff`/`molrs-io` deferred to a follow-up spec.
