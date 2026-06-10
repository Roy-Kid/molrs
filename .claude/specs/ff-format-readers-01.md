# Spec: Force-field format readers in molrs (sink molpy's FF IO)

## Summary
Give molrs the ability to read force-field definitions from external formats
directly into a `molrs.ForceField` (Style/Type/Params), with unit normalization
to molrs units (Å, kcal/mol). Land the reader framework plus the first concrete
reader — OPLS-AA / GROMACS (`oplsaa.xml`, GROMACS nm/kJ units) — since that is
what the OPLS-optimization goal needs. Other formats (LAMMPS, AMBER prmtop) are
enumerated as follow-on readers behind the same trait.

## Motivation
`molrs 要有读xml的能力` and `molpy的io也要下沉到molrs` (`feedback-core-sinks-to-molrs`).
Today molpy owns the FF readers (`molpy/io/forcefield/…`: XML, LAMMPS, AMBER
prmtop) and produces a molpy `ForceField`; the rejected emitter then tried to
shuttle that into molrs via an XML string. Instead, **molrs reads the FF format
itself** into its own `ForceField`. With `ForceField.to_potentials()`
(`ff-potentials-oop-01`) and the SMARTS typifier sink (B-line, later), the whole
OPLS pipeline becomes pure molrs.

A critical job of the reader is **unit normalization**: molpy's `oplsaa.xml` is
GROMACS units (nm, kJ/mol); molrs is Å, kcal/mol. Normalization is the reader's
responsibility, declared per source format — not a downstream converter.

## Scope
- **Crates affected**: `molrs-ff` (reader trait + OPLS/GROMACS reader; molrs-ff
  already has `read_forcefield_xml[_str]` for molrs's own schema, extend the
  surface), `molrs-python` (expose the new reader(s)), downstream molpy (retire
  its FF readers — tracked, not done here).
- **Traits/data**: `ForceFieldReader` trait (`read(path) -> ForceField`); an
  `OplsXmlReader`. Reuses existing `ForceField`/`Style`/`Type`/`Params`.
- **Feature flags**: none (XML dep already present via `roxmltree`).

## Technical Design

### API Surface
```rust
/// Read a force-field definition from an external format into a molrs
/// ForceField, normalized to molrs units (Å, kcal/mol, radians, e).
pub trait ForceFieldReader {
    fn read_str(&self, text: &str) -> Result<ForceField, String>;
    fn read(&self, path: &str) -> Result<ForceField, String> {
        self.read_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
    }
}

/// OPLS-AA / GROMACS XML (nm, kJ/mol). Maps OPLS atom/bond/angle/dihedral(RB)/
/// pair definitions to molrs Styles + Types, converting units and RB→OPLS
/// 4-cosine on the way in.
pub struct OplsXmlReader;
impl ForceFieldReader for OplsXmlReader { /* … */ }
```
PyO3: `molrs.read_opls_xml(path) -> ForceField` (and `_str`). Keeps the existing
`read_forcefield_xml` (molrs's native schema) untouched.

### Unit normalization (OPLS/GROMACS → molrs)
- length nm → Å (×10): bond `r0`, pair `sigma`.
- energy kJ/mol → kcal/mol (÷4.184): pair `epsilon`, angle/bond k (with the
  length-squared factor for bond k: ÷4.184/100), dihedral.
- dihedral RB `c0..c5` → OPLS 4-cosine `f1..f4` (the math molpy's `rb_to_opls`
  encodes), in kcal/mol — matching the `dihedral:opls` kernel from `opls-ef-01`.
- charge e, angle `theta0` rad: unchanged.
- The mapping table (which OPLS element → which molrs style/param keys) is the
  reader's; combining rules / 1-4 scaling are NOT baked here (they are applied
  when the nonbonded `Potential` evaluates against a frame — `ff-potentials-oop-01`).

### Data Flow
```
oplsaa.xml (nm/kJ, RB) --OplsXmlReader.read--> molrs.ForceField (Å/kcal, OPLS f1..f4)
                                                   └─ .to_potentials() (ff-potentials-oop-01)
```

### Integration Points
- New module `molrs-ff/src/forcefield/readers/opls.rs` (+ trait in
  `forcefield/readers/mod.rs`); re-export from `molrs_ff`.
- molpy `io/forcefield/xml.py` (OPLS path) becomes a thin wrapper over the molrs
  reader or is retired — molpy follow-up.

## Constraints & Invariants
- Normalization to molrs units happens **in the reader**; the resulting
  `ForceField` is pure molrs units. No downstream unit fixups.
- Reading is total: malformed/unknown elements → `Err`, never silent skip of a
  parameter that would later read as 0.
- Round-trip sanity: a `ForceField` read here, `to_potentials()`-d, evaluated on
  a frame, must reproduce molpy's numpy OPLS energy within tolerance (the
  cross-library parity that the rejected emitter targeted — now native).

## Test Criteria
### Unit Tests
1. Parse a small OPLS XML fixture → `ForceField` with expected styles/types and
   **molrs-unit** param values (assert nm→Å, kJ→kcal conversions on a known row;
   RB→OPLS on a known dihedral).
2. Malformed XML / missing required attr → `Err` with a clear message.

### Integration / Parity
3. Read molpy's bundled `oplsaa.xml` → non-empty ForceField with bond/angle/
   dihedral/pair styles.
4. **Parity**: for butane and ethanol typified by molpy OPLS, the molrs energy
   (`read_opls_xml(...).to_potentials().calc_energy(frame)`) and each per-term
   energy match molpy's own numpy OPLS potentials within 1e-4 kcal/mol. (This is
   the seam contract; runs where the molpy↔molrs harness lives, e.g.
   `bm-molrs-molpy`.)

## Performance Requirements
- Parsing is one-time setup; no hot-path constraint.

## Migration & Compatibility
- Additive in molrs. Depends on `ff-potentials-oop-01` for `to_potentials` to
  realize value, but the reader itself can land independently.
- molpy FF-reader retirement is a follow-up.

## Out of Scope
- LAMMPS / AMBER-prmtop readers (same trait; follow-on specs once OPLS lands).
- SMARTS typifier sink (B-line).
- The `to_potentials`/`calc_*`/`LBFGS` surface (`ff-potentials-oop-01`).
