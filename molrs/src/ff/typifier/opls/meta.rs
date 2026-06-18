//! OPLS-AA typing metadata: per-type SMARTS definition, overrides, and
//! specificity-priority inputs.
//!
//! This is the typing-metadata half of an OPLS-AA force field, read *separately*
//! from the potential parameters (mirroring
//! [`MMFFParams`](crate::ff::typifier::mmff::MMFFParams) versus the
//! [`ForceField`](crate::ff::forcefield::ForceField)). The OPLS potential reader
//! ([`OplsXmlReader`](crate::ff::forcefield::readers::opls::OplsXmlReader))
//! deliberately drops the `def` / `overrides` / `priority` / `layer` attributes;
//! [`read_opls_typing_xml_str`](crate::ff::forcefield::xml::read_opls_typing_xml_str)
//! reads them into the [`OplsTypingMeta`] table here.
//!
//! # Scope
//!
//! Only types carrying a SMARTS `def` participate in automatic SMARTS typing.
//! Legacy OPLS rows (the `opls_001`–`opls_134` block in `oplsaa.xml`) have no
//! `def` and are **out of scope** for auto-typing — they can only be assigned by
//! hand or read back from a LAMMPS data file.

use std::collections::HashMap;

/// Layer-priority stride. A type tagged `layer=L` adds `L * STRIDE` to its
/// priority so it strictly outranks every lower-layer type regardless of
/// specificity (CL&P / CL&Pol overlays read on top of OPLS-AA). Mirrors molpy's
/// `_LAYER_PRIORITY_STRIDE`.
pub const LAYER_PRIORITY_STRIDE: i64 = 1000;

/// One `<Type>` row of an OPLS-AA `<AtomTypes>` section, holding the typing
/// metadata (not the potential parameters).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OplsTypeRow {
    /// Chemical class (the `class` attribute, e.g. `"CT"`). Bonded forces key on
    /// this class vocabulary, distinct from the `opls_NNN` type vocabulary.
    pub class: String,
    /// SMARTS definition (the `def` attribute), or `None` for legacy rows that
    /// carry no `def` and therefore cannot be matched automatically.
    pub def: Option<String>,
    /// Type names this row overrides (parsed from a comma-separated `overrides`
    /// attribute); empty when absent.
    pub overrides: Vec<String>,
    /// Explicit `priority` attribute, if present. When set it wins outright over
    /// the overrides-derived priority.
    pub priority: Option<i64>,
    /// Overlay layer (the `layer` attribute); `0` (base force field) when absent.
    pub layer: u32,
}

/// Parsed OPLS-AA typing metadata: [`OplsTypeRow`]s keyed by `opls_NNN` type
/// name.
///
/// Read from the same XML as the potential [`ForceField`](crate::ff::forcefield::ForceField)
/// but kept separate — this table drives SMARTS atom typing, the `ForceField`
/// drives energy evaluation.
#[derive(Debug, Clone, Default)]
pub struct OplsTypingMeta {
    rows: HashMap<String, OplsTypeRow>,
}

impl OplsTypingMeta {
    /// Create an empty metadata table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from pre-parsed rows keyed by `opls_NNN`.
    pub fn from_rows(rows: HashMap<String, OplsTypeRow>) -> Self {
        Self { rows }
    }

    /// Insert or replace a row.
    pub fn insert(&mut self, name: impl Into<String>, row: OplsTypeRow) {
        self.rows.insert(name.into(), row);
    }

    /// Look up a row by `opls_NNN` name.
    pub fn get(&self, name: &str) -> Option<&OplsTypeRow> {
        self.rows.get(name)
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Iterate `(name, row)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &OplsTypeRow)> {
        self.rows.iter()
    }

    /// Compute the specificity priority for every type, replicating molpy's
    /// `_OplsAtomTypifier._extract_patterns`:
    ///
    /// - an explicit `priority` attribute wins outright;
    /// - otherwise `priority = (number of other types this row overrides)
    ///   − (number of types that override this row)`;
    /// - plus `layer * LAYER_PRIORITY_STRIDE`.
    ///
    /// Higher priority wins when an atom matches multiple defs. Returns a map
    /// from `opls_NNN` name to its integer priority.
    pub fn priorities(&self) -> HashMap<String, i64> {
        // Names that each row overrides (the row is the "overrider").
        let mut out = HashMap::with_capacity(self.rows.len());
        for (name, row) in &self.rows {
            if let Some(p) = row.priority {
                // Explicit priority wins outright (the layer boost is folded into
                // the explicit value upstream if intended — molpy `continue`s here).
                out.insert(name.clone(), p);
                continue;
            }
            let mut priority: i64 = 0;
            // −1 for every other row that overrides this one.
            for other in self.rows.values() {
                if other.overrides.iter().any(|o| o == name) {
                    priority -= 1;
                }
            }
            // +len(overrides) for the rows this one overrides.
            priority += row.overrides.len() as i64;
            // Overlay-layer boost (lifts an overlay strictly above lower layers).
            priority += i64::from(row.layer) * LAYER_PRIORITY_STRIDE;
            out.insert(name.clone(), priority);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        class: &str,
        def: Option<&str>,
        overrides: &[&str],
        priority: Option<i64>,
        layer: u32,
    ) -> OplsTypeRow {
        OplsTypeRow {
            class: class.to_string(),
            def: def.map(str::to_string),
            overrides: overrides.iter().map(|s| s.to_string()).collect(),
            priority,
            layer,
        }
    }

    #[test]
    fn priority_explicit_wins() {
        // An explicit `priority` is taken verbatim, ignoring overrides/layer.
        let mut m = OplsTypingMeta::new();
        m.insert(
            "opls_900",
            row("CT", Some("[C]"), &["opls_135"], Some(42), 3),
        );
        let p = m.priorities();
        assert_eq!(p["opls_900"], 42);
    }

    #[test]
    fn priority_from_overrides() {
        // opls_146 overrides opls_144; so opls_146 gets +1, opls_144 gets -1.
        let mut m = OplsTypingMeta::new();
        m.insert("opls_144", row("HA", Some("[H][C;X3]"), &[], None, 0));
        m.insert(
            "opls_146",
            row("HA", Some("[H][c]"), &["opls_144"], None, 0),
        );
        let p = m.priorities();
        assert_eq!(p["opls_146"], 1, "overrider gains +len(overrides)");
        assert_eq!(p["opls_144"], -1, "overridden loses 1");
    }

    #[test]
    fn priority_overrides_multiple() {
        // A row that overrides two types gains +2; each overridden loses 1.
        let mut m = OplsTypingMeta::new();
        m.insert("opls_a", row("X", Some("[*]"), &[], None, 0));
        m.insert("opls_b", row("X", Some("[*]"), &[], None, 0));
        m.insert(
            "opls_c",
            row("X", Some("[*]"), &["opls_a", "opls_b"], None, 0),
        );
        let p = m.priorities();
        assert_eq!(p["opls_c"], 2);
        assert_eq!(p["opls_a"], -1);
        assert_eq!(p["opls_b"], -1);
    }

    #[test]
    fn priority_layer_stride() {
        // layer=2 with no overrides => 2 * STRIDE.
        let mut m = OplsTypingMeta::new();
        m.insert("opls_overlay", row("CT", Some("[C]"), &[], None, 2));
        let p = m.priorities();
        assert_eq!(p["opls_overlay"], 2 * LAYER_PRIORITY_STRIDE);
    }

    #[test]
    fn priority_layer_plus_overrides() {
        // layer boost adds to (not replaces) the overrides-derived priority.
        let mut m = OplsTypingMeta::new();
        m.insert("opls_base", row("X", Some("[*]"), &[], None, 0));
        m.insert("opls_hi", row("X", Some("[*]"), &["opls_base"], None, 1));
        let p = m.priorities();
        assert_eq!(p["opls_hi"], 1 + LAYER_PRIORITY_STRIDE);
        assert_eq!(p["opls_base"], -1);
    }
}
