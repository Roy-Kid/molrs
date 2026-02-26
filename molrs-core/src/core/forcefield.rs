//! Force field definition types.
//!
//! Provides a declarative layer for defining atom types, bond types, pair types,
//! etc. with their parameters. A [`ForceField`] holds [`Style`]s, each of which
//! holds typed parameter sets ([`TypeDef`]s). The forcefield can be converted
//! into computational [`Potential`](super::potential::Potential) objects via
//! [`ForceField::to_potentials`].

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

/// Key-value parameter bag for type definitions.
#[derive(Debug, Clone, Default)]
pub struct Params {
    inner: HashMap<String, f64>,
}

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_pairs(pairs: &[(&str, f64)]) -> Self {
        let mut inner = HashMap::new();
        for &(k, v) in pairs {
            inner.insert(k.to_owned(), v);
        }
        Self { inner }
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        self.inner.get(key).copied()
    }

    pub fn set(&mut self, key: &str, value: f64) {
        self.inner.insert(key.to_owned(), value);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> + '_ {
        self.inner.iter().map(|(k, v)| (k.as_str(), *v))
    }

    pub fn inner(&self) -> &HashMap<String, f64> {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Type definitions
// ---------------------------------------------------------------------------

/// Atom type definition.
#[derive(Debug, Clone)]
pub struct AtomType {
    pub name: String,
    pub params: Params,
}

/// Bond type definition (references two atom type names).
#[derive(Debug, Clone)]
pub struct BondType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub params: Params,
}

/// Angle type definition (references three atom type names).
#[derive(Debug, Clone)]
pub struct AngleType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub ktom: String,
    pub params: Params,
}

/// Dihedral type definition (references four atom type names).
#[derive(Debug, Clone)]
pub struct DihedralType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub ktom: String,
    pub ltom: String,
    pub params: Params,
}

/// Improper type definition (references four atom type names).
#[derive(Debug, Clone)]
pub struct ImproperType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub ktom: String,
    pub ltom: String,
    pub params: Params,
}

/// Pair type definition (one or two atom type names).
#[derive(Debug, Clone)]
pub struct PairType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub params: Params,
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

/// A style groups a named interaction method with its type definitions.
#[derive(Debug, Clone)]
pub struct Style {
    pub category: StyleCategory,
    pub name: String,
    pub params: Params,
    pub atomtypes: Vec<AtomType>,
    pub bondtypes: Vec<BondType>,
    pub angletypes: Vec<AngleType>,
    pub dihedraltypes: Vec<DihedralType>,
    pub impropertypes: Vec<ImproperType>,
    pub pairtypes: Vec<PairType>,
}

/// Style category enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StyleCategory {
    Atom,
    Bond,
    Angle,
    Dihedral,
    Improper,
    Pair,
}

impl StyleCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Atom => "atom",
            Self::Bond => "bond",
            Self::Angle => "angle",
            Self::Dihedral => "dihedral",
            Self::Improper => "improper",
            Self::Pair => "pair",
        }
    }
}

impl Style {
    fn new(category: StyleCategory, name: &str, params: Params) -> Self {
        Self {
            category,
            name: name.to_owned(),
            params,
            atomtypes: Vec::new(),
            bondtypes: Vec::new(),
            angletypes: Vec::new(),
            dihedraltypes: Vec::new(),
            impropertypes: Vec::new(),
            pairtypes: Vec::new(),
        }
    }

    // -- atom --

    pub fn def_atomtype(&mut self, name: &str, params: &[(&str, f64)]) -> &AtomType {
        let at = AtomType {
            name: name.to_owned(),
            params: Params::from_pairs(params),
        };
        self.atomtypes.push(at);
        self.atomtypes.last().unwrap()
    }

    pub fn get_atomtype(&self, name: &str) -> Option<&AtomType> {
        self.atomtypes.iter().find(|t| t.name == name)
    }

    // -- bond --

    pub fn def_bondtype(&mut self, itom: &str, jtom: &str, params: &[(&str, f64)]) -> &BondType {
        let name = format!("{}-{}", itom, jtom);
        let bt = BondType {
            name,
            itom: itom.to_owned(),
            jtom: jtom.to_owned(),
            params: Params::from_pairs(params),
        };
        self.bondtypes.push(bt);
        self.bondtypes.last().unwrap()
    }

    pub fn get_bondtype(&self, itom: &str, jtom: &str) -> Option<&BondType> {
        self.bondtypes
            .iter()
            .find(|t| (t.itom == itom && t.jtom == jtom) || (t.itom == jtom && t.jtom == itom))
    }

    // -- angle --

    pub fn def_angletype(
        &mut self,
        itom: &str,
        jtom: &str,
        ktom: &str,
        params: &[(&str, f64)],
    ) -> &AngleType {
        let name = format!("{}-{}-{}", itom, jtom, ktom);
        let at = AngleType {
            name,
            itom: itom.to_owned(),
            jtom: jtom.to_owned(),
            ktom: ktom.to_owned(),
            params: Params::from_pairs(params),
        };
        self.angletypes.push(at);
        self.angletypes.last().unwrap()
    }

    // -- dihedral --

    pub fn def_dihedraltype(
        &mut self,
        itom: &str,
        jtom: &str,
        ktom: &str,
        ltom: &str,
        params: &[(&str, f64)],
    ) -> &DihedralType {
        let name = format!("{}-{}-{}-{}", itom, jtom, ktom, ltom);
        let dt = DihedralType {
            name,
            itom: itom.to_owned(),
            jtom: jtom.to_owned(),
            ktom: ktom.to_owned(),
            ltom: ltom.to_owned(),
            params: Params::from_pairs(params),
        };
        self.dihedraltypes.push(dt);
        self.dihedraltypes.last().unwrap()
    }

    // -- improper --

    pub fn def_impropertype(
        &mut self,
        itom: &str,
        jtom: &str,
        ktom: &str,
        ltom: &str,
        params: &[(&str, f64)],
    ) -> &ImproperType {
        let name = format!("{}-{}-{}-{}", itom, jtom, ktom, ltom);
        let it = ImproperType {
            name,
            itom: itom.to_owned(),
            jtom: jtom.to_owned(),
            ktom: ktom.to_owned(),
            ltom: ltom.to_owned(),
            params: Params::from_pairs(params),
        };
        self.impropertypes.push(it);
        self.impropertypes.last().unwrap()
    }

    // -- pair --

    pub fn def_pairtype(
        &mut self,
        itom: &str,
        jtom: Option<&str>,
        params: &[(&str, f64)],
    ) -> &PairType {
        let jtom_str = jtom.unwrap_or(itom);
        let name = if itom == jtom_str {
            itom.to_owned()
        } else {
            format!("{}-{}", itom, jtom_str)
        };
        let pt = PairType {
            name,
            itom: itom.to_owned(),
            jtom: jtom_str.to_owned(),
            params: Params::from_pairs(params),
        };
        self.pairtypes.push(pt);
        self.pairtypes.last().unwrap()
    }

    // -- unified def_type --

    /// Define a type using the unified name format.
    ///
    /// The `name` encodes atom types based on the style's category:
    /// - **Atom**: `"A"` → atom type name
    /// - **Bond**: `"A-B"` → itom=A, jtom=B
    /// - **Angle**: `"A-B-C"` → itom=A, jtom=B, ktom=C
    /// - **Pair**: `"A"` → self-pair (itom=A, jtom=A); `"A-B"` → cross-pair
    /// - **Dihedral/Improper**: `"A-B-C-D"` → itom=A, jtom=B, ktom=C, ltom=D
    pub fn def_type(&mut self, name: &str, params: &[(&str, f64)]) -> &mut Self {
        let parts: Vec<&str> = name.split('-').collect();
        match self.category {
            StyleCategory::Atom => {
                self.def_atomtype(name, params);
            }
            StyleCategory::Bond => {
                assert!(
                    parts.len() == 2,
                    "bond type name must be \"A-B\", got \"{}\"",
                    name
                );
                self.def_bondtype(parts[0], parts[1], params);
            }
            StyleCategory::Angle => {
                assert!(
                    parts.len() == 3,
                    "angle type name must be \"A-B-C\", got \"{}\"",
                    name
                );
                self.def_angletype(parts[0], parts[1], parts[2], params);
            }
            StyleCategory::Dihedral => {
                assert!(
                    parts.len() == 4,
                    "dihedral type name must be \"A-B-C-D\", got \"{}\"",
                    name
                );
                self.def_dihedraltype(parts[0], parts[1], parts[2], parts[3], params);
            }
            StyleCategory::Improper => {
                assert!(
                    parts.len() == 4,
                    "improper type name must be \"A-B-C-D\", got \"{}\"",
                    name
                );
                self.def_impropertype(parts[0], parts[1], parts[2], parts[3], params);
            }
            StyleCategory::Pair => match parts.len() {
                1 => {
                    self.def_pairtype(parts[0], None, params);
                }
                2 => {
                    self.def_pairtype(parts[0], Some(parts[1]), params);
                }
                _ => panic!("pair type name must be \"A\" or \"A-B\", got \"{}\"", name),
            },
        }
        self
    }

    pub fn get_pairtype(&self, itom: &str, jtom: Option<&str>) -> Option<&PairType> {
        let jtom_str = jtom.unwrap_or(itom);
        self.pairtypes.iter().find(|t| {
            (t.itom == itom && t.jtom == jtom_str) || (t.itom == jtom_str && t.jtom == itom)
        })
    }
}

// ---------------------------------------------------------------------------
// ForceField
// ---------------------------------------------------------------------------

/// Top-level forcefield container holding styles and their type definitions.
///
/// # Example
///
/// ```
/// use molrs::core::forcefield::ForceField;
///
/// let mut ff = ForceField::new("example");
/// ff.def_bondstyle("harmonic")
///     .def_type("A-B", &[("k", 300.0), ("r0", 1.5)]);
/// ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
///     .def_type("A", &[("epsilon", 0.5), ("sigma", 1.0)]);
///
/// // Compile into a PotentialSet (no Frame needed)
/// let pots = ff.to_potentials();
/// // Then use pots.energy(&frame) with a Frame that has atoms + topology blocks
/// ```
#[derive(Debug, Clone)]
pub struct ForceField {
    pub name: String,
    styles: Vec<Style>,
}

impl ForceField {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            styles: Vec::new(),
        }
    }

    // -- def_*style: register or retrieve existing style --

    fn def_style(&mut self, category: StyleCategory, name: &str, params: Params) -> &mut Style {
        // Return existing if same category+name
        if let Some(idx) = self
            .styles
            .iter()
            .position(|s| s.category == category && s.name == name)
        {
            return &mut self.styles[idx];
        }
        self.styles.push(Style::new(category, name, params));
        self.styles.last_mut().unwrap()
    }

    pub fn def_atomstyle(&mut self, name: &str) -> &mut Style {
        self.def_style(StyleCategory::Atom, name, Params::new())
    }

    pub fn def_bondstyle(&mut self, name: &str) -> &mut Style {
        self.def_style(StyleCategory::Bond, name, Params::new())
    }

    pub fn def_anglestyle(&mut self, name: &str) -> &mut Style {
        self.def_style(StyleCategory::Angle, name, Params::new())
    }

    pub fn def_dihedralstyle(&mut self, name: &str) -> &mut Style {
        self.def_style(StyleCategory::Dihedral, name, Params::new())
    }

    pub fn def_improperstyle(&mut self, name: &str) -> &mut Style {
        self.def_style(StyleCategory::Improper, name, Params::new())
    }

    pub fn def_pairstyle(&mut self, name: &str, params: &[(&str, f64)]) -> &mut Style {
        self.def_style(StyleCategory::Pair, name, Params::from_pairs(params))
    }

    // -- with_* builder pattern (consumes and returns self) --

    pub fn with_atomstyle(mut self, name: &str) -> Self {
        self.def_atomstyle(name);
        self
    }

    pub fn with_bondstyle(mut self, name: &str) -> Self {
        self.def_bondstyle(name);
        self
    }

    pub fn with_anglestyle(mut self, name: &str) -> Self {
        self.def_anglestyle(name);
        self
    }

    pub fn with_pairstyle(mut self, name: &str, params: &[(&str, f64)]) -> Self {
        self.def_pairstyle(name, params);
        self
    }

    // -- queries --

    pub fn styles(&self) -> &[Style] {
        &self.styles
    }

    pub fn get_style(&self, category: StyleCategory, name: &str) -> Option<&Style> {
        self.styles
            .iter()
            .find(|s| s.category == category && s.name == name)
    }

    pub fn get_styles(&self, category: StyleCategory) -> Vec<&Style> {
        self.styles
            .iter()
            .filter(|s| s.category == category)
            .collect()
    }

    pub fn get_atomtypes(&self) -> Vec<&AtomType> {
        self.styles
            .iter()
            .filter(|s| s.category == StyleCategory::Atom)
            .flat_map(|s| s.atomtypes.iter())
            .collect()
    }

    pub fn get_bondtypes(&self) -> Vec<&BondType> {
        self.styles
            .iter()
            .filter(|s| s.category == StyleCategory::Bond)
            .flat_map(|s| s.bondtypes.iter())
            .collect()
    }

    pub fn get_angletypes(&self) -> Vec<&AngleType> {
        self.styles
            .iter()
            .filter(|s| s.category == StyleCategory::Angle)
            .flat_map(|s| s.angletypes.iter())
            .collect()
    }

    pub fn get_pairtypes(&self) -> Vec<&PairType> {
        self.styles
            .iter()
            .filter(|s| s.category == StyleCategory::Pair)
            .flat_map(|s| s.pairtypes.iter())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params() {
        let p = Params::from_pairs(&[("k", 300.0), ("r0", 1.5)]);
        assert_eq!(p.get("k"), Some(300.0));
        assert_eq!(p.get("r0"), Some(1.5));
        assert_eq!(p.get("missing"), None);
    }

    #[test]
    fn test_def_atomstyle_and_types() {
        let mut ff = ForceField::new("test");
        let style = ff.def_atomstyle("full");
        style.def_atomtype("CT", &[("mass", 12.011), ("charge", -0.12)]);
        style.def_atomtype("HC", &[("mass", 1.008), ("charge", 0.06)]);

        let style = ff.get_style(StyleCategory::Atom, "full").unwrap();
        assert_eq!(style.atomtypes.len(), 2);

        let ct = style.get_atomtype("CT").unwrap();
        assert_eq!(ct.params.get("mass"), Some(12.011));
        assert_eq!(ct.params.get("charge"), Some(-0.12));
    }

    #[test]
    fn test_def_bondstyle_and_types() {
        let mut ff = ForceField::new("test");
        let style = ff.def_bondstyle("harmonic");
        style.def_bondtype("CT", "CT", &[("k", 268.0), ("r0", 1.529)]);
        style.def_bondtype("CT", "HC", &[("k", 340.0), ("r0", 1.09)]);

        let style = ff.get_style(StyleCategory::Bond, "harmonic").unwrap();
        assert_eq!(style.bondtypes.len(), 2);

        let bt = style.get_bondtype("CT", "CT").unwrap();
        assert_eq!(bt.params.get("k"), Some(268.0));

        // Order-independent lookup
        let bt2 = style.get_bondtype("HC", "CT").unwrap();
        assert_eq!(bt2.params.get("r0"), Some(1.09));
    }

    #[test]
    fn test_def_anglestyle_and_types() {
        let mut ff = ForceField::new("test");
        let style = ff.def_anglestyle("harmonic");
        style.def_angletype("HC", "CT", "HC", &[("k", 33.0), ("theta0", 107.8)]);

        let types = ff.get_angletypes();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].params.get("theta0"), Some(107.8));
    }

    #[test]
    fn test_def_pairstyle_and_types() {
        let mut ff = ForceField::new("test");
        let style = ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)]);
        style.def_pairtype("CT", None, &[("epsilon", 0.066), ("sigma", 3.5)]);
        style.def_pairtype("CT", Some("OH"), &[("epsilon", 0.1), ("sigma", 3.3)]);

        let style = ff.get_style(StyleCategory::Pair, "lj/cut").unwrap();
        assert_eq!(style.params.get("cutoff"), Some(10.0));
        assert_eq!(style.pairtypes.len(), 2);

        // self-interaction
        let pt = style.get_pairtype("CT", None).unwrap();
        assert_eq!(pt.itom, "CT");
        assert_eq!(pt.jtom, "CT");

        // cross-interaction (order-independent)
        let pt2 = style.get_pairtype("OH", Some("CT")).unwrap();
        assert_eq!(pt2.params.get("epsilon"), Some(0.1));
    }

    #[test]
    fn test_duplicate_style_returns_existing() {
        let mut ff = ForceField::new("test");
        ff.def_bondstyle("harmonic")
            .def_bondtype("A", "B", &[("k", 1.0), ("r0", 1.0)]);

        // Second call returns the same style, not a new one
        ff.def_bondstyle("harmonic")
            .def_bondtype("C", "D", &[("k", 2.0), ("r0", 2.0)]);

        let styles = ff.get_styles(StyleCategory::Bond);
        assert_eq!(styles.len(), 1);
        assert_eq!(styles[0].bondtypes.len(), 2);
    }

    #[test]
    fn test_builder_pattern() {
        let ff = ForceField::new("TIP3P")
            .with_atomstyle("full")
            .with_bondstyle("harmonic")
            .with_pairstyle("lj/cut", &[("cutoff", 10.0)]);

        assert_eq!(ff.styles().len(), 3);
        assert!(ff.get_style(StyleCategory::Atom, "full").is_some());
        assert!(ff.get_style(StyleCategory::Bond, "harmonic").is_some());
        assert!(ff.get_style(StyleCategory::Pair, "lj/cut").is_some());
    }

    // --- def_type unified tests ---

    #[test]
    fn test_def_type_bond() {
        let mut ff = ForceField::new("test");
        let style = ff.def_bondstyle("harmonic");
        style.def_type("CT-OH", &[("k", 300.0), ("r0", 1.4)]);

        let bt = style.get_bondtype("CT", "OH").unwrap();
        assert_eq!(bt.itom, "CT");
        assert_eq!(bt.jtom, "OH");
        assert_eq!(bt.params.get("k"), Some(300.0));
        assert_eq!(bt.params.get("r0"), Some(1.4));
    }

    #[test]
    fn test_def_type_angle() {
        let mut ff = ForceField::new("test");
        let style = ff.def_anglestyle("harmonic");
        style.def_type("HC-CT-HC", &[("k", 33.0), ("theta0", 107.8)]);

        assert_eq!(style.angletypes.len(), 1);
        assert_eq!(style.angletypes[0].itom, "HC");
        assert_eq!(style.angletypes[0].jtom, "CT");
        assert_eq!(style.angletypes[0].ktom, "HC");
        assert_eq!(style.angletypes[0].params.get("theta0"), Some(107.8));
    }

    #[test]
    fn test_def_type_pair_self() {
        let mut ff = ForceField::new("test");
        let style = ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)]);
        style.def_type("Ar", &[("epsilon", 1.0), ("sigma", 3.4)]);

        let pt = style.get_pairtype("Ar", None).unwrap();
        assert_eq!(pt.itom, "Ar");
        assert_eq!(pt.jtom, "Ar");
        assert_eq!(pt.params.get("epsilon"), Some(1.0));
    }

    #[test]
    fn test_def_type_pair_cross() {
        let mut ff = ForceField::new("test");
        let style = ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)]);
        style.def_type("CT-OH", &[("epsilon", 0.1), ("sigma", 3.3)]);

        let pt = style.get_pairtype("CT", Some("OH")).unwrap();
        assert_eq!(pt.itom, "CT");
        assert_eq!(pt.jtom, "OH");
    }

    #[test]
    fn test_def_type_dihedral() {
        let mut ff = ForceField::new("test");
        let style = ff.def_dihedralstyle("opls");
        style.def_type("HC-CT-CT-HC", &[("k1", 0.0), ("k2", 0.0), ("k3", 0.3)]);

        assert_eq!(style.dihedraltypes.len(), 1);
        assert_eq!(style.dihedraltypes[0].itom, "HC");
        assert_eq!(style.dihedraltypes[0].ltom, "HC");
    }

    #[test]
    fn test_def_type_chaining() {
        let mut ff = ForceField::new("test");
        let style = ff.def_bondstyle("harmonic");
        style
            .def_type("A-B", &[("k", 1.0), ("r0", 1.0)])
            .def_type("C-D", &[("k", 2.0), ("r0", 2.0)]);

        assert_eq!(style.bondtypes.len(), 2);
    }

    #[test]
    #[should_panic(expected = "bond type name must be")]
    fn test_def_type_bond_invalid_format() {
        let mut ff = ForceField::new("test");
        let style = ff.def_bondstyle("harmonic");
        style.def_type("CT", &[("k", 300.0), ("r0", 1.4)]);
    }

    #[test]
    fn test_get_all_types() {
        let mut ff = ForceField::new("test");

        let style = ff.def_atomstyle("full");
        style.def_atomtype("CT", &[("mass", 12.0)]);
        style.def_atomtype("OH", &[("mass", 16.0)]);

        let style = ff.def_bondstyle("harmonic");
        style.def_bondtype("CT", "OH", &[("k", 300.0), ("r0", 1.4)]);

        assert_eq!(ff.get_atomtypes().len(), 2);
        assert_eq!(ff.get_bondtypes().len(), 1);
        assert_eq!(ff.get_pairtypes().len(), 0);
    }
}
