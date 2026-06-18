//! OPLS-AA SMARTS atom typifier.
//!
//! Mirrors [`mmff`](crate::ff::typifier::mmff): typing metadata
//! ([`OplsTypingMeta`]) is read *separately* from the potential
//! [`ForceField`](crate::ff::forcefield::ForceField), both from the same OPLS-AA
//! XML. [`OplsTypifier`] owns both and implements [`Typifier`], assigning
//! `opls_NNN` atom types by SMARTS matching with overrides / priority / layer
//! conflict resolution (replicating molpy's `_OplsAtomTypifier`).
//!
//! # Scope
//!
//! Only types carrying a SMARTS `def` participate; legacy `oplsaa.xml` rows
//! (`opls_001`–`opls_134`, no `def`) are out of scope for auto-typing. Bonded
//! parameter assignment (bonds / angles / dihedrals) is chain 2.

use molrs::Atomistic;

use crate::ff::forcefield::ForceField;
use crate::ff::forcefield::readers::{ForceFieldReader, opls::OplsXmlReader};

use super::Typifier;

pub mod meta;
pub mod typing;

pub use meta::{LAYER_PRIORITY_STRIDE, OplsTypeRow, OplsTypingMeta};
pub use typing::annotate_opls;

/// OPLS-AA typifier — owns typing metadata and force-field parameters.
///
/// Primary constructor [`from_xml_str`](Self::from_xml_str) parses both the
/// typing metadata ([`OplsTypingMeta`]) and the potential parameters
/// ([`ForceField`]) from a single OPLS-AA XML string.
pub struct OplsTypifier {
    meta: OplsTypingMeta,
    ff: ForceField,
}

impl OplsTypifier {
    /// Build a typifier from an OPLS-AA / GROMACS XML string.
    ///
    /// Reads typing metadata and potential parameters in one call. The two are
    /// read by independent parsers from the same XML and never share state.
    ///
    /// # Errors
    ///
    /// Returns `Err` if either parse fails.
    pub fn from_xml_str(xml: &str) -> Result<Self, String> {
        let meta = crate::ff::forcefield::xml::read_opls_typing_xml_str(xml)?;
        let ff = OplsXmlReader::new().read_str(xml)?;
        Ok(Self { meta, ff })
    }

    /// Construct directly from already-parsed metadata and force field.
    pub fn new(meta: OplsTypingMeta, ff: ForceField) -> Self {
        Self { meta, ff }
    }

    /// Access the typing metadata.
    pub fn meta(&self) -> &OplsTypingMeta {
        &self.meta
    }

    /// Access the potential force field.
    pub fn ff(&self) -> &ForceField {
        &self.ff
    }
}

impl Typifier for OplsTypifier {
    fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        annotate_opls(mol, &self.meta, &self.ff)
    }
}
