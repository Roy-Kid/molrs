//! Python wrappers for the `MolGraph` hierarchy.
//!
//! `molrs-core` models molecular topology as a general graph
//! ([`molrs::molgraph::MolGraph`]) whose nodes are atoms *or* beads and whose
//! higher-order topology (bonds, angles, dihedrals, impropers) each carries a
//! generic property bag. This module exposes that hierarchy faithfully:
//!
//! - [`PyGraph`] (`molrs.Graph`) — the general base carrying **all** methods.
//! - [`PyAtomistic`] (`molrs.Atomistic`) — `extends=Graph`, atom-flavoured
//!   convenience constructor (`add_atom` writes `"element"`).
//! - [`PyCoarseGrain`] (`molrs.CoarseGrain`) — `extends=Graph`, bead-flavoured
//!   convenience constructor (`add_bead` writes `"bead_type"`).
//!
//! # Index model
//!
//! The Python API speaks **0-based indices** = position in iteration order.
//! `index_to_atom_id(i)` is the `i`-th item from `inner.atoms()`;
//! `atom_id_to_index(id)` is its position. Bonds / angles / dihedrals /
//! impropers expose the same index ↔ id resolution over their own collections.
//! Resolution is O(n) per call, which is fine for the sizes molpy builds.

use pyo3::prelude::*;
use pyo3::types::PyList;

use molrs::molgraph::{AngleId, Atom, AtomId, BondId, DihedralId, ImproperId, MolGraph, PropValue};

use crate::frame::PyFrame;
use crate::helpers::molrs_error_to_pyerr;

// ---------------------------------------------------------------------------
// Shared property conversion helpers
// ---------------------------------------------------------------------------

/// Convert a Python scalar to a [`PropValue`].
///
/// Dispatch order matters: a Python `bool` is an `int` subclass and `1`/`1.0`
/// must stay distinct, so `int` is tried before `float`.
fn py_to_prop(value: &Bound<'_, PyAny>) -> PyResult<PropValue> {
    if let Ok(i) = value.extract::<i64>() {
        Ok(PropValue::Int(i as i32))
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(PropValue::F64(f))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(PropValue::Str(s))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "property value must be int, float, or str",
        ))
    }
}

/// Convert a [`PropValue`] to a Python object.
fn prop_to_py(py: Python<'_>, value: &PropValue) -> PyResult<Py<PyAny>> {
    match value {
        PropValue::Int(i) => Ok(i.into_pyobject(py)?.into_any().unbind()),
        PropValue::F64(f) => Ok(f.into_pyobject(py)?.into_any().unbind()),
        PropValue::Str(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
    }
}

fn py_none(py: Python<'_>) -> Py<PyAny> {
    py.None()
}

// ---------------------------------------------------------------------------
// PyGraph — the general base
// ---------------------------------------------------------------------------

/// General molecular graph, exposed to Python as `molrs.Graph`.
///
/// Nodes are atoms *or* beads (a generic property bag each); higher-order
/// topology (bonds, angles, dihedrals, impropers) each carry their own
/// property bag. This is the base class for :class:`Atomistic` and
/// :class:`CoarseGrain`; it carries the full method surface.
///
/// # Python Examples
///
/// ```python
/// import molrs
///
/// g = molrs.Graph()
/// c = g.add_atom("C")
/// o = g.add_atom("O")
/// g.add_bond(c, o)
/// g.set_bond_order(c, o, 2.0)
/// print(g)  # Graph(atoms=2, bonds=1, angles=0, dihedrals=0, impropers=0)
/// ```
#[pyclass(name = "Graph", subclass, unsendable)]
pub struct PyGraph {
    pub(crate) inner: MolGraph,
}

#[pymethods]
impl PyGraph {
    /// Create an empty graph.
    ///
    /// Extra positional / keyword arguments are accepted and ignored so that a
    /// Python subclass `class S(molrs.Graph, SomeMixin)` needs **no**
    /// `__new__` shim — cooperative `super().__init__(...)` just works.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyAny>, _kwargs: Option<&Bound<'_, PyAny>>) -> Self {
        Self {
            inner: MolGraph::new(),
        }
    }

    // ---- atoms -----------------------------------------------------------

    /// Add an atom with the given element symbol and optional 3D coordinates.
    ///
    /// Sets ``"element"`` to ``symbol`` and, if all of ``x``/``y``/``z`` are
    /// provided, the ``"x"``/``"y"``/``"z"`` coordinate properties.
    ///
    /// Returns the 0-based index of the new atom.
    #[pyo3(signature = (symbol, x=None, y=None, z=None))]
    fn add_atom(&mut self, symbol: &str, x: Option<f64>, y: Option<f64>, z: Option<f64>) -> usize {
        self.add_node("element", symbol, x, y, z)
    }

    /// Add a coarse-grained bead with the given bead type and optional coords.
    ///
    /// Sets ``"bead_type"`` to ``bead_type``; otherwise identical to
    /// :meth:`add_atom`. Returns the 0-based index of the new node.
    #[pyo3(signature = (bead_type, x=None, y=None, z=None))]
    fn add_bead(
        &mut self,
        bead_type: &str,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
    ) -> usize {
        self.add_node("bead_type", bead_type, x, y, z)
    }

    /// Remove an atom (by index), cascade-deleting any topology that uses it.
    fn remove_atom(&mut self, i: usize) -> PyResult<()> {
        let aid = self.index_to_atom_id(i)?;
        self.inner.remove_atom(aid).map_err(molrs_error_to_pyerr)?;
        Ok(())
    }

    /// Number of atoms (nodes) in the graph.
    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Neighbours of atom ``i``, returned as a list of 0-based indices.
    fn neighbors(&self, i: usize) -> PyResult<Vec<usize>> {
        let aid = self.index_to_atom_id(i)?;
        let ids: Vec<AtomId> = self.inner.neighbors(aid).collect();
        Ok(ids
            .into_iter()
            .map(|id| self.atom_id_to_index(id))
            .collect())
    }

    /// Property keys present on atom ``i``.
    fn atom_keys(&self, i: usize) -> PyResult<Vec<String>> {
        let aid = self.index_to_atom_id(i)?;
        let atom = self.inner.get_atom(aid).map_err(molrs_error_to_pyerr)?;
        Ok(atom.keys().map(|k| k.to_owned()).collect())
    }

    /// Read a property from atom ``i`` (``None`` if absent).
    fn get_atom_prop(&self, py: Python<'_>, i: usize, key: &str) -> PyResult<Py<PyAny>> {
        let aid = self.index_to_atom_id(i)?;
        let atom = self.inner.get_atom(aid).map_err(molrs_error_to_pyerr)?;
        atom_prop_to_py(py, atom, key)
    }

    /// Set an arbitrary property on atom ``i``. ``value`` is ``int|float|str``.
    fn set_atom_prop(&mut self, index: usize, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let pv = py_to_prop(value)?;
        let aid = self.index_to_atom_id(index)?;
        let atom = self.inner.get_atom_mut(aid).map_err(molrs_error_to_pyerr)?;
        atom.set(key, pv);
        Ok(())
    }

    /// Remove a property from atom ``i`` (no-op if absent).
    fn del_atom_prop(&mut self, i: usize, key: &str) -> PyResult<()> {
        let aid = self.index_to_atom_id(i)?;
        let atom = self.inner.get_atom_mut(aid).map_err(molrs_error_to_pyerr)?;
        atom.remove(key);
        Ok(())
    }

    /// Column of ``key`` across all atoms in order (value-or-``None`` per atom).
    fn atom_column(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let mut items: Vec<Py<PyAny>> = Vec::with_capacity(self.inner.n_atoms());
        for (_, atom) in self.inner.atoms() {
            items.push(atom_prop_to_py(py, atom, key)?);
        }
        Ok(PyList::new(py, items)?.into_any().unbind())
    }

    /// N×3 list of ``[x, y, z]`` per atom (missing coordinate → ``0.0``).
    fn coords(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut rows: Vec<[f64; 3]> = Vec::with_capacity(self.inner.n_atoms());
        for (_, atom) in self.inner.atoms() {
            rows.push([
                atom.get_f64("x").unwrap_or(0.0),
                atom.get_f64("y").unwrap_or(0.0),
                atom.get_f64("z").unwrap_or(0.0),
            ]);
        }
        Ok(PyList::new(py, rows)?.into_any().unbind())
    }

    /// Overwrite per-atom coordinates from an N×3 list of rows.
    fn set_coords(&mut self, rows: Vec<[f64; 3]>) -> PyResult<()> {
        let ids: Vec<AtomId> = self.inner.atoms().map(|(id, _)| id).collect();
        if rows.len() != ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "set_coords expects {} rows, got {}",
                ids.len(),
                rows.len()
            )));
        }
        for (aid, row) in ids.into_iter().zip(rows.into_iter()) {
            let atom = self.inner.get_atom_mut(aid).map_err(molrs_error_to_pyerr)?;
            atom.set("x", row[0]);
            atom.set("y", row[1]);
            atom.set("z", row[2]);
        }
        Ok(())
    }

    /// Translate every atom by ``vec`` (added to ``x``/``y``/``z``).
    fn translate(&mut self, vec: [f64; 3]) -> PyResult<()> {
        let ids: Vec<AtomId> = self.inner.atoms().map(|(id, _)| id).collect();
        for aid in ids {
            let atom = self.inner.get_atom_mut(aid).map_err(molrs_error_to_pyerr)?;
            let x = atom.get_f64("x").unwrap_or(0.0) + vec[0];
            let y = atom.get_f64("y").unwrap_or(0.0) + vec[1];
            let z = atom.get_f64("z").unwrap_or(0.0) + vec[2];
            atom.set("x", x);
            atom.set("y", y);
            atom.set("z", z);
        }
        Ok(())
    }

    // ---- bonds -----------------------------------------------------------

    /// Add a bond between atoms ``i`` and ``j`` (by index).
    fn add_bond(&mut self, i: usize, j: usize) -> PyResult<()> {
        let ai = self.index_to_atom_id(i)?;
        let aj = self.index_to_atom_id(j)?;
        self.inner.add_bond(ai, aj).map_err(molrs_error_to_pyerr)?;
        Ok(())
    }

    /// Set the bond order of the bond between atoms ``i`` and ``j``.
    fn set_bond_order(&mut self, i: usize, j: usize, order: f64) -> PyResult<()> {
        let ai = self.index_to_atom_id(i)?;
        let aj = self.index_to_atom_id(j)?;
        let target_bid = self.inner.bonds().find_map(|(bid, bond)| {
            if (bond.atoms[0] == ai && bond.atoms[1] == aj)
                || (bond.atoms[0] == aj && bond.atoms[1] == ai)
            {
                Some(bid)
            } else {
                None
            }
        });
        match target_bid {
            Some(bid) => {
                self.inner
                    .get_bond_mut(bid)
                    .map_err(molrs_error_to_pyerr)?
                    .props
                    .insert("order".into(), PropValue::F64(order));
                Ok(())
            }
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "no bond between atoms {} and {}",
                i, j
            ))),
        }
    }

    /// Endpoint atom indices ``(i, j)`` of bond ``b``.
    fn get_bond_atoms(&self, b: usize) -> PyResult<(usize, usize)> {
        let bid = self.index_to_bond_id(b)?;
        let bond = self.inner.get_bond(bid).map_err(molrs_error_to_pyerr)?;
        Ok((
            self.atom_id_to_index(bond.atoms[0]),
            self.atom_id_to_index(bond.atoms[1]),
        ))
    }

    /// Property keys present on bond ``b``.
    fn bond_keys(&self, b: usize) -> PyResult<Vec<String>> {
        let bid = self.index_to_bond_id(b)?;
        let bond = self.inner.get_bond(bid).map_err(molrs_error_to_pyerr)?;
        Ok(bond.props.keys().cloned().collect())
    }

    /// Read a property from bond ``b`` (``None`` if absent).
    fn get_bond_prop(&self, py: Python<'_>, b: usize, key: &str) -> PyResult<Py<PyAny>> {
        let bid = self.index_to_bond_id(b)?;
        let bond = self.inner.get_bond(bid).map_err(molrs_error_to_pyerr)?;
        match bond.props.get(key) {
            Some(v) => prop_to_py(py, v),
            None => Ok(py_none(py)),
        }
    }

    /// Set a property on bond ``b``. ``value`` is ``int|float|str``.
    fn set_bond_prop(&mut self, b: usize, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let pv = py_to_prop(value)?;
        let bid = self.index_to_bond_id(b)?;
        self.inner
            .get_bond_mut(bid)
            .map_err(molrs_error_to_pyerr)?
            .props
            .insert(key.to_owned(), pv);
        Ok(())
    }

    /// Remove a property from bond ``b`` (no-op if absent).
    fn del_bond_prop(&mut self, b: usize, key: &str) -> PyResult<()> {
        let bid = self.index_to_bond_id(b)?;
        self.inner
            .get_bond_mut(bid)
            .map_err(molrs_error_to_pyerr)?
            .props
            .remove(key);
        Ok(())
    }

    /// Remove bond ``b``.
    fn remove_bond(&mut self, b: usize) -> PyResult<()> {
        let bid = self.index_to_bond_id(b)?;
        self.inner.remove_bond(bid).map_err(molrs_error_to_pyerr)?;
        Ok(())
    }

    /// Number of bonds in the graph.
    #[getter]
    fn n_bonds(&self) -> usize {
        self.inner.n_bonds()
    }

    // ---- angles ----------------------------------------------------------

    /// Add an angle over atoms ``i``-``j``-``k`` (``j`` is the vertex).
    fn add_angle(&mut self, i: usize, j: usize, k: usize) -> PyResult<usize> {
        let ai = self.index_to_atom_id(i)?;
        let aj = self.index_to_atom_id(j)?;
        let ak = self.index_to_atom_id(k)?;
        let id = self
            .inner
            .add_angle(ai, aj, ak)
            .map_err(molrs_error_to_pyerr)?;
        Ok(self.angle_id_to_index(id))
    }

    /// Endpoint atom indices ``(i, j, k)`` of angle ``a``.
    fn get_angle_atoms(&self, a: usize) -> PyResult<(usize, usize, usize)> {
        let id = self.index_to_angle_id(a)?;
        let angle = self.inner.get_angle(id).map_err(molrs_error_to_pyerr)?;
        Ok((
            self.atom_id_to_index(angle.atoms[0]),
            self.atom_id_to_index(angle.atoms[1]),
            self.atom_id_to_index(angle.atoms[2]),
        ))
    }

    /// Property keys present on angle ``a``.
    fn angle_keys(&self, a: usize) -> PyResult<Vec<String>> {
        let id = self.index_to_angle_id(a)?;
        let angle = self.inner.get_angle(id).map_err(molrs_error_to_pyerr)?;
        Ok(angle.props.keys().cloned().collect())
    }

    /// Read a property from angle ``a`` (``None`` if absent).
    fn get_angle_prop(&self, py: Python<'_>, a: usize, key: &str) -> PyResult<Py<PyAny>> {
        let id = self.index_to_angle_id(a)?;
        let angle = self.inner.get_angle(id).map_err(molrs_error_to_pyerr)?;
        match angle.props.get(key) {
            Some(v) => prop_to_py(py, v),
            None => Ok(py_none(py)),
        }
    }

    /// Set a property on angle ``a``. ``value`` is ``int|float|str``.
    fn set_angle_prop(&mut self, a: usize, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let pv = py_to_prop(value)?;
        let id = self.index_to_angle_id(a)?;
        self.inner
            .get_angle_mut(id)
            .map_err(molrs_error_to_pyerr)?
            .props
            .insert(key.to_owned(), pv);
        Ok(())
    }

    /// Remove a property from angle ``a`` (no-op if absent).
    fn del_angle_prop(&mut self, a: usize, key: &str) -> PyResult<()> {
        let id = self.index_to_angle_id(a)?;
        self.inner
            .get_angle_mut(id)
            .map_err(molrs_error_to_pyerr)?
            .props
            .remove(key);
        Ok(())
    }

    /// Remove angle ``a``.
    fn remove_angle(&mut self, a: usize) -> PyResult<()> {
        let id = self.index_to_angle_id(a)?;
        self.inner.remove_angle(id).map_err(molrs_error_to_pyerr)?;
        Ok(())
    }

    /// Number of angles in the graph.
    #[getter]
    fn n_angles(&self) -> usize {
        self.inner.n_angles()
    }

    // ---- dihedrals -------------------------------------------------------

    /// Add a dihedral over atoms ``i``-``j``-``k``-``l``.
    fn add_dihedral(&mut self, i: usize, j: usize, k: usize, l: usize) -> PyResult<usize> {
        let ai = self.index_to_atom_id(i)?;
        let aj = self.index_to_atom_id(j)?;
        let ak = self.index_to_atom_id(k)?;
        let al = self.index_to_atom_id(l)?;
        let id = self
            .inner
            .add_dihedral(ai, aj, ak, al)
            .map_err(molrs_error_to_pyerr)?;
        Ok(self.dihedral_id_to_index(id))
    }

    /// Endpoint atom indices ``(i, j, k, l)`` of dihedral ``d``.
    fn get_dihedral_atoms(&self, d: usize) -> PyResult<(usize, usize, usize, usize)> {
        let id = self.index_to_dihedral_id(d)?;
        let dih = self.inner.get_dihedral(id).map_err(molrs_error_to_pyerr)?;
        Ok((
            self.atom_id_to_index(dih.atoms[0]),
            self.atom_id_to_index(dih.atoms[1]),
            self.atom_id_to_index(dih.atoms[2]),
            self.atom_id_to_index(dih.atoms[3]),
        ))
    }

    /// Property keys present on dihedral ``d``.
    fn dihedral_keys(&self, d: usize) -> PyResult<Vec<String>> {
        let id = self.index_to_dihedral_id(d)?;
        let dih = self.inner.get_dihedral(id).map_err(molrs_error_to_pyerr)?;
        Ok(dih.props.keys().cloned().collect())
    }

    /// Read a property from dihedral ``d`` (``None`` if absent).
    fn get_dihedral_prop(&self, py: Python<'_>, d: usize, key: &str) -> PyResult<Py<PyAny>> {
        let id = self.index_to_dihedral_id(d)?;
        let dih = self.inner.get_dihedral(id).map_err(molrs_error_to_pyerr)?;
        match dih.props.get(key) {
            Some(v) => prop_to_py(py, v),
            None => Ok(py_none(py)),
        }
    }

    /// Set a property on dihedral ``d``. ``value`` is ``int|float|str``.
    fn set_dihedral_prop(&mut self, d: usize, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let pv = py_to_prop(value)?;
        let id = self.index_to_dihedral_id(d)?;
        self.inner
            .get_dihedral_mut(id)
            .map_err(molrs_error_to_pyerr)?
            .props
            .insert(key.to_owned(), pv);
        Ok(())
    }

    /// Remove a property from dihedral ``d`` (no-op if absent).
    fn del_dihedral_prop(&mut self, d: usize, key: &str) -> PyResult<()> {
        let id = self.index_to_dihedral_id(d)?;
        self.inner
            .get_dihedral_mut(id)
            .map_err(molrs_error_to_pyerr)?
            .props
            .remove(key);
        Ok(())
    }

    /// Remove dihedral ``d``.
    fn remove_dihedral(&mut self, d: usize) -> PyResult<()> {
        let id = self.index_to_dihedral_id(d)?;
        self.inner
            .remove_dihedral(id)
            .map_err(molrs_error_to_pyerr)?;
        Ok(())
    }

    /// Number of dihedrals in the graph.
    #[getter]
    fn n_dihedrals(&self) -> usize {
        self.inner.n_dihedrals()
    }

    // ---- impropers -------------------------------------------------------

    /// Add an improper over atoms ``i``-``j``-``k``-``l``.
    fn add_improper(&mut self, i: usize, j: usize, k: usize, l: usize) -> PyResult<usize> {
        let ai = self.index_to_atom_id(i)?;
        let aj = self.index_to_atom_id(j)?;
        let ak = self.index_to_atom_id(k)?;
        let al = self.index_to_atom_id(l)?;
        let id = self
            .inner
            .add_improper(ai, aj, ak, al)
            .map_err(molrs_error_to_pyerr)?;
        Ok(self.improper_id_to_index(id))
    }

    /// Endpoint atom indices ``(i, j, k, l)`` of improper ``m``.
    fn get_improper_atoms(&self, m: usize) -> PyResult<(usize, usize, usize, usize)> {
        let id = self.index_to_improper_id(m)?;
        let imp = self.inner.get_improper(id).map_err(molrs_error_to_pyerr)?;
        Ok((
            self.atom_id_to_index(imp.atoms[0]),
            self.atom_id_to_index(imp.atoms[1]),
            self.atom_id_to_index(imp.atoms[2]),
            self.atom_id_to_index(imp.atoms[3]),
        ))
    }

    /// Property keys present on improper ``m``.
    fn improper_keys(&self, m: usize) -> PyResult<Vec<String>> {
        let id = self.index_to_improper_id(m)?;
        let imp = self.inner.get_improper(id).map_err(molrs_error_to_pyerr)?;
        Ok(imp.props.keys().cloned().collect())
    }

    /// Read a property from improper ``m`` (``None`` if absent).
    fn get_improper_prop(&self, py: Python<'_>, m: usize, key: &str) -> PyResult<Py<PyAny>> {
        let id = self.index_to_improper_id(m)?;
        let imp = self.inner.get_improper(id).map_err(molrs_error_to_pyerr)?;
        match imp.props.get(key) {
            Some(v) => prop_to_py(py, v),
            None => Ok(py_none(py)),
        }
    }

    /// Set a property on improper ``m``. ``value`` is ``int|float|str``.
    fn set_improper_prop(&mut self, m: usize, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let pv = py_to_prop(value)?;
        let id = self.index_to_improper_id(m)?;
        self.inner
            .get_improper_mut(id)
            .map_err(molrs_error_to_pyerr)?
            .props
            .insert(key.to_owned(), pv);
        Ok(())
    }

    /// Remove a property from improper ``m`` (no-op if absent).
    fn del_improper_prop(&mut self, m: usize, key: &str) -> PyResult<()> {
        let id = self.index_to_improper_id(m)?;
        self.inner
            .get_improper_mut(id)
            .map_err(molrs_error_to_pyerr)?
            .props
            .remove(key);
        Ok(())
    }

    /// Remove improper ``m``.
    fn remove_improper(&mut self, m: usize) -> PyResult<()> {
        let id = self.index_to_improper_id(m)?;
        self.inner
            .remove_improper(id)
            .map_err(molrs_error_to_pyerr)?;
        Ok(())
    }

    /// Number of impropers in the graph.
    #[getter]
    fn n_impropers(&self) -> usize {
        self.inner.n_impropers()
    }

    // ---- geometry / merge / export --------------------------------------

    /// Rotate all atom coordinates about ``axis`` by ``angle`` radians.
    ///
    /// ``about`` is the optional pivot point (defaults to the origin).
    #[pyo3(signature = (axis, angle, about=None))]
    fn rotate(&mut self, axis: [f64; 3], angle: f64, about: Option<[f64; 3]>) {
        self.inner.rotate(axis, angle, about);
    }

    /// Merge ``other`` into ``self`` (atoms, bonds, angles, dihedrals,
    /// impropers and their properties are all copied; indices are re-mapped).
    fn extend(&mut self, other: PyRef<'_, PyGraph>) -> PyResult<()> {
        let src = &other.inner;

        // Atoms: clone each, building a src AtomId → dst AtomId map.
        let mut id_map: std::collections::HashMap<AtomId, AtomId> =
            std::collections::HashMap::new();
        for (src_id, atom) in src.atoms() {
            let new_id = self.inner.add_atom(atom.clone());
            id_map.insert(src_id, new_id);
        }

        // Bonds.
        for (_, bond) in src.bonds() {
            let a = id_map[&bond.atoms[0]];
            let b = id_map[&bond.atoms[1]];
            let new_id = self.inner.add_bond(a, b).map_err(molrs_error_to_pyerr)?;
            self.inner
                .get_bond_mut(new_id)
                .map_err(molrs_error_to_pyerr)?
                .props = bond.props.clone();
        }

        // Angles.
        for (_, angle) in src.angles() {
            let new_id = self
                .inner
                .add_angle(
                    id_map[&angle.atoms[0]],
                    id_map[&angle.atoms[1]],
                    id_map[&angle.atoms[2]],
                )
                .map_err(molrs_error_to_pyerr)?;
            self.inner
                .get_angle_mut(new_id)
                .map_err(molrs_error_to_pyerr)?
                .props = angle.props.clone();
        }

        // Dihedrals.
        for (_, dih) in src.dihedrals() {
            let new_id = self
                .inner
                .add_dihedral(
                    id_map[&dih.atoms[0]],
                    id_map[&dih.atoms[1]],
                    id_map[&dih.atoms[2]],
                    id_map[&dih.atoms[3]],
                )
                .map_err(molrs_error_to_pyerr)?;
            self.inner
                .get_dihedral_mut(new_id)
                .map_err(molrs_error_to_pyerr)?
                .props = dih.props.clone();
        }

        // Impropers.
        for (_, imp) in src.impropers() {
            let new_id = self
                .inner
                .add_improper(
                    id_map[&imp.atoms[0]],
                    id_map[&imp.atoms[1]],
                    id_map[&imp.atoms[2]],
                    id_map[&imp.atoms[3]],
                )
                .map_err(molrs_error_to_pyerr)?;
            self.inner
                .get_improper_mut(new_id)
                .map_err(molrs_error_to_pyerr)?
                .props = imp.props.clone();
        }

        Ok(())
    }

    /// Export the graph as a :class:`Frame` (atoms / bonds / angles /
    /// dihedrals / impropers blocks).
    fn to_frame(&self) -> PyResult<PyFrame> {
        let frame = self.inner.to_frame();
        PyFrame::from_core_frame(frame)
    }

    fn __repr__(&self) -> String {
        format!(
            "Graph(atoms={}, bonds={}, angles={}, dihedrals={}, impropers={})",
            self.inner.n_atoms(),
            self.inner.n_bonds(),
            self.inner.n_angles(),
            self.inner.n_dihedrals(),
            self.inner.n_impropers(),
        )
    }
}

// Internal (non-`#[pymethods]`) helpers.
impl PyGraph {
    /// Build an atom with `key=symbol` (+ optional xyz) and add it; return its
    /// 0-based index.
    fn add_node(
        &mut self,
        key: &str,
        symbol: &str,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
    ) -> usize {
        let mut atom = Atom::new();
        atom.set(key, symbol);
        if let (Some(xv), Some(yv), Some(zv)) = (x, y, z) {
            atom.set("x", xv);
            atom.set("y", yv);
            atom.set("z", zv);
        }
        let id = self.inner.add_atom(atom);
        self.atom_id_to_index(id)
    }

    fn atom_id_to_index(&self, target: AtomId) -> usize {
        self.inner
            .atoms()
            .position(|(id, _)| id == target)
            .expect("atom id not found")
    }

    fn index_to_atom_id(&self, index: usize) -> PyResult<AtomId> {
        self.inner
            .atoms()
            .nth(index)
            .map(|(id, _)| id)
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "atom index {} out of range",
                    index
                ))
            })
    }

    fn index_to_bond_id(&self, index: usize) -> PyResult<BondId> {
        self.inner
            .bonds()
            .nth(index)
            .map(|(id, _)| id)
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "bond index {} out of range",
                    index
                ))
            })
    }

    fn angle_id_to_index(&self, target: AngleId) -> usize {
        self.inner
            .angles()
            .position(|(id, _)| id == target)
            .expect("angle id not found")
    }

    fn index_to_angle_id(&self, index: usize) -> PyResult<AngleId> {
        self.inner
            .angles()
            .nth(index)
            .map(|(id, _)| id)
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "angle index {} out of range",
                    index
                ))
            })
    }

    fn dihedral_id_to_index(&self, target: DihedralId) -> usize {
        self.inner
            .dihedrals()
            .position(|(id, _)| id == target)
            .expect("dihedral id not found")
    }

    fn index_to_dihedral_id(&self, index: usize) -> PyResult<DihedralId> {
        self.inner
            .dihedrals()
            .nth(index)
            .map(|(id, _)| id)
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "dihedral index {} out of range",
                    index
                ))
            })
    }

    fn improper_id_to_index(&self, target: ImproperId) -> usize {
        self.inner
            .impropers()
            .position(|(id, _)| id == target)
            .expect("improper id not found")
    }

    fn index_to_improper_id(&self, index: usize) -> PyResult<ImproperId> {
        self.inner
            .impropers()
            .nth(index)
            .map(|(id, _)| id)
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "improper index {} out of range",
                    index
                ))
            })
    }
}

/// Read property `key` from an `Atom`, returning the matching Python scalar or
/// `None`. Atom's `props` field is private in core, so we read through its
/// typed accessors and probe `Str`/`Int`/`F64` in turn.
fn atom_prop_to_py(py: Python<'_>, atom: &Atom, key: &str) -> PyResult<Py<PyAny>> {
    if let Some(s) = atom.get_str(key) {
        Ok(s.into_pyobject(py)?.into_any().unbind())
    } else if let Some(i) = atom.get_int(key) {
        Ok(i.into_pyobject(py)?.into_any().unbind())
    } else if let Some(f) = atom.get_f64(key) {
        Ok(f.into_pyobject(py)?.into_any().unbind())
    } else {
        Ok(py_none(py))
    }
}

// ---------------------------------------------------------------------------
// PyAtomistic — marker subclass
// ---------------------------------------------------------------------------

/// All-atom molecular graph, exposed to Python as `molrs.Atomistic`.
///
/// A marker subclass of :class:`Graph`; it inherits the full method surface.
/// The ``"element"`` invariant is established by :meth:`Graph.add_atom`.
#[pyclass(name = "Atomistic", extends = PyGraph, unsendable)]
pub struct PyAtomistic;

#[pymethods]
impl PyAtomistic {
    /// Create an empty all-atom graph. Extra args are accepted and ignored.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyAny>, _kwargs: Option<&Bound<'_, PyAny>>) -> (Self, PyGraph) {
        (
            PyAtomistic,
            PyGraph {
                inner: MolGraph::new(),
            },
        )
    }
}

// ---------------------------------------------------------------------------
// PyCoarseGrain — marker subclass
// ---------------------------------------------------------------------------

/// Coarse-grained molecular graph, exposed to Python as `molrs.CoarseGrain`.
///
/// A marker subclass of :class:`Graph`; it inherits the full method surface.
/// The ``"bead_type"`` invariant is established by :meth:`Graph.add_bead`.
#[pyclass(name = "CoarseGrain", extends = PyGraph, unsendable)]
pub struct PyCoarseGrain;

#[pymethods]
impl PyCoarseGrain {
    /// Create an empty coarse-grained graph. Extra args are accepted/ignored.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyAny>, _kwargs: Option<&Bound<'_, PyAny>>) -> (Self, PyGraph) {
        (
            PyCoarseGrain,
            PyGraph {
                inner: MolGraph::new(),
            },
        )
    }
}
