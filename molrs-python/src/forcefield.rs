//! Python wrappers for MMFF94 force-field typification and compiled potentials.
//!
//! The workflow is:
//!
//! 1. Create an [`PyMMFFTypifier`] (loads embedded MMFF94 parameters).
//! 2. Call [`PyMMFFTypifier::typify`] to assign atom types, producing a typed
//!    [`PyFrame`].
//! 3. Call [`PyMMFFTypifier::build`] to compile potentials directly from an
//!    [`PyAtomistic`] graph.
//! 4. Use [`PyPotentials::eval`] to evaluate energy and forces on flat
//!    coordinate arrays.
//!
//! # References
//!
//! - Halgren, T.A. (1996). J. Comput. Chem. 17, 490-519. (MMFF94 force field)

use pyo3::prelude::*;

use molrs_ff::ForceField;
use molrs_ff::mmff::{MmffForceField, MmffMolProperties, MmffVariant};
use molrs_ff::potential::{Potentials, extract_coords};
use molrs_ff::typifier::Typifier;
use molrs_ff::typifier::mmff::MMFFTypifier;
use molrs_ff::{MinimizeOptions, OptReport, minimize, minimize_batch};

use crate::frame::PyFrame;
use crate::helpers::NpF;
use crate::molgraph::PyAtomistic;

use ndarray::{Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArrayDyn, ToPyArray};

/// Outcome of a geometry optimization, exposed to Python as `molrs.OptReport`.
#[pyclass(name = "OptReport")]
pub struct PyOptReport {
    inner: OptReport,
}

#[pymethods]
impl PyOptReport {
    /// Whether ``fmax`` convergence was reached within ``max_steps``.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Number of outer L-BFGS iterations performed.
    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps
    }

    /// Potential energy at the returned geometry (kcal/mol).
    #[getter]
    fn final_energy(&self) -> f64 {
        self.inner.final_energy
    }

    /// Maximum per-atom force magnitude at the returned geometry
    /// (kcal/mol/angstrom).
    #[getter]
    fn final_fmax(&self) -> f64 {
        self.inner.final_fmax
    }

    fn __repr__(&self) -> String {
        format!(
            "OptReport(converged={}, n_steps={}, final_energy={:.6}, final_fmax={:.6})",
            if self.inner.converged {
                "True"
            } else {
                "False"
            },
            self.inner.n_steps,
            self.inner.final_energy,
            self.inner.final_fmax
        )
    }
}

impl From<OptReport> for PyOptReport {
    fn from(inner: OptReport) -> Self {
        Self { inner }
    }
}

/// Compiled force-field potentials for energy and force evaluation.
///
/// Exposed to Python as `molrs.Potentials`.
///
/// Operates on flat coordinate arrays in the layout
/// ``[x0, y0, z0, x1, y1, z1, ...]`` (length 3N).
///
/// Examples
/// --------
/// >>> typifier = MMFFTypifier()
/// >>> potentials = typifier.build(mol)
/// >>> energy, forces = potentials.eval(coords)
#[pyclass(name = "Potentials")]
pub struct PyPotentials {
    inner: Potentials,
}

/// Force-field definition metadata exposed to Python as `molrs.ForceField`.
#[pyclass(name = "ForceField")]
pub struct PyForceField {
    pub(crate) inner: ForceField,
}

#[pymethods]
impl PyPotentials {
    /// Number of individual potential kernels in this compiled set.
    ///
    /// Returns
    /// -------
    /// int
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Evaluate energy and forces for the given coordinates.
    ///
    /// Parameters
    /// ----------
    /// coords : numpy.ndarray, shape (3*N,), dtype float
    ///     Flat coordinate array ``[x0, y0, z0, x1, y1, z1, ...]``.
    ///
    /// Returns
    /// -------
    /// tuple[float, numpy.ndarray]
    ///     ``(energy, forces)`` where ``energy`` is a scalar in kcal/mol and
    ///     ``forces`` is a flat array of the same shape as ``coords`` in
    ///     kcal/(mol*angstrom).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``coords`` is not contiguous in memory.
    ///
    /// Examples
    /// --------
    /// >>> energy, forces = potentials.eval(coords)
    /// >>> energy
    /// -12.34
    fn eval<'py>(
        &self,
        py: Python<'py>,
        coords: numpy::PyReadonlyArray1<'_, NpF>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<NpF>>)> {
        let slice = coords.as_slice()?;
        let (energy, forces) = self.inner.eval(slice);
        let forces_arr = forces.to_pyarray(py);
        Ok((energy, forces_arr))
    }

    /// Evaluate energy only (no force computation).
    ///
    /// Parameters
    /// ----------
    /// coords : numpy.ndarray, shape (3*N,), dtype float
    ///     Flat coordinate array.
    ///
    /// Returns
    /// -------
    /// float
    ///     Energy in kcal/mol.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``coords`` is not contiguous in memory.
    fn energy(&self, coords: numpy::PyReadonlyArray1<'_, NpF>) -> PyResult<f64> {
        let slice = coords.as_slice()?;
        Ok(self.inner.energy(slice))
    }

    /// Relax structures by L-BFGS energy minimization. Single or batch,
    /// dispatched on the input rank.
    ///
    /// * ``(N, 3)`` (or flat ``(3N,)``) → optimize one structure; returns
    ///   ``((N, 3) array, OptReport)``.
    /// * ``(B, N, 3)`` → optimize a homogeneous batch (all structures share
    ///   this :class:`Potentials`, hence the same atom count and topology);
    ///   returns ``((B, N, 3) array, list[OptReport])``. With the ``rayon``
    ///   build feature the batch runs in parallel.
    ///
    /// Heterogeneous systems (different topologies) cannot be batched — loop
    /// over single calls with a separate :class:`Potentials` each.
    ///
    /// Parameters
    /// ----------
    /// coords : numpy.ndarray, shape (N, 3), (3*N,), or (B, N, 3), dtype float
    ///     Starting coordinates. Not modified in place.
    /// fmax : float, optional
    ///     Convergence threshold on the maximum per-atom force magnitude
    ///     ``max_i ||F_i||`` (kcal/mol/angstrom). Default ``0.05``.
    /// max_steps : int, optional
    ///     Maximum outer L-BFGS iterations. Default ``500``.
    /// max_step : float, optional
    ///     Per-step displacement cap in angstrom (trust region). Default
    ///     ``0.2``.
    /// memory : int, optional
    ///     L-BFGS correction-pair history size. Default ``8``.
    ///
    /// Returns
    /// -------
    /// tuple[numpy.ndarray, OptReport] or tuple[numpy.ndarray, list[OptReport]]
    ///     Optimized coordinates (same rank as a fresh ``(N, 3)`` / ``(B, N, 3)``
    ///     array) and the per-structure report(s).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the coordinate count is not a multiple of three, the rank is not
    ///     1/2/3, the trailing axis of a 3-D array is not 3, or a batch's atom
    ///     count ``N`` does not match this :class:`Potentials`.
    ///
    /// Examples
    /// --------
    /// >>> potentials = MMFFTypifier().build(mol)
    /// >>> opt_coords, report = potentials.minimize(coords)          # (N, 3)
    /// >>> opt_batch, reports = potentials.minimize(batch)           # (B, N, 3)
    #[pyo3(signature = (coords, *, fmax = 0.05, max_steps = 500, max_step = 0.2, memory = 8))]
    fn minimize<'py>(
        &self,
        py: Python<'py>,
        coords: PyReadonlyArrayDyn<'_, NpF>,
        fmax: f64,
        max_steps: usize,
        max_step: f64,
        memory: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let opts = MinimizeOptions {
            fmax,
            max_steps,
            max_step,
            memory,
        };
        let arr = coords.as_array();
        let shape = arr.shape();

        match shape.len() {
            // Single structure: flat (3N,) or (N, 3).
            1 | 2 => {
                let mut flat: Vec<NpF> = arr.iter().copied().collect();
                let n_elem = flat.len();
                if n_elem % 3 != 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "coords has {n_elem} elements, not a multiple of 3 (expected (N, 3) or (3N,))"
                    )));
                }
                let report = minimize(&self.inner, &mut flat, &opts)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                let out: Bound<'py, PyArray2<NpF>> = Array2::from_shape_vec((n_elem / 3, 3), flat)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    .to_pyarray(py);
                Ok((out, PyOptReport::from(report))
                    .into_pyobject(py)?
                    .into_any())
            }
            // Homogeneous batch: (B, N, 3).
            3 => {
                if shape[2] != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "batch coords must be (B, N, 3); trailing axis is {} not 3",
                        shape[2]
                    )));
                }
                let (b, n) = (shape[0], shape[1]);
                // Validate N against the compiled topology when known
                // (n_atoms == 0 means built without a frame, e.g. via push()).
                let expected = self.inner.n_atoms();
                if expected != 0 && n != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "structure atom count N={n} does not match this Potentials' atom count {expected}"
                    )));
                }
                let mut flat: Vec<NpF> = arr.iter().copied().collect();
                let reports = minimize_batch(&self.inner, &mut flat, n, b, &opts)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                let out: Bound<'py, PyArray3<NpF>> = Array3::from_shape_vec((b, n, 3), flat)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    .to_pyarray(py);
                let reports: Vec<PyOptReport> =
                    reports.into_iter().map(PyOptReport::from).collect();
                Ok((out, reports).into_pyobject(py)?.into_any())
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coords must be 1-D (3N,), 2-D (N, 3), or 3-D (B, N, 3); got {other}-D"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!("Potentials(n_kernels={})", self.inner.len())
    }
}

/// MMFF94 atom-type assigner and potential builder.
///
/// Exposed to Python as `molrs.MMFFTypifier`.
///
/// Loads embedded MMFF94 parameter tables at construction time. Use
/// :meth:`typify` to assign atom types to a molecular graph, or
/// :meth:`build` as a one-step shortcut that also compiles potentials.
///
/// # References
///
/// - Halgren, T.A. (1996). J. Comput. Chem. 17, 490-519.
///
/// Examples
/// --------
/// >>> typifier = MMFFTypifier()
/// >>> frame = typifier.typify(mol)   # typed Frame
/// >>> potentials = typifier.build(mol)  # compiled Potentials
#[pyclass(name = "MMFFTypifier")]
pub struct PyMMFFTypifier {
    inner: MMFFTypifier,
}

#[pymethods]
impl PyMMFFTypifier {
    /// Create an MMFF94 typifier with embedded parameter tables.
    ///
    /// Returns
    /// -------
    /// MMFFTypifier
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If parameter initialization fails (should not happen with
    ///     embedded data).
    #[new]
    fn new() -> PyResult<Self> {
        let typifier = MMFFTypifier::mmff94().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to initialize MMFF94: {}", e))
        })?;
        Ok(Self { inner: typifier })
    }

    /// Assign MMFF94 atom types to a molecular graph.
    ///
    /// Returns a typed :class:`Frame` with an ``"atoms"`` block containing
    /// the assigned ``type`` (int) column and topology blocks
    /// (``"bonds"``, ``"angles"``, ``"dihedrals"``, ``"impropers"``).
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     Molecular graph with element symbols and bonds.
    ///
    /// Returns
    /// -------
    /// Frame
    ///     Typed molecular data.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If atom types cannot be determined (e.g. unsupported elements).
    fn typify(&self, mol: &PyAtomistic) -> PyResult<PyFrame> {
        let frame = self
            .inner
            .typify(mol.core())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        PyFrame::from_core_frame(frame)
    }

    /// Typify and compile potentials in one step.
    ///
    /// Equivalent to calling :meth:`typify` followed by force-field
    /// compilation, but avoids the intermediate :class:`Frame`.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     Molecular graph with element symbols and bonds.
    ///
    /// Returns
    /// -------
    /// Potentials
    ///     Compiled energy/force evaluator.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If typification or compilation fails.
    ///
    /// Examples
    /// --------
    /// >>> potentials = typifier.build(mol)
    /// >>> energy, forces = potentials.eval(coords)
    fn build(&self, mol: &PyAtomistic) -> PyResult<PyPotentials> {
        let potentials = self
            .inner
            .build(mol.core())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyPotentials { inner: potentials })
    }

    /// Return the underlying force-field definition.
    fn forcefield(&self) -> PyForceField {
        PyForceField {
            inner: self.inner.ff().clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("MMFFTypifier(forcefield='{}')", self.inner.ff().name)
    }
}

/// Extract a flat coordinate array from a Frame's ``"atoms"`` block.
///
/// Reads the ``x``, ``y``, ``z`` columns from the ``"atoms"`` block and
/// interleaves them into a flat 1D array: ``[x0, y0, z0, x1, y1, z1, ...]``.
///
/// Parameters
/// ----------
/// frame : Frame
///     Frame with an ``"atoms"`` block containing ``x``, ``y``, ``z``
///     float columns.
///
/// Returns
/// -------
/// numpy.ndarray, shape (3*N,), dtype float
///     Flat coordinate array suitable for :meth:`Potentials.eval`.
///
/// Raises
/// ------
/// ValueError
///     If the ``"atoms"`` block or required columns are missing.
///
/// Examples
/// --------
/// >>> coords = extract_coords(frame)
/// >>> energy, forces = potentials.eval(coords)
#[pyfunction]
#[pyo3(name = "extract_coords")]
pub fn extract_coords_py<'py>(
    py: Python<'py>,
    frame: &PyFrame,
) -> PyResult<Bound<'py, PyArray1<NpF>>> {
    let core_frame = frame.clone_core_frame()?;
    let coords = extract_coords(&core_frame)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(coords.to_pyarray(py))
}

/// Build ready-to-use MMFF94 potentials for a molecule.
///
/// Uses the assembled :rust:`MmffForceField` energy model (the RDKit-validated
/// MMFF94 path used by conformer generation), wrapped as a :class:`Potentials`
/// so it can be evaluated and minimized directly. This is the recommended way
/// to obtain potentials for :meth:`Potentials.minimize`.
///
/// Parameters
/// ----------
/// mol : Atomistic
///     Molecular graph with element symbols, bonds, and 3D coordinates.
/// variant : str, optional
///     ``"MMFF94"`` (default) or ``"MMFF94s"`` (static variant).
///
/// Returns
/// -------
/// Potentials
///     Compiled energy/force evaluator for ``mol``.
///
/// Raises
/// ------
/// ValueError
///     If ``variant`` is unknown, or MMFF94 typing / assembly fails.
///
/// Examples
/// --------
/// >>> mol = molrs.parse_smiles("CCO").to_atomistic()
/// >>> mol, _ = molrs.Conformer(seed=7).generate(mol)
/// >>> pots = molrs.build_mmff_potentials(mol)
/// >>> coords = molrs.extract_coords(molrs.MMFFTypifier().typify(mol)).reshape(-1, 3)
/// >>> opt, report = pots.minimize(coords, fmax=0.05)
#[pyfunction]
#[pyo3(name = "build_mmff_potentials", signature = (mol, variant = "MMFF94"))]
pub fn build_mmff_potentials_py(mol: &PyAtomistic, variant: &str) -> PyResult<PyPotentials> {
    let var = match variant.to_ascii_uppercase().as_str() {
        "MMFF94" => MmffVariant::Mmff94,
        "MMFF94S" => MmffVariant::Mmff94s,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown MMFF variant '{other}' (expected 'MMFF94' or 'MMFF94s')"
            )));
        }
    };
    let core = mol.core();
    let props = MmffMolProperties::compute(core, var)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let ff = MmffForceField::build(core, &props)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let mut pots = Potentials::new();
    pots.set_n_atoms(core.n_atoms());
    pots.push(Box::new(ff));
    Ok(PyPotentials { inner: pots })
}

/// Read a force-field definition from an XML file.
#[pyfunction]
#[pyo3(name = "read_forcefield_xml")]
pub fn read_forcefield_xml_py(path: &str) -> PyResult<PyForceField> {
    let forcefield = molrs_ff::read_forcefield_xml(path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyForceField { inner: forcefield })
}

#[pymethods]
impl PyForceField {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    fn style_names(&self) -> Vec<String> {
        self.inner
            .styles()
            .iter()
            .map(|style| format!("{}:{}", style.category(), style.name))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ForceField(name='{}', styles={})",
            self.inner.name,
            self.inner.styles().len()
        )
    }
}
