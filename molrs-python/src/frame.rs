//! Python wrapper for `Frame`, a hierarchical data container backed by the
//! shared FFI store.
//!
//! A [`PyFrame`] maps string keys (e.g. `"atoms"`, `"bonds"`, `"angles"`) to
//! [`PyBlock`] column stores. It may optionally carry a [`PyBox`] (simulation
//! box) and a string-to-string metadata dictionary.
//!
//! # Conventional Block Layout
//!
//! | Block key   | Expected columns                                      | Notes                                    |
//! |-------------|-------------------------------------------------------|------------------------------------------|
//! | `"atoms"`   | `symbol` (str), `x`/`y`/`z` (float), `mass` (float)  | Atom positions and properties             |
//! | `"bonds"`   | `i`/`j` (uint), `order` (float)                       | Bond topology (indices into atoms)        |
//! | `"angles"`  | `i`/`j`/`k` (uint), `type` (int)                      | Angle topology                            |
//!
//! The frame itself does **not** enforce cross-block row consistency; that is
//! the caller's responsibility (use [`PyFrame::validate`] to check).

use std::collections::HashMap;
use std::ffi::CString;

use crate::block::PyBlock;
use crate::helpers::molrs_error_to_pyerr;
use crate::simbox::PyBox;
use crate::store::ffi_error_to_pyerr;
use molrs::frame::Frame as CoreFrame;
use molrs_ffi::FrameRef;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

/// Hierarchical data container exposed to Python as `molrs.Frame`.
///
/// A `Frame` is a dictionary of named [`Block`](crate::block::PyBlock)s with
/// optional simulation box and metadata. It is the primary exchange format for
/// molecular data across the molrs ecosystem.
///
/// # Python Examples
///
/// ```python
/// import numpy as np
/// from molrs import Frame, Block, Box
///
/// frame = Frame()
/// atoms = Block()
/// atoms.insert("symbol", ["O", "H", "H"])
/// atoms.insert("x", np.array([0.0, 0.76, -0.76], dtype=np.float32))
/// atoms.insert("y", np.array([0.0, 0.59,  0.59], dtype=np.float32))
/// atoms.insert("z", np.zeros(3, dtype=np.float32))
/// frame["atoms"] = atoms
///
/// frame.box = Box.cube(10.0)
/// print(frame)          # Frame(blocks=['atoms'], simbox=yes)
/// print(frame.keys())   # ['atoms']
/// ```
#[pyclass(name = "Frame", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyFrame {
    pub(crate) inner: FrameRef,
}

#[pymethods]
impl PyFrame {
    /// Create an empty frame with no blocks, no simulation box, and empty
    /// metadata.
    ///
    /// Returns
    /// -------
    /// Frame
    #[new]
    fn new() -> Self {
        Self {
            inner: FrameRef::new_standalone(),
        }
    }

    /// Retrieve a block by name.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name (e.g. ``"atoms"``).
    ///
    /// Returns
    /// -------
    /// Block
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist.
    ///
    /// Examples
    /// --------
    /// >>> atoms = frame["atoms"]
    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Py<PyAny>> {
        if let Ok(inner) = self.inner.block(key) {
            return Ok(Py::new(py, PyBlock { inner })?.into_any());
        }
        Err(PyKeyError::new_err(key.to_string()))
    }

    /// Assign a block under the given name.
    ///
    /// If a block with the same key already exists it is replaced.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name.
    /// block : Block
    ///     The block to store.
    ///
    /// Examples
    /// --------
    /// >>> frame["atoms"] = atoms_block
    fn __setitem__(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(block) = value.extract::<PyRef<'_, PyBlock>>() {
            let core_block = block.clone_core_block()?;
            return self
                .inner
                .store
                .borrow_mut()
                .set_block(self.inner.id, key, core_block)
                .map_err(ffi_error_to_pyerr);
        }
        Err(PyTypeError::new_err("value must be a Block"))
    }

    /// Delete a block by name.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name to remove.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist.
    ///
    /// Examples
    /// --------
    /// >>> del frame["bonds"]
    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        self.inner
            .store
            .borrow_mut()
            .remove_block(self.inner.id, key)
            .map_err(ffi_error_to_pyerr)
    }

    /// Test whether a block name is present.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name.
    ///
    /// Returns
    /// -------
    /// bool
    ///
    /// Examples
    /// --------
    /// >>> "atoms" in frame
    /// True
    fn __contains__(&self, key: &str) -> PyResult<bool> {
        self.with_frame(|f| f.contains_key(key))
    }

    /// Number of blocks stored in this frame.
    ///
    /// Returns
    /// -------
    /// int
    fn __len__(&self) -> PyResult<usize> {
        self.with_frame(|f| f.len())
    }

    /// List all block names.
    ///
    /// Returns
    /// -------
    /// list[str]
    fn keys(&self) -> PyResult<Vec<String>> {
        self.with_frame(|f| f.keys().map(|s| s.to_string()).collect())
    }

    /// The simulation box attached to this frame, or ``None``.
    ///
    /// Returns
    /// -------
    /// Box | None
    ///     Periodic simulation box, if set.
    ///
    /// Examples
    /// --------
    /// >>> if frame.box is not None:
    /// ...     print(frame.box.volume())
    #[getter(simbox)]
    fn get_box(&self) -> PyResult<Option<PyBox>> {
        Ok(self
            .inner
            .simbox_clone()
            .map_err(ffi_error_to_pyerr)?
            .map(|inner| PyBox { inner }))
    }

    /// Alias for ``simbox`` — matches the molpy API.
    #[getter(r#box)]
    fn get_box_alias(&self) -> PyResult<Option<PyBox>> {
        self.get_box()
    }

    /// Set (or clear) the simulation box.
    ///
    /// Parameters
    /// ----------
    /// box : Box | None
    ///     Pass ``None`` to remove the simulation box.
    ///
    /// Examples
    /// --------
    /// >>> frame.box = Box.cube(20.0)
    /// >>> frame.box = None  # remove
    #[setter(simbox)]
    fn set_box(&mut self, simbox: Option<&PyBox>) -> PyResult<()> {
        self.inner
            .set_simbox(simbox.map(|sb| sb.inner.clone()))
            .map_err(ffi_error_to_pyerr)
    }

    /// Setter alias for ``simbox`` — matches the molpy API.
    #[setter(r#box)]
    fn set_box_alias(&mut self, simbox: Option<&PyBox>) -> PyResult<()> {
        self.set_box(simbox)
    }

    /// Metadata dictionary (``dict[str, str]``).
    ///
    /// Returns
    /// -------
    /// dict[str, str]
    ///     String key-value metadata attached to this frame.
    #[getter]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let meta = self.with_frame(|f| f.meta.clone())?;
        for (k, v) in meta {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    /// Replace the metadata dictionary.
    ///
    /// Parameters
    /// ----------
    /// meta : dict[str, str]
    ///     New metadata. All keys and values must be strings.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If any key or value cannot be converted to ``str``.
    #[setter]
    fn set_meta(&mut self, meta: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut map = HashMap::new();
        for (k, v) in meta.iter() {
            let key: String = k.extract()?;
            let val: String = v.extract()?;
            map.insert(key, val);
        }
        self.inner
            .with_mut(|f| {
                f.meta = map;
            })
            .map_err(ffi_error_to_pyerr)?;
        Ok(())
    }

    /// Validate cross-block consistency.
    ///
    /// Checks that referenced indices (e.g. bond atom indices) are within
    /// bounds and that required columns exist.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If validation fails.
    fn validate(&self) -> PyResult<()> {
        self.with_frame(|f| f.validate().map_err(molrs_error_to_pyerr))?
    }

    fn __repr__(&self) -> PyResult<String> {
        self.with_frame(|f| {
            let keys: Vec<&str> = f.keys().collect();
            format!(
                "Frame(blocks={:?}, simbox={})",
                keys,
                if f.simbox.is_some() { "yes" } else { "no" }
            )
        })
    }

    /// Export this frame's FFI handle as a ``PyCapsule``.
    ///
    /// The capsule wraps a *clone* of this frame's ``FrameRef`` handle. The
    /// clone shares the same underlying ``Store`` (``Rc<RefCell<Store>>``),
    /// so a consumer (e.g. Atomiverse C++ via the molrs-cxxapi bridge) that
    /// resolves the capsule reads and writes the *same* frame data: no deep
    /// copy is made. The capsule's destructor reclaims the boxed
    /// ``FrameRef`` on capsule destruction, dropping its two ``Rc``
    /// references.
    ///
    /// Pointer indirection: PyO3's ``PyCapsule::new`` heap-boxes its
    /// payload, and the payload here is a ``#[repr(transparent)]``
    /// ``FrameRefPtr`` (itself ``*mut FrameRef``). The capsule's ``void*``
    /// is therefore ``*mut FrameRefPtr`` ≡ ``*mut *mut FrameRef``: one
    /// dereference yields the ``*mut FrameRef`` clone. Atomiverse's
    /// ``frame_clone_from_addr`` does exactly that double-resolve.
    ///
    /// The capsule name is the C string ``"molrs.FrameRef"``.
    ///
    /// Returns
    /// -------
    /// capsule
    ///     A ``PyCapsule`` named ``"molrs.FrameRef"`` whose pointer is
    ///     ``*mut *mut`` :class:`molrs_ffi.FrameRef` (a cloned handle).
    fn _ffi_frameref_capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        // Box a clone of the handle and hand the raw pointer to the capsule.
        // `FrameRef` holds an `Rc` and is therefore not `Send`; a bare
        // `*mut FrameRef` is not `Send` either, so wrap it in `FrameRefPtr`
        // which asserts `Send`. This is sound because the capsule is only
        // ever touched under the GIL (molrs FFI is single-threaded — see the
        // threading note in `molrs_ffi::shared`).
        let raw = FrameRefPtr(Box::into_raw(Box::new(self.inner.clone())));
        let name = CString::new("molrs.FrameRef").expect("static capsule name");
        PyCapsule::new_with_destructor(py, raw, Some(name), |ptr: FrameRefPtr, _ctx| {
            // SAFETY: `ptr.0` is the pointer produced by `Box::into_raw`
            // above and is reclaimed exactly once when the capsule dies.
            drop(unsafe { Box::from_raw(ptr.0) });
        })
    }
}

/// `Send` wrapper around a `*mut FrameRef` so it can ride inside a
/// `PyCapsule` (whose payload must be `Send`).
///
/// `FrameRef` is `!Send` (it holds an `Rc`), and raw pointers are `!Send` by
/// default. The capsule is only ever created, read, and destroyed while the
/// Python GIL is held, so no cross-thread access of the `Rc` ever occurs —
/// the `unsafe impl Send` is upheld by that single-threaded discipline.
///
/// `#[repr(transparent)]` guarantees this newtype has exactly the layout of
/// the wrapped `*mut FrameRef`. PyO3's `PyCapsule::new` heap-boxes the
/// payload, so the capsule's `void*` is `*mut FrameRefPtr`; the transparent
/// repr makes that pointer reinterpretable as `*mut *mut FrameRef`, which is
/// how Atomiverse's molrs-cxxapi bridge resolves it
/// (`frame_clone_from_addr`).
#[repr(transparent)]
struct FrameRefPtr(*mut FrameRef);

// SAFETY: see the type-level doc — single-threaded, GIL-guarded use only.
unsafe impl Send for FrameRefPtr {}

impl PyFrame {
    /// Create a `PyFrame` from a Rust `CoreFrame`, allocating a new FFI store.
    pub(crate) fn from_core_frame(frame: CoreFrame) -> PyResult<Self> {
        let store = molrs_ffi::new_shared();
        let id = store.borrow_mut().frame_new();
        store
            .borrow_mut()
            .set_frame(id, frame)
            .map_err(ffi_error_to_pyerr)?;
        Ok(Self {
            inner: FrameRef::new(store, id),
        })
    }

    /// Clone the underlying `CoreFrame` out of the store (deep copy).
    pub(crate) fn clone_core_frame(&self) -> PyResult<CoreFrame> {
        self.inner.clone_frame().map_err(ffi_error_to_pyerr)
    }

    /// Run a read-only closure on the underlying `CoreFrame`.
    fn with_frame<R>(&self, f: impl FnOnce(&CoreFrame) -> R) -> PyResult<R> {
        self.inner.with(f).map_err(ffi_error_to_pyerr)
    }
}
