use molrs::core::locality::nblist::linked_cell::LinkedCell as RustLinkedCell;
use molrs::core::region::simbox::SimBox as RustBox;
use ndarray::{array, Array1};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod pipeline;
use pipeline::{PyModifier, PyPipeline};

// All explicit helper conversion functions removed; we rely directly on rust-numpy's
// as_array()/to_owned_array() methods for zero-copy views or owned clones.

/// Python wrapper for the Rust Box struct
#[pyclass(name = "Box")]
#[derive(Clone)]
pub struct PyBox {
    inner: RustBox,
}

#[pymethods]
impl PyBox {
    /// Create a new triclinic box
    ///
    /// Parameters
    /// ----------
    /// h : array_like, shape (3, 3), dtype=float32
    ///     Cell matrix with lattice vectors as columns
    /// origin : array_like, shape (3,), dtype=float32, optional
    ///     Origin of the box in Cartesian coordinates
    /// pbc : array_like, shape (3,), optional
    ///     Periodic boundary conditions for x, y, z axes
    #[new]
    #[pyo3(signature = (h, origin=None, pbc=None))]
    fn new(
        h: PyReadonlyArray2<f32>,
        origin: Option<PyReadonlyArray1<f32>>,
        pbc: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<Self> {
        let h_view = h.as_array();
        if h_view.dim() != (3, 3) {
            return Err(PyValueError::new_err("h must be a 3x3 matrix"));
        }
        let h_matrix = h_view.to_owned();

        let origin_vec: Array1<f32> = if let Some(o) = origin {
            let s = o.as_slice()?;
            if s.len() != 3 {
                return Err(PyValueError::new_err("origin must have length 3"));
            }
            array![s[0], s[1], s[2]]
        } else {
            array![0.0, 0.0, 0.0]
        };

        let pbc_array = if let Some(p) = pbc {
            let p_slice = p.as_slice()?;
            if p_slice.len() != 3 {
                return Err(PyValueError::new_err("pbc must have 3 elements"));
            }
            [p_slice[0], p_slice[1], p_slice[2]]
        } else {
            [true, true, true]
        };

        Ok(PyBox {
            inner: RustBox::new(h_matrix, origin_vec, pbc_array),
        })
    }

    /// Create a cubic box
    ///
    /// Parameters
    /// ----------
    /// a : float
    ///     Edge length of the cube
    /// origin : array_like, shape (3,), dtype=float32, optional
    ///     Origin of the box
    /// pbc : array_like, shape (3,), optional
    ///     Periodic boundary conditions
    #[staticmethod]
    #[pyo3(signature = (a, origin=None, pbc=None))]
    fn cube(
        a: f32,
        origin: Option<PyReadonlyArray1<f32>>,
        pbc: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<Self> {
        let origin_vec: Array1<f32> = if let Some(o) = origin {
            let s = o.as_slice()?;
            if s.len() != 3 {
                return Err(PyValueError::new_err("origin must have length 3"));
            }
            array![s[0], s[1], s[2]]
        } else {
            array![0.0, 0.0, 0.0]
        };
        let pbc_array = if let Some(p) = pbc {
            let p_slice = p.as_slice()?;
            if p_slice.len() != 3 {
                return Err(PyValueError::new_err("pbc must have 3 elements"));
            }
            [p_slice[0], p_slice[1], p_slice[2]]
        } else {
            [true, true, true]
        };

        Ok(PyBox {
            inner: RustBox::cube(a, origin_vec, pbc_array),
        })
    }

    /// Create an orthorhombic box
    ///
    /// Parameters
    /// ----------
    /// lengths : array_like, shape (3,), dtype=float32
    ///     Box lengths along x, y, z axes
    /// origin : array_like, shape (3,), dtype=float32, optional
    ///     Origin of the box
    /// pbc : array_like, shape (3,), optional
    ///     Periodic boundary conditions
    #[staticmethod]
    #[pyo3(signature = (lengths, origin=None, pbc=None))]
    fn ortho(
        lengths: PyReadonlyArray1<f32>,
        origin: Option<PyReadonlyArray1<f32>>,
        pbc: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<Self> {
        let lv = lengths.as_slice()?;
        if lv.len() != 3 {
            return Err(PyValueError::new_err("lengths must have length 3"));
        }
        let lengths_vec = array![lv[0], lv[1], lv[2]];
        let origin_vec: Array1<f32> = if let Some(o) = origin {
            let s = o.as_slice()?;
            if s.len() != 3 {
                return Err(PyValueError::new_err("origin must have length 3"));
            }
            array![s[0], s[1], s[2]]
        } else {
            array![0.0, 0.0, 0.0]
        };
        let pbc_array = if let Some(p) = pbc {
            let p_slice = p.as_slice()?;
            if p_slice.len() != 3 {
                return Err(PyValueError::new_err("pbc must have 3 elements"));
            }
            [p_slice[0], p_slice[1], p_slice[2]]
        } else {
            [true, true, true]
        };

        Ok(PyBox {
            inner: RustBox::ortho(lengths_vec, origin_vec, pbc_array),
        })
    }

    /// Get the volume of the box
    fn volume(&self) -> f32 {
        self.inner.volume()
    }

    /// Get a lattice vector
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     Index of the lattice vector (0, 1, or 2)
    ///
    /// Returns
    /// -------
    /// vector : ndarray, shape (3,), dtype=float32
    ///     Lattice vector
    fn lattice<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if index >= 3 {
            return Err(PyValueError::new_err("index must be 0, 1, or 2"));
        }
        let vec = self.inner.lattice(index);
        Ok(vec.into_pyarray_bound(py))
    }

    /// Convert Cartesian coordinates to fractional coordinates
    /// LAMMPS convention: x,y,z -> xs,ys,zs
    ///
    /// Parameters
    /// ----------
    /// xyz : array_like, shape (N, 3), dtype=float32
    ///     Cartesian coordinates (x, y, z columns)
    ///
    /// Returns
    /// -------
    /// xyzs : ndarray, shape (N, 3), dtype=float32
    ///     Fractional coordinates (xs, ys, zs columns)
    fn to_frac<'py>(
        &self,
        py: Python<'py>,
        xyz: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let view = xyz.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let frac = self.inner.to_frac(view);
        Ok(frac.into_pyarray_bound(py))
    }

    /// Convert scaled (fractional) coordinates to Cartesian
    /// LAMMPS convention: xs,ys,zs -> x,y,z
    ///
    /// Parameters
    /// ----------
    /// xyzs : array_like, shape (N, 3), dtype=float32
    ///     Scaled/fractional coordinates (xs, ys, zs columns)
    ///
    /// Returns
    /// -------
    /// xyz : ndarray, shape (N, 3), dtype=float32
    ///     Cartesian coordinates (x, y, z columns)
    fn to_cart<'py>(
        &self,
        py: Python<'py>,
        xyzs: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let view = xyzs.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let cart = self.inner.to_cart(view);
        Ok(cart.into_pyarray_bound(py))
    }

    /// Wrap unwrapped coordinates into the primary cell
    /// LAMMPS convention: xu,yu,zu -> x,y,z
    ///
    /// Parameters
    /// ----------
    /// xyzu : array_like, shape (N, 3), dtype=float32
    ///     Unwrapped Cartesian coordinates (xu, yu, zu columns)
    ///
    /// Returns
    /// -------
    /// xyz : ndarray, shape (N, 3), dtype=float32
    ///     Wrapped Cartesian coordinates (x, y, z columns)
    fn wrap<'py>(
        &self,
        py: Python<'py>,
        xyzu: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let view = xyzu.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let wrapped = self.inner.wrap(view);
        Ok(wrapped.into_pyarray_bound(py))
    }

    /// Calculate displacement vectors with optional minimum image convention
    /// LAMMPS convention: uses unwrapped coordinates (xu, yu, zu)
    ///
    /// Parameters
    /// ----------
    /// xyzu1 : array_like, shape (N, 3)
    ///     Starting unwrapped points (xu1, yu1, zu1 columns)
    /// xyzu2 : array_like, shape (N, 3)
    ///     Ending unwrapped points (xu2, yu2, zu2 columns)
    /// minimum_image : bool, optional
    ///     Whether to use minimum image convention (default: False)
    ///
    /// Returns
    /// -------
    /// dxyz : ndarray, shape (N, 3)
    ///     Displacement vectors (dx, dy, dz columns)
    #[pyo3(signature = (xyzu1, xyzu2, minimum_image=false))]
    fn delta<'py>(
        &self,
        py: Python<'py>,
        xyzu1: PyReadonlyArray2<f32>,
        xyzu2: PyReadonlyArray2<f32>,
        minimum_image: bool,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let v1 = xyzu1.as_array();
        let v2 = xyzu2.as_array();
        if v1.raw_dim() != v2.raw_dim() {
            return Err(PyValueError::new_err(
                "xyzu1 and xyzu2 must have the same shape",
            ));
        }
        if v1.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let d = self.inner.delta(v1, v2, minimum_image);
        Ok(d.into_pyarray_bound(py))
    }

    /// Check if points are inside the primary cell
    ///
    /// Parameters
    /// ----------
    /// xyz : array_like, shape (N, 3), dtype=float32
    ///     Points in Cartesian coordinates (x, y, z columns)
    ///
    /// Returns
    /// -------
    /// inside : ndarray, shape (N,), dtype=bool
    ///     Boolean array indicating if each point is inside
    fn isin<'py>(
        &self,
        py: Python<'py>,
        xyz: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let view = xyz.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let inside = self.inner.isin(view);
        Ok(inside.into_pyarray_bound(py))
    }

    fn __repr__(&self) -> String {
        format!("Box(volume={:.2})", self.volume())
    }
}

/// Python module definition
#[pymodule]
fn molrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBox>()?;
    m.add_class::<PyLinkedCell>()?;
    m.add_class::<PyModifier>()?;
    m.add_class::<PyPipeline>()?;
    Ok(())
}

/// Python wrapper for the Rust LinkedCell struct
#[pyclass(name = "LinkedCell")]
#[derive(Clone)]
pub struct PyLinkedCell {
    inner: RustLinkedCell,
}

#[pymethods]
impl PyLinkedCell {
    /// Build a linked-cell data structure from points, cutoff, and a simulation box.
    ///
    /// Parameters
    /// ----------
    /// points : ndarray, shape (N, 3), dtype=float32
    ///     Particle positions (x, y, z columns)
    /// cutoff : float
    ///     Cutoff radius for neighbor search
    /// box : Box
    ///     Simulation box describing PBC and geometry
    #[new]
    fn new(points: PyReadonlyArray2<f32>, cutoff: f32, r#box: &PyBox) -> PyResult<Self> {
        let view = points.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("points must have shape (N,3)"));
        }
        let pts = view.to_owned();
        let lc = RustLinkedCell::build(&pts, cutoff, &r#box.inner);
        Ok(Self { inner: lc })
    }

    /// Compute unique neighbor pairs (i<j) within cutoff using MIC.
    ///
    /// Parameters
    /// ----------
    /// points : ndarray, shape (N, 3), dtype=float32
    ///     Particle positions (x, y, z columns). Should match the set used to build.
    /// cutoff : float
    ///     Cutoff radius (can be the same as used to build)
    ///
    /// Returns
    /// -------
    /// pairs : ndarray, shape (M, 2), dtype=int64
    ///     Array of index pairs (i, j) with i < j within cutoff.
    fn pairs<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<f32>,
        cutoff: f32,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let view = points.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("points must have shape (N,3)"));
        }
        let pts = view.to_owned();
        let pairs = self.inner.pairs(&pts, cutoff);
        // Convert Vec<(usize, usize)> to ndarray for NumPy
        let out_vec: Vec<[i64; 2]> = pairs
            .into_iter()
            .map(|(i, j)| [i as i64, j as i64])
            .collect();
        let n = out_vec.len();
        let array =
            ndarray::Array2::from_shape_vec((n, 2), out_vec.into_iter().flatten().collect())
                .map_err(|_| PyValueError::new_err("failed to build output array"))?;
        Ok(array.into_pyarray_bound(py))
    }

    fn __repr__(&self) -> String {
        format!("LinkedCell(dims={:?})", self.inner.grid.dims)
    }
}
