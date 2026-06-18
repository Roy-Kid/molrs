//! Python wrappers for `molrs-compute::dielectric`.

use molrs::compute::dielectric as diel;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;

use crate::helpers::{py_value_err, warn_deprecated};

#[pyfunction]
pub(crate) fn dielectric_compute_dipole_moment<'py>(
    py: Python<'py>,
    charges: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = charges.as_array().to_owned();
    let p = positions.as_array().to_owned();
    let result = diel::compute_dipole_moment(&c, &p).map_err(py_value_err)?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn dielectric_compute_current_density<'py>(
    py: Python<'py>,
    dipole_moments: PyReadonlyArray2<'py, f64>,
    dt: f64,
    volume: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dm = dipole_moments.as_array().to_owned();
    let result = diel::compute_current_density(&dm, dt, volume).map_err(py_value_err)?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn dielectric_static_dielectric_constant<'py>(
    dipole_moments: PyReadonlyArray2<'py, f64>,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
) -> PyResult<f64> {
    let dm = dipole_moments.as_array().to_owned();
    diel::static_dielectric_constant(&dm, volume, temperature, epsilon_inf).map_err(py_value_err)
}

/// **Deprecated** (phase-02 compute/fit repoint): the bundled spectrum free
/// function is superseded by the explicit raw-compute + spectral-transform
/// composition. Build the raw dipole-flux ACF and pass it to
/// :class:`molrs.IRSpectrum` / :class:`molrs.PowerSpectrum`. This binding still
/// works and returns the unchanged dict, but emits a ``DeprecationWarning``.
#[pyfunction]
#[pyo3(signature = (dipole_moments, dt, volume, temperature, epsilon_inf, max_correlation_time, window_type="hann"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dielectric_einstein_helfand_spectrum<'py>(
    py: Python<'py>,
    dipole_moments: PyReadonlyArray2<'py, f64>,
    dt: f64,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
    max_correlation_time: usize,
    window_type: &str,
) -> PyResult<Py<PyAny>> {
    warn_deprecated(
        py,
        "dielectric_einstein_helfand_spectrum is deprecated; compose the raw \
         dipole-flux ACF with molrs.IRSpectrum / molrs.PowerSpectrum instead.",
    )?;
    let dm = dipole_moments.as_array().to_owned();
    let result = diel::einstein_helfand_spectrum(
        &dm,
        dt,
        volume,
        temperature,
        epsilon_inf,
        max_correlation_time,
        window_type,
    )
    .map_err(py_value_err)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("frequencies", result.frequencies.into_pyarray(py))?;
    dict.set_item("epsilon_real", result.epsilon_real.into_pyarray(py))?;
    dict.set_item("epsilon_imag", result.epsilon_imag.into_pyarray(py))?;
    dict.set_item("n_frames", result.n_frames)?;
    dict.set_item("n_correlation_steps", result.n_correlation_steps)?;
    Ok(dict.into())
}

/// **Deprecated** (phase-02 compute/fit repoint): superseded by the explicit
/// raw current ACF (:class:`molrs.GreenKuboConductivity`) + spectral transform
/// composition. This binding still works and returns the unchanged dict, but
/// emits a ``DeprecationWarning``.
#[pyfunction]
#[pyo3(signature = (current_density, dt, volume, temperature, epsilon_inf, max_correlation_time, window_type="hann"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dielectric_green_kubo_spectrum<'py>(
    py: Python<'py>,
    current_density: PyReadonlyArray2<'py, f64>,
    dt: f64,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
    max_correlation_time: usize,
    window_type: &str,
) -> PyResult<Py<PyAny>> {
    warn_deprecated(
        py,
        "dielectric_green_kubo_spectrum is deprecated; compose the raw current \
         ACF (molrs.GreenKuboConductivity) with a spectral transform instead.",
    )?;
    let cd = current_density.as_array().to_owned();
    let result = diel::green_kubo_spectrum(
        &cd,
        dt,
        volume,
        temperature,
        epsilon_inf,
        max_correlation_time,
        window_type,
    )
    .map_err(py_value_err)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("frequencies", result.frequencies.into_pyarray(py))?;
    dict.set_item("epsilon_real", result.epsilon_real.into_pyarray(py))?;
    dict.set_item("epsilon_imag", result.epsilon_imag.into_pyarray(py))?;
    dict.set_item("n_frames", result.n_frames)?;
    dict.set_item("n_correlation_steps", result.n_correlation_steps)?;
    Ok(dict.into())
}

// The legacy bundled `dielectric_einstein_helfand_conductivity` binding (raw MSD
// + fitted sigma/slope/fit_start/fit_end) was removed in compute-fit-03-cleanup.
// Compose `molrs.EinsteinConductivity` (raw collective-dipole MSD) with
// `molrs.LinearFit` and a caller-applied `slope/(6·V·k_B·T)` MD→SI prefactor.

#[pyfunction]
#[allow(clippy::type_complexity)]
pub(crate) fn dielectric_decompose_current<'py>(
    py: Python<'py>,
    per_particle_current: PyReadonlyArray3<'py, f64>,
    water_mask: PyReadonlyArray1<'py, bool>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let current = per_particle_current.as_array().to_owned();
    let mask = water_mask.as_array().to_owned();
    let (j_w, j_i) = diel::decompose_current(&current, &mask).map_err(py_value_err)?;
    Ok((j_w.into_pyarray(py), j_i.into_pyarray(py)))
}

pub fn register_dielectric(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dielectric_compute_dipole_moment, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric_compute_current_density, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric_static_dielectric_constant, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric_einstein_helfand_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric_green_kubo_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric_decompose_current, m)?)?;
    Ok(())
}
