//! Python wrappers for `molrs-compute::dielectric`.

use molrs_compute::dielectric as diel;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;

use crate::helpers::py_value_err;

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

#[pyfunction]
#[pyo3(signature = (translational_dipole, dt, volume, temperature, max_correlation_time, fit_start_frac=0.1, fit_end_frac=0.5))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dielectric_einstein_helfand_conductivity<'py>(
    py: Python<'py>,
    translational_dipole: PyReadonlyArray2<'py, f64>,
    dt: f64,
    volume: f64,
    temperature: f64,
    max_correlation_time: usize,
    fit_start_frac: f64,
    fit_end_frac: f64,
) -> PyResult<Py<PyAny>> {
    let td = translational_dipole.as_array().to_owned();
    let result = diel::einstein_helfand_conductivity(
        &td,
        dt,
        volume,
        temperature,
        max_correlation_time,
        fit_start_frac,
        fit_end_frac,
    )
    .map_err(py_value_err)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("lag_times", result.lag_times.into_pyarray(py))?;
    dict.set_item("msd", result.msd.into_pyarray(py))?;
    dict.set_item("sigma", result.sigma)?;
    dict.set_item("slope", result.slope)?;
    dict.set_item("fit_start", result.fit_start)?;
    dict.set_item("fit_end", result.fit_end)?;
    Ok(dict.into())
}

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
    m.add_function(wrap_pyfunction!(
        dielectric_einstein_helfand_conductivity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(dielectric_green_kubo_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric_decompose_current, m)?)?;
    Ok(())
}
