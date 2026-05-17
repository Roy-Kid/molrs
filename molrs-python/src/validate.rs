//! Domain validation checks for dielectric spectra.
//!
//! Three pure functions exposed to Python under `molrs.validate`:
//!
//! * [`validate_kramers_kronig_check`] — recover ε'(ω) from ε''(ω) via a
//!   discrete Kramers-Kronig integral and compare to the supplied real part.
//! * [`validate_conductivity_sum_rule_check`] — verify the sum rule
//!   ∫₀^∞ σ(ω) dω = (π/2)·⟨J²⟩/(3 V kB T).
//! * [`validate_route_agreement_check`] — compute pairwise RMS distances
//!   between named ε(ω) arrays.
//!
//! Each function returns a Python dict with at least a `passed: bool` key
//! and route-specific metrics. All inputs are treated as immutable views.

use ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict, PyDictMethods};

const KB_KCAL_MOL_K: f64 = 1.987_204e-3;

fn require_same_len(name_a: &str, a: usize, name_b: &str, b: usize) -> PyResult<()> {
    if a != b {
        return Err(PyValueError::new_err(format!(
            "length mismatch: {name_a}={a}, {name_b}={b}"
        )));
    }
    Ok(())
}

fn require_positive(name: &str, value: f64) -> PyResult<()> {
    if !(value.is_finite() && value > 0.0) {
        return Err(PyValueError::new_err(format!(
            "{name} must be positive and finite, got {value}"
        )));
    }
    Ok(())
}

fn discrete_kramers_kronig(
    omega: &Array1<f64>,
    eps_imag: &Array1<f64>,
    eps_inf: f64,
) -> Array1<f64> {
    let n = omega.len();
    let mut recovered = Array1::<f64>::from_elem(n, eps_inf);
    for i in 0..n {
        let omega_i = omega[i];
        let mut acc = 0.0;
        for j in 0..n {
            if j == i {
                continue;
            }
            let omega_j = omega[j];
            let denom = omega_j * omega_j - omega_i * omega_i;
            if denom.abs() < 1e-30 {
                continue;
            }
            // Trapezoidal weight: half of the neighbouring spacing.
            let dw = if j == 0 {
                omega[1] - omega[0]
            } else if j == n - 1 {
                omega[n - 1] - omega[n - 2]
            } else {
                0.5 * (omega[j + 1] - omega[j - 1])
            };
            acc += eps_imag[j] * omega_j / denom * dw;
        }
        recovered[i] += (2.0 / std::f64::consts::PI) * acc;
    }
    recovered
}

#[pyfunction]
pub(crate) fn validate_kramers_kronig_check<'py>(
    py: Python<'py>,
    frequency: PyReadonlyArray1<'py, f64>,
    eps_real: PyReadonlyArray1<'py, f64>,
    eps_imag: PyReadonlyArray1<'py, f64>,
    eps_inf: f64,
) -> PyResult<Py<PyAny>> {
    let omega = frequency.as_array().to_owned();
    let real = eps_real.as_array().to_owned();
    let imag = eps_imag.as_array().to_owned();
    require_same_len("frequency", omega.len(), "eps_real", real.len())?;
    require_same_len("frequency", omega.len(), "eps_imag", imag.len())?;
    if omega.len() < 3 {
        return Err(PyValueError::new_err(
            "kramers_kronig_check requires at least 3 frequency points",
        ));
    }

    let recovered = discrete_kramers_kronig(&omega, &imag, eps_inf);
    let mut sum_abs = 0.0;
    for i in 0..real.len() {
        sum_abs += (recovered[i] - real[i]).abs();
    }
    let mae = sum_abs / (real.len() as f64);
    // Pass threshold: scale by the dynamic range of the supplied real part so
    // small absolute residuals on bounded dielectrics are forgiven.
    let dynamic_range = real.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - real.iter().copied().fold(f64::INFINITY, f64::min);
    let tol = (dynamic_range.abs() * 0.1).max(1e-2);
    let passed = mae < tol;

    let dict = PyDict::new(py);
    dict.set_item("passed", passed)?;
    dict.set_item("mae", mae)?;
    dict.set_item("eps_real_recovered", recovered.into_pyarray(py))?;
    Ok(dict.into())
}

#[pyfunction]
pub(crate) fn validate_conductivity_sum_rule_check<'py>(
    py: Python<'py>,
    frequency: PyReadonlyArray1<'py, f64>,
    conductivity: PyReadonlyArray1<'py, f64>,
    current_sq_mean: f64,
    volume: f64,
    temperature: f64,
) -> PyResult<Py<PyAny>> {
    let omega = frequency.as_array().to_owned();
    let sigma = conductivity.as_array().to_owned();
    require_same_len("frequency", omega.len(), "conductivity", sigma.len())?;
    require_positive("volume", volume)?;
    require_positive("temperature", temperature)?;
    if !current_sq_mean.is_finite() || current_sq_mean < 0.0 {
        return Err(PyValueError::new_err(
            "current_sq_mean must be finite and non-negative",
        ));
    }
    if omega.len() < 2 {
        return Err(PyValueError::new_err(
            "conductivity_sum_rule_check requires at least 2 frequency points",
        ));
    }

    let mut integral = 0.0;
    for i in 1..omega.len() {
        let dw = omega[i] - omega[i - 1];
        integral += 0.5 * (sigma[i] + sigma[i - 1]) * dw;
    }
    let expected =
        std::f64::consts::PI * 0.5 * current_sq_mean / (3.0 * volume * KB_KCAL_MOL_K * temperature);
    let denom = expected.abs().max(1e-30);
    let relative_error = (integral - expected) / denom;
    let passed = relative_error.abs() < 0.05;

    let dict = PyDict::new(py);
    dict.set_item("passed", passed)?;
    dict.set_item("relative_error", relative_error)?;
    dict.set_item("integral", integral)?;
    dict.set_item("expected", expected)?;
    Ok(dict.into())
}

#[pyfunction]
pub(crate) fn validate_route_agreement_check<'py>(
    py: Python<'py>,
    results: &Bound<'py, PyDict>,
) -> PyResult<Py<PyAny>> {
    let mut entries: Vec<(String, Array1<f64>)> = Vec::with_capacity(results.len());
    for (key, value) in results.iter() {
        let name: String = key.extract()?;
        let arr: PyReadonlyArray1<f64> = value.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "route_agreement_check: value for '{name}' must be a 1-D float64 array"
            ))
        })?;
        entries.push((name, arr.as_array().to_owned()));
    }
    if entries.len() < 2 {
        return Err(PyValueError::new_err(
            "route_agreement_check needs at least two named result arrays",
        ));
    }
    let expected_len = entries[0].1.len();
    for (name, arr) in &entries {
        if arr.len() != expected_len {
            return Err(PyValueError::new_err(format!(
                "route_agreement_check: '{name}' has length {} != expected {expected_len}",
                arr.len()
            )));
        }
    }

    let pairwise = PyDict::new(py);
    let mut max_rms = 0.0_f64;
    for i in 0..entries.len() {
        for j in (i + 1)..entries.len() {
            let (name_i, arr_i) = &entries[i];
            let (name_j, arr_j) = &entries[j];
            let mut sum_sq = 0.0;
            let mut norm = 0.0;
            for k in 0..expected_len {
                let diff = arr_i[k] - arr_j[k];
                sum_sq += diff * diff;
                norm += 0.5 * (arr_i[k].abs() + arr_j[k].abs());
            }
            let rms_abs = (sum_sq / (expected_len as f64)).sqrt();
            let scale = (norm / (expected_len as f64)).max(1e-30);
            let rms_rel = rms_abs / scale;
            max_rms = max_rms.max(rms_rel);
            pairwise.set_item(format!("{name_i}_vs_{name_j}"), rms_rel)?;
        }
    }
    let passed = max_rms < 0.10;

    let dict = PyDict::new(py);
    dict.set_item("passed", passed)?;
    dict.set_item("pairwise_rms", pairwise)?;
    Ok(dict.into())
}

pub fn register_validate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_kramers_kronig_check, m)?)?;
    m.add_function(wrap_pyfunction!(validate_conductivity_sum_rule_check, m)?)?;
    m.add_function(wrap_pyfunction!(validate_route_agreement_check, m)?)?;
    Ok(())
}
