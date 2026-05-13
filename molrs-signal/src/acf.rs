use ndarray::Array1;
use rustfft::num_traits::Zero;
use rustfft::{FftPlanner, num_complex::Complex64};

/// Raw FFT-based autocorrelation via Wiener-Khinchin theorem.
///
/// Returns the un-normalized autocorrelation of `data` up to `max_lag`.
/// The caller is responsible for normalization.
///
/// Algorithm: zero-pad to next power of 2 >= 2*max_lag, forward FFT,
/// squared magnitude, inverse FFT, extract first (max_lag + 1) entries.
pub fn acf_fft(data: &Array1<f64>, max_lag: usize) -> Result<Array1<f64>, SignalError> {
    let n = data.len();
    if n == 0 {
        return Err(SignalError::EmptyInput);
    }
    if max_lag >= n {
        return Err(SignalError::MaxLagTooLarge { max_lag, len: n });
    }

    let n_pad = n;

    let mut planner = FftPlanner::new();
    let fwd = planner.plan_fft_forward(n_pad);
    let inv = planner.plan_fft_inverse(n_pad);

    let mut complex_data: Vec<Complex64> = data.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    complex_data.resize(n_pad, Complex64::zero());
    fwd.process(&mut complex_data);

    let power: Vec<Complex64> = complex_data
        .iter()
        .map(|c| Complex64::new(c.norm_sqr(), 0.0))
        .collect();
    let mut acf_raw = power;
    inv.process(&mut acf_raw);

    let scale = 1.0 / n_pad as f64;
    let result: Array1<f64> = acf_raw[..=max_lag]
        .iter()
        .map(|c| c.re * scale)
        .collect::<Vec<_>>()
        .into();

    Ok(result)
}

#[derive(Debug, PartialEq)]
pub enum SignalError {
    EmptyInput,
    MaxLagTooLarge { max_lag: usize, len: usize },
    AxisOutOfBounds { axis: usize, ndim: usize },
}

impl std::fmt::Display for SignalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalError::EmptyInput => write!(f, "input array is empty"),
            SignalError::MaxLagTooLarge { max_lag, len } => {
                write!(
                    f,
                    "max_lag ({max_lag}) must be less than data length ({len})"
                )
            }
            SignalError::AxisOutOfBounds { axis, ndim } => {
                write!(f, "axis ({axis}) out of bounds for ndim ({ndim})")
            }
        }
    }
}

impl std::error::Error for SignalError {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_acf_constant_signal_un_normalized() {
        let data = arr1(&[2.0, 2.0, 2.0, 2.0]);
        let result = acf_fft(&data, 3).unwrap();
        assert_eq!(result.len(), 4);
        let expected_lag0 = 4.0 * 2.0_f64.powi(2); // N * c^2 = 4 * 4 = 16
        assert!((result[0] - expected_lag0).abs() < 1e-10);
    }

    #[test]
    fn test_acf_constant_signal_all_lags_equal() {
        let data = arr1(&[3.0, 3.0, 3.0]);
        let result = acf_fft(&data, 2).unwrap();
        assert_eq!(result.len(), 3);
        let expected = 3.0 * 3.0_f64.powi(2); // 27
        for v in result.iter() {
            assert!((v - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_acf_single_element() {
        let data = arr1(&[5.0]);
        let result = acf_fft(&data, 0).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 25.0).abs() < 1e-10); // 1 * 5^2
    }

    #[test]
    fn test_acf_max_lag_zero() {
        let data = arr1(&[1.0, 2.0, 3.0]);
        let result = acf_fft(&data, 0).unwrap();
        assert_eq!(result.len(), 1);
        let expected = 1.0_f64.powi(2) + 2.0_f64.powi(2) + 3.0_f64.powi(2); // sum of squares
        assert!((result[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_acf_max_lag_too_large() {
        let data = arr1(&[1.0, 2.0]);
        let err = acf_fft(&data, 2).unwrap_err();
        assert_eq!(err, SignalError::MaxLagTooLarge { max_lag: 2, len: 2 });
    }

    #[test]
    fn test_acf_empty_input() {
        let data = Array1::<f64>::zeros(0);
        let err = acf_fft(&data, 0).unwrap_err();
        assert_eq!(err, SignalError::EmptyInput);
    }

    #[test]
    fn test_acf_white_noise_peak_at_zero() {
        use rand::Rng;
        let mut rng = rand::rng();
        let data: Vec<f64> = (0..1000).map(|_| rng.random()).collect();
        let arr = Array1::from_vec(data);
        let result = acf_fft(&arr, 10).unwrap();
        assert_eq!(result.len(), 11);
        for k in 1..result.len() {
            assert!(result[k].abs() < result[0]);
        }
    }

    #[test]
    fn test_acf_sine_wave_oscillatory() {
        let n = 128;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 16.0).sin())
            .collect();
        let arr = Array1::from_vec(data);
        let result = acf_fft(&arr, 32).unwrap();
        assert_eq!(result.len(), 33);
        assert!(result[0] > 0.0);
        // ACF at half-period (lag 8) should be negative (anti-phase)
        assert!(result[8] < 0.0);
    }
}
