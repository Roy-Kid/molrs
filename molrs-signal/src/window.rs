use ndarray::ArrayD;

use super::acf::SignalError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    Hann,
    Blackman,
}

/// Apply a window function to `data` along the specified `axis`.
///
/// Returns a new array; the input is never mutated.
pub fn apply_window(
    data: &ArrayD<f64>,
    window: WindowType,
    axis: usize,
) -> Result<ArrayD<f64>, SignalError> {
    let ndim = data.ndim();
    if axis >= ndim {
        return Err(SignalError::AxisOutOfBounds { axis, ndim });
    }

    let window_len = data.shape()[axis];
    if window_len == 0 {
        return Ok(data.clone());
    }

    let w: Vec<f64> = match window {
        WindowType::Hann => (0..window_len)
            .map(|n| {
                if window_len == 1 {
                    1.0
                } else {
                    0.5 * (1.0
                        - (2.0 * std::f64::consts::PI * n as f64 / (window_len - 1) as f64).cos())
                }
            })
            .collect(),
        WindowType::Blackman => (0..window_len)
            .map(|n| {
                if window_len == 1 {
                    1.0
                } else {
                    0.42 - 0.5
                        * (2.0 * std::f64::consts::PI * n as f64 / (window_len - 1) as f64).cos()
                        + 0.08
                            * (4.0 * std::f64::consts::PI * n as f64 / (window_len - 1) as f64)
                                .cos()
                }
            })
            .collect(),
    };

    let mut result = data.clone();
    for idx in ndarray::indices_of(&result) {
        let axis_idx = idx[axis];
        result[idx] *= w[axis_idx];
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    fn hann_analytical(n: usize, idx: usize) -> f64 {
        if n == 1 {
            return 1.0;
        }
        0.5 * (1.0 - (2.0 * std::f64::consts::PI * idx as f64 / (n - 1) as f64).cos())
    }

    fn blackman_analytical(n: usize, idx: usize) -> f64 {
        if n == 1 {
            return 1.0;
        }
        0.42 - 0.5 * (2.0 * std::f64::consts::PI * idx as f64 / (n - 1) as f64).cos()
            + 0.08 * (4.0 * std::f64::consts::PI * idx as f64 / (n - 1) as f64).cos()
    }

    #[test]
    fn test_hann_1d_matches_analytical() {
        let n = 5;
        let data = ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), vec![1.0; n]).unwrap();
        let result = apply_window(&data, WindowType::Hann, 0).unwrap();
        assert_eq!(result.shape(), &[5]);
        for (i, v) in result.iter().enumerate() {
            let expected = hann_analytical(n, i);
            assert!((v - expected).abs() < 1e-12, "idx {i}: {v} != {expected}");
        }
    }

    #[test]
    fn test_blackman_1d_matches_analytical() {
        let n = 5;
        let data = ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), vec![1.0; n]).unwrap();
        let result = apply_window(&data, WindowType::Blackman, 0).unwrap();
        for (i, v) in result.iter().enumerate() {
            let expected = blackman_analytical(n, i);
            assert!((v - expected).abs() < 1e-12, "idx {i}: {v} != {expected}");
        }
    }

    #[test]
    fn test_hann_2d_axis0_each_column_is_window() {
        let data = ArrayD::from_elem(ndarray::IxDyn(&[5, 3]), 1.0);
        let result = apply_window(&data, WindowType::Hann, 0).unwrap();
        assert_eq!(result.shape(), &[5, 3]);
        let expected_col = (0..5).map(|i| hann_analytical(5, i)).collect::<Vec<_>>();
        for col in 0..3 {
            for row in 0..5 {
                let v = result[[row, col]];
                assert!((v - expected_col[row]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_hann_2d_axis1_each_row_is_window() {
        let data = ArrayD::from_elem(ndarray::IxDyn(&[3, 5]), 1.0);
        let result = apply_window(&data, WindowType::Hann, 1).unwrap();
        assert_eq!(result.shape(), &[3, 5]);
        let expected_row = (0..5).map(|i| hann_analytical(5, i)).collect::<Vec<_>>();
        for row in 0..3 {
            for col in 0..5 {
                let v = result[[row, col]];
                assert!((v - expected_row[col]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_hann_boundary_values() {
        let n = 100;
        let data = ArrayD::from_elem(ndarray::IxDyn(&[n]), 1.0);
        let result = apply_window(&data, WindowType::Hann, 0).unwrap();
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[n - 1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_blackman_boundary_values() {
        let n = 100;
        let data = ArrayD::from_elem(ndarray::IxDyn(&[n]), 1.0);
        let result = apply_window(&data, WindowType::Blackman, 0).unwrap();
        let expected_first = 0.42 - 0.5 + 0.08; // = 0.0 at n=0
        assert!((result[0] - expected_first).abs() < 1e-10);
        let expected_last = 0.42
            - 0.5 * (2.0 * std::f64::consts::PI * (n - 1) as f64 / (n - 1) as f64).cos()
            + 0.08 * (4.0 * std::f64::consts::PI * (n - 1) as f64 / (n - 1) as f64).cos();
        assert!((result[n - 1] - expected_last).abs() < 1e-10);
    }

    #[test]
    fn test_axis_out_of_bounds() {
        let data = ArrayD::from_elem(ndarray::IxDyn(&[5, 3]), 1.0);
        let err = apply_window(&data, WindowType::Hann, 2).unwrap_err();
        assert!(matches!(
            err,
            SignalError::AxisOutOfBounds { axis: 2, ndim: 2 }
        ));
    }
}
