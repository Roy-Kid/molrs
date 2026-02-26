use ndarray::{Array2, ArrayView2};
use wasm_bindgen::prelude::*;

/// Owned numeric array in WASM memory with an ndarray-compatible shape.
#[wasm_bindgen]
pub struct WasmArray {
    data: Vec<f32>,
    shape: Box<[usize]>,
}

#[wasm_bindgen]
impl WasmArray {
    /// Create a zero-initialized array with the given shape.
    #[wasm_bindgen(constructor)]
    pub fn new(shape: Box<[usize]>) -> Self {
        let len: usize = shape.iter().product();
        let data = vec![0.0; len];
        Self { data, shape }
    }

    /// Create an array from a JS Float32Array.
    ///
    /// If `shape` is omitted, a 1D shape `[len]` is used.
    #[wasm_bindgen(js_name = from)]
    pub fn from_js(
        data: &js_sys::Float32Array,
        shape: Option<Box<[usize]>>,
    ) -> Result<WasmArray, JsValue> {
        let shape = shape.unwrap_or_else(|| Box::new([data.length() as usize]));
        let expected: usize = shape.iter().product();
        if expected != data.length() as usize {
            return Err(JsValue::from_str(&format!(
                "Shape mismatch: shape product {} but data length {}",
                expected,
                data.length()
            )));
        }
        Ok(WasmArray {
            data: data.to_vec(),
            shape,
        })
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a copy of shape metadata.
    pub fn shape(&self) -> Box<[usize]> {
        self.shape.clone()
    }

    /// Get a raw pointer to the underlying data.
    pub fn ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Overwrite the current array data from a JS Float32Array.
    pub fn write_from(&mut self, arr: &js_sys::Float32Array) -> Result<(), JsValue> {
        if arr.length() as usize != self.data.len() {
            return Err(JsValue::from_str(&format!(
                "Array length mismatch: expected {}, got {}",
                self.data.len(),
                arr.length()
            )));
        }
        arr.copy_to(&mut self.data);
        Ok(())
    }

    /// Zero-copy JS typed array view over this array's backing storage.
    ///
    /// The returned view becomes invalid if WASM memory grows.
    #[wasm_bindgen(js_name = toTypedArray)]
    pub fn to_typed_array(&self) -> js_sys::Float32Array {
        // SAFETY:
        // - `self.data` is contiguous and lives in WASM linear memory.
        // - JS callers must treat this as a short-lived view.
        unsafe { js_sys::Float32Array::view(self.data.as_slice()) }
    }

    /// Owned JS copy of the data.
    #[wasm_bindgen(js_name = toCopy)]
    pub fn to_copy(&self) -> js_sys::Float32Array {
        js_sys::Float32Array::from(self.data.as_slice())
    }

    /// Compute the sum of all elements (for testing).
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }
}

// Internal methods not exposed to JavaScript
impl WasmArray {
    pub(crate) fn from_vec(data: Vec<f32>, shape: Box<[usize]>) -> Self {
        Self { data, shape }
    }

    pub(crate) fn as_array2(
        &self,
        rows: usize,
        cols: usize,
    ) -> Result<ArrayView2<'_, f32>, String> {
        if rows * cols != self.data.len() {
            return Err(format!(
                "Shape mismatch: {}x{} = {} but data has {} elements",
                rows,
                cols,
                rows * cols,
                self.data.len()
            ));
        }
        ArrayView2::from_shape((rows, cols), &self.data)
            .map_err(|e| format!("Failed to create array view: {}", e))
    }

    pub(crate) fn from_array2(arr: Array2<f32>) -> Self {
        let shape = Box::new([arr.nrows(), arr.ncols()]);
        let (data, _offset) = arr.into_raw_vec_and_offset();
        Self { data, shape }
    }

    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::WasmArray;
    use js_sys::Float32Array;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_array_basic_ops() {
        let mut view = WasmArray::new(Box::new([2_usize, 3_usize]));
        assert_eq!(view.len(), 6);
        assert_eq!(&*view.shape(), &[2, 3]);

        let data = Float32Array::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..]);
        view.write_from(&data).expect("write_from failed");
        assert!((view.sum() - 21.0).abs() < 1.0e-5);

        let js_array = view.to_copy();
        assert_eq!(js_array.length(), 6);
        assert!((js_array.get_index(0) - 1.0).abs() < 1.0e-5);
    }
}
