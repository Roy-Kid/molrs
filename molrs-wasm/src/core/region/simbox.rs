use crate::core::block::Block;
use crate::core::types::WasmArray;
use molrs::core::region::simbox::SimBox;
use wasm_bindgen::prelude::*;

/// Simulation box defining the periodic boundary conditions and coordinate transformations.
///
/// Represents a parallelepiped defined by a 3x3 matrix `h` and an origin.
/// Supports periodic boundary conditions (PBC) independently in x, y, z directions.
#[wasm_bindgen]
pub struct Box {
    pub(crate) inner: SimBox,
}

#[inline]
fn vec3_from_slice(v: &[f32]) -> ndarray::Array1<f32> {
    ndarray::arr1(&[v[0], v[1], v[2]])
}

#[inline]
fn mat3_from_slice(v: &[f32]) -> ndarray::Array2<f32> {
    ndarray::arr2(&[[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]])
}

#[inline]
fn array2_into_parts(arr: ndarray::Array2<f32>) -> (Vec<f32>, std::boxed::Box<[usize]>) {
    let shape = std::boxed::Box::new([arr.nrows(), arr.ncols()]);
    let (data, _offset) = arr.into_raw_vec_and_offset();
    (data, shape)
}

#[wasm_bindgen]
impl Box {
    /// Create a new box from a 3x3 matrix and origin.
    ///
    /// # Arguments
    /// * `h` - 3x3 matrix as Float32Array with 9 elements (row-major: [h00, h01, h02, h10, ...])
    /// * `origin` - 3D origin vector as Float32Array with 3 elements [x, y, z]
    /// * `pbc_x`, `pbc_y`, `pbc_z` - Periodic boundary conditions for each axis
    #[wasm_bindgen(constructor)]
    pub fn new(
        h: &js_sys::Float32Array,
        origin: &js_sys::Float32Array,
        pbc_x: bool,
        pbc_y: bool,
        pbc_z: bool,
    ) -> Result<Box, JsValue> {
        if h.length() != 9 {
            return Err(JsValue::from_str("h must have 9 elements (3x3 matrix)"));
        }
        if origin.length() != 3 {
            return Err(JsValue::from_str("origin must have 3 elements"));
        }

        let h_vec = h.to_vec();
        let origin_vec = origin.to_vec();

        let h_mat = mat3_from_slice(&h_vec);
        let origin_arr = vec3_from_slice(&origin_vec);
        let inner = SimBox::new(h_mat, origin_arr, [pbc_x, pbc_y, pbc_z])
            .map_err(|e| JsValue::from_str(&format!("box new error: {:?}", e)))?;
        Ok(Box { inner })
    }

    /// Create a cubic box.
    ///
    /// # Arguments
    /// * `a` - Side length of the cube
    /// * `origin` - 3D origin vector as Float32Array with 3 elements
    /// * `pbc_x`, `pbc_y`, `pbc_z` - Periodic boundary conditions for each axis
    pub fn cube(
        a: f32,
        origin: &js_sys::Float32Array,
        pbc_x: bool,
        pbc_y: bool,
        pbc_z: bool,
    ) -> Result<Box, JsValue> {
        if origin.length() != 3 {
            return Err(JsValue::from_str("origin must have 3 elements"));
        }
        let origin_vec = origin.to_vec();
        let inner = SimBox::cube(a, vec3_from_slice(&origin_vec), [pbc_x, pbc_y, pbc_z])
            .map_err(|e| JsValue::from_str(&format!("box cube error: {:?}", e)))?;
        Ok(Box { inner })
    }

    /// Create an orthorhombic box (rectangular box with axis-aligned edges).
    ///
    /// # Arguments
    /// * `lengths` - Box dimensions as Float32Array with 3 elements [lx, ly, lz]
    /// * `origin` - 3D origin vector as Float32Array with 3 elements
    /// * `pbc_x`, `pbc_y`, `pbc_z` - Periodic boundary conditions for each axis
    pub fn ortho(
        lengths: &js_sys::Float32Array,
        origin: &js_sys::Float32Array,
        pbc_x: bool,
        pbc_y: bool,
        pbc_z: bool,
    ) -> Result<Box, JsValue> {
        if lengths.length() != 3 {
            return Err(JsValue::from_str("lengths must have 3 elements"));
        }
        if origin.length() != 3 {
            return Err(JsValue::from_str("origin must have 3 elements"));
        }
        let lengths_vec = lengths.to_vec();
        let origin_vec = origin.to_vec();
        let inner = SimBox::ortho(
            vec3_from_slice(&lengths_vec),
            vec3_from_slice(&origin_vec),
            [pbc_x, pbc_y, pbc_z],
        )
        .map_err(|e| JsValue::from_str(&format!("box ortho error: {:?}", e)))?;
        Ok(Box { inner })
    }

    /// Get the volume of the box.
    pub fn volume(&self) -> f32 {
        self.inner.volume()
    }

    /// Get the box origin as WasmArray [x, y, z].
    pub fn origin(&self) -> WasmArray {
        let o = self.inner.origin_view();
        WasmArray::from_vec(o.to_vec(), std::boxed::Box::new([3]))
    }

    /// Get the box edge lengths as WasmArray [lx, ly, lz].
    pub fn lengths(&self) -> WasmArray {
        let l = self.inner.lengths();
        WasmArray::from_vec(l.to_vec(), std::boxed::Box::new([3]))
    }

    /// Get the box tilt factors as WasmArray [xy, xz, yz].
    pub fn tilts(&self) -> WasmArray {
        let t = self.inner.tilts();
        WasmArray::from_vec(t.to_vec(), std::boxed::Box::new([3]))
    }

    /// Convert Cartesian coordinates to fractional coordinates.
    ///
    /// # Arguments
    /// * `coords` - WasmArray with Nx3 coordinates
    ///
    /// # Returns
    /// WasmArray with Nx3 fractional coordinates
    pub fn to_frac(&self, coords: &WasmArray) -> Result<WasmArray, JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;

        let result = self.inner.to_frac(coords_arr);
        Ok(WasmArray::from_array2(result))
    }

    /// Convert fractional coordinates to Cartesian coordinates.
    ///
    /// # Arguments
    /// * `coords` - WasmArray with Nx3 coordinates
    ///
    /// # Returns
    /// WasmArray with Nx3 Cartesian coordinates
    pub fn to_cart(&self, coords: &WasmArray) -> Result<WasmArray, JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;

        let result = self.inner.to_cart(coords_arr);
        Ok(WasmArray::from_array2(result))
    }

    /// Wrap coordinates into the primary simulation box.
    ///
    /// # Arguments
    /// * `coords` - WasmArray with Nx3 coordinates
    ///
    /// # Returns
    /// WasmArray with Nx3 wrapped coordinates
    pub fn wrap(&self, coords: &WasmArray) -> Result<WasmArray, JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;

        let result = self.inner.wrap(coords_arr);
        Ok(WasmArray::from_array2(result))
    }

    #[wasm_bindgen(js_name = wrapToBlock)]
    pub fn wrap_to_block(
        &self,
        coords: &WasmArray,
        out_block: &mut Block,
        out_key: &str,
    ) -> Result<(), JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;
        let result = self.inner.wrap(coords_arr);
        let (data, shape) = array2_into_parts(result);
        out_block.set_owned_column(out_key, data, shape)
    }

    /// Calculate displacement vectors between two sets of coordinates.
    ///
    /// # Arguments
    /// * `a` - WasmArray with Nx3 coordinates
    /// * `b` - WasmArray with Nx3 coordinates
    /// * `minimum_image` - If true, apply minimum image convention
    ///
    /// # Returns
    /// WasmArray with Nx3 displacement vectors (b - a)
    pub fn delta(
        &self,
        a: &WasmArray,
        b: &WasmArray,
        minimum_image: bool,
    ) -> Result<WasmArray, JsValue> {
        let shape_a = a.shape();
        let shape_b = b.shape();
        if shape_a.len() != 2 || shape_a[1] != 3 {
            return Err(JsValue::from_str("a must have shape [N, 3]"));
        }
        if shape_b.len() != 2 || shape_b[1] != 3 {
            return Err(JsValue::from_str("b must have shape [N, 3]"));
        }
        if shape_a[0] != shape_b[0] {
            return Err(JsValue::from_str(
                "a and b must have the same number of atoms",
            ));
        }

        let n_atoms = shape_a[0];
        let a_arr = a.as_array2(n_atoms, 3).map_err(|e| JsValue::from_str(&e))?;
        let b_arr = b.as_array2(n_atoms, 3).map_err(|e| JsValue::from_str(&e))?;

        let result = self.inner.delta(a_arr, b_arr, minimum_image);
        Ok(WasmArray::from_array2(result))
    }

    #[wasm_bindgen(js_name = deltaToBlock)]
    pub fn delta_to_block(
        &self,
        a: &WasmArray,
        b: &WasmArray,
        minimum_image: bool,
        out_block: &mut Block,
        out_key: &str,
    ) -> Result<(), JsValue> {
        let shape_a = a.shape();
        let shape_b = b.shape();
        if shape_a.len() != 2 || shape_a[1] != 3 {
            return Err(JsValue::from_str("a must have shape [N, 3]"));
        }
        if shape_b.len() != 2 || shape_b[1] != 3 {
            return Err(JsValue::from_str("b must have shape [N, 3]"));
        }
        if shape_a[0] != shape_b[0] {
            return Err(JsValue::from_str(
                "a and b must have the same number of atoms",
            ));
        }
        let n_atoms = shape_a[0];
        let a_arr = a.as_array2(n_atoms, 3).map_err(|e| JsValue::from_str(&e))?;
        let b_arr = b.as_array2(n_atoms, 3).map_err(|e| JsValue::from_str(&e))?;
        let result = self.inner.delta(a_arr, b_arr, minimum_image);
        let (data, shape) = array2_into_parts(result);
        out_block.set_owned_column(out_key, data, shape)
    }

    #[wasm_bindgen(js_name = toFracToBlock)]
    pub fn to_frac_to_block(
        &self,
        coords: &WasmArray,
        out_block: &mut Block,
        out_key: &str,
    ) -> Result<(), JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;
        let result = self.inner.to_frac(coords_arr);
        let (data, shape) = array2_into_parts(result);
        out_block.set_owned_column(out_key, data, shape)
    }

    #[wasm_bindgen(js_name = toCartToBlock)]
    pub fn to_cart_to_block(
        &self,
        coords: &WasmArray,
        out_block: &mut Block,
        out_key: &str,
    ) -> Result<(), JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;
        let result = self.inner.to_cart(coords_arr);
        let (data, shape) = array2_into_parts(result);
        out_block.set_owned_column(out_key, data, shape)
    }

    pub fn get_corners(&self) -> WasmArray {
        let corners = self.inner.get_corners();
        WasmArray::from_array2(corners)
    }
}

#[cfg(test)]
mod tests {
    use super::Box as WasmBox;
    use crate::{Frame, WasmArray};
    use js_sys::Float32Array;
    use wasm_bindgen_test::wasm_bindgen_test;

    fn f32(values: &[f32]) -> Float32Array {
        Float32Array::from(values)
    }

    fn assert_eq_array(actual: &Float32Array, expected: &[f32]) {
        assert_eq!(actual.length() as usize, expected.len());
        for (i, value) in expected.iter().enumerate() {
            let got = actual.get_index(i as u32);
            assert!(
                (got - value).abs() < 1.0e-5,
                "index {}: {} != {}",
                i,
                got,
                value
            );
        }
    }

    #[wasm_bindgen_test]
    fn box_coordinate_ops() {
        let mut view = WasmArray::new(Box::new([2_usize, 3_usize]));
        let view_data = f32(&[0.0, 0.0, 0.0, 2.0, 3.0, 4.0]);
        view.write_from(&view_data).expect("write_from failed");

        let h = f32(&[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]);
        let origin = f32(&[0.0, 0.0, 0.0]);
        let sim_box = WasmBox::new(&h, &origin, true, true, true).expect("box new");

        let frac = sim_box.to_frac(&view).expect("to_frac failed");
        let frac_js = frac.to_copy();
        assert_eq_array(&frac_js, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let cart = sim_box.to_cart(&frac).expect("to_cart failed");
        let cart_js = cart.to_copy();
        assert_eq_array(&cart_js, &[0.0, 0.0, 0.0, 2.0, 3.0, 4.0]);

        let mut wrap_view = WasmArray::new(Box::new([1_usize, 3_usize]));
        wrap_view
            .write_from(&f32(&[2.5, 3.5, 4.5]))
            .expect("write wrap_view");
        let wrapped = sim_box.wrap(&wrap_view).expect("wrap failed");
        let wrapped_js = wrapped.to_copy();
        assert_eq_array(&wrapped_js, &[0.5, 0.5, 0.5]);

        let corners = sim_box.get_corners();
        assert_eq!(corners.len(), 24);

        let cube = WasmBox::cube(10.0, &origin, true, true, true).expect("cube");
        assert!((cube.volume() - 1000.0).abs() < 1.0e-5);

        let lengths = cube.lengths().to_copy();
        assert_eq_array(&lengths, &[10.0, 10.0, 10.0]);

        let tilts = cube.tilts().to_copy();
        assert_eq_array(&tilts, &[0.0, 0.0, 0.0]);

        let ortho_lengths = f32(&[3.0, 4.0, 5.0]);
        let ortho = WasmBox::ortho(&ortho_lengths, &origin, true, true, true).expect("ortho");
        let ortho_lengths_out = ortho.lengths().to_copy();
        assert_eq_array(&ortho_lengths_out, &[3.0, 4.0, 5.0]);

        let mut a = WasmArray::new(Box::new([1_usize, 3_usize]));
        let mut b = WasmArray::new(Box::new([1_usize, 3_usize]));
        a.write_from(&f32(&[1.0, 1.0, 1.0])).expect("write a");
        b.write_from(&f32(&[9.0, 9.0, 9.0])).expect("write b");

        let delta_mi = cube.delta(&a, &b, true).expect("delta mi").to_copy();
        let delta_no = cube.delta(&a, &b, false).expect("delta no").to_copy();
        assert!(delta_mi.get_index(0).abs() < delta_no.get_index(0).abs());

        let frame = Frame::new();
        let mut out = frame.create_block("out").expect("create out");
        cube.wrap_to_block(&wrap_view, &mut out, "wrapped")
            .expect("wrapToBlock");
        let wrapped_out = out
            .column_view("wrapped")
            .expect("columnView")
            .to_copy()
            .expect("toCopy");
        assert_eq_array(&wrapped_out, &[0.5, 0.5, 0.5]);
    }
}
