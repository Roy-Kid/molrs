//! WASM bindings for Block and zero-copy ColumnView.

use js_sys::{Array as JsArray, Float32Array};
use ndarray::Array1;
use wasm_bindgen::prelude::*;

use molrs::core::block::{Block as RsBlock, DType};
use molrs_ffi::BlockHandle as FFIBlockHandle;

use super::types::WasmArray;
use super::{SharedStore, js_err};

/// Zero-copy borrowed view of a block column.
#[wasm_bindgen]
pub struct ColumnView {
    block: FFIBlockHandle,
    key: String,
    store: SharedStore,
    len: usize,
    shape: Box<[usize]>,
}

#[wasm_bindgen]
impl ColumnView {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn shape(&self) -> Box<[usize]> {
        self.shape.clone()
    }

    #[wasm_bindgen(js_name = isValid)]
    pub fn is_valid(&self) -> bool {
        self.store
            .borrow()
            .with_block(&self.block, |b| b.get_f32(&self.key).is_some())
            .unwrap_or(false)
    }

    /// Zero-copy JS typed array view over this column.
    ///
    /// The returned view becomes invalid when the owning block is invalidated.
    #[wasm_bindgen(js_name = toTypedArray)]
    pub fn to_typed_array(&self) -> Result<Float32Array, JsValue> {
        self.with_slice(|slice| {
            // SAFETY:
            // - Column storage is contiguous and backed by WASM linear memory.
            // - The caller is responsible for not keeping this view after invalidation.
            unsafe { Float32Array::view(slice) }
        })
    }

    /// Owned JS copy of this column.
    #[wasm_bindgen(js_name = toCopy)]
    pub fn to_copy(&self) -> Result<Float32Array, JsValue> {
        self.with_slice(|slice| Float32Array::from(slice))
    }
}

impl ColumnView {
    fn with_slice<R>(&self, f: impl FnOnce(&[f32]) -> R) -> Result<R, JsValue> {
        self.store
            .borrow()
            .with_block(&self.block, |b| {
                let arr = b.get_f32(&self.key).ok_or_else(|| {
                    JsValue::from_str(&format!("Column '{}' not found", self.key))
                })?;
                let slice = arr
                    .as_slice_memory_order()
                    .ok_or_else(|| JsValue::from_str("Non-contiguous array"))?;
                Ok::<R, JsValue>(f(slice))
            })
            .map_err(js_err)?
    }
}

/// A block containing columnar data.
#[wasm_bindgen]
pub struct Block {
    pub(crate) handle: FFIBlockHandle,
    pub(crate) store: SharedStore,
}

#[wasm_bindgen]
impl Block {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let store = std::rc::Rc::new(std::cell::RefCell::new(molrs_ffi::Store::new()));
        let fid = store.borrow_mut().frame_new();
        store
            .borrow_mut()
            .set_block(fid, "temp", RsBlock::new())
            .unwrap();
        let handle = store.borrow().get_block(fid, "temp").unwrap();
        Block { handle, store }
    }

    #[wasm_bindgen(js_name = len)]
    pub fn len(&self) -> Result<usize, JsValue> {
        self.with(|b| b.len())
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }

    #[wasm_bindgen(js_name = nrows)]
    pub fn nrows(&self) -> Result<usize, JsValue> {
        self.with(|b| b.nrows().unwrap_or(0))
    }

    #[wasm_bindgen(js_name = keys)]
    pub fn keys(&self) -> Result<JsArray, JsValue> {
        self.with(|b| {
            let arr = JsArray::new();
            for k in b.keys() {
                arr.push(&JsValue::from_str(k));
            }
            arr
        })
    }

    /// Returns the dtype string for the given column key.
    #[wasm_bindgen(js_name = getDtype)]
    pub fn get_dtype(&self, key: &str) -> Option<String> {
        self.store
            .borrow()
            .with_block(&self.handle, |b| {
                b.dtype(key).map(|dt| {
                    match dt {
                        DType::F32 => "f32",
                        DType::F64 => "f64",
                        DType::I64 => "i64",
                        DType::U32 => "u32",
                        DType::U8 => "u8",
                        DType::Bool => "bool",
                        DType::String => "string",
                    }
                    .to_string()
                })
            })
            .ok()
            .flatten()
    }

    #[wasm_bindgen(js_name = renameColumn)]
    pub fn rename_column(&mut self, old_key: &str, new_key: &str) -> Result<bool, JsValue> {
        self.store
            .borrow_mut()
            .with_block_mut(&mut self.handle, |b| b.rename_column(old_key, new_key))
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = setColumn)]
    pub fn set_column(
        &mut self,
        key: &str,
        data: &Float32Array,
        shape: Option<Box<[usize]>>,
    ) -> Result<(), JsValue> {
        self.insert_f32(key, data.to_vec(), shape)
    }

    #[wasm_bindgen(js_name = setFromView)]
    pub fn set_from_view(&mut self, key: &str, view: &WasmArray) -> Result<(), JsValue> {
        self.insert_f32(key, view.as_slice().to_vec(), Some(view.shape()))
    }

    #[wasm_bindgen(js_name = column)]
    pub fn column(&self, key: &str) -> Result<WasmArray, JsValue> {
        let (data, shape) = self
            .store
            .borrow()
            .block_col_f32(&self.handle, key)
            .map_err(js_err)?;
        Ok(WasmArray::from_vec(data, shape.into_boxed_slice()))
    }

    #[wasm_bindgen(js_name = columnCopy)]
    pub fn column_copy(&self, key: &str) -> Result<Float32Array, JsValue> {
        let (data, _shape) = self
            .store
            .borrow()
            .block_col_f32(&self.handle, key)
            .map_err(js_err)?;
        Ok(Float32Array::from(&data[..]))
    }

    #[wasm_bindgen(js_name = columnView)]
    pub fn column_view(&self, key: &str) -> Result<ColumnView, JsValue> {
        let (_ptr, len, shape) = self
            .store
            .borrow()
            .block_col_f32_view(&self.handle, key)
            .map_err(js_err)?;

        Ok(ColumnView {
            block: self.handle.clone(),
            key: key.to_string(),
            store: self.store.clone(),
            len,
            shape: shape.into_boxed_slice(),
        })
    }

    #[wasm_bindgen(js_name = setColumnStrings)]
    pub fn set_column_strings(&mut self, key: &str, data: JsArray) -> Result<(), JsValue> {
        let strings: Vec<String> = data
            .iter()
            .map(|v| v.as_string().unwrap_or_default())
            .collect();
        self.insert_col(key, Array1::from(strings).into_dyn())
    }

    #[wasm_bindgen(js_name = getColumnStrings)]
    pub fn get_column_strings(&self, key: &str) -> Option<JsArray> {
        self.store
            .borrow()
            .with_block(&self.handle, |b| {
                let arr = b.get_string(key)?;
                let js = JsArray::new();
                for s in arr.iter() {
                    js.push(&JsValue::from_str(s));
                }
                Some(js)
            })
            .ok()?
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

impl Block {
    fn with<R>(&self, f: impl FnOnce(&RsBlock) -> R) -> Result<R, JsValue> {
        self.store
            .borrow()
            .with_block(&self.handle, f)
            .map_err(js_err)
    }

    fn insert_col<T: molrs::core::block::BlockDtype>(
        &mut self,
        key: &str,
        array: ndarray::ArrayD<T>,
    ) -> Result<(), JsValue> {
        self.store
            .borrow_mut()
            .with_block_mut(&mut self.handle, |b| {
                b.insert(key, array)
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            })
            .map_err(js_err)?
    }

    fn insert_f32(
        &mut self,
        key: &str,
        data: Vec<f32>,
        shape: Option<Box<[usize]>>,
    ) -> Result<(), JsValue> {
        let array = if let Some(dims) = shape {
            let dims_vec = Vec::from(dims);
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims_vec), data)
                .map_err(|e| JsValue::from_str(&e.to_string()))?
        } else {
            Array1::from(data).into_dyn()
        };
        self.insert_col(key, array)
    }

    pub(crate) fn set_owned_column(
        &mut self,
        key: &str,
        data: Vec<f32>,
        shape: Box<[usize]>,
    ) -> Result<(), JsValue> {
        self.insert_f32(key, data, Some(shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::frame::Frame;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_block_operations() {
        let frame = Frame::new();
        let mut block = frame.create_block("atoms").unwrap();
        let data = Float32Array::from(&[1.0_f32, 2.0, 3.0][..]);
        block.set_column("x", &data, None).unwrap();

        let retrieved = block.column_copy("x").unwrap();
        assert_eq!(retrieved.length(), 3);

        let view = block.column_view("x").unwrap();
        assert!(view.is_valid());
        assert_eq!(view.len(), 3);
    }
}
