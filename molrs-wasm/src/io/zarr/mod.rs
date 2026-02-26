use crate::core::frame::Frame;
use molrs::io::zarr::store::ZarrStoreReader;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::storage::WritableStorageTraits;
use zarrs::storage::store::MemoryStore;

/// Zarr trajectory reader for WASM.
#[wasm_bindgen(js_name = ZarrReader)]
pub struct ZarrReader {
    inner: ZarrStoreReader,
}

#[wasm_bindgen(js_class = ZarrReader)]
impl ZarrReader {
    /// Create a new ZarrReader from a map of file paths to content.
    ///
    /// @param {Map<string, Uint8Array>} files - Map of relative paths to file content
    #[wasm_bindgen(constructor)]
    pub fn new(files: js_sys::Map) -> Result<ZarrReader, JsValue> {
        let store = Arc::new(MemoryStore::new());

        for key_res in files.keys() {
            let key = key_res.map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
            let path = key
                .as_string()
                .ok_or_else(|| JsValue::from_str("Invalid path key"))?;
            let content_value = files.get(&key);
            let content = js_sys::Uint8Array::new(&content_value).to_vec();

            let store_path = path.strip_prefix('/').unwrap_or(&path);

            let skey = zarrs::storage::StoreKey::new(store_path)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            store
                .set(&skey, content.into())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        let inner = ZarrStoreReader::open_store(store as ReadableWritableListableStorage)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(ZarrReader { inner })
    }

    /// Reads a frame at the given index.
    /// @param {number} step - Frame index
    /// @returns {Frame | undefined}
    #[wasm_bindgen]
    pub fn read(&self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_frame(step as u64)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Some(Frame::from_rs_frame(rs_frame)))
    }

    /// Returns the number of frames in the trajectory.
    /// @returns {number}
    #[wasm_bindgen]
    pub fn len(&self) -> usize {
        self.inner.len() as usize
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
