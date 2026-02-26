//! WASM bindings for Frame.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use molrs::core::block::Block as RsBlock;
use molrs_ffi::{FrameId as FFIFrameId, Store as FFIStore};

use super::block::Block;
use super::{SharedStore, js_err};

/// A frame that contains blocks of data.
#[wasm_bindgen]
pub struct Frame {
    id: FFIFrameId,
    store: SharedStore,
}

#[wasm_bindgen]
impl Frame {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let store = Rc::new(RefCell::new(FFIStore::new()));
        let id = store.borrow_mut().frame_new();
        Frame { id, store }
    }

    /// @param key - block name
    #[wasm_bindgen(js_name = createBlock)]
    pub fn create_block(&self, key: &str) -> Result<Block, JsValue> {
        let rs_block = RsBlock::new();
        self.store
            .borrow_mut()
            .set_block(self.id, key, rs_block)
            .map_err(js_err)?;
        let handle = self
            .store
            .borrow()
            .get_block(self.id, key)
            .map_err(js_err)?;
        Ok(Block {
            handle,
            store: self.store.clone(),
        })
    }

    /// @param key - block name
    #[wasm_bindgen(js_name = getBlock)]
    pub fn get_block(&self, key: &str) -> Option<Block> {
        let handle = self.store.borrow().get_block(self.id, key).ok()?;
        Some(Block {
            handle,
            store: self.store.clone(),
        })
    }

    /// Inserts a block (copies data into this frame's store).
    #[wasm_bindgen(js_name = insertBlock)]
    pub fn insert_block(&self, key: &str, block: Block) -> Result<(), JsValue> {
        let rs_block = block
            .store
            .borrow()
            .clone_block(&block.handle)
            .map_err(js_err)?;
        self.store
            .borrow_mut()
            .set_block(self.id, key, rs_block)
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = removeBlock)]
    pub fn remove_block(&self, key: &str) -> Result<(), JsValue> {
        self.store
            .borrow_mut()
            .remove_block(self.id, key)
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = clear)]
    pub fn clear(&self) -> Result<(), JsValue> {
        self.store.borrow_mut().clear_frame(self.id).map_err(js_err)
    }

    #[wasm_bindgen(js_name = renameBlock)]
    pub fn rename_block(&self, old_key: &str, new_key: &str) -> Result<bool, JsValue> {
        self.store
            .borrow_mut()
            .with_frame_mut(self.id, |f| f.rename_block(old_key, new_key))
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = renameColumn)]
    pub fn rename_column(
        &self,
        block_key: &str,
        old_col: &str,
        new_col: &str,
    ) -> Result<bool, JsValue> {
        self.store
            .borrow_mut()
            .with_frame_mut(self.id, |f| f.rename_column(block_key, old_col, new_col))
            .map_err(js_err)
    }

    #[wasm_bindgen(getter, js_name = simbox)]
    pub fn get_simbox(&self) -> Option<super::region::simbox::Box> {
        self.store
            .borrow()
            .with_frame_simbox(self.id, |sb| {
                sb.map(|s| super::region::simbox::Box { inner: s.clone() })
            })
            .ok()?
    }

    #[wasm_bindgen(setter, js_name = simbox)]
    pub fn set_simbox(&self, simbox: Option<super::region::simbox::Box>) {
        let _ = self.store.borrow_mut().with_frame_mut(self.id, |f| {
            f.simbox = simbox.map(|b| b.inner);
        });
    }

    #[wasm_bindgen(js_name = drop)]
    pub fn drop_frame(&self) -> Result<(), JsValue> {
        self.store.borrow_mut().frame_drop(self.id).map_err(js_err)
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal helpers (not exposed to JS).
impl Frame {
    pub(crate) fn from_rs_frame(rs_frame: molrs::core::frame::Frame) -> Self {
        let store = Rc::new(RefCell::new(FFIStore::new()));
        let id = store.borrow_mut().frame_new();
        store
            .borrow_mut()
            .set_frame(id, rs_frame)
            .expect("set_frame");
        Frame { id, store }
    }

    pub(crate) fn clone_core_frame(&self) -> Result<molrs::core::frame::Frame, JsValue> {
        self.store.borrow().clone_frame(self.id).map_err(js_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_frame_lifecycle() {
        let frame = Frame::new();
        assert!(frame.clear().is_ok());
        frame.drop_frame().unwrap();
        assert!(frame.clear().is_err());
    }
}
