use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::JsValue;

use molrs_ffi::{FfiError, Store as FFIStore};

pub mod block;
pub mod frame;
pub mod region;
pub mod types;

pub use block::{Block, ColumnView};
pub use frame::Frame;
pub use region::simbox::Box;
pub use types::WasmArray;

pub(crate) type SharedStore = Rc<RefCell<FFIStore>>;

pub(crate) fn js_err(err: FfiError) -> JsValue {
    JsValue::from_str(&err.to_string())
}
