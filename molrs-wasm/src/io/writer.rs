//! File writers for WASM using FFI-based API.

use crate::core::frame::Frame;
use molrs::io::pdb::PDBWriter;
use molrs::io::writer::{FrameWriter, Writer};
use molrs::io::xyz::XYZFrameWriter;
use wasm_bindgen::prelude::*;

/// Writes a frame to a string in the specified format.
///
/// # Example (JavaScript)
/// ```js
/// const output = writeFrame(frame, "xyz");
/// console.log(output); // XYZ format string
/// ```
///
/// @param {Frame} frame - Frame to write
/// @param {string} format - Output format ("xyz" or "pdb")
/// @returns {string} Formatted output
#[wasm_bindgen(js_name = writeFrame)]
pub fn write_frame_export(frame: &Frame, format: &str) -> Result<String, JsValue> {
    // Get the core frame data by cloning all blocks
    let rs_frame = frame_to_core(frame)?;

    let mut buffer: Vec<u8> = Vec::new();
    match format.to_lowercase().as_str() {
        "xyz" => {
            let mut writer = <XYZFrameWriter<_> as Writer>::new(&mut buffer);
            writer
                .write_frame(&rs_frame)
                .map_err(|e| JsValue::from_str(&format!("XYZ writing error: {}", e)))?;
        }
        "pdb" => {
            let mut writer = <PDBWriter<_> as Writer>::new(&mut buffer);
            writer
                .write_frame(&rs_frame)
                .map_err(|e| JsValue::from_str(&format!("PDB writing error: {}", e)))?;
        }
        _ => {
            return Err(JsValue::from_str(&format!(
                "unsupported format: {}",
                format
            )));
        }
    }

    String::from_utf8(buffer)
        .map_err(|e| JsValue::from_str(&format!("UTF-8 conversion error: {}", e)))
}

/// Helper to convert FFI Frame to core Frame
fn frame_to_core(frame: &Frame) -> Result<molrs::core::frame::Frame, JsValue> {
    frame.clone_core_frame()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::frame::Frame;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_write_frame_formats() {
        use js_sys::Array as JsArray;

        let frame = Frame::new();
        let mut atoms = frame.create_block("atoms").expect("atoms block");

        let x = js_sys::Float32Array::from(&[0.0_f32, 1.0][..]);
        let y = js_sys::Float32Array::from(&[0.0_f32, 0.0][..]);
        let z = js_sys::Float32Array::from(&[0.0_f32, 0.5][..]);

        atoms.set_column("x", &x, None).expect("x");
        atoms.set_column("y", &y, None).expect("y");
        atoms.set_column("z", &z, None).expect("z");

        let elements = JsArray::new();
        elements.push(&JsValue::from_str("H"));
        elements.push(&JsValue::from_str("O"));
        atoms
            .set_column_strings("element", elements)
            .expect("element");

        let xyz_output = write_frame_export(&frame, "xyz").expect("xyz output");
        assert!(xyz_output.lines().next().unwrap_or("").starts_with('2'));

        let pdb_output = write_frame_export(&frame, "pdb").expect("pdb output");
        assert!(pdb_output.contains("ATOM"));
    }
}
