//! File readers for WASM using FFI-based API.

use crate::core::frame::Frame;
use molrs::io::lammps_data::LAMMPSDataReader;
use molrs::io::pdb::PDBReader;
use molrs::io::reader::{FrameReader, TrajReader};
use molrs::io::xyz::XYZReader;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

/// XYZ/ExtXYZ file reader for WASM
///
/// # Example (JavaScript)
/// ```js
/// const reader = new XyzReader(fileContent);
/// const frame = reader.read(0);
/// ```
#[wasm_bindgen(js_name = XyzReader)]
pub struct XyzReader {
    inner: XYZReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = XyzReader)]
impl XyzReader {
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> XyzReader {
        let bytes = content.as_bytes().to_vec();
        XyzReader {
            inner: XYZReader::new(Cursor::new(bytes)),
        }
    }

    /// Reads a frame at the given step.
    /// @param {number} step - Frame index
    /// @returns {Frame | undefined}
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_step(step)
            .map_err(|e| JsValue::from_str(&format!("XYZ read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs_frame(frame_data))),
            None => Ok(None),
        }
    }

    /// Returns the number of frames in the file.
    /// @returns {number}
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("XYZ len error: {}", e)))
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// PDB file reader for WASM
///
/// # Example (JavaScript)
/// ```js
/// const reader = new PdbReader(fileContent);
/// const frame = reader.read(0);
/// ```
#[wasm_bindgen(js_name = PdbReader)]
pub struct PdbReader {
    content: Vec<u8>,
}

#[wasm_bindgen(js_class = PdbReader)]
impl PdbReader {
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> PdbReader {
        PdbReader {
            content: content.as_bytes().to_vec(),
        }
    }

    /// Reads a frame at the given step.
    /// @param {number} step - Frame index (PDB only supports single frame, so only step=0 is valid)
    /// @returns {Frame | undefined}
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }

        let mut reader = PDBReader::new(Cursor::new(self.content.as_slice()));
        let rs_frame = reader
            .read_frame()
            .map_err(|e| JsValue::from_str(&format!("PDB read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs_frame(frame_data))),
            None => Ok(None),
        }
    }

    /// Returns the number of frames (always 1 for PDB).
    /// @returns {number}
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        let has_frame = self.read(0)?.is_some();
        Ok(if has_frame { 1 } else { 0 })
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// LAMMPS data file reader for WASM
///
/// # Example (JavaScript)
/// ```js
/// const reader = new LammpsReader(fileContent);
/// const frame = reader.read(0);
/// ```
#[wasm_bindgen(js_name = LammpsReader)]
pub struct LammpsReader {
    inner: LAMMPSDataReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = LammpsReader)]
impl LammpsReader {
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> LammpsReader {
        let bytes = content.as_bytes().to_vec();
        LammpsReader {
            inner: LAMMPSDataReader::new(Cursor::new(bytes)),
        }
    }

    /// Reads a frame at the given step.
    /// @param {number} step - Frame index (LAMMPS only supports single frame, so only step=0 is valid)
    /// @returns {Frame | undefined}
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }

        let rs_frame = self
            .inner
            .read_frame()
            .map_err(|e| JsValue::from_str(&format!("LAMMPS read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs_frame(frame_data))),
            None => Ok(None),
        }
    }

    /// Returns the number of frames (always 1 for LAMMPS).
    /// @returns {number}
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        let has_frame = self.read(0)?.is_some();
        Ok(if has_frame { 1 } else { 0 })
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_pdb_reader() {
        let pdb_content = r#"ATOM      1  C   MOL     1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  N   MOL     1       4.000   5.000   6.000  1.00  0.00           N
END"#;

        let mut reader = PdbReader::new(pdb_content);

        let frame = reader.read(0).expect("read failed").expect("no frame");

        let block = frame.get_block("atoms").expect("no atoms block");

        let x = block.column_copy("x").expect("no x column");
        assert_eq!(x.length(), 2);
        assert_eq!(x.get_index(0), 1.0);
        assert_eq!(x.get_index(1), 4.0);
    }
}
