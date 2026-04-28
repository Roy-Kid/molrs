//! Molecular file readers for the WASM API.
//!
//! Each reader takes a file's text content as a string (since WASM has
//! no filesystem access) and produces [`Frame`] objects. All readers
//! implement a uniform two-method interface:
//!
//! - `read(step)` -- read a specific frame by index
//! - `len()` -- return the number of available frames
//!
//! # Supported formats
//!
//! | JS class | Format | Multi-frame? | Produces |
//! |----------|--------|-------------|----------|
//! | `XYZReader` | XYZ / ExtXYZ | Yes | `"atoms"` block with `element`, `x`, `y`, `z` |
//! | `PDBReader` | Protein Data Bank | No (step=0 only) | `"atoms"` block with `name`, `resname`, `x`, `y`, `z`, etc. |
//! | `CIFReader` | Crystallographic Information File | Yes (per `data_` block) | `"atoms"` block + simbox from unit cell |
//! | `LAMMPSReader` | LAMMPS data file | No (step=0 only) | `"atoms"` block + `"bonds"` block + simbox |
//! | `LAMMPSTrajReader` | LAMMPS dump trajectory | Yes | `"atoms"` block with columns from dump header |

use crate::core::frame::Frame;
use molrs_io::cif::CifReader as RsCifReader;
use molrs_io::dcd::DcdReader as RsDcdReader;
use molrs_io::lammps_data::LAMMPSDataReader;
use molrs_io::lammps_dump::LAMMPSTrajReader;
use molrs_io::pdb::PDBReader;
use molrs_io::reader::{FrameReader, Reader, TrajReader};
use molrs_io::sdf::SDFReader;
use molrs_io::xyz::XYZReader;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

/// XYZ / Extended XYZ file reader.
///
/// Supports multi-frame trajectory files. Each frame produces a
/// [`Frame`] with an `"atoms"` block containing `element` (string)
/// and `x`, `y`, `z` (F, coordinates in angstrom) columns.
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text(); // read file in browser
/// const reader = new XYZReader(content);
/// console.log(reader.len()); // number of frames
///
/// const frame = reader.read(0); // first frame
/// const atoms = frame.getBlock("atoms");
/// const x = atoms.copyColF("x");
/// ```
#[wasm_bindgen(js_name = XYZReader)]
pub struct XyzReader {
    inner: XYZReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = XYZReader)]
impl XyzReader {
    /// Create a new XYZ reader from a string containing the file content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of an XYZ or Extended XYZ file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new XYZReader(fileContent);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> XyzReader {
        let bytes = content.as_bytes().to_vec();
        XyzReader {
            inner: XYZReader::new(Cursor::new(bytes)),
        }
    }

    /// Read a frame at the given step index.
    ///
    /// # Arguments
    ///
    /// * `step` - Zero-based frame index
    ///
    /// # Returns
    ///
    /// A [`Frame`] if the step exists, or `undefined` if `step` is
    /// out of range.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = reader.read(0);
    /// if (frame) {
    ///   const atoms = frame.getBlock("atoms");
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_step(step)
            .map_err(|e| JsValue::from_str(&format!("XYZ read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames in the file.
    ///
    /// # Returns
    ///
    /// The total frame count.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the file cannot be scanned.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(reader.len()); // e.g., 100
    /// ```
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("XYZ len error: {}", e)))
    }

    /// Check whether the file contains no frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the file cannot be scanned.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// Protein Data Bank (PDB) file reader.
///
/// PDB files contain a single molecular structure. The reader produces
/// a [`Frame`] with an `"atoms"` block containing columns such as
/// `name` (string), `resname` (string), `x`, `y`, `z` (F, angstrom),
/// and optionally `occupancy` and `bfactor` (F).
///
/// # Example (JavaScript)
///
/// ```js
/// const reader = new PDBReader(pdbContent);
/// const frame = reader.read(0);
/// const atoms = frame.getBlock("atoms");
/// const names = atoms.copyColStr("name"); // ["CA", "CB", ...]
/// const x = atoms.copyColF("x");
/// ```
#[wasm_bindgen(js_name = PDBReader)]
pub struct PdbReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = PDBReader)]
impl PdbReader {
    /// Create a new PDB reader from a string containing the file content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of a PDB file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new PDBReader(pdbString);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> PdbReader {
        PdbReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at the given step index.
    ///
    /// PDB files contain a single structure, so only `step = 0` is
    /// valid. Passing any other step returns `undefined`.
    ///
    /// # Arguments
    ///
    /// * `step` - Frame index (must be `0` for PDB files)
    ///
    /// # Returns
    ///
    /// A [`Frame`] if `step == 0` and the file is valid, or `undefined`.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = reader.read(0);
    /// ```
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
            Some(frame_data) => Ok(Some(Frame::from_rs(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames (always 0 or 1 for PDB files).
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let has_frame = self.read(0)?.is_some();
        let n = if has_frame { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// LAMMPS data file reader.
///
/// Reads LAMMPS data files (the format written by `write_data`). The
/// reader produces a [`Frame`] containing:
///
/// - `"atoms"` block: `type` (i32), `x`, `y`, `z` (F, angstrom),
///   and optionally `charge` (F)
/// - `"bonds"` block (if present): `i`, `j` (u32), `type` (i32)
/// - Simulation box (`simbox`) with PBC
///
/// Only a single frame is supported (`step = 0`).
///
/// # Example (JavaScript)
///
/// ```js
/// const reader = new LAMMPSReader(dataFileContent);
/// const frame = reader.read(0);
/// const atoms = frame.getBlock("atoms");
/// const bonds = frame.getBlock("bonds");
/// const box   = frame.simbox;
/// ```
#[wasm_bindgen(js_name = LAMMPSReader)]
pub struct LammpsReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = LAMMPSReader)]
impl LammpsReader {
    /// Create a new LAMMPS data file reader from string content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of a LAMMPS data file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new LAMMPSReader(dataFileString);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> LammpsReader {
        LammpsReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at the given step index.
    ///
    /// LAMMPS data files contain a single configuration, so only
    /// `step = 0` is valid. Passing any other step returns `undefined`.
    ///
    /// # Arguments
    ///
    /// * `step` - Frame index (must be `0`)
    ///
    /// # Returns
    ///
    /// A [`Frame`] with atoms, optional bonds, and simbox, or `undefined`.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = reader.read(0);
    /// ```
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }

        // Build a fresh rs reader each call: `FrameReader::read_frame` is a
        // single-shot cursor API (sets `returned=true` after first call), so
        // reusing `inner` across `len()` and `read(0)` returned None on the
        // second call. PDBReader already uses this pattern — mirror it here.
        let mut reader = LAMMPSDataReader::new(Cursor::new(self.content.as_slice()));
        let rs_frame = reader
            .read_frame()
            .map_err(|e| JsValue::from_str(&format!("LAMMPS read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames (always 0 or 1 for LAMMPS data files).
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let has_frame = self.read(0)?.is_some();
        let n = if has_frame { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// LAMMPS dump trajectory file reader.
///
/// Reads multi-frame LAMMPS dump files (the format produced by the
/// `dump` command). Each frame produces a [`Frame`] containing an
/// `"atoms"` block with columns matching the dump header (e.g.
/// `id`, `type`, `x`, `y`, `z`, `vx`, `vy`, `vz`).
///
/// # Example (JavaScript)
///
/// ```js
/// const reader = new LAMMPSTrajReader(dumpContent);
/// console.log(reader.len()); // number of timesteps
/// const frame = reader.read(0);
/// const atoms = frame.getBlock("atoms");
/// ```
#[wasm_bindgen(js_name = LAMMPSTrajReader)]
pub struct LammpsDumpReader {
    inner: LAMMPSTrajReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = LAMMPSTrajReader)]
impl LammpsDumpReader {
    /// Create a new LAMMPS dump reader from string content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> LammpsDumpReader {
        let bytes = content.as_bytes().to_vec();
        LammpsDumpReader {
            inner: LAMMPSTrajReader::new(Cursor::new(bytes)),
        }
    }

    /// Read a frame at the given step index.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_step(step)
            .map_err(|e| JsValue::from_str(&format!("LAMMPS dump read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames in the dump file.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("LAMMPS dump len error: {}", e)))
    }

    /// Check whether the file contains no frames.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// MDL molfile / SDF (V2000 CTAB) reader.
///
/// Parses the connection table found in `.mol` files and the record
/// blocks of `.sdf` files. Coordinates come directly from the file —
/// no 3D generation is performed. Only V2000 is supported; V3000
/// records throw on read.
///
/// Produces a [`Frame`] with:
/// - `"atoms"` block: `element` (string), `id` (u32, 1-based),
///   `x`, `y`, `z` (F, angstrom)
/// - `"bonds"` block (if present): `atomi`, `atomj` (u32, 0-based),
///   `order` (u32)
///
/// Multi-record SDF files expose each record as a separate frame via
/// `read(step)`.
///
/// # Example (JavaScript)
///
/// ```js
/// const reader = new SDFReader(sdfContent);
/// const frame = reader.read(0);
/// const atoms = frame.getBlock("atoms");
/// const x = atoms.copyColF("x");
/// ```
#[wasm_bindgen(js_name = SDFReader)]
pub struct SdfReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = SDFReader)]
impl SdfReader {
    /// Create a new SDF reader from a string containing the file content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> SdfReader {
        SdfReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame (SDF record) at the given step index.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let mut reader = SDFReader::new(Cursor::new(self.content.as_slice()));
        for current in 0..=step {
            let rs_frame = reader
                .read_frame()
                .map_err(|e| JsValue::from_str(&format!("SDF read error: {}", e)))?;
            match rs_frame {
                Some(frame) if current == step => return Ok(Some(Frame::from_rs(frame)?)),
                Some(_) => continue, // drop records before the target
                None => return Ok(None),
            }
        }
        Ok(None)
    }

    /// Return the total number of records in the SDF file.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let mut reader = SDFReader::new(Cursor::new(self.content.as_slice()));
        let mut count = 0usize;
        while reader
            .read_frame()
            .map_err(|e| JsValue::from_str(&format!("SDF len error: {}", e)))?
            .is_some()
        {
            count += 1;
        }
        self.cached_len = Some(count);
        Ok(count)
    }

    /// Check whether the file contains no records.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// DCD trajectory file reader.
///
/// DCD is a binary multi-frame trajectory format originally used by
/// CHARMM and now widely produced by NAMD / OpenMM / GROMACS-via-VMD.
/// This wrapper accepts the file as raw bytes (`Uint8Array`) since
/// DCD is not text-encoded — passing a JS string would corrupt the
/// fixed-width Fortran record markers.
///
/// Each frame produces a [`Frame`] with an `"atoms"` block carrying
/// `x`, `y`, `z` (F, angstrom). Box/cell information, when the DCD
/// header declares it present, is attached as the frame's `simbox`.
///
/// # Example (JavaScript)
///
/// ```js
/// const bytes = new Uint8Array(await blob.arrayBuffer());
/// const reader = new DCDReader(bytes);
/// console.log(reader.len()); // number of frames
///
/// const frame = reader.read(0); // first frame
/// const atoms = frame.getBlock("atoms");
/// const x = atoms.copyColF("x");
/// ```
#[wasm_bindgen(js_name = DCDReader)]
pub struct DcdReader {
    inner: RsDcdReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = DCDReader)]
impl DcdReader {
    /// Create a new DCD reader from the file's raw bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The full binary content of a DCD file. The reader
    ///   takes ownership of an internal copy, so the caller is free
    ///   to discard the buffer immediately after this returns.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const bytes = new Uint8Array(await file.arrayBuffer());
    /// const reader = new DCDReader(bytes);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> DcdReader {
        DcdReader {
            inner: RsDcdReader::new(Cursor::new(bytes.to_vec())),
        }
    }

    /// Read a frame at the given step index.
    ///
    /// # Arguments
    ///
    /// * `step` - Zero-based frame index
    ///
    /// # Returns
    ///
    /// A [`Frame`] if the step exists, or `undefined` if `step` is
    /// out of range.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors (truncated record,
    /// malformed header, byte-order mismatch).
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_step(step)
            .map_err(|e| JsValue::from_str(&format!("DCD read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames in the DCD file.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the header cannot be parsed.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("DCD len error: {}", e)))
    }

    /// Check whether the file contains no frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the header cannot be parsed.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// Crystallographic Information File (CIF / mmCIF) reader.
///
/// Each `data_*` block in the file becomes one [`Frame`]. Most CIF files
/// contain a single block (one structure), but multi-block files (e.g.
/// polymorphs of the same compound) are also supported and are exposed
/// as a multi-frame sequence. The unit cell parameters
/// (`_cell_length_a`, `_b`, `_c`, `_cell_angle_alpha`, `_beta`,
/// `_gamma`) are converted to a 3x3 h-matrix on the Rust side and
/// surface on the JS side as `frame.simbox`.
///
/// Produces a [`Frame`] with an `"atoms"` block containing
/// `element` (string), `x`, `y`, `z` (F, angstrom in Cartesian
/// coordinates) and (when present in the file) `label`, `occupancy`,
/// `bfactor` columns.
///
/// CIF parsing reads the entire file on each `read(step)` call --
/// random access is therefore O(file_size), but typical CIF files are
/// small (< 1 MB) and the molvis lazy trajectory caches frames at the
/// JS level, so this is rarely a bottleneck.
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text();
/// const reader = new CIFReader(content);
/// const frame  = reader.read(0);
/// const atoms  = frame.getBlock("atoms");
/// const box    = frame.simbox;        // populated from the unit cell
/// ```
#[wasm_bindgen(js_name = CIFReader)]
pub struct CifReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = CIFReader)]
impl CifReader {
    /// Create a new CIF reader from a string containing the file content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of a CIF file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new CIFReader(cifString);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> CifReader {
        CifReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at the given block index.
    ///
    /// # Arguments
    ///
    /// * `step` - 0-based index of the `data_*` block to return
    ///
    /// # Returns
    ///
    /// A [`Frame`] for the requested block, or `undefined` when
    /// `step >= len()`.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let mut reader = RsCifReader::new(Cursor::new(self.content.as_slice()));
        let frames = reader
            .read_all()
            .map_err(|e| JsValue::from_str(&format!("CIF read error: {}", e)))?;
        if step >= frames.len() {
            return Ok(None);
        }
        let rs_frame = frames.into_iter().nth(step).expect("bounds checked");
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    /// Return the number of `data_*` blocks in the file.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let mut reader = RsCifReader::new(Cursor::new(self.content.as_slice()));
        let n = reader
            .read_all()
            .map_err(|e| JsValue::from_str(&format!("CIF len error: {}", e)))?
            .len();
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid blocks.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
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
    fn test_lammps_reader_len_then_read() {
        // Regression: `len()` used to call `read(0)` which flipped the
        // inner reader's `returned` flag, causing a subsequent `read(0)`
        // from the lazy FrameProvider in reader.ts to return None
        // ("lammps reader returned no frame at step 0").
        let data = "LAMMPS data\n\n\
                    2 atoms\n1 atom types\n\n\
                    0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n\
                    Atoms\n\n\
                    1 1 1.0 2.0 3.0\n2 1 4.0 5.0 6.0\n";
        let mut reader = LammpsReader::new(data);
        assert_eq!(reader.len().expect("len"), 1);
        let frame = reader.read(0).expect("read").expect("frame");
        let atoms = frame.get_block("atoms").expect("atoms");
        assert_eq!(atoms.copy_col_f("x").expect("x").length(), 2);
    }

    #[wasm_bindgen_test]
    fn test_pdb_reader() {
        let pdb_content = r#"ATOM      1  C   MOL     1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  N   MOL     1       4.000   5.000   6.000  1.00  0.00           N
END"#;

        let mut reader = PdbReader::new(pdb_content);

        let frame = reader.read(0).expect("read failed").expect("no frame");

        let block = frame.get_block("atoms").expect("no atoms block");

        let x = block.copy_col_f("x").expect("no x column");
        assert_eq!(x.length(), 2);
        assert_eq!(x.get_index(0), 1.0);
        assert_eq!(x.get_index(1), 4.0);
    }
}
