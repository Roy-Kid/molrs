//! LAMMPS dump trajectory file reader and writer.
//!
//! Implements support for LAMMPS dump files as output by the `dump` command:
//! <https://docs.lammps.org/dump.html>
//!
//! # Supported Features
//!
//! - Multi-frame trajectory reading with random access via `TrajReader`
//! - Orthogonal and triclinic simulation boxes
//! - Automatic column type detection (integer vs float)
//! - Boundary condition flag parsing (`pp`, `ff`, `ss`, etc.)
//! - Gzip-compressed files via `open_lammps_dump`
//!
//! # Examples
//!
//! ```no_run
//! use molrs::io::lammps_dump::{read_lammps_dump, open_lammps_dump, write_lammps_dump};
//!
//! # fn main() -> std::io::Result<()> {
//! // Read all frames
//! let frames = read_lammps_dump("trajectory.lammpstrj")?;
//!
//! // Random access via TrajReader
//! use molrs::io::reader::TrajReader;
//! let mut reader = open_lammps_dump("trajectory.lammpstrj")?;
//! let frame_5 = reader.read_step(5)?;
//!
//! // Write frames
//! write_lammps_dump("output.lammpstrj", &frames)?;
//! # Ok(())
//! # }
//! ```

use crate::block::Block;
use crate::frame::Frame;
use crate::io::reader::{FrameIndex, FrameReader, ReadSeek, Reader, TrajReader};
use crate::io::writer::{FrameWriter, Writer};
use crate::types::{F, I};
use ndarray::{Array1, IxDyn};
use once_cell::sync::OnceCell;
use std::fs::File;
use std::io::{BufRead, Seek, SeekFrom, Write};
use std::path::Path;

// ============================================================================
// Helpers
// ============================================================================

fn err_mapper<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

/// Column type classification for LAMMPS dump columns.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ColumnType {
    Integer,
    Float,
}

/// Classify a LAMMPS dump column by name.
///
/// Integer columns: id, type, mol, proc, procp1, ix, iy, iz, element.
/// Everything else (coordinates, velocities, forces, charges, custom computes)
/// defaults to float.
fn classify_column(name: &str) -> ColumnType {
    match name {
        "id" | "type" | "mol" | "proc" | "procp1" | "ix" | "iy" | "iz" | "element" => {
            ColumnType::Integer
        }
        _ => ColumnType::Float,
    }
}

// ============================================================================
// Parsing
// ============================================================================

/// Parsed box bounds from a single LAMMPS dump frame header.
#[derive(Debug, Clone)]
struct DumpBoxBounds {
    xlo: f64,
    xhi: f64,
    ylo: f64,
    yhi: f64,
    zlo: f64,
    zhi: f64,
    xy: Option<f64>,
    xz: Option<f64>,
    yz: Option<f64>,
    boundary_raw: [String; 3],
}

impl DumpBoxBounds {
    /// Parse the BOX BOUNDS header line to detect triclinic and boundary flags.
    ///
    /// Format: `ITEM: BOX BOUNDS [xy xz yz] bb bb bb`
    /// where bb is pp, ff, ss, fs, sf, etc.
    fn parse_header(header: &str) -> std::io::Result<(bool, [String; 3])> {
        // Strip "ITEM: BOX BOUNDS" prefix
        let rest = header.strip_prefix("ITEM: BOX BOUNDS").unwrap_or("").trim();

        let tokens: Vec<&str> = rest.split_whitespace().collect();

        // Detect triclinic: header contains "xy xz yz" before boundary flags
        let (is_triclinic, boundary_tokens) =
            if tokens.len() >= 6 && tokens[0] == "xy" && tokens[1] == "xz" && tokens[2] == "yz" {
                (true, &tokens[3..])
            } else {
                (false, tokens.as_slice())
            };

        let boundary_raw = if boundary_tokens.len() >= 3 {
            [
                boundary_tokens[0].to_string(),
                boundary_tokens[1].to_string(),
                boundary_tokens[2].to_string(),
            ]
        } else {
            ["pp".to_string(), "pp".to_string(), "pp".to_string()]
        };

        Ok((is_triclinic, boundary_raw))
    }

    /// Parse 3 box bound lines (orthogonal or triclinic).
    fn parse_lines<R: BufRead>(
        reader: &mut R,
        is_triclinic: bool,
        boundary_raw: [String; 3],
    ) -> std::io::Result<Self> {
        let mut line = String::new();

        // Line 1: xlo xhi [xy]
        line.clear();
        reader.read_line(&mut line)?;
        let vals: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().map_err(err_mapper))
            .collect::<Result<_, _>>()?;

        let (xlo_bound, xhi_bound) = (vals[0], vals[1]);
        let xy = if is_triclinic { Some(vals[2]) } else { None };

        // Line 2: ylo yhi [xz]
        line.clear();
        reader.read_line(&mut line)?;
        let vals: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().map_err(err_mapper))
            .collect::<Result<_, _>>()?;

        let (ylo_bound, yhi_bound) = (vals[0], vals[1]);
        let xz = if is_triclinic { Some(vals[2]) } else { None };

        // Line 3: zlo zhi [yz]
        line.clear();
        reader.read_line(&mut line)?;
        let vals: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().map_err(err_mapper))
            .collect::<Result<_, _>>()?;

        let (zlo_bound, zhi_bound) = (vals[0], vals[1]);
        let yz = if is_triclinic { Some(vals[2]) } else { None };

        // For triclinic, convert bounds to actual box limits
        // See: https://docs.lammps.org/Howto_triclinic.html
        if is_triclinic {
            let xy_v = xy.unwrap_or(0.0);
            let xz_v = xz.unwrap_or(0.0);
            let yz_v = yz.unwrap_or(0.0);

            let xlo = xlo_bound - f64::min(0.0, f64::min(xy_v, f64::min(xz_v, xy_v + xz_v)));
            let xhi = xhi_bound - f64::max(0.0, f64::max(xy_v, f64::max(xz_v, xy_v + xz_v)));
            let ylo = ylo_bound - f64::min(0.0, yz_v);
            let yhi = yhi_bound - f64::max(0.0, yz_v);
            let zlo = zlo_bound;
            let zhi = zhi_bound;

            Ok(Self {
                xlo,
                xhi,
                ylo,
                yhi,
                zlo,
                zhi,
                xy,
                xz,
                yz,
                boundary_raw,
            })
        } else {
            Ok(Self {
                xlo: xlo_bound,
                xhi: xhi_bound,
                ylo: ylo_bound,
                yhi: yhi_bound,
                zlo: zlo_bound,
                zhi: zhi_bound,
                xy: None,
                xz: None,
                yz: None,
                boundary_raw,
            })
        }
    }
}

/// Parse a single LAMMPS dump frame from the current reader position.
///
/// Returns `Ok(None)` on EOF.
fn parse_single_frame<R: BufRead>(reader: &mut R) -> std::io::Result<Option<Frame>> {
    let mut line = String::new();

    // -- ITEM: TIMESTEP --
    line.clear();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None); // EOF
    }
    if !line.trim().starts_with("ITEM: TIMESTEP") {
        return Err(err_mapper(format!(
            "Expected 'ITEM: TIMESTEP', got: {}",
            line.trim()
        )));
    }

    line.clear();
    reader.read_line(&mut line)?;
    let timestep: i64 = line.trim().parse().map_err(err_mapper)?;

    // -- ITEM: NUMBER OF ATOMS --
    line.clear();
    reader.read_line(&mut line)?;
    if !line.trim().starts_with("ITEM: NUMBER OF ATOMS") {
        return Err(err_mapper(format!(
            "Expected 'ITEM: NUMBER OF ATOMS', got: {}",
            line.trim()
        )));
    }

    line.clear();
    reader.read_line(&mut line)?;
    let natoms: usize = line.trim().parse().map_err(err_mapper)?;

    // -- ITEM: BOX BOUNDS --
    line.clear();
    reader.read_line(&mut line)?;
    if !line.trim().starts_with("ITEM: BOX BOUNDS") {
        return Err(err_mapper(format!(
            "Expected 'ITEM: BOX BOUNDS', got: {}",
            line.trim()
        )));
    }

    let (is_triclinic, boundary_raw) = DumpBoxBounds::parse_header(line.trim())?;
    let bounds = DumpBoxBounds::parse_lines(reader, is_triclinic, boundary_raw)?;

    // -- ITEM: ATOMS --
    line.clear();
    reader.read_line(&mut line)?;
    if !line.trim().starts_with("ITEM: ATOMS") {
        return Err(err_mapper(format!(
            "Expected 'ITEM: ATOMS', got: {}",
            line.trim()
        )));
    }

    // Extract column names from "ITEM: ATOMS col1 col2 ..."
    let atoms_header = line.trim().strip_prefix("ITEM: ATOMS").unwrap_or("").trim();
    let col_names: Vec<String> = atoms_header.split_whitespace().map(String::from).collect();
    let col_types: Vec<ColumnType> = col_names.iter().map(|n| classify_column(n)).collect();

    if col_names.is_empty() {
        return Err(err_mapper("ITEM: ATOMS header has no column names"));
    }

    // Pre-allocate column storage
    let ncols = col_names.len();
    let mut int_cols: Vec<Option<Vec<I>>> = vec![None; ncols];
    let mut float_cols: Vec<Option<Vec<F>>> = vec![None; ncols];

    for (i, ct) in col_types.iter().enumerate() {
        match ct {
            ColumnType::Integer => int_cols[i] = Some(Vec::with_capacity(natoms)),
            ColumnType::Float => float_cols[i] = Some(Vec::with_capacity(natoms)),
        }
    }

    // Read atom data lines
    for row in 0..natoms {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Err(err_mapper(format!(
                "Unexpected EOF at atom line {} (expected {})",
                row, natoms
            )));
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < ncols {
            return Err(err_mapper(format!(
                "Atom line {} has {} tokens, expected {}",
                row,
                tokens.len(),
                ncols
            )));
        }

        for (i, ct) in col_types.iter().enumerate() {
            match ct {
                ColumnType::Integer => {
                    let v: I = tokens[i].parse().map_err(err_mapper)?;
                    int_cols[i].as_mut().unwrap().push(v);
                }
                ColumnType::Float => {
                    let v: F = tokens[i].parse().map_err(err_mapper)?;
                    float_cols[i].as_mut().unwrap().push(v);
                }
            }
        }
    }

    // Build Frame
    let mut frame = Frame::new();
    let mut atoms_block = Block::new();

    for (i, name) in col_names.iter().enumerate() {
        match col_types[i] {
            ColumnType::Integer => {
                let arr = Array1::from_vec(int_cols[i].take().unwrap())
                    .into_shape_with_order(IxDyn(&[natoms]))
                    .map_err(err_mapper)?
                    .into_dyn();
                atoms_block.insert(name.as_str(), arr).map_err(err_mapper)?;
            }
            ColumnType::Float => {
                let arr = Array1::from_vec(float_cols[i].take().unwrap())
                    .into_shape_with_order(IxDyn(&[natoms]))
                    .map_err(err_mapper)?
                    .into_dyn();
                atoms_block.insert(name.as_str(), arr).map_err(err_mapper)?;
            }
        }
    }

    frame.insert("atoms", atoms_block);

    // Store metadata (matching lammps_data.rs convention)
    frame
        .meta
        .insert("timestep".to_string(), timestep.to_string());

    let lx = bounds.xhi - bounds.xlo;
    let ly = bounds.yhi - bounds.ylo;
    let lz = bounds.zhi - bounds.zlo;
    frame
        .meta
        .insert("box".to_string(), format!("{} {} {}", lx, ly, lz));
    frame.meta.insert(
        "box_origin".to_string(),
        format!("{} {} {}", bounds.xlo, bounds.ylo, bounds.zlo),
    );

    if let (Some(xy), Some(xz), Some(yz)) = (bounds.xy, bounds.xz, bounds.yz) {
        frame
            .meta
            .insert("box_tilt".to_string(), format!("{} {} {}", xy, xz, yz));
    }

    frame
        .meta
        .insert("boundary".to_string(), bounds.boundary_raw.join(" "));

    Ok(Some(frame))
}

// ============================================================================
// Reader
// ============================================================================

/// LAMMPS dump trajectory reader implementing `TrajReader` for random access.
///
/// Supports multi-frame dump files with lazy index building for random access.
///
/// # Examples
///
/// ```no_run
/// use molrs::io::lammps_dump::open_lammps_dump;
/// use molrs::io::reader::TrajReader;
///
/// # fn main() -> std::io::Result<()> {
/// let mut reader = open_lammps_dump("traj.lammpstrj")?;
/// let n = reader.len()?;
/// println!("Trajectory has {} frames", n);
/// let frame = reader.read_step(0)?.expect("first frame");
/// # Ok(())
/// # }
/// ```
pub struct LAMMPSDumpReader<R: BufRead> {
    reader: R,
    index: OnceCell<FrameIndex>,
}

impl<R: BufRead + Seek> LAMMPSDumpReader<R> {
    /// Create a new LAMMPS dump reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            index: OnceCell::new(),
        }
    }

    /// Build frame index by scanning for `ITEM: TIMESTEP` markers.
    fn build_index_impl(&mut self) -> std::io::Result<()> {
        if self.index.get().is_some() {
            return Ok(());
        }

        let start_pos = self.reader.stream_position()?;
        self.reader.seek(SeekFrom::Start(0))?;

        let mut frame_index = FrameIndex::new();
        let mut current_pos: u64 = 0;
        let mut line = String::new();

        loop {
            // Record potential frame start
            let frame_start = current_pos;

            // Read next line
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break; // EOF
            }
            current_pos += bytes as u64;

            if !line.trim().starts_with("ITEM: TIMESTEP") {
                continue;
            }

            // Found a frame start
            frame_index.add_frame(frame_start);

            // Skip timestep value
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            current_pos += bytes as u64;

            // Skip "ITEM: NUMBER OF ATOMS"
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            current_pos += bytes as u64;

            // Read natoms
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            current_pos += bytes as u64;
            let natoms: usize = line.trim().parse().unwrap_or(0);

            // Skip "ITEM: BOX BOUNDS ..."
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            current_pos += bytes as u64;

            // Skip 3 box lines
            for _ in 0..3 {
                line.clear();
                let bytes = self.reader.read_line(&mut line)?;
                if bytes == 0 {
                    break;
                }
                current_pos += bytes as u64;
            }

            // Skip "ITEM: ATOMS ..."
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            current_pos += bytes as u64;

            // Skip natoms data lines
            for _ in 0..natoms {
                line.clear();
                let bytes = self.reader.read_line(&mut line)?;
                if bytes == 0 {
                    break;
                }
                current_pos += bytes as u64;
            }
        }

        self.reader.seek(SeekFrom::Start(start_pos))?;
        self.index
            .set(frame_index)
            .map_err(|_| std::io::Error::other("failed to set index"))?;

        Ok(())
    }

    /// Read frame at a specific byte offset.
    fn read_at_offset(&mut self, offset: u64) -> std::io::Result<Option<Frame>> {
        self.reader.seek(SeekFrom::Start(offset))?;
        parse_single_frame(&mut self.reader)
    }
}

impl<R: BufRead + Seek> Reader for LAMMPSDumpReader<R> {
    type R = R;
    type Frame = Frame;

    fn new(reader: Self::R) -> Self {
        Self::new(reader)
    }
}

impl<R: BufRead + Seek> FrameReader for LAMMPSDumpReader<R> {
    fn read_frame(&mut self) -> std::io::Result<Option<Self::Frame>> {
        parse_single_frame(&mut self.reader)
    }
}

impl<R: BufRead + Seek> TrajReader for LAMMPSDumpReader<R> {
    fn build_index(&mut self) -> std::io::Result<()> {
        self.build_index_impl()
    }

    fn read_step(&mut self, step: usize) -> std::io::Result<Option<Self::Frame>> {
        if self.index.get().is_none() {
            self.build_index_impl()?;
        }

        let index = self.index.get().unwrap();
        if step >= index.len() {
            return Ok(None);
        }

        let offset = index.get(step).unwrap();
        self.read_at_offset(offset)
    }

    fn len(&mut self) -> std::io::Result<usize> {
        if self.index.get().is_none() {
            self.build_index_impl()?;
        }
        Ok(self.index.get().unwrap().len())
    }
}

// ============================================================================
// Writer
// ============================================================================

/// LAMMPS dump trajectory writer.
///
/// Writes frames in LAMMPS dump format. Call `write_frame` for each timestep.
///
/// # Examples
///
/// ```no_run
/// use molrs::io::lammps_dump::write_lammps_dump;
/// use molrs::frame::Frame;
///
/// # fn main() -> std::io::Result<()> {
/// let frames: Vec<Frame> = vec![];
/// write_lammps_dump("output.lammpstrj", &frames)?;
/// # Ok(())
/// # }
/// ```
pub struct LAMMPSDumpWriter<W: Write> {
    writer: W,
}

impl<W: Write> LAMMPSDumpWriter<W> {
    /// Create a new LAMMPS dump writer.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> Writer for LAMMPSDumpWriter<W> {
    type W = W;
    type FrameLike = Frame;

    fn new(writer: Self::W) -> Self {
        Self::new(writer)
    }
}

impl<W: Write> FrameWriter for LAMMPSDumpWriter<W> {
    fn write_frame(&mut self, frame: &Frame) -> std::io::Result<()> {
        let atoms = frame
            .get("atoms")
            .ok_or_else(|| err_mapper("Frame must contain 'atoms' block"))?;
        let natoms = atoms.nrows().unwrap_or(0);

        // -- Timestep --
        let timestep = frame.meta.get("timestep").map_or("0", |s| s.as_str());
        writeln!(self.writer, "ITEM: TIMESTEP")?;
        writeln!(self.writer, "{}", timestep)?;

        // -- Number of atoms --
        writeln!(self.writer, "ITEM: NUMBER OF ATOMS")?;
        writeln!(self.writer, "{}", natoms)?;

        // -- Box bounds --
        let box_str = frame
            .meta
            .get("box")
            .ok_or_else(|| err_mapper("Frame metadata must contain 'box'"))?;
        let default_origin = "0.0 0.0 0.0".to_string();
        let origin_str = frame.meta.get("box_origin").unwrap_or(&default_origin);

        let dims: Vec<f64> = box_str
            .split_whitespace()
            .map(|s| s.parse().map_err(err_mapper))
            .collect::<Result<_, _>>()?;
        let origin: Vec<f64> = origin_str
            .split_whitespace()
            .map(|s| s.parse().map_err(err_mapper))
            .collect::<Result<_, _>>()?;

        if dims.len() != 3 || origin.len() != 3 {
            return Err(err_mapper("Invalid box dimensions in metadata"));
        }

        // Boundary flags
        let boundary = frame.meta.get("boundary");
        let pbc_str = boundary.map_or("pp pp pp", |s| s.as_str());

        let has_tilt = frame.meta.contains_key("box_tilt");

        if has_tilt {
            let tilt_str = frame.meta.get("box_tilt").unwrap();
            let tilts: Vec<f64> = tilt_str
                .split_whitespace()
                .map(|s| s.parse().map_err(err_mapper))
                .collect::<Result<_, _>>()?;

            if tilts.len() != 3 {
                return Err(err_mapper("Invalid box_tilt in metadata"));
            }

            let (xy, xz, yz) = (tilts[0], tilts[1], tilts[2]);

            // Convert actual limits to bound values for triclinic output
            let xlo = origin[0];
            let ylo = origin[1];
            let zlo = origin[2];
            let xhi = xlo + dims[0];
            let yhi = ylo + dims[1];
            let zhi = zlo + dims[2];

            let xlo_bound = xlo + f64::min(0.0, f64::min(xy, f64::min(xz, xy + xz)));
            let xhi_bound = xhi + f64::max(0.0, f64::max(xy, f64::max(xz, xy + xz)));
            let ylo_bound = ylo + f64::min(0.0, yz);
            let yhi_bound = yhi + f64::max(0.0, yz);

            writeln!(self.writer, "ITEM: BOX BOUNDS xy xz yz {}", pbc_str)?;
            writeln!(self.writer, "{} {} {}", xlo_bound, xhi_bound, xy)?;
            writeln!(self.writer, "{} {} {}", ylo_bound, yhi_bound, xz)?;
            writeln!(self.writer, "{} {} {}", zlo, zhi, yz)?;
        } else {
            let xlo = origin[0];
            let ylo = origin[1];
            let zlo = origin[2];
            let xhi = xlo + dims[0];
            let yhi = ylo + dims[1];
            let zhi = zlo + dims[2];

            writeln!(self.writer, "ITEM: BOX BOUNDS {}", pbc_str)?;
            writeln!(self.writer, "{} {}", xlo, xhi)?;
            writeln!(self.writer, "{} {}", ylo, yhi)?;
            writeln!(self.writer, "{} {}", zlo, zhi)?;
        }

        // -- Atoms --
        // Determine column ordering: id first, type second, then sorted remaining
        let col_names: Vec<&str> = atoms.keys().collect();
        let mut ordered: Vec<&str> = Vec::with_capacity(col_names.len());

        if col_names.contains(&"id") {
            ordered.push("id");
        }
        if col_names.contains(&"type") {
            ordered.push("type");
        }

        let mut remaining: Vec<&str> = col_names
            .iter()
            .filter(|&&n| n != "id" && n != "type")
            .copied()
            .collect();
        remaining.sort();
        ordered.extend(remaining);

        writeln!(self.writer, "ITEM: ATOMS {}", ordered.join(" "))?;

        let col_types: Vec<ColumnType> = ordered.iter().map(|n| classify_column(n)).collect();

        for row in 0..natoms {
            for (ci, &name) in ordered.iter().enumerate() {
                if ci > 0 {
                    write!(self.writer, " ")?;
                }
                match col_types[ci] {
                    ColumnType::Integer => {
                        if let Some(arr) = atoms.get_int(name) {
                            write!(self.writer, "{}", arr[row])?;
                        } else if let Some(arr) = atoms.get_float(name) {
                            // Fallback: column classified as int but stored as float
                            write!(self.writer, "{}", arr[row] as I)?;
                        } else {
                            write!(self.writer, "0")?;
                        }
                    }
                    ColumnType::Float => {
                        if let Some(arr) = atoms.get_float(name) {
                            write!(self.writer, "{:.6}", arr[row])?;
                        } else if let Some(arr) = atoms.get_int(name) {
                            // Fallback: column classified as float but stored as int
                            write!(self.writer, "{:.6}", arr[row] as F)?;
                        } else {
                            write!(self.writer, "0.000000")?;
                        }
                    }
                }
            }
            writeln!(self.writer)?;
        }

        Ok(())
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Read all frames from a LAMMPS dump file.
///
/// For large trajectories, prefer `open_lammps_dump` with `TrajReader::read_step`
/// for random access without loading all frames into memory.
pub fn read_lammps_dump<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Frame>> {
    let reader = crate::io::reader::open_seekable(path)?;
    let mut dump_reader = LAMMPSDumpReader::new(reader);
    dump_reader.read_all()
}

/// Open a LAMMPS dump file for trajectory-style random access.
///
/// Returns a reader implementing `TrajReader`. The index is built lazily
/// on first call to `read_step` or `len`.
pub fn open_lammps_dump<P: AsRef<Path>>(
    path: P,
) -> std::io::Result<LAMMPSDumpReader<Box<dyn ReadSeek>>> {
    let reader = crate::io::reader::open_seekable(path)?;
    Ok(LAMMPSDumpReader::new(reader))
}

/// Write frames to a LAMMPS dump file.
pub fn write_lammps_dump<P: AsRef<Path>>(path: P, frames: &[Frame]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = LAMMPSDumpWriter::new(file);
    for frame in frames {
        writer.write_frame(frame)?;
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Minimal orthogonal dump with 3 atoms.
    const ORTHO_DUMP: &str = "\
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
3 2 7.0 8.0 9.0
";

    /// Triclinic dump with tilt factors.
    const TRICLINIC_DUMP: &str = "\
ITEM: TIMESTEP
200
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp ff
-0.5 10.5 0.5
0.0 10.0 0.0
0.0 10.0 0.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
2 2 4.0 5.0 6.0
";

    /// Multi-frame dump (2 frames).
    const MULTI_DUMP: &str = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.5 2.5 3.5
2 1 4.5 5.5 6.5
";

    fn cursor(s: &str) -> Cursor<Vec<u8>> {
        Cursor::new(s.as_bytes().to_vec())
    }

    #[test]
    fn test_classify_column() {
        assert_eq!(classify_column("id"), ColumnType::Integer);
        assert_eq!(classify_column("type"), ColumnType::Integer);
        assert_eq!(classify_column("mol"), ColumnType::Integer);
        assert_eq!(classify_column("ix"), ColumnType::Integer);
        assert_eq!(classify_column("x"), ColumnType::Float);
        assert_eq!(classify_column("vx"), ColumnType::Float);
        assert_eq!(classify_column("q"), ColumnType::Float);
        assert_eq!(classify_column("c_pe"), ColumnType::Float);
        assert_eq!(classify_column("f_reax[1]"), ColumnType::Float);
    }

    #[test]
    fn test_read_single_frame_orthogonal() {
        let mut reader = LAMMPSDumpReader::new(cursor(ORTHO_DUMP));
        let frame = reader.read_frame().unwrap().expect("should parse frame");

        // Check timestep
        assert_eq!(frame.meta.get("timestep").unwrap(), "100");

        // Check atoms
        let atoms = frame.get("atoms").expect("atoms block");
        assert_eq!(atoms.nrows(), Some(3));

        let ids = atoms.get_int("id").expect("id column");
        assert_eq!(ids.as_slice().unwrap(), &[1, 2, 3]);

        let types = atoms.get_int("type").expect("type column");
        assert_eq!(types.as_slice().unwrap(), &[1, 1, 2]);

        let x = atoms.get_float("x").expect("x column");
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 4.0).abs() < 1e-6);
        assert!((x[2] - 7.0).abs() < 1e-6);

        let z = atoms.get_float("z").expect("z column");
        assert!((z[0] - 3.0).abs() < 1e-6);
        assert!((z[2] - 9.0).abs() < 1e-6);

        // Check box
        assert_eq!(frame.meta.get("box").unwrap(), "10 10 10");
        assert_eq!(frame.meta.get("box_origin").unwrap(), "0 0 0");
        assert!(frame.meta.get("box_tilt").is_none());
        assert_eq!(frame.meta.get("boundary").unwrap(), "pp pp pp");
    }

    #[test]
    fn test_read_single_frame_triclinic() {
        let mut reader = LAMMPSDumpReader::new(cursor(TRICLINIC_DUMP));
        let frame = reader.read_frame().unwrap().expect("should parse frame");

        assert_eq!(frame.meta.get("timestep").unwrap(), "200");

        // Triclinic should have tilt metadata
        let tilt = frame.meta.get("box_tilt").expect("should have box_tilt");
        let tilt_vals: Vec<f64> = tilt
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert!((tilt_vals[0] - 0.5).abs() < 1e-6); // xy
        assert!((tilt_vals[1] - 0.0).abs() < 1e-6); // xz
        assert!((tilt_vals[2] - 0.0).abs() < 1e-6); // yz

        // Boundary: pp pp ff
        assert_eq!(frame.meta.get("boundary").unwrap(), "pp pp ff");
    }

    #[test]
    fn test_read_multiframe() {
        let mut reader = LAMMPSDumpReader::new(cursor(MULTI_DUMP));
        let frames = reader.read_all().unwrap();

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].meta.get("timestep").unwrap(), "0");
        assert_eq!(frames[1].meta.get("timestep").unwrap(), "100");

        let x0 = frames[0].get("atoms").unwrap().get_float("x").unwrap();
        assert!((x0[0] - 1.0).abs() < 1e-6);

        let x1 = frames[1].get("atoms").unwrap().get_float("x").unwrap();
        assert!((x1[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_build_index() {
        let mut reader = LAMMPSDumpReader::new(cursor(MULTI_DUMP));
        reader.build_index().unwrap();

        assert_eq!(reader.len().unwrap(), 2);
    }

    #[test]
    fn test_read_step_random_access() {
        let mut reader = LAMMPSDumpReader::new(cursor(MULTI_DUMP));

        // Read step 1 first (out of order)
        let f1 = reader.read_step(1).unwrap().expect("step 1");
        assert_eq!(f1.meta.get("timestep").unwrap(), "100");

        // Then step 0
        let f0 = reader.read_step(0).unwrap().expect("step 0");
        assert_eq!(f0.meta.get("timestep").unwrap(), "0");

        // Out of bounds
        assert!(reader.read_step(5).unwrap().is_none());
    }

    #[test]
    fn test_write_single_frame_orthogonal() {
        // Build a frame
        let mut frame = Frame::new();
        let mut atoms = Block::new();

        let ids = Array1::from_vec(vec![1 as I, 2 as I])
            .into_shape_with_order(IxDyn(&[2]))
            .unwrap()
            .into_dyn();
        let types = Array1::from_vec(vec![1 as I, 2 as I])
            .into_shape_with_order(IxDyn(&[2]))
            .unwrap()
            .into_dyn();
        let x = Array1::from_vec(vec![1.0 as F, 4.0])
            .into_shape_with_order(IxDyn(&[2]))
            .unwrap()
            .into_dyn();
        let y = Array1::from_vec(vec![2.0 as F, 5.0])
            .into_shape_with_order(IxDyn(&[2]))
            .unwrap()
            .into_dyn();
        let z = Array1::from_vec(vec![3.0 as F, 6.0])
            .into_shape_with_order(IxDyn(&[2]))
            .unwrap()
            .into_dyn();

        atoms.insert("id", ids).unwrap();
        atoms.insert("type", types).unwrap();
        atoms.insert("x", x).unwrap();
        atoms.insert("y", y).unwrap();
        atoms.insert("z", z).unwrap();
        frame.insert("atoms", atoms);

        frame.meta.insert("timestep".to_string(), "42".to_string());
        frame.meta.insert("box".to_string(), "10 10 10".to_string());
        frame
            .meta
            .insert("box_origin".to_string(), "0 0 0".to_string());

        let mut buf = Vec::new();
        let mut writer = LAMMPSDumpWriter::new(&mut buf);
        writer.write_frame(&frame).unwrap();

        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("ITEM: TIMESTEP\n42\n"));
        assert!(output.contains("ITEM: NUMBER OF ATOMS\n2\n"));
        assert!(output.contains("ITEM: BOX BOUNDS pp pp pp\n"));
        assert!(output.contains("ITEM: ATOMS id type"));
    }

    #[test]
    fn test_write_single_frame_triclinic() {
        let mut frame = Frame::new();
        let mut atoms = Block::new();

        let ids = Array1::from_vec(vec![1 as I])
            .into_shape_with_order(IxDyn(&[1]))
            .unwrap()
            .into_dyn();
        let types = Array1::from_vec(vec![1 as I])
            .into_shape_with_order(IxDyn(&[1]))
            .unwrap()
            .into_dyn();
        let x = Array1::from_vec(vec![1.0 as F])
            .into_shape_with_order(IxDyn(&[1]))
            .unwrap()
            .into_dyn();
        let y = Array1::from_vec(vec![2.0 as F])
            .into_shape_with_order(IxDyn(&[1]))
            .unwrap()
            .into_dyn();
        let z = Array1::from_vec(vec![3.0 as F])
            .into_shape_with_order(IxDyn(&[1]))
            .unwrap()
            .into_dyn();

        atoms.insert("id", ids).unwrap();
        atoms.insert("type", types).unwrap();
        atoms.insert("x", x).unwrap();
        atoms.insert("y", y).unwrap();
        atoms.insert("z", z).unwrap();
        frame.insert("atoms", atoms);

        frame.meta.insert("timestep".to_string(), "0".to_string());
        frame.meta.insert("box".to_string(), "10 10 10".to_string());
        frame
            .meta
            .insert("box_origin".to_string(), "0 0 0".to_string());
        frame
            .meta
            .insert("box_tilt".to_string(), "1.0 0.5 0.0".to_string());

        let mut buf = Vec::new();
        let mut writer = LAMMPSDumpWriter::new(&mut buf);
        writer.write_frame(&frame).unwrap();

        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("ITEM: BOX BOUNDS xy xz yz pp pp pp\n"));
    }

    #[test]
    fn test_roundtrip() {
        // Read original
        let mut reader = LAMMPSDumpReader::new(cursor(ORTHO_DUMP));
        let frame = reader.read_frame().unwrap().expect("read frame");

        // Write
        let mut buf = Vec::new();
        let mut writer = LAMMPSDumpWriter::new(&mut buf);
        writer.write_frame(&frame).unwrap();

        // Read back
        let mut reader2 = LAMMPSDumpReader::new(Cursor::new(buf));
        let frame2 = reader2.read_frame().unwrap().expect("read back");

        // Compare
        assert_eq!(frame.meta.get("timestep"), frame2.meta.get("timestep"));

        let atoms1 = frame.get("atoms").unwrap();
        let atoms2 = frame2.get("atoms").unwrap();
        assert_eq!(atoms1.nrows(), atoms2.nrows());

        let x1 = atoms1.get_float("x").unwrap();
        let x2 = atoms2.get_float("x").unwrap();
        for i in 0..x1.len() {
            assert!((x1[i] - x2[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_variable_atom_count() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
3 2 7.0 8.0 9.0
";
        let mut reader = LAMMPSDumpReader::new(cursor(dump));
        let frames = reader.read_all().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].get("atoms").unwrap().nrows(), Some(2));
        assert_eq!(frames[1].get("atoms").unwrap().nrows(), Some(3));
    }

    #[test]
    fn test_custom_columns() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z vx vy vz q c_pe
1 1 1.0 2.0 3.0 0.1 0.2 0.3 -0.5 -10.5
";
        let mut reader = LAMMPSDumpReader::new(cursor(dump));
        let frame = reader.read_frame().unwrap().expect("parse");
        let atoms = frame.get("atoms").unwrap();

        // Custom columns should be float
        let q = atoms.get_float("q").expect("q column");
        assert!((q[0] - (-0.5)).abs() < 1e-6);

        let pe = atoms.get_float("c_pe").expect("c_pe column");
        assert!((pe[0] - (-10.5)).abs() < 1e-4);

        // Velocities should be float
        let vx = atoms.get_float("vx").expect("vx column");
        assert!((vx[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input() {
        let mut reader = LAMMPSDumpReader::new(cursor(""));
        assert!(reader.read_frame().unwrap().is_none());
    }

    #[test]
    fn test_empty_index() {
        let mut reader = LAMMPSDumpReader::new(cursor(""));
        reader.build_index().unwrap();
        assert_eq!(reader.len().unwrap(), 0);
    }

    #[test]
    fn test_boundary_flags() {
        let dump = "\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS ff pp ss
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
";
        let mut reader = LAMMPSDumpReader::new(cursor(dump));
        let frame = reader.read_frame().unwrap().expect("parse");
        assert_eq!(frame.meta.get("boundary").unwrap(), "ff pp ss");
    }

    #[test]
    fn test_multiframe_roundtrip() {
        let mut reader = LAMMPSDumpReader::new(cursor(MULTI_DUMP));
        let frames = reader.read_all().unwrap();

        let mut buf = Vec::new();
        let mut writer = LAMMPSDumpWriter::new(&mut buf);
        for f in &frames {
            writer.write_frame(f).unwrap();
        }

        let mut reader2 = LAMMPSDumpReader::new(Cursor::new(buf));
        let frames2 = reader2.read_all().unwrap();

        assert_eq!(frames.len(), frames2.len());
        for (f1, f2) in frames.iter().zip(frames2.iter()) {
            assert_eq!(f1.meta.get("timestep"), f2.meta.get("timestep"));
        }
    }

    #[test]
    fn test_iter() {
        let mut reader = LAMMPSDumpReader::new(cursor(MULTI_DUMP));
        reader.build_index().unwrap();
        let mut count = 0;
        for result in reader.iter() {
            result.unwrap();
            count += 1;
        }
        assert_eq!(count, 2);
    }
}
