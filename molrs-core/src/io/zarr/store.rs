use std::collections::HashMap;
#[cfg(feature = "filesystem")]
use std::path::Path;
#[cfg(feature = "filesystem")]
use std::sync::Arc;

#[cfg(feature = "filesystem")]
use zarrs::array::ArrayBuilder;
use zarrs::array::{Array, ArraySubset, data_type};
#[cfg(feature = "filesystem")]
use zarrs::filesystem::FilesystemStore;
#[cfg(feature = "filesystem")]
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::ReadableWritableListableStorage;

use crate::core::frame::Frame;
use crate::error::MolRsError;

/// Default chunk size along the time axis.
///
/// This is intentionally large so trajectory arrays default to a single
/// time chunk for typical runs, minimizing the number of chunk files.
#[cfg(feature = "filesystem")]
const DEFAULT_TIME_CHUNK_SIZE: u64 = u32::MAX as u64;

#[cfg(feature = "filesystem")]
fn normalize_chunk_size(chunk_size: u64) -> u64 {
    if chunk_size == 0 {
        DEFAULT_TIME_CHUNK_SIZE
    } else {
        chunk_size
    }
}

// ---------------------------------------------------------------------------
// StoreData — per-frame payload for append()
// ---------------------------------------------------------------------------

/// Data for a single trajectory frame to be appended.
pub struct StoreData<'a> {
    /// Flat [3N] positions (f32), or None to skip.
    pub positions: Option<&'a [f32]>,
    /// Flat [3N] velocities (f32), or None to skip.
    pub velocities: Option<&'a [f32]>,
    /// Flat [3N] forces (f32), or None to skip.
    pub forces: Option<&'a [f32]>,
    /// Named scalar values (e.g. "pe", "ke").
    pub scalars: HashMap<String, f64>,
    /// [9] row-major 3x3 cell matrix, or None to skip.
    pub box_h: Option<&'a [f32]>,
}

// ---------------------------------------------------------------------------
// ZarrStoreWriter — builder + streaming writer
// ---------------------------------------------------------------------------

type ZarrArray = Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>;

/// Builder for configuring which fields to track in a trajectory.
#[cfg(feature = "filesystem")]
pub struct StoreBuilder {
    path: std::path::PathBuf,
    positions: bool,
    velocities: bool,
    forces: bool,
    scalars: Vec<String>,
    box_h: bool,
    chunk_size: u64,
}

#[cfg(feature = "filesystem")]
impl StoreBuilder {
    pub fn with_positions(mut self) -> Self {
        self.positions = true;
        self
    }

    pub fn with_velocities(mut self) -> Self {
        self.velocities = true;
        self
    }

    pub fn with_forces(mut self) -> Self {
        self.forces = true;
        self
    }

    pub fn with_scalars(mut self, names: &[&str]) -> Self {
        self.scalars = names.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn with_box_h(mut self) -> Self {
        self.box_h = true;
        self
    }

    pub fn chunk_size(mut self, c: u64) -> Self {
        self.chunk_size = normalize_chunk_size(c);
        self
    }

    /// Create the trajectory store and write the reference frame.
    pub fn create(self, reference_frame: &Frame) -> Result<ZarrStoreWriter, MolRsError> {
        let n_atoms = reference_frame
            .get("atoms")
            .and_then(|b| b.nrows())
            .ok_or_else(|| MolRsError::zarr("reference frame has no atoms block"))?
            as u64;

        let store: ReadableWritableListableStorage = Arc::new(
            FilesystemStore::new(&self.path).map_err(|e| MolRsError::zarr(e.to_string()))?,
        );

        // Root group
        let mut root_attrs = serde_json::Map::new();
        root_attrs.insert("molrs_format".into(), "traj".into());
        root_attrs.insert("version".into(), 3.into());
        GroupBuilder::new()
            .attributes(root_attrs)
            .build(store.clone(), "/")?
            .store_metadata()?;
        // Trajectory group
        GroupBuilder::new()
            .build(store.clone(), "/trajectory")?
            .store_metadata()?;

        // Write reference frame — convert x/y/z to f32
        let ref_f32 = coords_to_f32(reference_frame)?;
        write_reference_frame(&store, &ref_f32)?;

        let c = normalize_chunk_size(self.chunk_size);

        // step: [0] Int64, chunk [C]
        let step_arr = ArrayBuilder::new(vec![0u64], vec![c], data_type::int64(), 0i64)
            .build(store.clone(), "/trajectory/step")?;
        step_arr.store_metadata()?;

        // time: [0] Float64, chunk [C]
        let time_arr = ArrayBuilder::new(vec![0u64], vec![c], data_type::float64(), 0.0f64)
            .build(store.clone(), "/trajectory/time")?;
        time_arr.store_metadata()?;

        // Positions: /trajectory/x, /trajectory/y, /trajectory/z — each [0, N] Float32
        let pos_arrs = if self.positions {
            Some(make_xyz_arrays(&store, ["x", "y", "z"], n_atoms, c)?)
        } else {
            None
        };

        // Velocities: /trajectory/vx, /trajectory/vy, /trajectory/vz
        let vel_arrs = if self.velocities {
            Some(make_xyz_arrays(&store, ["vx", "vy", "vz"], n_atoms, c)?)
        } else {
            None
        };

        // Forces: /trajectory/fx, /trajectory/fy, /trajectory/fz
        let frc_arrs = if self.forces {
            Some(make_xyz_arrays(&store, ["fx", "fy", "fz"], n_atoms, c)?)
        } else {
            None
        };

        // Scalar arrays: [0] Float64, chunk [C]
        let mut scalar_arrs = HashMap::new();
        for name in &self.scalars {
            let a = ArrayBuilder::new(vec![0u64], vec![c], data_type::float64(), 0.0f64)
                .build(store.clone(), &format!("/trajectory/{}", name))?;
            a.store_metadata()?;
            scalar_arrs.insert(name.clone(), a);
        }

        // box_h: [0, 3, 3] Float32, chunk [C, 3, 3]
        let box_h_arr = if self.box_h {
            let a = ArrayBuilder::new(vec![0, 3, 3], vec![c, 3, 3], data_type::float32(), 0.0f32)
                .build(store.clone(), "/trajectory/box_h")?;
            a.store_metadata()?;
            Some(a)
        } else {
            None
        };

        Ok(ZarrStoreWriter {
            _store: store,
            n_atoms,
            n_frames: 0,
            step_arr,
            time_arr,
            pos_arrs,
            vel_arrs,
            frc_arrs,
            scalar_arrs,
            box_h_arr,
        })
    }
}

/// Streaming trajectory writer.
pub struct ZarrStoreWriter {
    _store: ReadableWritableListableStorage,
    n_atoms: u64,
    n_frames: u64,
    step_arr: ZarrArray,
    time_arr: ZarrArray,
    pos_arrs: Option<[ZarrArray; 3]>, // x, y, z
    vel_arrs: Option<[ZarrArray; 3]>, // vx, vy, vz
    frc_arrs: Option<[ZarrArray; 3]>, // fx, fy, fz
    scalar_arrs: HashMap<String, ZarrArray>,
    box_h_arr: Option<ZarrArray>,
}

impl ZarrStoreWriter {
    /// Create a new trajectory builder targeting the given path.
    #[cfg(feature = "filesystem")]
    pub fn builder(path: impl AsRef<Path>) -> StoreBuilder {
        StoreBuilder {
            path: path.as_ref().to_path_buf(),
            positions: false,
            velocities: false,
            forces: false,
            scalars: Vec::new(),
            box_h: false,
            chunk_size: DEFAULT_TIME_CHUNK_SIZE,
        }
    }

    /// Append one frame of data.
    pub fn append(&mut self, step: i64, time: f64, data: &StoreData) -> Result<(), MolRsError> {
        let t = self.n_frames;
        let n = self.n_atoms;
        let new_t = t + 1;

        // Extend step
        self.step_arr.set_shape(vec![new_t])?;
        self.step_arr.store_metadata()?;
        let step_range = t..new_t;
        self.step_arr.store_array_subset(
            &ArraySubset::new_with_ranges(std::slice::from_ref(&step_range)),
            &[step],
        )?;

        // Extend time
        self.time_arr.set_shape(vec![new_t])?;
        self.time_arr.store_metadata()?;
        let time_range = t..new_t;
        self.time_arr.store_array_subset(
            &ArraySubset::new_with_ranges(std::slice::from_ref(&time_range)),
            &[time],
        )?;

        // Positions → x, y, z
        if let (Some(arrs), Some(flat)) = (&mut self.pos_arrs, data.positions) {
            append_xyz(arrs, flat, n, t, new_t)?;
        }

        // Velocities → vx, vy, vz
        if let (Some(arrs), Some(flat)) = (&mut self.vel_arrs, data.velocities) {
            append_xyz(arrs, flat, n, t, new_t)?;
        }

        // Forces → fx, fy, fz
        if let (Some(arrs), Some(flat)) = (&mut self.frc_arrs, data.forces) {
            append_xyz(arrs, flat, n, t, new_t)?;
        }

        // Scalars
        for (name, arr) in &mut self.scalar_arrs {
            if let Some(&val) = data.scalars.get(name) {
                arr.set_shape(vec![new_t])?;
                arr.store_metadata()?;
                let scalar_range = t..new_t;
                arr.store_array_subset(
                    &ArraySubset::new_with_ranges(std::slice::from_ref(&scalar_range)),
                    &[val],
                )?;
            }
        }

        // box_h
        if let (Some(arr), Some(bh)) = (&mut self.box_h_arr, data.box_h) {
            arr.set_shape(vec![new_t, 3, 3])?;
            arr.store_metadata()?;
            arr.store_array_subset(&ArraySubset::new_with_ranges(&[t..new_t, 0..3, 0..3]), bh)?;
        }

        self.n_frames = new_t;
        Ok(())
    }

    /// Finalize the trajectory (no-op currently, but ensures metadata is flushed).
    pub fn close(self) -> Result<(), MolRsError> {
        Ok(())
    }

    /// Number of frames written so far.
    pub fn len(&self) -> u64 {
        self.n_frames
    }

    /// Whether no frame has been written yet.
    pub fn is_empty(&self) -> bool {
        self.n_frames == 0
    }
}

// ---------------------------------------------------------------------------
// XYZ array helpers (create / append / read)
// ---------------------------------------------------------------------------

/// Create three [0, n_atoms] Float32 arrays at /trajectory/{names[0..3]}.
#[cfg(feature = "filesystem")]
fn make_xyz_arrays(
    store: &ReadableWritableListableStorage,
    names: [&str; 3],
    n_atoms: u64,
    chunk_size: u64,
) -> Result<[ZarrArray; 3], MolRsError> {
    let mut out = Vec::with_capacity(3);
    for name in names {
        let a = ArrayBuilder::new(
            vec![0, n_atoms],
            vec![chunk_size, n_atoms],
            data_type::float32(),
            0.0f32,
        )
        .build(store.clone(), &format!("/trajectory/{}", name))?;
        a.store_metadata()?;
        out.push(a);
    }
    Ok([out.remove(0), out.remove(0), out.remove(0)])
}

/// Split flat [3N] into x/y/z and append one row to each array.
fn append_xyz(
    arrs: &mut [ZarrArray; 3],
    flat: &[f32],
    n: u64,
    t: u64,
    new_t: u64,
) -> Result<(), MolRsError> {
    let nu = n as usize;
    let mut bufs = [
        Vec::with_capacity(nu),
        Vec::with_capacity(nu),
        Vec::with_capacity(nu),
    ];
    for i in 0..nu {
        bufs[0].push(flat[3 * i]);
        bufs[1].push(flat[3 * i + 1]);
        bufs[2].push(flat[3 * i + 2]);
    }
    for (arr, buf) in arrs.iter_mut().zip(bufs.iter()) {
        arr.set_shape(vec![new_t, n])?;
        arr.store_metadata()?;
        arr.store_array_subset(&ArraySubset::new_with_ranges(&[t..new_t, 0..n]), buf)?;
    }
    Ok(())
}

/// Check whether a zarr array exists at the given path.
fn array_exists(store: &ReadableWritableListableStorage, path: &str) -> bool {
    Array::open(store.clone(), path).is_ok()
}

/// Read a 1D f32 slice at frame `t` from a [frames, n_atoms] array.
fn read_col_at(
    store: &ReadableWritableListableStorage,
    path: &str,
    t: u64,
    n: u64,
) -> Result<Vec<f32>, MolRsError> {
    let arr = Array::open(store.clone(), path)?;
    let subset = ArraySubset::new_with_ranges(&[t..t + 1, 0..n]);

    if arr.data_type() == &data_type::float64() {
        let data: Vec<f64> = arr.retrieve_array_subset(&subset)?;
        Ok(data.into_iter().map(|v| v as f32).collect())
    } else {
        let data: Vec<f32> = arr.retrieve_array_subset(&subset)?;
        Ok(data)
    }
}

// ---------------------------------------------------------------------------
// ZarrStoreReader — random-access reader
// ---------------------------------------------------------------------------

/// Random-access trajectory reader.
pub struct ZarrStoreReader {
    store: ReadableWritableListableStorage,
    n_atoms: u64,
    n_frames: u64,
}

impl ZarrStoreReader {
    /// Open an existing trajectory store with a specific backend.
    pub fn open_store(store: ReadableWritableListableStorage) -> Result<Self, MolRsError> {
        // Infer n_atoms from reference frame atoms block
        let ref_frame = read_reference_frame(&store)?;
        let n_atoms = ref_frame
            .get("atoms")
            .and_then(|b| b.nrows())
            .ok_or_else(|| MolRsError::zarr("reference frame has no atoms block"))?
            as u64;

        // Determine n_frames from the trajectory/step array shape
        let step_arr = Array::open(store.clone(), "/trajectory/step")?;
        let n_frames = step_arr.shape()[0];

        Ok(Self {
            store,
            n_atoms,
            n_frames,
        })
    }

    /// Open an existing trajectory store from a filesystem path.
    #[cfg(feature = "filesystem")]
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage = Arc::new(
            FilesystemStore::new(path.as_ref()).map_err(|e| MolRsError::zarr(e.to_string()))?,
        );
        Self::open_store(store)
    }

    /// Number of frames in the trajectory.
    pub fn len(&self) -> u64 {
        self.n_frames
    }

    /// Whether the trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.n_frames == 0
    }

    /// Read the reference (topology) frame.
    pub fn reference_frame(&self) -> Result<Frame, MolRsError> {
        read_reference_frame(&self.store)
    }

    /// Read a complete Frame at index `t` (reference frame with positions/velocities overlaid).
    pub fn read_frame(&self, t: u64) -> Result<Frame, MolRsError> {
        self.check_bounds(t)?;
        let mut frame = self.reference_frame()?;

        let atoms = frame
            .get_mut("atoms")
            .ok_or_else(|| MolRsError::zarr("reference frame missing atoms block"))?;

        // Overlay positions (required)
        for key in ["x", "y", "z"] {
            let col = read_col_at(
                &self.store,
                &format!("/trajectory/{}", key),
                t,
                self.n_atoms,
            )?;
            atoms
                .insert(key, ndarray::Array1::from_vec(col).into_dyn())
                .map_err(|e| MolRsError::zarr(format!("insert {} at t={}: {}", key, t, e)))?;
        }

        // Overlay velocities (only if the array exists in the store)
        if array_exists(&self.store, "/trajectory/vx") {
            for key in ["vx", "vy", "vz"] {
                let col = read_col_at(
                    &self.store,
                    &format!("/trajectory/{}", key),
                    t,
                    self.n_atoms,
                )?;
                atoms
                    .insert(key, ndarray::Array1::from_vec(col).into_dyn())
                    .map_err(|e| MolRsError::zarr(format!("insert {} at t={}: {}", key, t, e)))?;
            }
        }

        // Overlay per-frame box_h (if the array exists in the store)
        if array_exists(&self.store, "/trajectory/box_h") {
            let arr = Array::open(self.store.clone(), "/trajectory/box_h")?;
            let subset = ArraySubset::new_with_ranges(&[t..t + 1, 0..3, 0..3]);
            let h_data: Vec<f32> = arr.retrieve_array_subset(&subset)?;
            let h = ndarray::Array2::from_shape_vec((3, 3), h_data)
                .map_err(|e| MolRsError::zarr(format!("box_h reshape: {}", e)))?
                .mapv(|v| v as crate::core::types::F);

            let (origin, pbc) = match &frame.simbox {
                Some(sb) => {
                    let o = sb.origin_view().to_owned();
                    let p = [sb.pbc_view()[0], sb.pbc_view()[1], sb.pbc_view()[2]];
                    (o, p)
                }
                None => (ndarray::arr1(&[0.0, 0.0, 0.0]), [true, true, true]),
            };
            frame.simbox = Some(
                crate::core::region::simbox::SimBox::new(h, origin, pbc)
                    .map_err(|e| MolRsError::zarr(format!("box_h simbox: {:?}", e)))?,
            );
        }

        Ok(frame)
    }

    /// Read positions at frame `t` as a flat 3N vector (f32), interleaved from /trajectory/x, /trajectory/y, /trajectory/z.
    pub fn positions(&self, t: u64) -> Result<Vec<f32>, MolRsError> {
        self.check_bounds(t)?;
        let n = self.n_atoms;
        let xs = read_col_at(&self.store, "/trajectory/x", t, n)?;
        let ys = read_col_at(&self.store, "/trajectory/y", t, n)?;
        let zs = read_col_at(&self.store, "/trajectory/z", t, n)?;
        let mut out = Vec::with_capacity(3 * n as usize);
        for i in 0..n as usize {
            out.push(xs[i]);
            out.push(ys[i]);
            out.push(zs[i]);
        }
        Ok(out)
    }

    /// Read velocities at frame `t` as a flat 3N vector (f32), interleaved from /trajectory/vx, /trajectory/vy, /trajectory/vz.
    pub fn velocities(&self, t: u64) -> Result<Vec<f32>, MolRsError> {
        self.check_bounds(t)?;
        let n = self.n_atoms;
        let vxs = read_col_at(&self.store, "/trajectory/vx", t, n)?;
        let vys = read_col_at(&self.store, "/trajectory/vy", t, n)?;
        let vzs = read_col_at(&self.store, "/trajectory/vz", t, n)?;
        let mut out = Vec::with_capacity(3 * n as usize);
        for i in 0..n as usize {
            out.push(vxs[i]);
            out.push(vys[i]);
            out.push(vzs[i]);
        }
        Ok(out)
    }

    /// Read a scalar value at frame `t`.
    pub fn scalar(&self, name: &str, t: u64) -> Result<f64, MolRsError> {
        self.check_bounds(t)?;
        let arr = Array::open(self.store.clone(), &format!("/trajectory/{}", name))?;
        let scalar_range = t..t + 1;
        let subset = ArraySubset::new_with_ranges(std::slice::from_ref(&scalar_range));
        let data: Vec<f64> = arr.retrieve_array_subset(&subset)?;
        data.into_iter()
            .next()
            .ok_or_else(|| MolRsError::zarr("empty scalar read"))
    }

    /// Read all step values.
    pub fn steps(&self) -> Result<Vec<i64>, MolRsError> {
        let arr = Array::open(self.store.clone(), "/trajectory/step")?;
        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        let data: Vec<i64> = arr.retrieve_array_subset(&subset)?;
        Ok(data)
    }

    /// Read all time values.
    pub fn times(&self) -> Result<Vec<f64>, MolRsError> {
        let arr = Array::open(self.store.clone(), "/trajectory/time")?;
        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        let data: Vec<f64> = arr.retrieve_array_subset(&subset)?;
        Ok(data)
    }

    fn check_bounds(&self, t: u64) -> Result<(), MolRsError> {
        if t >= self.n_frames {
            return Err(MolRsError::zarr(format!(
                "frame index {} out of range [0, {})",
                t, self.n_frames
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Reference frame helpers (write under /frame, read back)
// ---------------------------------------------------------------------------

/// Convert f64 x/y/z columns in atoms block to f32.
#[cfg(feature = "filesystem")]
fn coords_to_f32(frame: &Frame) -> Result<Frame, MolRsError> {
    let mut out = frame.clone();
    if let Some(atoms) = out.get_mut("atoms") {
        for key in ["x", "y", "z"] {
            if let Some(arr) = atoms.get_f64(key) {
                let f32_data: Vec<f32> = arr.iter().map(|&v| v as f32).collect();
                atoms
                    .insert(key, ndarray::Array1::from_vec(f32_data).into_dyn())
                    .map_err(|e| {
                        MolRsError::zarr(format!("coords_to_f32 insert {}: {}", key, e))
                    })?;
            }
        }
    }
    Ok(out)
}

#[cfg(feature = "filesystem")]
fn write_reference_frame(
    store: &ReadableWritableListableStorage,
    frame: &Frame,
) -> Result<(), MolRsError> {
    GroupBuilder::new()
        .build(store.clone(), "/frame")?
        .store_metadata()?;

    // Meta
    if !frame.meta.is_empty() {
        let mut meta_attrs = serde_json::Map::new();
        for (k, v) in &frame.meta {
            meta_attrs.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        GroupBuilder::new()
            .attributes(meta_attrs)
            .build(store.clone(), "/frame/meta")?
            .store_metadata()?;
    }

    // SimBox
    if let Some(ref simbox) = frame.simbox {
        super::frame::write_simbox(store, "/frame/simbox", simbox)?;
    }

    // Blocks
    for (block_name, block) in frame.iter() {
        let group_path = format!("/frame/{}", block_name);
        GroupBuilder::new()
            .build(store.clone(), &group_path)?
            .store_metadata()?;

        for (col_name, col) in block.iter() {
            let arr_path = format!("/frame/{}/{}", block_name, col_name);
            super::frame::write_column(store, &arr_path, col)?;
        }
    }

    Ok(())
}

fn read_reference_frame(store: &ReadableWritableListableStorage) -> Result<Frame, MolRsError> {
    let mut frame = Frame::new();

    // Read meta
    if let Ok(meta_group) = zarrs::group::Group::open(store.clone(), "/frame/meta") {
        for (k, v) in meta_group.attributes() {
            if let Some(s) = v.as_str() {
                frame.meta.insert(k.clone(), s.to_string());
            } else {
                frame.meta.insert(k.clone(), v.to_string());
            }
        }
    }

    // Read simbox
    if zarrs::group::Group::open(store.clone(), "/frame/simbox").is_ok() {
        frame.simbox = Some(super::frame::read_simbox(store, "/frame/simbox")?);
    }

    // Read blocks
    let frame_node = Node::open(store, "/frame")?;
    for child in frame_node.children() {
        let child_name = child.path().as_str().rsplit('/').next().unwrap_or("");
        if child_name == "meta" || child_name == "simbox" || child_name.is_empty() {
            continue;
        }
        if !matches!(child.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let mut block = crate::core::block::Block::new();
        let block_node = Node::open(store, child.path().as_str())?;
        for col_child in block_node.children() {
            if !matches!(col_child.metadata(), NodeMetadata::Array(_)) {
                continue;
            }
            let col_name = col_child.path().as_str().rsplit('/').next().unwrap_or("");
            let col = super::frame::read_column(store, col_child.path().as_str())?;
            super::frame::insert_column_into_block(&mut block, col_name, col)?;
        }
        frame.insert(child_name, block);
    }

    Ok(frame)
}

#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    fn read_chunk_shape(path: &std::path::Path, array_name: &str) -> Vec<u64> {
        let meta_path = path.join("trajectory").join(array_name).join("zarr.json");
        let raw = std::fs::read_to_string(meta_path).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&raw).unwrap();
        meta["chunk_grid"]["configuration"]["chunk_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect()
    }

    fn make_ref_frame() -> Frame {
        let mut frame = Frame::new();

        let mut atoms = crate::core::block::Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![1.0f64, 2.0]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![3.0f64, 4.0]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(vec![5.0f64, 6.0]).into_dyn())
            .unwrap();
        atoms
            .insert("mass", Array1::from_vec(vec![12.0f64, 1.0]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);

        frame.simbox = Some(
            crate::core::region::simbox::SimBox::cube(
                10.0,
                array![0.0f32, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
        );

        frame
    }

    #[test]
    fn test_traj_write_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_traj.zarr");
        let ref_frame = make_ref_frame();

        let mut writer = ZarrStoreWriter::builder(&path)
            .with_positions()
            .with_velocities()
            .with_scalars(&["pe", "ke"])
            .chunk_size(10)
            .create(&ref_frame)
            .unwrap();

        // Reference frame x/y/z should be f32 on disk
        let reader_pre = ZarrStoreReader::open(&path).unwrap();
        let rf = reader_pre.reference_frame().unwrap();
        assert!(
            rf.get("atoms").unwrap().get_f32("x").is_some(),
            "ref frame x should be f32"
        );

        // Write 10 frames
        for i in 0..10u64 {
            let t = i as f64 * 0.001;
            let pos: Vec<f32> = vec![
                1.0 + i as f32 * 0.1,
                3.0,
                5.0,
                2.0,
                4.0,
                6.0 + i as f32 * 0.1,
            ];
            let vel: Vec<f32> = vec![0.1, 0.0, 0.0, -0.1, 0.0, 0.0];
            let mut scalars = HashMap::new();
            scalars.insert("pe".to_string(), -10.0 + i as f64);
            scalars.insert("ke".to_string(), 5.0 - i as f64 * 0.1);

            writer
                .append(
                    i as i64,
                    t,
                    &StoreData {
                        positions: Some(&pos),
                        velocities: Some(&vel),
                        forces: None,
                        scalars,
                        box_h: None,
                    },
                )
                .unwrap();
        }

        assert_eq!(writer.len(), 10);
        writer.close().unwrap();

        // Read back
        let reader = ZarrStoreReader::open(&path).unwrap();
        assert_eq!(reader.len(), 10);

        // Check steps
        let steps = reader.steps().unwrap();
        assert_eq!(steps.len(), 10);
        assert_eq!(steps[0], 0);
        assert_eq!(steps[9], 9);

        // Check positions at frame 5 (reconstructed flat 3N)
        let pos5 = reader.positions(5).unwrap();
        assert_eq!(pos5.len(), 6); // 2 atoms * 3
        assert!((pos5[0] - 1.5f32).abs() < 1e-5);

        // Check scalar
        let pe5 = reader.scalar("pe", 5).unwrap();
        assert!((pe5 - (-5.0)).abs() < 1e-10);

        // Check read_frame (x/y/z as f32 columns)
        let frame5 = reader.read_frame(5).unwrap();
        let x = frame5.get("atoms").unwrap().get_f32("x").unwrap();
        assert!((x[0] - 1.5f32).abs() < 1e-5);
        let y = frame5.get("atoms").unwrap().get_f32("y").unwrap();
        assert!((y[0] - 3.0f32).abs() < 1e-5);
    }

    /// Simulate the WASM path: write to filesystem, load into MemoryStore, read back.
    #[test]
    #[cfg(feature = "filesystem")]
    fn test_traj_memorystore_roundtrip() {
        use zarrs::storage::WritableStorageTraits;
        use zarrs::storage::store::MemoryStore;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_traj.zarr");
        let ref_frame = make_ref_frame();

        // Write trajectory to filesystem
        let mut writer = ZarrStoreWriter::builder(&path)
            .with_positions()
            .chunk_size(10)
            .create(&ref_frame)
            .unwrap();

        for i in 0..5u64 {
            let pos: Vec<f32> = vec![
                1.0 + i as f32 * 0.5,
                3.0,
                5.0,
                2.0,
                4.0 + i as f32 * 0.3,
                6.0,
            ];
            writer
                .append(
                    i as i64,
                    i as f64 * 0.01,
                    &StoreData {
                        positions: Some(&pos),
                        velocities: None,
                        forces: None,
                        scalars: HashMap::new(),
                        box_h: None,
                    },
                )
                .unwrap();
        }
        writer.close().unwrap();

        // Load all files into MemoryStore (same as WASM ZarrReader does)
        let mem_store = Arc::new(MemoryStore::new());
        fn visit(dir: &std::path::Path, root: &std::path::Path, store: &MemoryStore) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let entry = entry.unwrap();
                let p = entry.path();
                if p.is_dir() {
                    visit(&p, root, store);
                } else {
                    let rel = p.strip_prefix(root).unwrap().to_string_lossy().to_string();
                    let content = std::fs::read(&p).unwrap();
                    let key = zarrs::storage::StoreKey::new(&rel).unwrap();
                    store.set(&key, content.into()).unwrap();
                }
            }
        }
        visit(&path, &path, &mem_store);

        // Read via MemoryStore
        let reader =
            ZarrStoreReader::open_store(mem_store as ReadableWritableListableStorage).unwrap();
        assert_eq!(reader.len(), 5);

        // Frame 0: x[0] should be 1.0
        let f0 = reader.read_frame(0).unwrap();
        let x0 = f0.get("atoms").unwrap().get_f32("x").unwrap();
        println!("MemStore frame 0 x: {:?}", x0.as_slice().unwrap());
        assert!(
            (x0[0] - 1.0f32).abs() < 1e-5,
            "frame 0 x[0]={}, expected 1.0",
            x0[0]
        );

        // Frame 3: x[0] should be 1.0 + 3*0.5 = 2.5
        let f3 = reader.read_frame(3).unwrap();
        let x3 = f3.get("atoms").unwrap().get_f32("x").unwrap();
        println!("MemStore frame 3 x: {:?}", x3.as_slice().unwrap());
        assert!(
            (x3[0] - 2.5f32).abs() < 1e-5,
            "frame 3 x[0]={}, expected 2.5",
            x3[0]
        );

        // Verify positions() direct read
        let pos3 = reader.positions(3).unwrap();
        println!("MemStore frame 3 positions flat: {:?}", &pos3[..6]);
        assert!(
            (pos3[0] - 2.5f32).abs() < 1e-5,
            "positions(3)[0]={}, expected 2.5",
            pos3[0]
        );
    }

    #[test]
    #[cfg(feature = "filesystem")]
    fn test_chunk_size_applies_to_all_trajectory_arrays() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_chunk_shape.zarr");
        let ref_frame = make_ref_frame();
        let chunk_size = 7u64;

        let mut writer = ZarrStoreWriter::builder(&path)
            .with_positions()
            .with_velocities()
            .with_forces()
            .with_scalars(&["pe"])
            .with_box_h()
            .chunk_size(chunk_size)
            .create(&ref_frame)
            .unwrap();

        for i in 0..8u64 {
            let pos: Vec<f32> = vec![
                1.0 + i as f32,
                3.0,
                5.0, //
                2.0,
                4.0 + i as f32,
                6.0,
            ];
            let vel: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
            let frc: Vec<f32> = vec![-0.1, -0.2, -0.3, -0.4, -0.5, -0.6];
            let mut scalars = HashMap::new();
            scalars.insert("pe".to_string(), -10.0 + i as f64);
            let box_h = [
                10.0f32, 0.0, 0.0, //
                0.0, 10.0, 0.0, //
                0.0, 0.0, 10.0,
            ];

            writer
                .append(
                    i as i64,
                    i as f64 * 0.001,
                    &StoreData {
                        positions: Some(&pos),
                        velocities: Some(&vel),
                        forces: Some(&frc),
                        scalars,
                        box_h: Some(&box_h),
                    },
                )
                .unwrap();
        }
        writer.close().unwrap();

        assert_eq!(read_chunk_shape(&path, "step"), vec![chunk_size]);
        assert_eq!(read_chunk_shape(&path, "time"), vec![chunk_size]);
        assert_eq!(read_chunk_shape(&path, "x"), vec![chunk_size, 2]);
        assert_eq!(read_chunk_shape(&path, "vx"), vec![chunk_size, 2]);
        assert_eq!(read_chunk_shape(&path, "fx"), vec![chunk_size, 2]);
        assert_eq!(read_chunk_shape(&path, "pe"), vec![chunk_size]);
        assert_eq!(read_chunk_shape(&path, "box_h"), vec![chunk_size, 3, 3]);
    }
}
