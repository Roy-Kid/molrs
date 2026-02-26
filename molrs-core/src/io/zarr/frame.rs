#[cfg(feature = "filesystem")]
use std::path::Path;
#[cfg(feature = "filesystem")]
use std::sync::Arc;

use ndarray::ArrayD;
use zarrs::array::ArraySubset;
use zarrs::array::data_type::{
    BoolDataType, Float32DataType, Float64DataType, Int64DataType, StringDataType, UInt8DataType,
    UInt32DataType,
};
#[cfg(feature = "filesystem")]
use zarrs::array::{ArrayBuilder, data_type};
#[cfg(feature = "filesystem")]
use zarrs::filesystem::FilesystemStore;
#[cfg(feature = "filesystem")]
use zarrs::group::GroupBuilder;
#[cfg(feature = "filesystem")]
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::ReadableWritableListableStorage;

use crate::core::block::{Block, Column};
#[cfg(feature = "filesystem")]
use crate::core::frame::Frame;
use crate::core::region::simbox::SimBox;
use crate::core::types::F;
use crate::error::MolRsError;

/// Zarr-based single-frame I/O.
pub struct ZarrFrame;

impl ZarrFrame {
    /// Write a Frame to a Zarr V3 store on disk.
    #[cfg(feature = "filesystem")]
    pub fn write(path: impl AsRef<Path>, frame: &Frame) -> Result<(), MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);

        // Root group with format metadata
        let mut root_attrs = serde_json::Map::new();
        root_attrs.insert("molrs_format".into(), "frame".into());
        root_attrs.insert("version".into(), 1.into());
        GroupBuilder::new()
            .attributes(root_attrs)
            .build(store.clone(), "/")?
            .store_metadata()?;

        // Meta group — store all meta key-value pairs as group attributes
        if !frame.meta.is_empty() {
            let mut meta_attrs = serde_json::Map::new();
            for (k, v) in &frame.meta {
                meta_attrs.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            GroupBuilder::new()
                .attributes(meta_attrs)
                .build(store.clone(), "/meta")?
                .store_metadata()?;
        }

        // SimBox group
        if let Some(ref simbox) = frame.simbox {
            write_simbox(&store, "/simbox", simbox)?;
        }

        // Blocks
        for (block_name, block) in frame.iter() {
            let group_path = format!("/{}", block_name);
            GroupBuilder::new()
                .build(store.clone(), &group_path)?
                .store_metadata()?;

            for (col_name, col) in block.iter() {
                let arr_path = format!("/{}/{}", block_name, col_name);
                write_column(&store, &arr_path, col)?;
            }
        }

        Ok(())
    }

    /// Read a Frame from a Zarr V3 store on disk.
    #[cfg(feature = "filesystem")]
    pub fn read(path: impl AsRef<Path>) -> Result<Frame, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);

        let mut frame = Frame::new();

        // Read root attributes to verify format
        let root = zarrs::group::Group::open(store.clone(), "/")?;
        let root_attrs = root.attributes();
        if let Some(fmt) = root_attrs.get("molrs_format")
            && fmt.as_str() != Some("frame")
        {
            return Err(MolRsError::zarr(format!(
                "expected molrs_format='frame', got {:?}",
                fmt
            )));
        }

        // Read meta
        if let Ok(meta_group) = zarrs::group::Group::open(store.clone(), "/meta") {
            for (k, v) in meta_group.attributes() {
                if let Some(s) = v.as_str() {
                    frame.meta.insert(k.clone(), s.to_string());
                } else {
                    frame.meta.insert(k.clone(), v.to_string());
                }
            }
        }

        // Read simbox
        if zarrs::group::Group::open(store.clone(), "/simbox").is_ok() {
            frame.simbox = Some(read_simbox(&store, "/simbox")?);
        }

        // Discover block groups by listing children of root
        let node = Node::open(&store, "/")?;
        for child in node.children() {
            let child_name = child.path().as_str().trim_start_matches('/');
            // Skip internal groups
            if child_name == "meta" || child_name == "simbox" || child_name.is_empty() {
                continue;
            }
            // Only process groups (blocks), not arrays
            if !matches!(child.metadata(), NodeMetadata::Group(_)) {
                continue;
            }
            let mut block = Block::new();
            let block_node = Node::open(&store, child.path().as_str())?;
            for col_child in block_node.children() {
                if !matches!(col_child.metadata(), NodeMetadata::Array(_)) {
                    continue;
                }
                let col_name = col_child.path().as_str().rsplit('/').next().unwrap_or("");
                let col = read_column(&store, col_child.path().as_str())?;
                insert_column_into_block(&mut block, col_name, col)?;
            }
            frame.insert(child_name, block);
        }

        Ok(frame)
    }
}

// ---------------------------------------------------------------------------
// SimBox read/write helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
pub(crate) fn write_simbox(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    simbox: &SimBox,
) -> Result<(), MolRsError> {
    GroupBuilder::new()
        .build(store.clone(), prefix)?
        .store_metadata()?;

    // h: [3,3] Float32
    let h_view = simbox.h_view();
    #[allow(clippy::unnecessary_cast)]
    let h_data: Vec<f32> = h_view.iter().map(|&v| v as f32).collect();
    write_f32_array(store, &format!("{}/h", prefix), &[3, 3], &h_data)?;

    // origin: [3] Float32
    let origin_view = simbox.origin_view();
    #[allow(clippy::unnecessary_cast)]
    let origin_data: Vec<f32> = origin_view.iter().map(|&v| v as f32).collect();
    write_f32_array(store, &format!("{}/origin", prefix), &[3], &origin_data)?;

    // pbc: [3] UInt8 (bool → 0/1)
    let pbc_view = simbox.pbc_view();
    let pbc_data: Vec<u8> = pbc_view.iter().map(|&b| b as u8).collect();
    write_u8_array(store, &format!("{}/pbc", prefix), &[3], &pbc_data)?;

    Ok(())
}

pub(crate) fn read_simbox(
    store: &ReadableWritableListableStorage,
    prefix: &str,
) -> Result<SimBox, MolRsError> {
    use ndarray::{Array2, array};

    // h: [3,3]
    let h_arr = zarrs::array::Array::open(store.clone(), &format!("{}/h", prefix))?;
    let h_subset = ArraySubset::new_with_shape(vec![3, 3]);
    let h_data: Vec<f32> = h_arr.retrieve_array_subset(&h_subset)?;
    let h = Array2::from_shape_vec((3, 3), h_data.into_iter().map(|v| v as F).collect())
        .map_err(|e| MolRsError::zarr(format!("bad h shape: {}", e)))?;

    // origin: [3]
    let origin_arr = zarrs::array::Array::open(store.clone(), &format!("{}/origin", prefix))?;
    let origin_subset = ArraySubset::new_with_shape(vec![3]);
    let origin_data: Vec<f32> = origin_arr.retrieve_array_subset(&origin_subset)?;
    let origin = array![
        origin_data[0] as F,
        origin_data[1] as F,
        origin_data[2] as F
    ];

    // pbc: [3]
    let pbc_arr = zarrs::array::Array::open(store.clone(), &format!("{}/pbc", prefix))?;
    let pbc_subset = ArraySubset::new_with_shape(vec![3]);
    let pbc_data: Vec<u8> = pbc_arr.retrieve_array_subset(&pbc_subset)?;
    let pbc = [pbc_data[0] != 0, pbc_data[1] != 0, pbc_data[2] != 0];

    SimBox::new(h, origin, pbc).map_err(|e| MolRsError::zarr(format!("invalid simbox: {:?}", e)))
}

// ---------------------------------------------------------------------------
// Column read/write helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
pub(crate) fn write_column(
    store: &ReadableWritableListableStorage,
    path: &str,
    col: &Column,
) -> Result<(), MolRsError> {
    match col {
        Column::F32(a) => {
            let data = a.as_standard_layout();
            let shape: Vec<u64> = data.shape().iter().map(|&s| s as u64).collect();
            let chunk = shape.clone();
            let arr = ArrayBuilder::new(shape.clone(), chunk, data_type::float32(), 0.0f32)
                .build(store.clone(), path)?;
            arr.store_metadata()?;
            let subset = ArraySubset::new_with_shape(shape);
            arr.store_array_subset(&subset, data.as_slice().unwrap())?;
        }
        Column::F64(a) => {
            let data = a.as_standard_layout();
            let shape: Vec<u64> = data.shape().iter().map(|&s| s as u64).collect();
            let chunk = shape.clone();
            let arr = ArrayBuilder::new(shape.clone(), chunk, data_type::float64(), 0.0f64)
                .build(store.clone(), path)?;
            arr.store_metadata()?;
            let subset = ArraySubset::new_with_shape(shape);
            arr.store_array_subset(&subset, data.as_slice().unwrap())?;
        }
        Column::I64(a) => {
            let data = a.as_standard_layout();
            let shape: Vec<u64> = data.shape().iter().map(|&s| s as u64).collect();
            let chunk = shape.clone();
            let arr = ArrayBuilder::new(shape.clone(), chunk, data_type::int64(), 0i64)
                .build(store.clone(), path)?;
            arr.store_metadata()?;
            let subset = ArraySubset::new_with_shape(shape);
            arr.store_array_subset(&subset, data.as_slice().unwrap())?;
        }
        Column::U32(a) => {
            let data = a.as_standard_layout();
            let shape: Vec<u64> = data.shape().iter().map(|&s| s as u64).collect();
            let chunk = shape.clone();
            let arr = ArrayBuilder::new(shape.clone(), chunk, data_type::uint32(), 0u32)
                .build(store.clone(), path)?;
            arr.store_metadata()?;
            let subset = ArraySubset::new_with_shape(shape);
            arr.store_array_subset(&subset, data.as_slice().unwrap())?;
        }
        Column::U8(a) => {
            let data = a.as_standard_layout();
            let shape: Vec<u64> = data.shape().iter().map(|&s| s as u64).collect();
            let chunk = shape.clone();
            let arr = ArrayBuilder::new(shape.clone(), chunk, data_type::uint8(), 0u8)
                .build(store.clone(), path)?;
            arr.store_metadata()?;
            let subset = ArraySubset::new_with_shape(shape);
            arr.store_array_subset(&subset, data.as_slice().unwrap())?;
        }
        Column::Bool(a) => {
            // Store bools as UInt8 with attribute marker
            let data = a.as_standard_layout();
            let u8_data: Vec<u8> = data.iter().map(|&b| b as u8).collect();
            let shape: Vec<u64> = data.shape().iter().map(|&s| s as u64).collect();
            let chunk = shape.clone();
            let mut attrs = serde_json::Map::new();
            attrs.insert("molrs_dtype".into(), "bool".into());
            let arr = ArrayBuilder::new(shape.clone(), chunk, data_type::uint8(), 0u8)
                .attributes(attrs)
                .build(store.clone(), path)?;
            arr.store_metadata()?;
            let subset = ArraySubset::new_with_shape(shape);
            arr.store_array_subset(&subset, &u8_data)?;
        }
        Column::String(a) => {
            let data = a.as_standard_layout();
            let strings: Vec<String> = data.iter().cloned().collect();
            let shape: Vec<u64> = data.shape().iter().map(|&s| s as u64).collect();
            let chunk = shape.clone();
            let arr = ArrayBuilder::new(shape.clone(), chunk, data_type::string(), "")
                .build(store.clone(), path)?;
            arr.store_metadata()?;
            let subset = ArraySubset::new_with_shape(shape);
            arr.store_array_subset(&subset, &strings)?;
        }
    }
    Ok(())
}

pub(crate) fn read_column(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<Column, MolRsError> {
    let arr = zarrs::array::Array::open(store.clone(), path)?;
    let shape: Vec<usize> = arr.shape().iter().map(|&s| s as usize).collect();
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());

    // Check for bool marker
    let is_bool = arr.attributes().get("molrs_dtype").and_then(|v| v.as_str()) == Some("bool");

    let dt = arr.data_type();
    if dt.is::<Float32DataType>() {
        let data: Vec<f32> = arr.retrieve_array_subset(&subset)?;
        let nd = ArrayD::from_shape_vec(shape, data)
            .map_err(|e| MolRsError::zarr(format!("shape error: {}", e)))?;
        Ok(Column::F32(nd))
    } else if dt.is::<Float64DataType>() {
        let data: Vec<f64> = arr.retrieve_array_subset(&subset)?;
        let nd = ArrayD::from_shape_vec(shape, data)
            .map_err(|e| MolRsError::zarr(format!("shape error: {}", e)))?;
        Ok(Column::F64(nd))
    } else if dt.is::<Int64DataType>() {
        let data: Vec<i64> = arr.retrieve_array_subset(&subset)?;
        let nd = ArrayD::from_shape_vec(shape, data)
            .map_err(|e| MolRsError::zarr(format!("shape error: {}", e)))?;
        Ok(Column::I64(nd))
    } else if dt.is::<UInt32DataType>() {
        let data: Vec<u32> = arr.retrieve_array_subset(&subset)?;
        let nd = ArrayD::from_shape_vec(shape, data)
            .map_err(|e| MolRsError::zarr(format!("shape error: {}", e)))?;
        Ok(Column::U32(nd))
    } else if dt.is::<UInt8DataType>() || dt.is::<BoolDataType>() {
        if is_bool {
            let data: Vec<u8> = arr.retrieve_array_subset(&subset)?;
            let bool_data: Vec<bool> = data.into_iter().map(|v| v != 0).collect();
            let nd = ArrayD::from_shape_vec(shape, bool_data)
                .map_err(|e| MolRsError::zarr(format!("shape error: {}", e)))?;
            Ok(Column::Bool(nd))
        } else {
            let data: Vec<u8> = arr.retrieve_array_subset(&subset)?;
            let nd = ArrayD::from_shape_vec(shape, data)
                .map_err(|e| MolRsError::zarr(format!("shape error: {}", e)))?;
            Ok(Column::U8(nd))
        }
    } else if dt.is::<StringDataType>() {
        let data: Vec<String> = arr.retrieve_array_subset(&subset)?;
        let nd = ArrayD::from_shape_vec(shape, data)
            .map_err(|e| MolRsError::zarr(format!("shape error: {}", e)))?;
        Ok(Column::String(nd))
    } else {
        Err(MolRsError::zarr(format!("unsupported dtype: {:?}", dt)))
    }
}

pub(crate) fn insert_column_into_block(
    block: &mut Block,
    name: &str,
    col: Column,
) -> Result<(), MolRsError> {
    match col {
        Column::F32(a) => block.insert(name, a).map_err(MolRsError::Block),
        Column::F64(a) => block.insert(name, a).map_err(MolRsError::Block),
        Column::I64(a) => block.insert(name, a).map_err(MolRsError::Block),
        Column::U32(a) => block.insert(name, a).map_err(MolRsError::Block),
        Column::U8(a) => block.insert(name, a).map_err(MolRsError::Block),
        Column::Bool(a) => block.insert(name, a).map_err(MolRsError::Block),
        Column::String(a) => block.insert(name, a).map_err(MolRsError::Block),
    }
}

// ---------------------------------------------------------------------------
// Primitive array write helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
fn write_f32_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    shape: &[u64],
    data: &[f32],
) -> Result<(), MolRsError> {
    let arr = ArrayBuilder::new(shape.to_vec(), shape.to_vec(), data_type::float32(), 0.0f32)
        .build(store.clone(), path)?;
    arr.store_metadata()?;
    let subset = ArraySubset::new_with_shape(shape.to_vec());
    arr.store_array_subset(&subset, data)?;
    Ok(())
}

#[cfg(feature = "filesystem")]
fn write_u8_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    shape: &[u64],
    data: &[u8],
) -> Result<(), MolRsError> {
    let arr = ArrayBuilder::new(shape.to_vec(), shape.to_vec(), data_type::uint8(), 0u8)
        .build(store.clone(), path)?;
    arr.store_metadata()?;
    let subset = ArraySubset::new_with_shape(shape.to_vec());
    arr.store_array_subset(&subset, data)?;
    Ok(())
}

/// Helper to convert arbitrary zarrs errors into MolRsError::Zarr.
#[cfg(feature = "filesystem")]
fn zerr(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(e.to_string())
}

// Auto-convert zarrs error types
impl From<zarrs::group::GroupCreateError> for MolRsError {
    fn from(e: zarrs::group::GroupCreateError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

impl From<zarrs::storage::StorageError> for MolRsError {
    fn from(e: zarrs::storage::StorageError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

impl From<zarrs::array::ArrayCreateError> for MolRsError {
    fn from(e: zarrs::array::ArrayCreateError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

impl From<zarrs::array::ArrayError> for MolRsError {
    fn from(e: zarrs::array::ArrayError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

impl From<zarrs::node::NodeCreateError> for MolRsError {
    fn from(e: zarrs::node::NodeCreateError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_test_frame() -> Frame {
        let mut frame = Frame::new();

        // atoms block
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![1.0f64, 2.0, 3.0]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![4.0f64, 5.0, 6.0]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(vec![7.0f64, 8.0, 9.0]).into_dyn())
            .unwrap();
        atoms
            .insert(
                "mass",
                Array1::from_vec(vec![12.0f64, 1.0, 16.0]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["C".to_string(), "H".to_string(), "O".to_string()])
                    .into_dyn(),
            )
            .unwrap();
        frame.insert("atoms", atoms);

        // bonds block
        let mut bonds = Block::new();
        bonds
            .insert("i", Array1::from_vec(vec![0u32, 1]).into_dyn())
            .unwrap();
        bonds
            .insert("j", Array1::from_vec(vec![1u32, 2]).into_dyn())
            .unwrap();
        bonds
            .insert(
                "type",
                Array1::from_vec(vec!["single".to_string(), "double".to_string()]).into_dyn(),
            )
            .unwrap();
        frame.insert("bonds", bonds);

        // meta
        frame.meta.insert("title".into(), "test molecule".into());
        frame.meta.insert("step".into(), "42".into());

        // simbox
        use ndarray::array;
        frame.simbox =
            Some(SimBox::cube(10.0, array![0.0f32, 0.0, 0.0], [true, true, true]).unwrap());

        frame
    }

    #[test]
    fn test_frame_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_frame.zarr");

        let frame = make_test_frame();
        ZarrFrame::write(&path, &frame).unwrap();
        let loaded = ZarrFrame::read(&path).unwrap();

        // Check meta
        assert_eq!(loaded.meta.get("title").unwrap(), "test molecule");
        assert_eq!(loaded.meta.get("step").unwrap(), "42");

        // Check atoms block
        let atoms = loaded.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(3));
        let x = atoms.get_f64("x").unwrap();
        assert_eq!(x.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        let mass = atoms.get_f64("mass").unwrap();
        assert_eq!(mass.as_slice().unwrap(), &[12.0, 1.0, 16.0]);
        let types = atoms.get_string("type").unwrap();
        assert_eq!(types[0], "C");
        assert_eq!(types[1], "H");
        assert_eq!(types[2], "O");

        // Check bonds block
        let bonds = loaded.get("bonds").unwrap();
        assert_eq!(bonds.nrows(), Some(2));
        let i = bonds.get_u32("i").unwrap();
        assert_eq!(i.as_slice().unwrap(), &[0, 1]);

        // Check simbox
        let sb = loaded.simbox.as_ref().unwrap();
        assert!((sb.volume() - 1000.0).abs() < 1e-3);
    }
}
