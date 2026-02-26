//! Store: owns frames and mediates access via handles.

use crate::error::FfiError;
use crate::handle::{BlockHandle, FrameId};
use molrs::core::{block::Block, frame::Frame};
use slotmap::SlotMap;
use std::collections::HashMap;

/// Entry storing a frame and its invalidation tracking.
struct FrameEntry {
    frame: Frame,
    /// Version counter for each block key. Increments on remove/replace.
    block_versions: HashMap<String, u64>,
}

/// Store owns all frames and mediates access via handles.
pub struct Store {
    frames: SlotMap<FrameId, FrameEntry>,
}

impl Store {
    /// Creates a new empty Store.
    pub fn new() -> Self {
        Self {
            frames: SlotMap::with_key(),
        }
    }

    /// Creates a new frame and returns its stable ID.
    pub fn frame_new(&mut self) -> FrameId {
        let entry = FrameEntry {
            frame: Frame::new(),
            block_versions: HashMap::new(),
        };
        self.frames.insert(entry)
    }

    /// Drops a frame from the store, invalidating all handles to it.
    pub fn frame_drop(&mut self, id: FrameId) -> Result<(), FfiError> {
        self.frames
            .remove(id)
            .ok_or(FfiError::InvalidFrameId)
            .map(|_| ())
    }

    /// Sets a block in a frame. If the key exists, increments version to invalidate old handles.
    pub fn set_block(&mut self, id: FrameId, key: &str, block: Block) -> Result<(), FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;

        // If key exists, increment version to invalidate old handles
        if entry.frame.contains_key(key) {
            let version = entry.block_versions.entry(key.to_string()).or_insert(0);
            *version += 1;
        }

        entry.frame.insert(key, block);
        Ok(())
    }

    /// Removes a block from a frame, incrementing version to invalidate handles.
    pub fn remove_block(&mut self, id: FrameId, key: &str) -> Result<(), FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;

        if entry.frame.remove(key).is_some() {
            // Increment version to invalidate old handles
            let version = entry.block_versions.entry(key.to_string()).or_insert(0);
            *version += 1;
            Ok(())
        } else {
            Err(FfiError::KeyNotFound {
                key: key.to_string(),
            })
        }
    }

    /// Gets a block handle for a key in a frame.
    pub fn get_block(&self, id: FrameId, key: &str) -> Result<BlockHandle, FfiError> {
        let entry = self.frames.get(id).ok_or(FfiError::InvalidFrameId)?;

        if entry.frame.contains_key(key) {
            let version = entry.block_versions.get(key).copied().unwrap_or(0);
            Ok(BlockHandle::new(id, key.to_string(), version))
        } else {
            Err(FfiError::KeyNotFound {
                key: key.to_string(),
            })
        }
    }

    /// Clears all blocks from a frame, invalidating all block handles.
    pub fn clear_frame(&mut self, id: FrameId) -> Result<(), FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;

        // Increment all block versions to invalidate handles
        for key in entry.frame.keys() {
            let version = entry.block_versions.entry(key.to_string()).or_insert(0);
            *version += 1;
        }

        entry.frame.clear();
        Ok(())
    }

    /// Returns an owned clone of a frame.
    pub fn clone_frame(&self, id: FrameId) -> Result<Frame, FfiError> {
        let entry = self.frames.get(id).ok_or(FfiError::InvalidFrameId)?;
        Ok(entry.frame.clone())
    }

    /// Borrows a frame mutably and runs a closure on it.
    pub fn with_frame_mut<R>(
        &mut self,
        id: FrameId,
        f: impl FnOnce(&mut Frame) -> R,
    ) -> Result<R, FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;
        let result = f(&mut entry.frame);
        Ok(result)
    }

    /// Borrows a frame's simbox immutably and runs a closure on it.
    pub fn with_frame_simbox<R>(
        &self,
        id: FrameId,
        f: impl FnOnce(Option<&molrs::core::region::simbox::SimBox>) -> R,
    ) -> Result<R, FfiError> {
        let entry = self.frames.get(id).ok_or(FfiError::InvalidFrameId)?;
        Ok(f(entry.frame.simbox.as_ref()))
    }

    /// Sets an entire frame, replacing the existing one.
    pub fn set_frame(&mut self, id: FrameId, frame: Frame) -> Result<(), FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;

        // Increment all block versions to invalidate old handles
        for key in entry.frame.keys() {
            let version = entry.block_versions.entry(key.to_string()).or_insert(0);
            *version += 1;
        }

        entry.frame = frame;
        Ok(())
    }

    /// Returns an owned clone of a block.
    pub fn clone_block(&self, handle: &BlockHandle) -> Result<Block, FfiError> {
        self.validate_block_handle(handle)?;
        let entry = self.frames.get(handle.frame_id).unwrap();
        let block = entry.frame.get(&handle.key).unwrap();
        Ok(block.clone())
    }

    /// Borrows a block immutably and runs a closure on it.
    pub fn with_block<R>(
        &self,
        handle: &BlockHandle,
        f: impl FnOnce(&Block) -> R,
    ) -> Result<R, FfiError> {
        self.validate_block_handle(handle)?;
        let entry = self.frames.get(handle.frame_id).unwrap();
        let block = entry.frame.get(&handle.key).unwrap();
        Ok(f(block))
    }

    /// Borrows a block mutably, runs a closure, then bumps the version and updates the handle.
    pub fn with_block_mut<R>(
        &mut self,
        handle: &mut BlockHandle,
        f: impl FnOnce(&mut Block) -> R,
    ) -> Result<R, FfiError> {
        self.validate_block_handle(handle)?;
        let entry = self.frames.get_mut(handle.frame_id).unwrap();
        let block = entry.frame.get_mut(&handle.key).unwrap();
        let result = f(block);

        // Bump version to invalidate other handles to this block
        let version = entry.block_versions.entry(handle.key.clone()).or_insert(0);
        *version += 1;
        handle.version = *version;

        Ok(result)
    }

    /// Gets an f32 column as an owned copy with shape.
    pub fn block_col_f32(
        &self,
        handle: &BlockHandle,
        col: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), FfiError> {
        self.validate_block_handle(handle)?;
        let entry = self.frames.get(handle.frame_id).unwrap();
        let block = entry.frame.get(&handle.key).unwrap();

        let arr = block.get_f32(col).ok_or_else(|| FfiError::KeyNotFound {
            key: col.to_string(),
        })?;

        let shape = arr.shape().to_vec();
        let data = arr.iter().copied().collect();

        Ok((data, shape))
    }

    /// Gets a u32 column as an owned copy with shape.
    pub fn block_col_u32(
        &self,
        handle: &BlockHandle,
        col: &str,
    ) -> Result<(Vec<u32>, Vec<usize>), FfiError> {
        self.validate_block_handle(handle)?;
        let entry = self.frames.get(handle.frame_id).unwrap();
        let block = entry.frame.get(&handle.key).unwrap();

        let arr = block.get_u32(col).ok_or_else(|| FfiError::KeyNotFound {
            key: col.to_string(),
        })?;

        let shape = arr.shape().to_vec();
        let data = arr.iter().copied().collect();

        Ok((data, shape))
    }

    /// Gets a u8 column as an owned copy with shape.
    pub fn block_col_u8(
        &self,
        handle: &BlockHandle,
        col: &str,
    ) -> Result<(Vec<u8>, Vec<usize>), FfiError> {
        self.validate_block_handle(handle)?;
        let entry = self.frames.get(handle.frame_id).unwrap();
        let block = entry.frame.get(&handle.key).unwrap();

        let arr = block.get_u8(col).ok_or_else(|| FfiError::KeyNotFound {
            key: col.to_string(),
        })?;

        let shape = arr.shape().to_vec();
        let data = arr.iter().copied().collect();

        Ok((data, shape))
    }

    /// Gets an f64 column as an owned copy with shape.
    pub fn block_col_f64(
        &self,
        handle: &BlockHandle,
        col: &str,
    ) -> Result<(Vec<f64>, Vec<usize>), FfiError> {
        self.validate_block_handle(handle)?;
        let entry = self.frames.get(handle.frame_id).unwrap();
        let block = entry.frame.get(&handle.key).unwrap();

        let arr = block.get_f64(col).ok_or_else(|| FfiError::KeyNotFound {
            key: col.to_string(),
        })?;

        let shape = arr.shape().to_vec();
        let data = arr.iter().copied().collect();

        Ok((data, shape))
    }

    /// Gets an f32 column as a zero-copy view (ptr, len, shape).
    /// Fails if array is not contiguous.
    pub fn block_col_f32_view(
        &self,
        handle: &BlockHandle,
        col: &str,
    ) -> Result<(*const f32, usize, Vec<usize>), FfiError> {
        self.validate_block_handle(handle)?;
        let entry = self.frames.get(handle.frame_id).unwrap();
        let block = entry.frame.get(&handle.key).unwrap();

        let arr = block.get_f32(col).ok_or_else(|| FfiError::KeyNotFound {
            key: col.to_string(),
        })?;

        // Check if contiguous
        let slice = arr
            .as_slice_memory_order()
            .ok_or_else(|| FfiError::NonContiguous {
                key: col.to_string(),
            })?;

        let shape = arr.shape().to_vec();
        Ok((slice.as_ptr(), slice.len(), shape))
    }

    /// Validates a block handle.
    fn validate_block_handle(&self, handle: &BlockHandle) -> Result<(), FfiError> {
        // Check frame exists
        let entry = self
            .frames
            .get(handle.frame_id)
            .ok_or(FfiError::InvalidBlockHandle)?;

        // Check key exists
        if !entry.frame.contains_key(&handle.key) {
            return Err(FfiError::InvalidBlockHandle);
        }

        // Check version matches (block hasn't been replaced)
        let current_version = entry.block_versions.get(&handle.key).copied().unwrap_or(0);
        if current_version != handle.version {
            return Err(FfiError::InvalidBlockHandle);
        }

        Ok(())
    }
}

impl Default for Store {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_frame_lifecycle() {
        let mut store = Store::new();

        let id = store.frame_new();
        assert!(store.clone_frame(id).is_ok());

        store.frame_drop(id).unwrap();
        assert!(store.clone_frame(id).is_err());
    }

    #[test]
    fn test_block_operations() {
        let mut store = Store::new();
        let id = store.frame_new();

        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn())
            .unwrap();

        store.set_block(id, "atoms", block).unwrap();

        let handle = store.get_block(id, "atoms").unwrap();
        assert!(store.clone_block(&handle).is_ok());
    }

    #[test]
    fn test_block_invalidation_on_remove() {
        let mut store = Store::new();
        let id = store.frame_new();

        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn())
            .unwrap();

        store.set_block(id, "atoms", block).unwrap();
        let handle = store.get_block(id, "atoms").unwrap();

        // Remove block
        store.remove_block(id, "atoms").unwrap();

        // Old handle should be invalid
        assert!(matches!(
            store.clone_block(&handle),
            Err(FfiError::InvalidBlockHandle)
        ));
    }

    #[test]
    fn test_block_invalidation_on_replace() {
        let mut store = Store::new();
        let id = store.frame_new();

        let mut block1 = Block::new();
        block1
            .insert("x", Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn())
            .unwrap();

        store.set_block(id, "atoms", block1).unwrap();
        let handle = store.get_block(id, "atoms").unwrap();

        // Replace block
        let mut block2 = Block::new();
        block2
            .insert("x", Array1::from_vec(vec![3.0_f32, 4.0]).into_dyn())
            .unwrap();
        store.set_block(id, "atoms", block2).unwrap();

        // Old handle should be invalid
        assert!(matches!(
            store.clone_block(&handle),
            Err(FfiError::InvalidBlockHandle)
        ));
    }

    #[test]
    fn test_reinsert_does_not_resurrect() {
        let mut store = Store::new();
        let id = store.frame_new();

        let mut block1 = Block::new();
        block1
            .insert("x", Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn())
            .unwrap();

        store.set_block(id, "atoms", block1).unwrap();
        let old_handle = store.get_block(id, "atoms").unwrap();

        // Remove
        store.remove_block(id, "atoms").unwrap();

        // Reinsert with same key
        let mut block2 = Block::new();
        block2
            .insert("x", Array1::from_vec(vec![3.0_f32, 4.0]).into_dyn())
            .unwrap();
        store.set_block(id, "atoms", block2).unwrap();

        // Old handle should still be invalid
        assert!(matches!(
            store.clone_block(&old_handle),
            Err(FfiError::InvalidBlockHandle)
        ));

        // New handle should work
        let new_handle = store.get_block(id, "atoms").unwrap();
        assert!(store.clone_block(&new_handle).is_ok());
    }

    #[test]
    fn test_clear_invalidates_all_blocks() {
        let mut store = Store::new();
        let id = store.frame_new();

        let mut block1 = Block::new();
        block1
            .insert("x", Array1::from_vec(vec![1.0_f32]).into_dyn())
            .unwrap();
        let mut block2 = Block::new();
        block2
            .insert("y", Array1::from_vec(vec![2.0_f32]).into_dyn())
            .unwrap();

        store.set_block(id, "atoms", block1).unwrap();
        store.set_block(id, "bonds", block2).unwrap();

        let handle1 = store.get_block(id, "atoms").unwrap();
        let handle2 = store.get_block(id, "bonds").unwrap();

        // Clear frame
        store.clear_frame(id).unwrap();

        // Both handles should be invalid
        assert!(matches!(
            store.clone_block(&handle1),
            Err(FfiError::InvalidBlockHandle)
        ));
        assert!(matches!(
            store.clone_block(&handle2),
            Err(FfiError::InvalidBlockHandle)
        ));
    }

    #[test]
    fn test_col_f32_copy() {
        let mut store = Store::new();
        let id = store.frame_new();

        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn())
            .unwrap();

        store.set_block(id, "atoms", block).unwrap();
        let handle = store.get_block(id, "atoms").unwrap();

        let (data, shape) = store.block_col_f32(&handle, "x").unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert_eq!(shape, vec![3]);
    }

    #[test]
    fn test_col_f32_view_contiguous() {
        let mut store = Store::new();
        let id = store.frame_new();

        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn())
            .unwrap();

        store.set_block(id, "atoms", block).unwrap();
        let handle = store.get_block(id, "atoms").unwrap();

        let (ptr, len, shape) = store.block_col_f32_view(&handle, "x").unwrap();
        assert_eq!(len, 3);
        assert_eq!(shape, vec![3]);

        // Verify data via pointer
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_with_block_read() {
        let mut store = Store::new();
        let id = store.frame_new();

        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn())
            .unwrap();
        store.set_block(id, "atoms", block).unwrap();

        let handle = store.get_block(id, "atoms").unwrap();
        let len = store.with_block(&handle, |b| b.len()).unwrap();
        assert_eq!(len, 1);

        let nrows = store.with_block(&handle, |b| b.nrows()).unwrap();
        assert_eq!(nrows, Some(3));
    }

    #[test]
    fn test_with_block_mut_insert() {
        let mut store = Store::new();
        let id = store.frame_new();

        let block = Block::new();
        store.set_block(id, "atoms", block).unwrap();
        let mut handle = store.get_block(id, "atoms").unwrap();
        let old_version = handle.version;

        store
            .with_block_mut(&mut handle, |b| {
                b.insert("y", Array1::from_vec(vec![4.0_f32, 5.0]).into_dyn())
                    .unwrap();
            })
            .unwrap();

        // Version should have bumped
        assert!(handle.version > old_version);

        // Data should be accessible
        let nrows = store.with_block(&handle, |b| b.nrows()).unwrap();
        assert_eq!(nrows, Some(2));
    }

    #[test]
    fn test_with_block_mut_invalidates_old_handles() {
        let mut store = Store::new();
        let id = store.frame_new();

        let block = Block::new();
        store.set_block(id, "atoms", block).unwrap();
        let old_handle = store.get_block(id, "atoms").unwrap();
        let mut handle = old_handle.clone();

        store
            .with_block_mut(&mut handle, |b| {
                b.insert("x", Array1::from_vec(vec![1.0_f32]).into_dyn())
                    .unwrap();
            })
            .unwrap();

        // Old handle should be invalid
        assert!(matches!(
            store.with_block(&old_handle, |_| ()),
            Err(FfiError::InvalidBlockHandle)
        ));
    }

    #[test]
    fn test_with_frame_mut() {
        let mut store = Store::new();
        let id = store.frame_new();

        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0_f32]).into_dyn())
            .unwrap();
        store.set_block(id, "atoms", block).unwrap();

        let renamed = store
            .with_frame_mut(id, |f| f.rename_block("atoms", "particles"))
            .unwrap();
        assert!(renamed);

        // Old key gone, new key exists
        assert!(store.get_block(id, "atoms").is_err());
        assert!(store.get_block(id, "particles").is_ok());
    }
}
