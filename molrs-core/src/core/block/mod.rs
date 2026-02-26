//! Block: dict-like keyed arrays with consistent axis-0 length and heterogeneous types.
//!
//! A Block stores heterogeneous arrays (f32, f64, i64, bool) keyed by strings,
//! enforcing that all stored arrays share the same axis-0 length (nrows).
//!
//! # Examples
//!
//! ```
//! use molrs::core::block::Block;
//! use ndarray::{Array1, ArrayD};
//!
//! let mut block = Block::new();
//!
//! // Insert different types - generic dispatch handles the conversion
//! let pos = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn();
//! let ids = Array1::from_vec(vec![10_i64, 20, 30]).into_dyn();
//!
//! block.insert("pos", pos).unwrap();
//! block.insert("id", ids).unwrap();
//!
//! // Type-safe retrieval
//! let pos_ref = block.get_f32("pos").unwrap();
//! let ids_ref = block.get_i64("id").unwrap();
//!
//! assert_eq!(block.nrows(), Some(3));
//! assert_eq!(block.len(), 2);
//! ```

mod column;
mod dtype;
mod error;

pub use column::Column;
pub use dtype::{BlockDtype, DType};
pub use error::BlockError;

use ndarray::ArrayD;
use std::collections::HashMap;
use std::ops::{Index, IndexMut};

/// A dictionary from string keys to ndarray arrays with a consistent axis-0 length.
///
/// This Block supports heterogeneous column types (f32, f64, i64, bool).
#[derive(Default, Clone)]
pub struct Block {
    map: HashMap<String, Column>,
    nrows: Option<usize>,
}

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (k, v) in &self.map {
            map.entry(k, &format!("{}(shape={:?})", v.dtype(), v.shape()));
        }
        map.finish()
    }
}

impl Block {
    /// Creates an empty Block.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            nrows: None,
        }
    }

    /// Creates an empty Block with the specified capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: HashMap::with_capacity(cap),
            nrows: None,
        }
    }

    /// Number of keys (columns).
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if there are no arrays in the block.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the common axis-0 length of all arrays, or `None` if empty.
    #[inline]
    pub fn nrows(&self) -> Option<usize> {
        self.nrows
    }

    /// Returns true if the Block contains the specified key.
    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Inserts an array under `key`, enforcing consistent axis-0 length.
    ///
    /// This method uses generic dispatch via the `BlockDtype` trait to accept
    /// any supported type (f32, f64, i64, bool) without requiring users to
    /// manually construct Column enums.
    ///
    /// # Errors
    ///
    /// - Returns `BlockError::RankZero` if the array has rank 0
    /// - Returns `BlockError::RaggedAxis0` if the array's axis-0 length doesn't
    ///   match the Block's existing `nrows`
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::block::Block;
    /// use ndarray::Array1;
    ///
    /// let mut block = Block::new();
    ///
    /// // Insert f32 array
    /// let arr_f32 = Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn();
    /// block.insert("x", arr_f32).unwrap();
    ///
    /// // Insert i64 array with same nrows
    /// let arr_i64 = Array1::from_vec(vec![10_i64, 20]).into_dyn();
    /// block.insert("id", arr_i64).unwrap();
    ///
    /// // This would error - different nrows
    /// let arr_bad = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn();
    /// assert!(block.insert("bad", arr_bad).is_err());
    /// ```
    pub fn insert<T: BlockDtype>(
        &mut self,
        key: impl Into<String>,
        arr: ArrayD<T>,
    ) -> Result<(), BlockError> {
        let key = key.into();
        let shape = arr.shape();

        // Check rank >= 1
        if shape.is_empty() {
            return Err(BlockError::RankZero { key });
        }

        let len0 = shape[0];

        // Check axis-0 consistency
        match self.nrows {
            None => {
                // First insertion defines nrows
                self.nrows = Some(len0);
            }
            Some(expected) => {
                if len0 != expected {
                    return Err(BlockError::RaggedAxis0 {
                        key,
                        expected,
                        got: len0,
                    });
                }
            }
        }

        let col = T::into_column(arr);
        self.map.insert(key, col);
        Ok(())
    }

    /// Gets an immutable reference to the column for `key` if present.
    ///
    /// For type-safe access, prefer using `get_f32()`, `get_i64()`, etc.
    #[inline]
    pub fn get(&self, key: &str) -> Option<&Column> {
        self.map.get(key)
    }

    /// Gets a mutable reference to the column for `key` if present.
    ///
    /// For type-safe access, prefer using `get_f32_mut()`, `get_i64_mut()`, etc.
    ///
    /// # Warning
    ///
    /// Mutating the column's shape through this reference is allowed but NOT
    /// revalidated. It's the caller's responsibility to maintain axis-0 consistency.
    #[inline]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Column> {
        self.map.get_mut(key)
    }

    // Type-specific getters for f32

    /// Gets an immutable reference to an f32 array for `key` if present and of correct type.
    pub fn get_f32(&self, key: &str) -> Option<&ArrayD<f32>> {
        self.get(key).and_then(|c| c.as_f32())
    }

    /// Gets a mutable reference to an f32 array for `key` if present and of correct type.
    pub fn get_f32_mut(&mut self, key: &str) -> Option<&mut ArrayD<f32>> {
        self.get_mut(key).and_then(|c| c.as_f32_mut())
    }

    // Type-specific getters for f64

    /// Gets an immutable reference to an f64 array for `key` if present and of correct type.
    pub fn get_f64(&self, key: &str) -> Option<&ArrayD<f64>> {
        self.get(key).and_then(|c| c.as_f64())
    }

    /// Gets a mutable reference to an f64 array for `key` if present and of correct type.
    pub fn get_f64_mut(&mut self, key: &str) -> Option<&mut ArrayD<f64>> {
        self.get_mut(key).and_then(|c| c.as_f64_mut())
    }

    // Type-specific getters for i64

    /// Gets an immutable reference to an i64 array for `key` if present and of correct type.
    pub fn get_i64(&self, key: &str) -> Option<&ArrayD<i64>> {
        self.get(key).and_then(|c| c.as_i64())
    }

    /// Gets a mutable reference to an i64 array for `key` if present and of correct type.
    pub fn get_i64_mut(&mut self, key: &str) -> Option<&mut ArrayD<i64>> {
        self.get_mut(key).and_then(|c| c.as_i64_mut())
    }

    // Type-specific getters for bool

    /// Gets an immutable reference to a bool array for `key` if present and of correct type.
    pub fn get_bool(&self, key: &str) -> Option<&ArrayD<bool>> {
        self.get(key).and_then(|c| c.as_bool())
    }

    /// Gets a mutable reference to a bool array for `key` if present and of correct type.
    pub fn get_bool_mut(&mut self, key: &str) -> Option<&mut ArrayD<bool>> {
        self.get_mut(key).and_then(|c| c.as_bool_mut())
    }

    // Type-specific getters for u32

    /// Gets an immutable reference to a u32 array for `key` if present and of correct type.
    pub fn get_u32(&self, key: &str) -> Option<&ArrayD<u32>> {
        self.get(key).and_then(|c| c.as_u32())
    }

    /// Gets a mutable reference to a u32 array for `key` if present and of correct type.
    pub fn get_u32_mut(&mut self, key: &str) -> Option<&mut ArrayD<u32>> {
        self.get_mut(key).and_then(|c| c.as_u32_mut())
    }

    // Type-specific getters for u8

    /// Gets an immutable reference to a u8 array for `key` if present and of correct type.
    pub fn get_u8(&self, key: &str) -> Option<&ArrayD<u8>> {
        self.get(key).and_then(|c| c.as_u8())
    }

    /// Gets a mutable reference to a u8 array for `key` if present and of correct type.
    pub fn get_u8_mut(&mut self, key: &str) -> Option<&mut ArrayD<u8>> {
        self.get_mut(key).and_then(|c| c.as_u8_mut())
    }

    // Type-specific getters for String

    /// Gets an immutable reference to a String array for `key` if present and of correct type.
    pub fn get_string(&self, key: &str) -> Option<&ArrayD<String>> {
        self.get(key).and_then(|c| c.as_string())
    }

    /// Gets a mutable reference to a String array for `key` if present and of correct type.
    pub fn get_string_mut(&mut self, key: &str) -> Option<&mut ArrayD<String>> {
        self.get_mut(key).and_then(|c| c.as_string_mut())
    }

    /// Removes and returns the column for `key`, if present.
    ///
    /// If the Block becomes empty after removal, resets `nrows` to `None`.
    pub fn remove(&mut self, key: &str) -> Option<Column> {
        let out = self.map.remove(key);
        if self.map.is_empty() {
            self.nrows = None;
        }
        out
    }

    /// Renames a column from `old_key` to `new_key`.
    ///
    /// Returns `true` if the column was successfully renamed, `false` if `old_key` doesn't exist
    /// or `new_key` already exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::block::Block;
    /// use ndarray::Array1;
    ///
    /// let mut block = Block::new();
    /// block.insert("x", Array1::from_vec(vec![1.0_f32]).into_dyn()).unwrap();
    ///
    /// assert!(block.rename_column("x", "position_x"));
    /// assert!(!block.contains_key("x"));
    /// assert!(block.contains_key("position_x"));
    /// ```
    pub fn rename_column(&mut self, old_key: &str, new_key: &str) -> bool {
        // Check if old_key exists and new_key doesn't exist
        if !self.map.contains_key(old_key) || self.map.contains_key(new_key) {
            return false;
        }

        // Remove the old key and re-insert with new key
        if let Some(column) = self.map.remove(old_key) {
            self.map.insert(new_key.to_string(), column);
            true
        } else {
            false
        }
    }

    /// Clears the Block, removing all keys and resetting `nrows`.
    pub fn clear(&mut self) {
        self.map.clear();
        self.nrows = None;
    }

    /// Returns an iterator over (&str, &Column).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns an iterator over keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(|k| k.as_str())
    }

    /// Returns an iterator over column references.
    pub fn values(&self) -> impl Iterator<Item = &Column> {
        self.map.values()
    }

    /// Returns the data type of the column with the given key, if it exists.
    pub fn dtype(&self, key: &str) -> Option<DType> {
        self.get(key).map(|c| c.dtype())
    }

    /// Safely resize all columns to a new number of rows.
    ///
    /// This method ensures that all columns are resized consistently,
    /// maintaining the Block's nrows invariant.
    ///
    /// # Arguments
    /// * `new_nrows` - The new number of rows
    /// * `fill_fn` - Function that provides fill values for each column
    ///
    /// # Returns
    /// * `Ok(())` if resize succeeds
    /// * `Err(BlockError)` if resize fails
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::block::Block;
    /// use ndarray::Array1;
    ///
    /// let mut block = Block::new();
    /// block.insert("x", Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn()).unwrap();
    ///
    /// // Note: resize is not yet fully implemented - this is a placeholder
    /// // block.resize(3, |_key| 0.0).unwrap();
    /// ```
    pub fn resize(&mut self, new_nrows: usize) -> Result<(), crate::error::MolRsError> {
        // For now, we only allow resizing empty blocks
        // Full implementation would require cloning and resizing each column
        if self.is_empty() {
            self.nrows = Some(new_nrows);
            Ok(())
        } else {
            Err(crate::error::MolRsError::validation(
                "Block resize is not yet fully implemented for non-empty blocks",
            ))
        }
    }

    /// Merge another block into this one by concatenating columns along axis-0.
    ///
    /// Both blocks must have the same set of column keys and matching dtypes.
    /// The resulting block will have nrows = self.nrows + other.nrows.
    ///
    /// # Arguments
    /// * `other` - The block to merge into this one
    ///
    /// # Returns
    /// * `Ok(())` if merge succeeds
    /// * `Err(BlockError)` if blocks have incompatible columns
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::block::Block;
    /// use ndarray::Array1;
    ///
    /// let mut block1 = Block::new();
    /// block1.insert("x", Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn()).unwrap();
    ///
    /// let mut block2 = Block::new();
    /// block2.insert("x", Array1::from_vec(vec![3.0_f32, 4.0]).into_dyn()).unwrap();
    ///
    /// block1.merge(&block2).unwrap();
    /// assert_eq!(block1.nrows(), Some(4));
    /// ```
    pub fn merge(&mut self, other: &Block) -> Result<(), BlockError> {
        use ndarray::Axis;
        use ndarray::concatenate;

        // If other is empty, nothing to do
        if other.is_empty() {
            return Ok(());
        }

        // If self is empty, clone other
        if self.is_empty() {
            self.map = other.map.clone();
            self.nrows = other.nrows;
            return Ok(());
        }

        // Check that both blocks have the same keys
        let self_keys: std::collections::HashSet<_> = self.keys().collect();
        let other_keys: std::collections::HashSet<_> = other.keys().collect();

        if self_keys != other_keys {
            return Err(BlockError::validation(format!(
                "Cannot merge blocks with different keys. Self has {:?}, other has {:?}",
                self_keys, other_keys
            )));
        }

        // Merge each column
        let mut new_map = HashMap::new();
        for key in self.keys() {
            let self_col = &self.map[key];
            let other_col = &other.map[key];

            // Check dtype compatibility
            if self_col.dtype() != other_col.dtype() {
                return Err(BlockError::validation(format!(
                    "Column '{}' has incompatible dtypes: {:?} vs {:?}",
                    key,
                    self_col.dtype(),
                    other_col.dtype()
                )));
            }

            // Concatenate based on dtype
            let merged_col = match (self_col, other_col) {
                (Column::F32(a), Column::F32(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate f32 column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::F32(merged)
                }
                (Column::F64(a), Column::F64(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate f64 column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::F64(merged)
                }
                (Column::I64(a), Column::I64(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate i64 column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::I64(merged)
                }
                (Column::U32(a), Column::U32(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate u32 column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::U32(merged)
                }
                (Column::U8(a), Column::U8(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate u8 column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::U8(merged)
                }
                (Column::Bool(a), Column::Bool(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate bool column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::Bool(merged)
                }
                (Column::String(a), Column::String(b)) => {
                    let merged = concatenate(Axis(0), &[a.view(), b.view()]).map_err(|e| {
                        BlockError::validation(format!(
                            "Failed to concatenate string column '{}': {}",
                            key, e
                        ))
                    })?;
                    Column::String(merged)
                }
                _ => unreachable!("dtype mismatch already checked"),
            };

            new_map.insert(key.to_string(), merged_col);
        }

        // Update nrows
        let new_nrows = self.nrows.unwrap() + other.nrows.unwrap();
        self.map = new_map;
        self.nrows = Some(new_nrows);

        Ok(())
    }
}

// Index trait for convenient access: block["key"]
impl Index<&str> for Block {
    type Output = Column;

    fn index(&self, key: &str) -> &Self::Output {
        self.get(key)
            .unwrap_or_else(|| panic!("key '{}' not found in Block", key))
    }
}

impl IndexMut<&str> for Block {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        self.get_mut(key)
            .unwrap_or_else(|| panic!("key '{}' not found in Block", key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_insert_mixed_dtypes() {
        let mut block = Block::new();

        let arr_f32 = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn();
        let arr_f64 = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]).into_dyn();
        let arr_i64 = Array1::from_vec(vec![10_i64, 20, 30]).into_dyn();
        let arr_bool = Array1::from_vec(vec![true, false, true]).into_dyn();

        assert!(block.insert("x", arr_f32).is_ok());
        assert!(block.insert("y", arr_f64).is_ok());
        assert!(block.insert("id", arr_i64).is_ok());
        assert!(block.insert("mask", arr_bool).is_ok());

        assert_eq!(block.len(), 4);
        assert_eq!(block.nrows(), Some(3));
    }

    #[test]
    fn test_axis0_mismatch_error() {
        let mut block = Block::new();

        let arr1 = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn();
        let arr2 = Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn();

        block.insert("x", arr1).unwrap();
        let result = block.insert("y", arr2);

        assert!(result.is_err());
        match result {
            Err(BlockError::RaggedAxis0 { expected, got, .. }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            _ => panic!("Expected RaggedAxis0 error"),
        }
    }

    #[test]
    fn test_typed_getters() {
        let mut block = Block::new();

        let arr_f32 = Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn();
        let arr_i64 = Array1::from_vec(vec![10_i64, 20]).into_dyn();

        block.insert("x", arr_f32).unwrap();
        block.insert("id", arr_i64).unwrap();

        // Correct type access
        assert!(block.get_f32("x").is_some());
        assert!(block.get_i64("id").is_some());

        // Wrong type access returns None
        assert!(block.get_i64("x").is_none());
        assert!(block.get_f32("id").is_none());

        // Mutable access
        if let Some(x_mut) = block.get_f32_mut("x") {
            x_mut[[0]] = 99.0;
        }
        assert_eq!(block.get_f32("x").unwrap()[[0]], 99.0);
    }

    #[test]
    fn test_index_access() {
        let mut block = Block::new();

        let arr = Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn();
        block.insert("x", arr).unwrap();

        // Immutable index
        let col = &block["x"];
        assert_eq!(col.dtype(), DType::F32);

        // Mutable index
        let col_mut = &mut block["x"];
        if let Some(arr_mut) = col_mut.as_f32_mut() {
            arr_mut[[0]] = 42.0;
        }
        assert_eq!(block.get_f32("x").unwrap()[[0]], 42.0);
    }

    #[test]
    #[should_panic(expected = "key 'missing' not found")]
    fn test_index_panic_on_missing_key() {
        let block = Block::new();
        let _ = &block["missing"];
    }

    #[test]
    fn test_remove_resets_nrows() {
        let mut block = Block::new();

        let arr = Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn();
        block.insert("x", arr).unwrap();

        assert_eq!(block.nrows(), Some(2));

        block.remove("x");
        assert_eq!(block.nrows(), None);
        assert!(block.is_empty());
    }

    #[test]
    fn test_iter_keys_values() {
        let mut block = Block::new();

        let arr1 = Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn();
        let arr2 = Array1::from_vec(vec![10_i64, 20]).into_dyn();

        block.insert("x", arr1).unwrap();
        block.insert("id", arr2).unwrap();

        let keys: Vec<&str> = block.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"x"));
        assert!(keys.contains(&"id"));

        let dtypes: Vec<DType> = block.values().map(|c| c.dtype()).collect();
        assert!(dtypes.contains(&DType::F32));
        assert!(dtypes.contains(&DType::I64));
    }

    #[test]
    fn test_rank_zero_error() {
        let mut block = Block::new();

        // Create a rank-0 array (scalar)
        let arr = ArrayD::<f32>::zeros(vec![]);

        let result = block.insert("scalar", arr);
        assert!(result.is_err());
        match result {
            Err(BlockError::RankZero { key }) => {
                assert_eq!(key, "scalar");
            }
            _ => panic!("Expected RankZero error"),
        }
    }

    #[test]
    fn test_dtype_query() {
        let mut block = Block::new();

        let arr_f32 = Array1::from_vec(vec![1.0_f32]).into_dyn();
        let arr_i64 = Array1::from_vec(vec![10_i64]).into_dyn();

        block.insert("x", arr_f32).unwrap();
        block.insert("id", arr_i64).unwrap();

        assert_eq!(block.dtype("x"), Some(DType::F32));
        assert_eq!(block.dtype("id"), Some(DType::I64));
        assert_eq!(block.dtype("missing"), None);
    }

    #[test]
    fn test_merge_basic() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        let arr1 = Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn();
        let arr2 = Array1::from_vec(vec![3.0_f32, 4.0]).into_dyn();

        block1.insert("x", arr1).unwrap();
        block2.insert("x", arr2).unwrap();

        block1.merge(&block2).unwrap();

        assert_eq!(block1.nrows(), Some(4));
        let x = block1.get_f32("x").unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_merge_empty_blocks() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        // Merge empty into empty
        block1.merge(&block2).unwrap();
        assert_eq!(block1.nrows(), None);

        // Merge non-empty into empty
        let arr = Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn();
        block2.insert("x", arr).unwrap();
        block1.merge(&block2).unwrap();
        assert_eq!(block1.nrows(), Some(2));

        // Merge empty into non-empty
        let block3 = Block::new();
        block1.merge(&block3).unwrap();
        assert_eq!(block1.nrows(), Some(2));
    }

    #[test]
    fn test_merge_incompatible_keys() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        let arr1 = Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn();
        let arr2 = Array1::from_vec(vec![3.0_f32, 4.0]).into_dyn();

        block1.insert("x", arr1).unwrap();
        block2.insert("y", arr2).unwrap();

        let result = block1.merge(&block2);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_incompatible_dtypes() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        let arr1 = Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn();
        let arr2 = Array1::from_vec(vec![3_i64, 4]).into_dyn();

        block1.insert("x", arr1).unwrap();
        block2.insert("x", arr2).unwrap();

        let result = block1.merge(&block2);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_multiple_columns() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        block1
            .insert("x", Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn())
            .unwrap();
        block1
            .insert("id", Array1::from_vec(vec![10_i64, 20]).into_dyn())
            .unwrap();

        block2
            .insert("x", Array1::from_vec(vec![3.0_f32, 4.0]).into_dyn())
            .unwrap();
        block2
            .insert("id", Array1::from_vec(vec![30_i64, 40]).into_dyn())
            .unwrap();

        block1.merge(&block2).unwrap();

        assert_eq!(block1.nrows(), Some(4));
        let x = block1.get_f32("x").unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        let id = block1.get_i64("id").unwrap();
        assert_eq!(id.as_slice_memory_order().unwrap(), &[10, 20, 30, 40]);
    }

    #[test]
    fn test_rename_column() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn())
            .unwrap();
        block
            .insert("y", Array1::from_vec(vec![3.0_f32, 4.0]).into_dyn())
            .unwrap();

        // Successful rename
        assert!(block.rename_column("x", "position_x"));
        assert!(!block.contains_key("x"));
        assert!(block.contains_key("position_x"));
        assert_eq!(
            block
                .get_f32("position_x")
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[1.0, 2.0]
        );

        // Try to rename non-existent column
        assert!(!block.rename_column("nonexistent", "new_name"));

        // Try to rename to existing column name
        assert!(!block.rename_column("position_x", "y"));
    }
}
