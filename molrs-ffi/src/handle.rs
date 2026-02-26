//! Handle types for stable cross-language references.

use slotmap::new_key_type;

// Define a new key type for frames using slotmap
new_key_type! {
    /// Stable identifier for a stored frame.
    ///
    /// Structure: (index: u32, generation: u32)
    /// - index: slot in the store
    /// - generation: invalidation counter (detects use-after-free)
    ///
    /// Properties:
    /// - Copy, comparable, FFI-safe
    /// - Valid until frame_drop() is called
    /// - JS can represent without precision loss (two u32 instead of u64)
    pub struct FrameId;
}

/// View handle that refers to a block inside a frame.
///
/// Structure: (frame_id, key)
///
/// Becomes invalid when:
/// - Frame is dropped
/// - Block is removed via remove_block(key)
/// - Block is replaced via set_block(key, new_block)
/// - Frame is cleared
///
/// Not resurrected if same key is reinserted later.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockHandle {
    pub frame_id: FrameId,
    pub key: String,
    // Internal: version counter to detect invalidation
    pub(crate) version: u64,
}

impl BlockHandle {
    pub(crate) fn new(frame_id: FrameId, key: String, version: u64) -> Self {
        Self {
            frame_id,
            key,
            version,
        }
    }
}
