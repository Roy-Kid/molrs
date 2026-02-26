//! Internal column representation for heterogeneous data.

use ndarray::ArrayD;

use super::dtype::DType;

/// Internal enum representing a column of data in a Block.
///
/// This type is exposed in the public API but users typically don't need to
/// interact with it directly. Instead, use the type-specific getters like
/// `get_f32()`, `get_i64()`, etc.
#[derive(Clone)]
pub enum Column {
    /// 32-bit floating point column
    F32(ArrayD<f32>),
    /// 64-bit floating point column
    F64(ArrayD<f64>),
    /// 64-bit signed integer column
    I64(ArrayD<i64>),
    /// Boolean column
    Bool(ArrayD<bool>),
    /// 32-bit unsigned integer column
    U32(ArrayD<u32>),
    /// 8-bit unsigned integer column
    U8(ArrayD<u8>),
    /// String column
    String(ArrayD<String>),
}

impl Column {
    /// Returns the number of rows (axis-0 length) of this column.
    ///
    /// Returns `None` if the array has rank 0 (which should never happen
    /// in a valid Block, as rank-0 arrays are rejected during insertion).
    pub fn nrows(&self) -> Option<usize> {
        match self {
            Column::F32(a) => a.shape().first().copied(),
            Column::F64(a) => a.shape().first().copied(),
            Column::I64(a) => a.shape().first().copied(),
            Column::Bool(a) => a.shape().first().copied(),
            Column::U32(a) => a.shape().first().copied(),
            Column::U8(a) => a.shape().first().copied(),
            Column::String(a) => a.shape().first().copied(),
        }
    }

    /// Returns the data type of this column.
    pub fn dtype(&self) -> DType {
        match self {
            Column::F32(_) => DType::F32,
            Column::F64(_) => DType::F64,
            Column::I64(_) => DType::I64,
            Column::Bool(_) => DType::Bool,
            Column::U32(_) => DType::U32,
            Column::U8(_) => DType::U8,
            Column::String(_) => DType::String,
        }
    }

    /// Returns the shape of the underlying array.
    pub fn shape(&self) -> &[usize] {
        match self {
            Column::F32(a) => a.shape(),
            Column::F64(a) => a.shape(),
            Column::I64(a) => a.shape(),
            Column::Bool(a) => a.shape(),
            Column::U32(a) => a.shape(),
            Column::U8(a) => a.shape(),
            Column::String(a) => a.shape(),
        }
    }

    // Type-specific accessors for f32
    pub fn as_f32(&self) -> Option<&ArrayD<f32>> {
        match self {
            Column::F32(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut ArrayD<f32>> {
        match self {
            Column::F32(a) => Some(a),
            _ => None,
        }
    }

    // Type-specific accessors for f64
    pub fn as_f64(&self) -> Option<&ArrayD<f64>> {
        match self {
            Column::F64(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_f64_mut(&mut self) -> Option<&mut ArrayD<f64>> {
        match self {
            Column::F64(a) => Some(a),
            _ => None,
        }
    }

    // Type-specific accessors for i64
    pub fn as_i64(&self) -> Option<&ArrayD<i64>> {
        match self {
            Column::I64(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_i64_mut(&mut self) -> Option<&mut ArrayD<i64>> {
        match self {
            Column::I64(a) => Some(a),
            _ => None,
        }
    }

    // Type-specific accessors for bool
    pub fn as_bool(&self) -> Option<&ArrayD<bool>> {
        match self {
            Column::Bool(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_bool_mut(&mut self) -> Option<&mut ArrayD<bool>> {
        match self {
            Column::Bool(a) => Some(a),
            _ => None,
        }
    }

    // Type-specific accessors for u32
    pub fn as_u32(&self) -> Option<&ArrayD<u32>> {
        match self {
            Column::U32(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_u32_mut(&mut self) -> Option<&mut ArrayD<u32>> {
        match self {
            Column::U32(a) => Some(a),
            _ => None,
        }
    }

    // Type-specific accessors for u8
    pub fn as_u8(&self) -> Option<&ArrayD<u8>> {
        match self {
            Column::U8(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_u8_mut(&mut self) -> Option<&mut ArrayD<u8>> {
        match self {
            Column::U8(a) => Some(a),
            _ => None,
        }
    }

    // Type-specific accessors for String
    pub fn as_string(&self) -> Option<&ArrayD<String>> {
        match self {
            Column::String(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_string_mut(&mut self) -> Option<&mut ArrayD<String>> {
        match self {
            Column::String(a) => Some(a),
            _ => None,
        }
    }
}

impl std::fmt::Debug for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Column::F32(a) => write!(f, "Column::F32(shape={:?})", a.shape()),
            Column::F64(a) => write!(f, "Column::F64(shape={:?})", a.shape()),
            Column::I64(a) => write!(f, "Column::I64(shape={:?})", a.shape()),
            Column::Bool(a) => write!(f, "Column::Bool(shape={:?})", a.shape()),
            Column::U32(a) => write!(f, "Column::U32(shape={:?})", a.shape()),
            Column::U8(a) => write!(f, "Column::U8(shape={:?})", a.shape()),
            Column::String(a) => write!(f, "Column::String(shape={:?})", a.shape()),
        }
    }
}
