//! Data type enumeration and trait for Block columns.

use ndarray::ArrayD;

use super::column::Column;

/// Supported data types for Block columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 64-bit signed integer
    I64,
    /// Boolean
    Bool,
    /// 32-bit unsigned integer
    U32,
    /// 8-bit unsigned integer
    U8,
    /// String
    String,
}

impl DType {
    /// Returns the name of the data type as a string.
    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I64 => "i64",
            DType::Bool => "bool",
            DType::U32 => "u32",
            DType::U8 => "u8",
            DType::String => "string",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Trait for types that can be stored in a Block column.
///
/// This trait provides the mechanism for generic dispatch when inserting
/// arrays into a Block. Users don't need to interact with this trait directly.
pub trait BlockDtype: Sized + 'static {
    /// Returns the DType for this type.
    fn dtype() -> DType;

    /// Converts an ArrayD of this type into a Column.
    fn into_column(arr: ArrayD<Self>) -> Column;

    /// Tries to extract a reference to an ArrayD of this type from a Column.
    fn from_column(col: &Column) -> Option<&ArrayD<Self>>;

    /// Tries to extract a mutable reference to an ArrayD of this type from a Column.
    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>>;
}

impl BlockDtype for f32 {
    fn dtype() -> DType {
        DType::F32
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::F32(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_f32()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_f32_mut()
    }
}

impl BlockDtype for f64 {
    fn dtype() -> DType {
        DType::F64
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::F64(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_f64()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_f64_mut()
    }
}

impl BlockDtype for i64 {
    fn dtype() -> DType {
        DType::I64
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::I64(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_i64()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_i64_mut()
    }
}

impl BlockDtype for bool {
    fn dtype() -> DType {
        DType::Bool
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::Bool(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_bool()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_bool_mut()
    }
}

impl BlockDtype for u32 {
    fn dtype() -> DType {
        DType::U32
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::U32(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_u32()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_u32_mut()
    }
}

impl BlockDtype for u8 {
    fn dtype() -> DType {
        DType::U8
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::U8(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_u8()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_u8_mut()
    }
}

impl BlockDtype for String {
    fn dtype() -> DType {
        DType::String
    }

    fn into_column(arr: ArrayD<Self>) -> Column {
        Column::String(arr)
    }

    fn from_column(col: &Column) -> Option<&ArrayD<Self>> {
        col.as_string()
    }

    fn from_column_mut(col: &mut Column) -> Option<&mut ArrayD<Self>> {
        col.as_string_mut()
    }
}
