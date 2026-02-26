//! Common numeric and geometry type aliases used across the crate.
//!
//! The **F-prefix family** provides a consistent naming convention for
//! ndarray-backed types parameterized by the float precision [`F`].

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Primary floating-point scalar type.
///
/// Defaults to `f32`. Enable the `f64` feature for double precision,
/// which is recommended for packing (GENCAN optimizer needs >6 significant
/// digits in finite-difference Hessian-vector products).
#[cfg(not(feature = "f64"))]
pub type F = f32;
#[cfg(feature = "f64")]
pub type F = f64;

// ---- Fixed-size 3D types ----

/// 3-element vector (position, velocity, force, displacement).
pub type F3 = Array1<F>;

/// 3×3 matrix (box matrix, rotation, stress tensor).
pub type F3x3 = Array2<F>;

// ---- Variable-size types ----

/// N-element vector.
pub type FN = Array1<F>;

/// N×3 matrix (collection of 3D vectors).
pub type FNx3 = Array2<F>;

// ---- Views ----

/// Borrowed view of a 3-element vector.
pub type F3View<'a> = ArrayView1<'a, F>;

/// Borrowed N×3 view.
pub type FNx3View<'a> = ArrayView2<'a, F>;

// ---- Non-float ----

/// Per-axis periodic boundary condition flags.
pub type Pbc3 = [bool; 3];

// ---- Backward-compatibility aliases ----
// These will be removed once all consumers are updated.

/// 3D vector backed by ndarray. Prefer [`F3`].
pub type Vec3 = F3;

/// 3×3 matrix backed by ndarray. Prefer [`F3x3`].
pub type Mat3 = F3x3;

/// N×3 matrix of points. Prefer [`FNx3`].
pub type PointsNx3f = FNx3;

/// Axis-aligned bounds as 3×2 array. Prefer [`FNx3`].
pub type Bounds3f = FNx3;

/// 3D point backed by ndarray. Prefer [`F3`].
pub type Point3f = F3;

// ---- Raw array types (used by molrs-pack, to be removed in Phase 6) ----

/// 3D coordinate as a fixed-size array.
pub type Coord3 = [F; 3];

/// 3×3 rotation matrix, row-major.
pub type RotMat3 = [[F; 3]; 3];

/// Variable-length list of 3D coordinates.
pub type Coords = Vec<Coord3>;

/// Axis-aligned bounding box: `[[min, max]; 3]` per axis.
pub type Bounds3 = [[F; 2]; 3];
