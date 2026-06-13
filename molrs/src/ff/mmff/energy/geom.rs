//! Minimal 3-vector helpers for the MMFF energy terms.
//!
//! Coordinates are the flat `[x0,y0,z0,x1,y1,z1,...]` layout used by the
//! [`crate::ff::potential::Potential`] trait. These helpers keep the term
//! kernels readable without pulling in a linear-algebra dependency.

pub(super) type V3 = [f64; 3];

/// `coords[i]` as a 3-vector.
#[inline]
pub(super) fn pt(coords: &[f64], i: usize) -> V3 {
    [coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]]
}

#[inline]
pub(super) fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(super) fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(super) fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub(super) fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub(super) fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Clip a value into `[-1, 1]` (RDKit `clipToOne`).
#[inline]
pub(super) fn clip_to_one(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

/// Accumulate `g` into the gradient slot of atom `i`.
#[inline]
pub(super) fn add_grad(grad: &mut [f64], i: usize, g: V3) {
    grad[3 * i] += g[0];
    grad[3 * i + 1] += g[1];
    grad[3 * i + 2] += g[2];
}
