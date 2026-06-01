//! Triangle-inequality bounds smoothing.
//!
//! Direct port of RDKit's `DistGeom::triangleSmoothBounds`
//! (`$RDBASE/Code/DistGeom/TriangleSmooth.cpp`, BSD-3, Copyright (C)
//! 2004-2025 Greg Landrum and other RDKit contributors), itself the O(N³)
//! algorithm from Crippen & Havel, *Distance Geometry and Molecular
//! Conformation* (1988), pp. 252-253.
//!
//! RDKit does **not** apply tetrangle (four-point) smoothing in the default
//! `GetMoleculeBoundsMatrix` path — only the triangle pass — so to stay
//! byte-comparable we port the triangle pass exactly and document that
//! tetrangle is intentionally omitted (it is not part of the reference
//! matrix this port is validated against).

use molrs::error::MolRsError;

use super::matrix::BoundsMatrix;

/// Triangle-smooth `bounds` in place (RDKit `triangleSmoothBounds`, `tol`).
///
/// Returns `Err` when the bounds are inconsistent (a lower bound exceeds the
/// corresponding upper bound by more than `tol`), exactly like RDKit's
/// boolean return mapped onto our error type.
pub fn smooth_bounds_tol(bounds: &mut BoundsMatrix, tol: f64) -> Result<(), MolRsError> {
    let npt = bounds.len();
    for k in 0..npt {
        for i in 0..npt.saturating_sub(1) {
            if i == k {
                continue;
            }
            let (ii, ik) = if i > k { (k, i) } else { (i, k) };
            let u_ik = bounds.raw(ii, ik); // upper(i,k)
            let l_ik = bounds.raw(ik, ii); // lower(i,k)
            for j in (i + 1)..npt {
                if j == k {
                    continue;
                }
                let (jj, jk) = if j > k { (k, j) } else { (j, k) };
                let u_kj = bounds.raw(jj, jk); // upper(j,k)
                let sum_u = u_ik + u_kj;
                if bounds.raw(i, j) > sum_u {
                    bounds.set_raw(i, j, sum_u);
                }

                let diff_lik_ukj = l_ik - u_kj;
                let diff_ljk_uik = bounds.raw(jk, jj) - u_ik; // lower(j,k) - upper(i,k)
                if bounds.raw(j, i) < diff_lik_ukj {
                    bounds.set_raw(j, i, diff_lik_ukj);
                } else if bounds.raw(j, i) < diff_ljk_uik {
                    bounds.set_raw(j, i, diff_ljk_uik);
                }

                let l_bound = bounds.raw(j, i); // lower(i,j)
                let u_bound = bounds.raw(i, j); // upper(i,j)
                if tol > 0.0
                    && (l_bound - u_bound) / l_bound > 0.0
                    && (l_bound - u_bound) / l_bound < tol
                {
                    bounds.set_raw(i, j, l_bound);
                } else if l_bound - u_bound > 0.0 {
                    return Err(MolRsError::validation(
                        "distance-geometry bounds are inconsistent (lower > upper) during triangle smoothing",
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Triangle-smooth `bounds` with `tol = 0` (the `GetMoleculeBoundsMatrix`
/// default).
pub fn smooth_bounds(bounds: &mut BoundsMatrix) -> Result<(), MolRsError> {
    smooth_bounds_tol(bounds, 0.0)
}
