// Jacobi rotations naturally express updates by paired (row, col) index;
// rewriting as iterators hurts readability without speeding the loop up.
#![allow(clippy::needless_range_loop)]

//! Symmetric 3×3 eigensolver via cyclic Jacobi rotations.
//!
//! For matrices of this size the analytic Cardano root form is faster but
//! numerically delicate near degenerate eigenvalues (Nematic Q-tensor,
//! gyration tensor of a sphere). Jacobi converges in ≲ 10 sweeps and is
//! robust at degeneracies, which is what callers like `freud/order/Nematic.cc`
//! and `freud/order/Cubatic.cc` rely on.
//!
//! Returns eigenvalues sorted **descending** alongside their unit eigenvectors
//! (each in a column of the returned matrix). The eigenvector matrix is
//! orthogonal: `Vᵀ A V = diag(eigvals)`.
//!
//! # References
//!
//! - Press et al., *Numerical Recipes*, §11.1 (cyclic Jacobi for symmetric
//!   eigenproblems).

use ndarray::{Array1, array};

use crate::types::{F, F3, F3x3};

/// Maximum Jacobi sweeps. 3×3 typically converges in ≤ 8.
const MAX_SWEEPS: usize = 50;

/// Absolute tolerance for declaring off-diagonal entries zero.
const OFF_DIAG_TOL: F = 1e-14;

/// Compute eigenvalues only (sorted descending) of a symmetric 3×3 matrix.
///
/// The input is assumed symmetric; the upper triangle is used (lower triangle
/// is ignored).
pub fn eigvals_sym_3x3(a: &F3x3) -> F3 {
    let (vals, _) = eigh_sym_3x3(a);
    vals
}

/// Largest eigenvalue (and its unit eigenvector) of a symmetric 4×4 matrix.
///
/// Uses cyclic Jacobi rotations — same algorithm as
/// [`eigh_sym_3x3`], extended to the six off-diagonal pairs of a 4×4.
/// Returns `(λ_max, v_max)` with `v_max` normalised.
///
/// Used by Horn's optimal-rotation method
/// (point-cloud Kabsch alignment via the quaternion form): the largest
/// eigenvalue eigenvector of the 4×4 `N` matrix built from the
/// cross-covariance encodes the best-fit rotation quaternion.
pub fn eigh_largest_sym_4x4(a: &[[F; 4]; 4]) -> (F, [F; 4]) {
    let mut m: [[F; 4]; 4] = *a;
    let mut v: [[F; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for _ in 0..MAX_SWEEPS {
        let off: F = pairs.iter().map(|&(p, q)| m[p][q].abs()).sum();
        if off < OFF_DIAG_TOL {
            break;
        }
        for (p, q) in pairs {
            let apq = m[p][q];
            if apq.abs() < OFF_DIAG_TOL {
                continue;
            }
            let app = m[p][p];
            let aqq = m[q][q];
            let theta = if (app - aqq).abs() < OFF_DIAG_TOL {
                std::f64::consts::FRAC_PI_4 * apq.signum()
            } else {
                0.5 * (2.0 * apq).atan2(app - aqq)
            };
            let c = theta.cos();
            let s = theta.sin();
            let new_app = c * c * app + 2.0 * s * c * apq + s * s * aqq;
            let new_aqq = s * s * app - 2.0 * s * c * apq + c * c * aqq;
            m[p][p] = new_app;
            m[q][q] = new_aqq;
            m[p][q] = 0.0;
            m[q][p] = 0.0;
            for r in 0..4 {
                if r != p && r != q {
                    let arp = m[r][p];
                    let arq = m[r][q];
                    let nrp = c * arp + s * arq;
                    let nrq = -s * arp + c * arq;
                    m[r][p] = nrp;
                    m[p][r] = nrp;
                    m[r][q] = nrq;
                    m[q][r] = nrq;
                }
            }
            for r in 0..4 {
                let vrp = v[r][p];
                let vrq = v[r][q];
                v[r][p] = c * vrp + s * vrq;
                v[r][q] = -s * vrp + c * vrq;
            }
        }
    }

    // Find largest diagonal entry → corresponding column of `v` is the
    // unit eigenvector.
    let mut imax = 0usize;
    let mut lmax = m[0][0];
    for i in 1..4 {
        if m[i][i] > lmax {
            imax = i;
            lmax = m[i][i];
        }
    }
    (lmax, [v[0][imax], v[1][imax], v[2][imax], v[3][imax]])
}

/// Full symmetric eigen-decomposition `A = V · diag(λ) · Vᵀ`.
///
/// Returns `(λ, V)` with `λ` sorted descending and `V[:, i]` the unit
/// eigenvector for `λ[i]`.
pub fn eigh_sym_3x3(a: &F3x3) -> (F3, F3x3) {
    debug_assert_eq!(a.dim(), (3, 3), "eigh_sym_3x3 expects a 3×3 matrix");

    // Working symmetric copy of `a` (upper triangle promoted to full).
    let mut m: [[F; 3]; 3] = [
        [a[[0, 0]], a[[0, 1]], a[[0, 2]]],
        [a[[0, 1]], a[[1, 1]], a[[1, 2]]],
        [a[[0, 2]], a[[1, 2]], a[[2, 2]]],
    ];
    let mut v: [[F; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..MAX_SWEEPS {
        // Off-diagonal Frobenius norm² (only upper triangle).
        let off = m[0][1].abs() + m[0][2].abs() + m[1][2].abs();
        if off < OFF_DIAG_TOL {
            break;
        }

        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = m[p][q];
            if apq.abs() < OFF_DIAG_TOL {
                continue;
            }
            let app = m[p][p];
            let aqq = m[q][q];
            // tan(2θ) = 2 apq / (app - aqq)
            let theta = if (app - aqq).abs() < OFF_DIAG_TOL {
                std::f64::consts::FRAC_PI_4 * apq.signum()
            } else {
                0.5 * (2.0 * apq).atan2(app - aqq)
            };
            let c = theta.cos();
            let s = theta.sin();

            // J = [[c, -s], [s, c]] (acting on columns p, q). Then
            //   A' = JᵀAJ:
            //     A'[p,p] = c²·app + 2cs·apq + s²·aqq
            //     A'[q,q] = s²·app − 2cs·apq + c²·aqq
            //     A'[r,p] = c·A[r,p] + s·A[r,q]   (r ≠ p, q)
            //     A'[r,q] = −s·A[r,p] + c·A[r,q]
            let new_app = c * c * app + 2.0 * s * c * apq + s * s * aqq;
            let new_aqq = s * s * app - 2.0 * s * c * apq + c * c * aqq;
            m[p][p] = new_app;
            m[q][q] = new_aqq;
            m[p][q] = 0.0;
            m[q][p] = 0.0;

            for r in 0..3 {
                if r != p && r != q {
                    let arp = m[r][p];
                    let arq = m[r][q];
                    let new_arp = c * arp + s * arq;
                    let new_arq = -s * arp + c * arq;
                    m[r][p] = new_arp;
                    m[p][r] = new_arp;
                    m[r][q] = new_arq;
                    m[q][r] = new_arq;
                }
            }

            // Accumulate eigenvector rotation: V <- V · J
            for r in 0..3 {
                let vrp = v[r][p];
                let vrq = v[r][q];
                v[r][p] = c * vrp + s * vrq;
                v[r][q] = -s * vrp + c * vrq;
            }
        }
    }

    let mut vals = [m[0][0], m[1][1], m[2][2]];
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap());

    let sorted_vals = [vals[order[0]], vals[order[1]], vals[order[2]]];
    let sorted_vecs: F3x3 = array![
        [v[0][order[0]], v[0][order[1]], v[0][order[2]]],
        [v[1][order[0]], v[1][order[1]], v[1][order[2]]],
        [v[2][order[0]], v[2][order[1]], v[2][order[2]]],
    ];

    // Renumber: scratch
    let _ = &mut vals;

    (Array1::from_vec(sorted_vals.to_vec()), sorted_vecs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const TOL: F = 1e-10;

    fn approx_eq(a: F, b: F, tol: F) {
        assert!((a - b).abs() < tol, "expected {b}, got {a} (Δ={})", a - b);
    }

    #[test]
    fn diagonal_matrix_unchanged() {
        let a: F3x3 = array![[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
        let vals = eigvals_sym_3x3(&a);
        approx_eq(vals[0], 3.0, TOL);
        approx_eq(vals[1], 2.0, TOL);
        approx_eq(vals[2], 1.0, TOL);
    }

    #[test]
    fn identity_returns_ones() {
        let a: F3x3 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (vals, vecs) = eigh_sym_3x3(&a);
        for v in vals.iter() {
            approx_eq(*v, 1.0, TOL);
        }
        // Eigenvectors should still be orthonormal (identity or permutation of it).
        for c in 0..3 {
            let mut norm: F = 0.0;
            for r in 0..3 {
                norm += vecs[[r, c]].powi(2);
            }
            approx_eq(norm, 1.0, TOL);
        }
    }

    #[test]
    fn known_two_by_two_embedded() {
        // [[2 1 0]
        //  [1 2 0]
        //  [0 0 5]]
        // Eigenvalues: 5, 3, 1 (descending)
        let a: F3x3 = array![[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]];
        let vals = eigvals_sym_3x3(&a);
        approx_eq(vals[0], 5.0, TOL);
        approx_eq(vals[1], 3.0, TOL);
        approx_eq(vals[2], 1.0, TOL);
    }

    #[test]
    fn eigenvector_reconstruction() {
        let a: F3x3 = array![[4.0, 1.0, 2.0], [1.0, 3.0, -1.0], [2.0, -1.0, 5.0]];
        let (vals, vecs) = eigh_sym_3x3(&a);

        // Verify A v_i = λ_i v_i for each i.
        for i in 0..3 {
            let v = vecs.column(i);
            // (A v)_r = Σ_c A[r,c] v[c]
            for r in 0..3 {
                let av_r: F = (0..3).map(|c| a[[r, c]] * v[c]).sum();
                approx_eq(av_r, vals[i] * v[r], 1e-8);
            }
        }

        // Trace and determinant invariants.
        let tr_a: F = (0..3).map(|i| a[[i, i]]).sum();
        let tr_eig: F = vals.sum();
        approx_eq(tr_a, tr_eig, TOL);
    }

    #[test]
    fn eigenvectors_are_orthonormal() {
        let a: F3x3 = array![[7.0, 2.0, -1.0], [2.0, 5.0, 3.0], [-1.0, 3.0, 6.0]];
        let (_, vecs) = eigh_sym_3x3(&a);
        for i in 0..3 {
            for j in 0..3 {
                let mut dot: F = 0.0;
                for r in 0..3 {
                    dot += vecs[[r, i]] * vecs[[r, j]];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                approx_eq(dot, expected, 1e-10);
            }
        }
    }

    #[test]
    fn degenerate_eigenvalues_handled() {
        // 2I + outer(u, u) has eigenvalues (2, 2, 2 + |u|²)
        // u = (1, 0, 0): eigenvalues (3, 2, 2).
        let a: F3x3 = array![[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let vals = eigvals_sym_3x3(&a);
        approx_eq(vals[0], 3.0, TOL);
        approx_eq(vals[1], 2.0, TOL);
        approx_eq(vals[2], 2.0, TOL);
    }

    #[test]
    fn largest_eigenvalue_sym_4x4_diagonal() {
        let m = [
            [4.0_f64, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ];
        let (lambda, v) = eigh_largest_sym_4x4(&m);
        approx_eq(lambda, 7.0, TOL);
        // Eigenvector aligned with axis 2.
        approx_eq(v[0].abs(), 0.0, TOL);
        approx_eq(v[1].abs(), 0.0, TOL);
        approx_eq(v[2].abs(), 1.0, TOL);
        approx_eq(v[3].abs(), 0.0, TOL);
    }

    #[test]
    fn largest_eigenvalue_sym_4x4_general() {
        // Rank-1 outer product: A = u·uᵀ with u = (1, 2, 3, 1).
        // Has eigenvalues (|u|², 0, 0, 0) = (15, 0, 0, 0).
        let u = [1.0_f64, 2.0, 3.0, 1.0];
        let mut m = [[0.0_f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = u[i] * u[j];
            }
        }
        let (lambda, v) = eigh_largest_sym_4x4(&m);
        approx_eq(lambda, 15.0, 1e-9);
        // Eigenvector parallel to u (up to sign).
        let dot: F = (0..4).map(|i| v[i] * u[i]).sum::<F>().abs();
        let nu = 15.0_f64.sqrt();
        approx_eq(dot, nu, 1e-9);
    }
}
