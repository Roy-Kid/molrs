//! Native periodic radical (Laguerre) Voronoi tessellation.
//!
//! Builds one [`VoronoiCells`] entry per generator by clipping a box-sized
//! convex polyhedron against the radical plane of every candidate periodic
//! neighbour ([`super::cell::Poly::clip`]). This is the cell-by-cell strategy
//! of voro++ (`container_periodic::compute_cell`, `src/v_container_prd.cpp`,
//! driven as in TRAVIS `vorowrapper.cpp`), with the radical/power plane offset
//! from `radius_poly` (`src/v_rad_option.h`).
//!
//! **Deliberate deviation from voro++:** voro++ keeps an incremental
//! vertex/edge structure and a sorted work-list of candidate particles for
//! speed; molrs clips against every candidate in a growing periodic shell
//! (simpler, WASM-clean, no FFI). The plane-cut semantics are identical, so
//! per-cell volumes, faces, areas and the neighbour relation match — verified
//! by `Σ volume == box volume` and face-area symmetry in the tests.
//!
//! # Radical plane
//! Between generators i, j at displacement `r = x_j − x_i` with radii `Rᵢ`,
//! `Rⱼ`, the radical plane is `|x|² − Rᵢ² = |x − r|² − Rⱼ²`, i.e.
//! `x·r = (|r|² + Rᵢ² − Rⱼ²)/2`; the cell keeps the generator side
//! (`x·r ≤ off`). Equal radii reduce to the plain Voronoi bisector at `r/2`.

use molrs::spatial::region::simbox::SimBox;
use molrs::types::F;
use ndarray::ArrayView2;

use super::cell::{BOUNDARY, Poly, VoronoiCells};
use crate::compute::error::ComputeError;

/// Builder for the periodic radical-Voronoi tessellation. Orthorhombic boxes
/// only (triclinic is out of scope; see the spec).
#[derive(Debug, Clone, Copy, Default)]
pub struct RadicalVoronoi;

/// Largest periodic shell searched before accepting a cell (a residual
/// box face at shell 3 means the configuration is pathologically sparse).
const MAX_SHELL: i32 = 3;
/// Area below which a residual boundary face is treated as numerical dust.
const FACE_EPS: F = 1e-9;

impl RadicalVoronoi {
    /// Tessellate `positions` (N×3, Å) with per-atom `radii` (length N; pass
    /// all-zero for a plain Voronoi diagram) in the periodic orthorhombic
    /// `simbox`.
    pub fn build(
        &self,
        positions: ArrayView2<F>,
        radii: &[F],
        simbox: &SimBox,
    ) -> Result<VoronoiCells, ComputeError> {
        let n = positions.nrows();
        if positions.ncols() != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: positions.ncols(),
                what: "positions columns",
            });
        }
        if radii.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: radii.len(),
                what: "radii length",
            });
        }
        let l = simbox.lengths();
        let (lx, ly, lz) = (l[0], l[1], l[2]);
        if lx <= 0.0 || ly <= 0.0 || lz <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "RadicalVoronoi::box_lengths",
                value: format!("{lx},{ly},{lz}"),
            });
        }

        let mut volumes = Vec::with_capacity(n);
        let mut faces = Vec::with_capacity(n);

        for i in 0..n {
            let gi = [positions[[i, 0]], positions[[i, 1]], positions[[i, 2]]];
            let ri2 = radii[i] * radii[i];

            // Grow the periodic search shell until no initial-box face
            // survives — guarantees the cell is bounded by real neighbours,
            // which is what makes Σ volume == box hold.
            let mut shell = 1;
            let cell_faces = loop {
                let mut poly = Poly::box_cell(lx, ly, lz);
                for j in 0..n {
                    let rj2 = radii[j] * radii[j];
                    for sx in -shell..=shell {
                        for sy in -shell..=shell {
                            for sz in -shell..=shell {
                                if i == j && sx == 0 && sy == 0 && sz == 0 {
                                    continue;
                                }
                                let dr = [
                                    positions[[j, 0]] + sx as F * lx - gi[0],
                                    positions[[j, 1]] + sy as F * ly - gi[1],
                                    positions[[j, 2]] + sz as F * lz - gi[2],
                                ];
                                let rr = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                                if rr < 1e-18 {
                                    continue;
                                }
                                let off = 0.5 * (rr + ri2 - rj2);
                                poly.clip(dr, off, j as i64);
                            }
                        }
                    }
                }
                let cf = poly.cell_faces();
                let unbounded = cf
                    .iter()
                    .any(|f| f.neighbor == BOUNDARY && f.area > FACE_EPS);
                if !unbounded || shell >= MAX_SHELL {
                    volumes.push(poly.volume());
                    break cf;
                }
                shell += 1;
            };
            faces.push(cell_faces);
        }

        Ok(VoronoiCells { volumes, faces })
    }
}
