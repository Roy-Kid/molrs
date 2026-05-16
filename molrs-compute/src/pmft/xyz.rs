//! 3-D `(x, y, z)` Pair Mode Fourier Transform.
//!
//! Mirrors `freud.pmft.PMFTXYZ`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/PMFTXYZ.cc)).
//!
//! For every neighbor pair the bond vector `(dx, dy, dz)` is binned into
//! a 3-D histogram on
//! `[−x_max, x_max] × [−y_max, y_max] × [−z_max, z_max]`. PMF is
//! `−ln(ρ(x, y, z) / ρ_ref)` with `ρ_ref = N² / V_box`. Empty bins → `+∞`.
//!
//! First-pass implementation — no per-particle orientations yet (lab frame
//! only). Adding a rotating reference frame follows the same pattern as
//! [`LocalBondProjection`](super::super::environment::LocalBondProjection).

use molrs::frame_access::FrameAccess;
use molrs::neighbors::NeighborList;
use molrs::region::simbox::BoxKind;
use molrs::types::F;
use ndarray::Array3;

use crate::error::ComputeError;
use crate::result::ComputeResult;
use crate::traits::Compute;

/// Per-frame PMFTXYZ result.
#[derive(Debug, Clone, Default)]
pub struct PMFTXYZResult {
    pub density: Array3<F>,
    pub raw_counts: Array3<u64>,
    pub pmf: Array3<F>,
    pub x_edges: Vec<F>,
    pub y_edges: Vec<F>,
    pub z_edges: Vec<F>,
}

impl ComputeResult for PMFTXYZResult {}

/// `PMFTXYZ` analyzer.
#[derive(Debug, Clone, Copy)]
pub struct PMFTXYZ {
    x_max: F,
    y_max: F,
    z_max: F,
    n_x: usize,
    n_y: usize,
    n_z: usize,
}

impl PMFTXYZ {
    pub fn new(
        x_max: F,
        y_max: F,
        z_max: F,
        n_x: usize,
        n_y: usize,
        n_z: usize,
    ) -> Result<Self, ComputeError> {
        if x_max.is_nan()
            || y_max.is_nan()
            || z_max.is_nan()
            || x_max <= 0.0
            || y_max <= 0.0
            || z_max <= 0.0
        {
            return Err(ComputeError::OutOfRange {
                field: "PMFTXYZ ranges",
                value: format!("x_max={x_max}, y_max={y_max}, z_max={z_max}"),
            });
        }
        if n_x == 0 || n_y == 0 || n_z == 0 {
            return Err(ComputeError::OutOfRange {
                field: "PMFTXYZ bin counts",
                value: format!("n_x={n_x}, n_y={n_y}, n_z={n_z}"),
            });
        }
        Ok(Self {
            x_max,
            y_max,
            z_max,
            n_x,
            n_y,
            n_z,
        })
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &NeighborList,
    ) -> Result<PMFTXYZResult, ComputeError> {
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let (lx, ly, lz) = match simbox.kind() {
            BoxKind::Ortho { len, .. } => (len[0], len[1], len[2]),
            BoxKind::Triclinic => {
                return Err(ComputeError::OutOfRange {
                    field: "PMFTXYZ::simbox",
                    value: "triclinic boxes not supported".into(),
                });
            }
        };

        let dx = 2.0 * self.x_max / self.n_x as F;
        let dy = 2.0 * self.y_max / self.n_y as F;
        let dz = 2.0 * self.z_max / self.n_z as F;
        let bin_vol = dx * dy * dz;

        let mut counts = Array3::<u64>::zeros((self.n_x, self.n_y, self.n_z));
        let vectors = nlist.vectors();
        let n_pairs = nlist.n_pairs();
        let symmetric = matches!(nlist.mode(), molrs::neighbors::QueryMode::SelfQuery);

        let mut push = |vx: F, vy: F, vz: F| {
            if vx.abs() >= self.x_max || vy.abs() >= self.y_max || vz.abs() >= self.z_max {
                return;
            }
            let bx = (((vx + self.x_max) / dx) as usize).min(self.n_x - 1);
            let by = (((vy + self.y_max) / dy) as usize).min(self.n_y - 1);
            let bz = (((vz + self.z_max) / dz) as usize).min(self.n_z - 1);
            counts[[bx, by, bz]] += 1;
        };

        for k in 0..n_pairs {
            let vx = vectors[[k, 0]];
            let vy = vectors[[k, 1]];
            let vz = vectors[[k, 2]];
            push(vx, vy, vz);
            if symmetric {
                push(-vx, -vy, -vz);
            }
        }

        let n_q = nlist.num_query_points() as F;
        let n_p = nlist.num_points() as F;
        let n_pairs_total = if symmetric {
            n_p * (n_p - 1.0)
        } else {
            n_q * n_p
        };
        let v_box = lx * ly * lz;
        let rho_ref = if v_box > 0.0 {
            n_pairs_total / v_box
        } else {
            0.0
        };

        let shape = (self.n_x, self.n_y, self.n_z);
        let mut density = Array3::<F>::zeros(shape);
        let mut pmf = Array3::<F>::from_elem(shape, F::INFINITY);
        for ix in 0..self.n_x {
            for iy in 0..self.n_y {
                for iz in 0..self.n_z {
                    let rho = counts[[ix, iy, iz]] as F / bin_vol;
                    density[[ix, iy, iz]] = rho;
                    if rho > 0.0 && rho_ref > 0.0 {
                        pmf[[ix, iy, iz]] = -(rho / rho_ref).ln();
                    }
                }
            }
        }

        let x_edges: Vec<F> = (0..=self.n_x).map(|i| -self.x_max + i as F * dx).collect();
        let y_edges: Vec<F> = (0..=self.n_y).map(|i| -self.y_max + i as F * dy).collect();
        let z_edges: Vec<F> = (0..=self.n_z).map(|i| -self.z_max + i as F * dz).collect();

        Ok(PMFTXYZResult {
            density,
            raw_counts: counts,
            pmf,
            x_edges,
            y_edges,
            z_edges,
        })
    }
}

impl Compute for PMFTXYZ {
    type Args<'a> = &'a Vec<NeighborList>;
    type Output = Vec<PMFTXYZResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a Vec<NeighborList>,
    ) -> Result<Vec<PMFTXYZResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if frames.len() != nlists.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: nlists.len(),
                what: "neighbor-list count",
            });
        }
        let mut out = Vec::with_capacity(frames.len());
        for (f, nl) in frames.iter().zip(nlists.iter()) {
            out.push(self.one_frame(*f, nl)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use molrs::block::Block;
    use molrs::neighbors::{LinkCell, NbListAlgo};
    use molrs::region::simbox::SimBox;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F, pbc: [bool; 3]) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], pbc).unwrap());
        frame
    }

    fn build_nlist(frame: &Frame, cutoff: F) -> NeighborList {
        let xp = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(<F as molrs::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let yp = frame
            .get("atoms")
            .unwrap()
            .get("y")
            .and_then(<F as molrs::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let zp = frame
            .get("atoms")
            .unwrap()
            .get("z")
            .and_then(<F as molrs::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let n = xp.len();
        let mut pos = ndarray::Array2::<F>::zeros((n, 3));
        for i in 0..n {
            pos[[i, 0]] = xp[i];
            pos[[i, 1]] = yp[i];
            pos[[i, 2]] = zp[i];
        }
        let simbox = frame.simbox.as_ref().unwrap();
        let mut lc = LinkCell::new().cutoff(cutoff);
        lc.build(pos.view(), simbox);
        lc.query().clone()
    }

    #[test]
    fn antiparallel_bonds_populate_symmetric_bins() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [0.5, 0.4, 0.3]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.5);
        let r = &PMFTXYZ::new(1.0, 1.0, 1.0, 8, 8, 8)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        let total: u64 = r.raw_counts.iter().copied().sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn out_of_range_dropped() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 5.0);
        let r = &PMFTXYZ::new(1.0, 1.0, 1.0, 4, 4, 4)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        assert_eq!(r.raw_counts.iter().copied().sum::<u64>(), 0);
    }

    #[test]
    fn invalid_args_error() {
        assert!(PMFTXYZ::new(0.0, 1.0, 1.0, 4, 4, 4).is_err());
        assert!(PMFTXYZ::new(1.0, 1.0, 1.0, 0, 4, 4).is_err());
    }
}
