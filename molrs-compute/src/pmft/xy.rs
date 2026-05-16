//! 2-D `(x, y)` Pair Mode Fourier Transform.
//!
//! Mirrors `freud.pmft.PMFTXY`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/PMFTXY.cc)).
//!
//! For every neighbor pair the bond vector `(dx, dy)` (ignoring `dz`) is
//! binned into a 2-D histogram on `[−x_max, x_max] × [−y_max, y_max]`.
//! The result is the **PMF** estimate
//!
//! ```text
//!   PMF(x, y) = −ln( ρ(x, y) / ρ_ref )
//! ```
//!
//! where `ρ_ref` is the bulk number density (`N / (Lx · Ly)` for an
//! orthorhombic 2-D box). Empty bins return `+∞` for the PMF; the raw
//! density and per-bin counts are also exposed.
//!
//! This first-pass implementation does **not** apply per-particle
//! orientations (freud's `query_orientations` argument); a rotating
//! reference frame will be added when the downstream consumers need it.

use molrs::frame_access::FrameAccess;
use molrs::neighbors::NeighborList;
use molrs::region::simbox::BoxKind;
use molrs::types::F;
use ndarray::Array2;

use crate::error::ComputeError;
use crate::result::ComputeResult;
use crate::traits::Compute;

/// Per-frame PMFTXY result.
#[derive(Debug, Clone, Default)]
pub struct PMFTXYResult {
    /// Number-density histogram, `(n_x, n_y)`, normalised to the bin area
    /// and total expected pair count.
    pub density: Array2<F>,
    /// Raw pair counts per bin.
    pub raw_counts: Array2<u64>,
    /// Potential of mean force, `−ln(density / ρ_ref)`. Empty bins → `+∞`.
    pub pmf: Array2<F>,
    /// `x` bin edges (length `n_x + 1`).
    pub x_edges: Vec<F>,
    /// `y` bin edges (length `n_y + 1`).
    pub y_edges: Vec<F>,
}

impl ComputeResult for PMFTXYResult {}

/// `PMFTXY` analyzer.
#[derive(Debug, Clone, Copy)]
pub struct PMFTXY {
    x_max: F,
    y_max: F,
    n_x: usize,
    n_y: usize,
}

impl PMFTXY {
    pub fn new(x_max: F, y_max: F, n_x: usize, n_y: usize) -> Result<Self, ComputeError> {
        if x_max.is_nan() || x_max <= 0.0 || y_max.is_nan() || y_max <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "PMFTXY ranges",
                value: format!("x_max={x_max}, y_max={y_max}"),
            });
        }
        if n_x == 0 || n_y == 0 {
            return Err(ComputeError::OutOfRange {
                field: "PMFTXY bin counts",
                value: format!("n_x={n_x}, n_y={n_y}"),
            });
        }
        Ok(Self {
            x_max,
            y_max,
            n_x,
            n_y,
        })
    }

    pub fn x_max(&self) -> F {
        self.x_max
    }
    pub fn y_max(&self) -> F {
        self.y_max
    }
    pub fn n_x(&self) -> usize {
        self.n_x
    }
    pub fn n_y(&self) -> usize {
        self.n_y
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &NeighborList,
    ) -> Result<PMFTXYResult, ComputeError> {
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let (lx, ly) = match simbox.kind() {
            BoxKind::Ortho { len, .. } => (len[0], len[1]),
            BoxKind::Triclinic => {
                return Err(ComputeError::OutOfRange {
                    field: "PMFTXY::simbox",
                    value: "triclinic boxes not supported".into(),
                });
            }
        };

        let dx = 2.0 * self.x_max / self.n_x as F;
        let dy = 2.0 * self.y_max / self.n_y as F;
        let bin_area = dx * dy;

        let mut counts = Array2::<u64>::zeros((self.n_x, self.n_y));
        let vectors = nlist.vectors();
        let n_pairs = nlist.n_pairs();
        let symmetric = matches!(nlist.mode(), molrs::neighbors::QueryMode::SelfQuery);

        let mut push = |dxp: F, dyp: F| {
            if dxp.abs() >= self.x_max || dyp.abs() >= self.y_max {
                return;
            }
            let bx = (((dxp + self.x_max) / dx) as usize).min(self.n_x - 1);
            let by = (((dyp + self.y_max) / dy) as usize).min(self.n_y - 1);
            counts[[bx, by]] += 1;
        };

        for k in 0..n_pairs {
            let vx = vectors[[k, 0]];
            let vy = vectors[[k, 1]];
            push(vx, vy);
            if symmetric {
                push(-vx, -vy);
            }
        }

        // ρ_ref = N_query · (N_points − 1) / area  (per-pair density in the box).
        let n_q = nlist.num_query_points() as F;
        let n_p = nlist.num_points() as F;
        let n_pairs_total = if symmetric {
            n_p * (n_p - 1.0)
        } else {
            n_q * n_p
        };
        let area_box = lx * ly;
        let rho_ref = if area_box > 0.0 {
            n_pairs_total / area_box
        } else {
            0.0
        };

        let mut density = Array2::<F>::zeros((self.n_x, self.n_y));
        let mut pmf = Array2::<F>::from_elem((self.n_x, self.n_y), F::INFINITY);
        for ix in 0..self.n_x {
            for iy in 0..self.n_y {
                let rho = counts[[ix, iy]] as F / bin_area;
                density[[ix, iy]] = rho;
                if rho > 0.0 && rho_ref > 0.0 {
                    pmf[[ix, iy]] = -(rho / rho_ref).ln();
                }
            }
        }

        let x_edges: Vec<F> = (0..=self.n_x).map(|i| -self.x_max + i as F * dx).collect();
        let y_edges: Vec<F> = (0..=self.n_y).map(|i| -self.y_max + i as F * dy).collect();

        Ok(PMFTXYResult {
            density,
            raw_counts: counts,
            pmf,
            x_edges,
            y_edges,
        })
    }
}

impl Compute for PMFTXY {
    type Args<'a> = &'a Vec<NeighborList>;
    type Output = Vec<PMFTXYResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a Vec<NeighborList>,
    ) -> Result<Vec<PMFTXYResult>, ComputeError> {
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
    fn two_particles_land_in_symmetric_bins() {
        // Two particles separated by (1, 0). In a self-query NL the pair
        // is visited once but accumulated symmetrically, so we expect
        // count = 1 in (+1, 0) bin and count = 1 in (−1, 0) bin.
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.5);
        let r = &PMFTXY::new(2.0, 2.0, 8, 8)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        let total: u64 = r.raw_counts.iter().copied().sum();
        assert_eq!(total, 2);
        // Bin for (+1, 0): dx = 0.5 → bx = ((1 + 2)/0.5) = 6.
        let bx_pos = ((1.0_f64 + 2.0) / 0.5) as usize;
        let by_zero = ((0.0_f64 + 2.0) / 0.5) as usize;
        assert_eq!(r.raw_counts[[bx_pos, by_zero]], 1);
        let bx_neg = ((-1.0_f64 + 2.0) / 0.5) as usize;
        assert_eq!(r.raw_counts[[bx_neg, by_zero]], 1);
    }

    #[test]
    fn pmf_is_finite_only_in_occupied_bins() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.5);
        let r = &PMFTXY::new(2.0, 2.0, 4, 4)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        let mut finite_count = 0;
        for v in r.pmf.iter() {
            if v.is_finite() {
                finite_count += 1;
            }
        }
        assert_eq!(finite_count, 2);
    }

    #[test]
    fn out_of_range_pairs_dropped() {
        // Bond length 5 → outside the (x_max=2, y_max=2) box → no count.
        let frame = frame_with(&[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 6.0);
        let r = &PMFTXY::new(2.0, 2.0, 8, 8)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        assert_eq!(r.raw_counts.iter().copied().sum::<u64>(), 0);
    }

    #[test]
    fn invalid_args_error() {
        assert!(PMFTXY::new(0.0, 2.0, 4, 4).is_err());
        assert!(PMFTXY::new(2.0, 2.0, 0, 4).is_err());
    }

    #[test]
    fn empty_input_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = PMFTXY::new(2.0, 2.0, 4, 4)
            .unwrap()
            .compute(&frames, &Vec::<NeighborList>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }
}
