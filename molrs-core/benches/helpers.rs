//! Shared test fixtures for benchmarks.

use molrs::core::region::simbox::SimBox;
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Generate N random f32 points inside a cubic box.
pub fn random_points(n: usize, box_size: f32, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pts = Array2::<f32>::zeros((n, 3));
    for i in 0..n {
        pts[[i, 0]] = rng.random::<f32>() * box_size;
        pts[[i, 1]] = rng.random::<f32>() * box_size;
        pts[[i, 2]] = rng.random::<f32>() * box_size;
    }
    pts
}

/// Create a non-PBC cubic SimBox.
#[allow(dead_code)]
pub fn make_simbox(size: f32) -> SimBox {
    SimBox::cube(size, array![0.0_f32, 0.0, 0.0], [false, false, false])
        .expect("invalid box length")
}

/// Create a PBC cubic SimBox.
pub fn make_pbc_simbox(size: f32) -> SimBox {
    SimBox::cube(size, array![0.0_f32, 0.0, 0.0], [true, true, true]).expect("invalid box length")
}

/// Generate N random `[f32; 3]` points inside a cubic box (native Vec layout).
#[allow(dead_code)]
pub fn random_points_native(n: usize, box_size: f32, seed: u64) -> Vec<[f32; 3]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            [
                rng.random::<f32>() * box_size,
                rng.random::<f32>() * box_size,
                rng.random::<f32>() * box_size,
            ]
        })
        .collect()
}

/// Generate a flat `Vec<f32>` of length `3*n` (MDState-style layout).
#[allow(dead_code)]
pub fn random_flat(n: usize, box_size: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * 3).map(|_| rng.random::<f32>() * box_size).collect()
}

/// Generate a 1D `Vec<f32>` of length `n` with values in [0, 1).
#[allow(dead_code)]
pub fn random_1d_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random::<f32>()).collect()
}

/// Generate a 1D `Array1<f32>` of length `n` with values in [0, 1).
#[allow(dead_code)]
pub fn random_1d_ndarray(n: usize, seed: u64) -> Array1<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array1::from_iter((0..n).map(|_| rng.random::<f32>()))
}
