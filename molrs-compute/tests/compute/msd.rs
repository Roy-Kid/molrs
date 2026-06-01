//! End-to-end MSD integration tests against known analytical results.
//!
//! Inputs are trajectories built in code with prescribed motion. For uniform
//! ballistic drift `r_i(t) = r_i(0) + v_i·t` the displacement is exactly known
//! at every lag, so both Direct and Window modes have closed-form references.

use molrs::Frame;
use molrs::block::Block;
use molrs::types::F;
use ndarray::Array1;

use molrs_compute::msd::{MSD, MsdMode};
use molrs_compute::traits::Compute;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn make_frame(x: &[F], y: &[F], z: &[F]) -> Frame {
    let mut block = Block::new();
    block.insert("x", Array1::from_vec(x.to_vec()).into_dyn()).unwrap();
    block.insert("y", Array1::from_vec(y.to_vec()).into_dyn()).unwrap();
    block.insert("z", Array1::from_vec(z.to_vec()).into_dyn()).unwrap();
    let mut frame = Frame::new();
    frame.insert("atoms", block);
    frame
}

/// Trajectory of `n_frames` snapshots where particle `i` drifts at constant
/// velocity `vel[i]` along +x from the origin: `x_i(t) = vel[i]·t`.
fn ballistic_x(vel: &[F], n_frames: usize) -> Vec<Frame> {
    let n = vel.len();
    (0..n_frames)
        .map(|t| {
            let xs: Vec<F> = (0..n).map(|i| vel[i] * t as F).collect();
            let zeros = vec![0.0 as F; n];
            make_frame(&xs, &zeros, &zeros)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Direct mode (single reference = frame 0)
// ---------------------------------------------------------------------------

#[test]
fn direct_linear_motion_matches_t_squared_v_squared() {
    // MSD_direct(t) = ⟨|r(t) - r(0)|²⟩ = ⟨(v·t)²⟩ = t²·⟨v²⟩.
    let vel = [1.0, 2.0, 3.0];
    let n_frames = 6;
    let frames_owned = ballistic_x(&vel, n_frames);
    let frames: Vec<&Frame> = frames_owned.iter().collect();

    let series = MSD::new().compute(&frames, ()).unwrap();
    assert_eq!(series.len(), n_frames);

    let mean_v2: F = vel.iter().map(|v| v * v).sum::<F>() / vel.len() as F;
    for t in 0..n_frames {
        let expected = (t as F) * (t as F) * mean_v2;
        assert!(
            (series.data[t].mean - expected).abs() < 1e-10,
            "direct MSD[{t}] = {}, expected {expected}",
            series.data[t].mean
        );
    }
    // Reference frame has exactly zero displacement.
    assert!(series.data[0].mean.abs() < 1e-12);
}

#[test]
fn direct_per_particle_displacement_is_exact() {
    // Particle i at lag t moves v_i·t along x, so per_particle[i] = (v_i·t)².
    let vel = [1.0, 4.0];
    let frames_owned = ballistic_x(&vel, 4);
    let frames: Vec<&Frame> = frames_owned.iter().collect();

    let series = MSD::new().compute(&frames, ()).unwrap();
    for t in 0..4 {
        for (i, v) in vel.iter().enumerate() {
            let expected = (v * t as F).powi(2);
            assert!(
                (series.data[t].per_particle[i] - expected).abs() < 1e-10,
                "per_particle[t={t}][i={i}] = {}, expected {expected}",
                series.data[t].per_particle[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Window mode (all time origins, FFT)
// ---------------------------------------------------------------------------

#[test]
fn window_linear_motion_matches_t_squared_v_squared() {
    // For pure ballistic drift, every time origin gives the same displacement
    // over lag t (= v·t), so the all-origins average still equals t²·⟨v²⟩.
    let vel = [1.0, 2.0];
    let n_frames = 8;
    let frames_owned = ballistic_x(&vel, n_frames);
    let frames: Vec<&Frame> = frames_owned.iter().collect();

    let series = MSD::windowed().compute(&frames, ()).unwrap();
    assert_eq!(series.len(), n_frames);

    let mean_v2: F = vel.iter().map(|v| v * v).sum::<F>() / vel.len() as F;
    for t in 0..n_frames {
        let expected = (t as F) * (t as F) * mean_v2;
        assert!(
            (series.data[t].mean - expected).abs() < 1e-9,
            "window MSD[{t}] = {}, expected {expected}",
            series.data[t].mean
        );
    }
    assert!(series.data[0].mean.abs() < 1e-10, "lag 0 must be zero");
}

#[test]
fn direct_and_window_agree_on_constant_velocity() {
    // Both estimators must coincide for ballistic motion (no statistical
    // averaging difference when displacement is origin-independent).
    let vel = [0.5, 1.5, 2.5];
    let frames_owned = ballistic_x(&vel, 7);
    let frames: Vec<&Frame> = frames_owned.iter().collect();

    let direct = MSD::with_mode(MsdMode::Direct).compute(&frames, ()).unwrap();
    let window = MSD::with_mode(MsdMode::Window).compute(&frames, ()).unwrap();
    assert_eq!(direct.len(), window.len());
    for t in 0..direct.len() {
        assert!(
            (direct.data[t].mean - window.data[t].mean).abs() < 1e-9,
            "lag {t}: direct={}, window={}",
            direct.data[t].mean,
            window.data[t].mean
        );
    }
}

#[test]
fn empty_input_is_error() {
    let frames: Vec<&Frame> = Vec::new();
    let err = MSD::new().compute(&frames, ()).unwrap_err();
    assert!(matches!(
        err,
        molrs_compute::error::ComputeError::EmptyInput
    ));
}
