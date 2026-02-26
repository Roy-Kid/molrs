use molrs::core::block::Block;
use molrs::core::forcefield::ForceField;
use molrs::core::frame::Frame;
use molrs::core::types::F;
use molrs_md::{
    CPU, DynamicsEngine, Fix, FixLangevin, FixNVE, FixNVT, MD, MDError, MDState, StageMask,
};
use ndarray::{Array1, ArrayD};

/// Get an F-typed column from a Block (works for both f32 and f64 builds).
#[cfg(not(feature = "f64"))]
fn get_f_col<'a>(block: &'a Block, key: &str) -> Option<&'a ArrayD<f32>> {
    block.get_f32(key)
}
#[cfg(feature = "f64")]
fn get_f_col<'a>(block: &'a Block, key: &str) -> Option<&'a ArrayD<f64>> {
    block.get_f64(key)
}

/// Helper: build a 2-atom LJ Frame with given positions and optional velocities.
#[allow(clippy::too_many_arguments)]
fn make_lj_frame_with_vel(
    x0: F,
    y0: F,
    z0: F,
    x1: F,
    y1: F,
    z1: F,
    vx0: F,
    vy0: F,
    vz0: F,
    vx1: F,
    vy1: F,
    vz1: F,
) -> Frame {
    let mut frame = Frame::new();

    let mut atoms = Block::new();
    atoms
        .insert("x", Array1::from_vec(vec![x0, x1]).into_dyn())
        .unwrap();
    atoms
        .insert("y", Array1::from_vec(vec![y0, y1]).into_dyn())
        .unwrap();
    atoms
        .insert("z", Array1::from_vec(vec![z0, z1]).into_dyn())
        .unwrap();
    atoms
        .insert("vx", Array1::from_vec(vec![vx0, vx1]).into_dyn())
        .unwrap();
    atoms
        .insert("vy", Array1::from_vec(vec![vy0, vy1]).into_dyn())
        .unwrap();
    atoms
        .insert("vz", Array1::from_vec(vec![vz0, vz1]).into_dyn())
        .unwrap();
    atoms
        .insert(
            "mass",
            Array1::from_vec(vec![1.0 as F, 1.0 as F]).into_dyn(),
        )
        .unwrap();
    frame.insert("atoms", atoms);

    let mut pairs = Block::new();
    pairs
        .insert("i", Array1::from_vec(vec![0_u32]).into_dyn())
        .unwrap();
    pairs
        .insert("j", Array1::from_vec(vec![1_u32]).into_dyn())
        .unwrap();
    pairs
        .insert("type", Array1::from_vec(vec!["A".to_string()]).into_dyn())
        .unwrap();
    frame.insert("pairs", pairs);

    frame
}

fn make_lj_ff() -> ForceField {
    let mut ff = ForceField::new("test");
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);
    ff
}

/// Test NVE energy conservation: 2-atom LJ, 1000 steps, small dt.
/// Total energy (PE + KE) should be conserved to high precision.
#[test]
fn test_nve_energy_conservation() {
    let ff = make_lj_ff();

    // Start near LJ minimum (r ≈ 1.12) with small initial velocity
    let r0 = 1.2;
    let frame = make_lj_frame_with_vel(0.0, 0.0, 0.0, r0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.0);

    let mut dynamics = MD::dynamics()
        .forcefield(&ff)
        .dt(0.001)
        .fix(FixNVE::new())
        .compile::<CPU>(())
        .expect("compile failed");

    let state = dynamics.init(&frame).expect("init failed");
    let e_initial = state.pe + state.ke;

    let state = dynamics.run(1000, state).expect("run failed");
    let e_final = state.pe + state.ke;

    let rel_error = ((e_final - e_initial) / e_initial).abs();
    assert!(
        rel_error < 1e-3,
        "NVE energy not conserved: E_init={}, E_final={}, rel_error={}",
        e_initial,
        e_final,
        rel_error
    );
}

/// Test that run(1000) gives identical results to run(500) + run(500).
#[test]
fn test_multi_segment_vs_single() {
    let ff = make_lj_ff();
    let r0 = 1.2;
    let frame = make_lj_frame_with_vel(0.0, 0.0, 0.0, r0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.0);

    // Method 1: single run(1000)
    let mut dynamics1 = MD::dynamics()
        .forcefield(&ff)
        .dt(0.001)
        .fix(FixNVE::new())
        .compile::<CPU>(())
        .expect("compile failed");

    let state1 = dynamics1.init(&frame).expect("init failed");
    let state1 = dynamics1.run(1000, state1).expect("run failed");
    let result1 = state1.to_frame(&frame);

    // Method 2: run(500) + run(500)
    let mut dynamics2 = MD::dynamics()
        .forcefield(&ff)
        .dt(0.001)
        .fix(FixNVE::new())
        .compile::<CPU>(())
        .expect("compile failed");

    let state2 = dynamics2.init(&frame).expect("init failed");
    let state2 = dynamics2.run(500, state2).expect("run failed");
    let state2 = dynamics2.run(500, state2).expect("run failed");
    let result2 = state2.to_frame(&frame);

    // Compare positions
    let atoms1 = result1.get("atoms").unwrap();
    let atoms2 = result2.get("atoms").unwrap();
    let x1 = get_f_col(atoms1, "x").unwrap();
    let x2 = get_f_col(atoms2, "x").unwrap();

    for i in 0..2 {
        assert!(
            (x1[i] - x2[i]).abs() < 1e-6,
            "Position mismatch at atom {}: {} vs {}",
            i,
            x1[i],
            x2[i]
        );
    }
}

/// Test custom Fix: verify post_force modification takes effect.
struct FixExternalForce {
    fx: F,
}

impl Fix for FixExternalForce {
    fn name(&self) -> &str {
        "external_force"
    }

    fn stages(&self) -> StageMask {
        StageMask::POST_FORCE
    }

    fn post_force(&mut self, s: &mut MDState) -> Result<(), MDError> {
        for i in 0..s.n_atoms {
            s.f[3 * i] += self.fx;
        }
        Ok(())
    }
}

#[test]
fn test_custom_fix_post_force() {
    let ff = make_lj_ff();

    // Start at LJ minimum, zero velocity
    let r_min = 2.0_f32.powf(1.0 / 6.0) as F;
    let frame =
        make_lj_frame_with_vel(0.0, 0.0, 0.0, r_min, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    // Run with external force pushing both atoms in +x
    let mut dynamics = MD::dynamics()
        .forcefield(&ff)
        .dt(0.001)
        .fix(FixNVE::new())
        .fix(FixExternalForce { fx: 1.0 })
        .compile::<CPU>(())
        .expect("compile failed");

    let state = dynamics.init(&frame).expect("init failed");
    let state = dynamics.run(100, state).expect("run failed");
    let result = state.to_frame(&frame);

    let atoms = result.get("atoms").unwrap();
    let x = get_f_col(atoms, "x").unwrap();

    // Both atoms should have moved in +x direction due to external force
    assert!(x[0] > 0.0, "atom 0 should move in +x, got {}", x[0]);
    assert!(
        x[1] > r_min,
        "atom 1 should move in +x, got {} (started at {})",
        x[1],
        r_min
    );
}

/// Test multi-segment run: equilibrate, then produce.
#[test]
fn test_multi_segment_run() {
    let ff = make_lj_ff();
    let r0 = 1.2;
    let frame = make_lj_frame_with_vel(0.0, 0.0, 0.0, r0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.0);

    let mut dynamics = MD::dynamics()
        .forcefield(&ff)
        .dt(0.001)
        .fix(FixNVE::new())
        .compile::<CPU>(())
        .expect("compile failed");

    let state = dynamics.init(&frame).expect("init failed");

    // Segment 1
    let state = dynamics.run(500, state).expect("segment 1 failed");
    assert_eq!(state.step, 500);
    let e_after_500 = state.pe + state.ke;

    // Segment 2 (state is preserved)
    let state = dynamics.run(500, state).expect("segment 2 failed");
    assert_eq!(state.step, 1000);
    let e_after_1000 = state.pe + state.ke;

    // Energy should still be conserved across segments
    let rel_error = ((e_after_1000 - e_after_500) / e_after_500).abs();
    assert!(
        rel_error < 1e-3,
        "Energy drift across segments: rel_error={}",
        rel_error
    );
}

/// Test missing forcefield is rejected.
#[test]
fn test_dynamics_missing_forcefield() {
    let result = MD::dynamics()
        .dt(0.001)
        .fix(FixNVE::new())
        .compile::<CPU>(());
    assert!(result.is_err());
}

/// Test output frame contains velocity and energy metadata.
#[test]
fn test_output_frame_metadata() {
    let ff = make_lj_ff();
    let r0 = 1.2;
    let frame = make_lj_frame_with_vel(0.0, 0.0, 0.0, r0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.0);

    let mut dynamics = MD::dynamics()
        .forcefield(&ff)
        .dt(0.001)
        .fix(FixNVE::new())
        .compile::<CPU>(())
        .expect("compile failed");

    let state = dynamics.init(&frame).expect("init failed");
    let state = dynamics.run(100, state).expect("run failed");
    let result = state.to_frame(&frame);

    // Check metadata
    assert!(result.meta.contains_key("pe"));
    assert!(result.meta.contains_key("ke"));
    assert!(result.meta.contains_key("total_energy"));
    assert!(result.meta.contains_key("step"));

    let step: usize = result.meta.get("step").unwrap().parse().unwrap();
    assert_eq!(step, 100);

    // Check velocities in output
    let atoms = result.get("atoms").unwrap();
    assert!(get_f_col(atoms, "vx").is_some());
    assert!(get_f_col(atoms, "vy").is_some());
    assert!(get_f_col(atoms, "vz").is_some());
}

/// Test NVT (Nosé-Hoover) thermostat: average temperature should converge
/// to the target temperature.
#[test]
fn test_nvt_temperature_convergence() {
    let ff = make_lj_ff();
    let target_temp: F = 1.0;
    let r0 = 1.2;
    let frame = make_lj_frame_with_vel(0.0, 0.0, 0.0, r0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0);

    let mut dynamics = MD::dynamics()
        .forcefield(&ff)
        .dt(0.001)
        .fix(FixNVT::new(target_temp, 0.1))
        .compile::<CPU>(())
        .expect("compile failed");

    let state = dynamics.init(&frame).expect("init failed");

    // Equilibrate
    let mut state = dynamics.run(10000, state).expect("equilibration failed");

    // Collect temperature samples
    let n_samples = 40000;
    let sample_interval = 10;
    let mut temp_sum: F = 0.0;
    let mut n_collected = 0;
    for _ in 0..n_samples {
        state = dynamics.run(sample_interval, state).expect("run failed");
        let temp = 2.0 * state.ke / state.n_dof as F;
        temp_sum += temp;
        n_collected += 1;
    }
    let avg_temp = temp_sum / n_collected as F;

    let rel_error = ((avg_temp - target_temp) / target_temp).abs();
    assert!(
        rel_error < 0.15,
        "NVT avg_temp={:.4}, target={}, rel_error={:.4}",
        avg_temp,
        target_temp,
        rel_error
    );
}

/// Test Langevin thermostat: NVE + FixLangevin should thermalize
/// to the target temperature.
#[test]
fn test_langevin_temperature_convergence() {
    let ff = make_lj_ff();
    let target_temp: F = 1.0;
    let r0 = 1.2;
    let frame = make_lj_frame_with_vel(0.0, 0.0, 0.0, r0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0);

    let mut dynamics = MD::dynamics()
        .forcefield(&ff)
        .dt(0.001)
        .fix(FixNVE::new())
        .fix(FixLangevin::new(target_temp, 0.1, 42))
        .compile::<CPU>(())
        .expect("compile failed");

    let state = dynamics.init(&frame).expect("init failed");

    // Equilibrate
    let mut state = dynamics.run(10000, state).expect("equilibration failed");

    // Collect temperature samples
    let n_samples = 40000;
    let sample_interval = 10;
    let mut temp_sum: F = 0.0;
    let mut n_collected = 0;
    for _ in 0..n_samples {
        state = dynamics.run(sample_interval, state).expect("run failed");
        let temp = 2.0 * state.ke / state.n_dof as F;
        temp_sum += temp;
        n_collected += 1;
    }
    let avg_temp = temp_sum / n_collected as F;

    let rel_error = ((avg_temp - target_temp) / target_temp).abs();
    assert!(
        rel_error < 0.15,
        "Langevin avg_temp={:.4}, target={}, rel_error={:.4}",
        avg_temp,
        target_temp,
        rel_error
    );
}

/// Test multi-segment consistency: run(1000) should give identical results
/// to init() + run(500) + run(500).
#[test]
fn test_multi_segment_consistency() {
    let ff = make_lj_ff();
    let r0 = 1.2;
    let frame = make_lj_frame_with_vel(0.0, 0.0, 0.0, r0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.0);

    // Method 1: single run(1000)
    let mut dynamics1 = MD::dynamics()
        .forcefield(&ff)
        .dt(0.001)
        .fix(FixNVE::new())
        .compile::<CPU>(())
        .expect("compile failed");

    let state1 = dynamics1.init(&frame).expect("init failed");
    let state1 = dynamics1.run(1000, state1).expect("run failed");
    let result1 = state1.to_frame(&frame);

    // Method 2: init + run(500) + run(500)
    let mut dynamics2 = MD::dynamics()
        .forcefield(&ff)
        .dt(0.001)
        .fix(FixNVE::new())
        .compile::<CPU>(())
        .expect("compile failed");

    let state2 = dynamics2.init(&frame).expect("init failed");
    let state2 = dynamics2.run(500, state2).expect("segment 1 failed");
    let state2 = dynamics2.run(500, state2).expect("segment 2 failed");
    let result2 = state2.to_frame(&frame);

    // Compare positions
    let atoms1 = result1.get("atoms").unwrap();
    let atoms2 = result2.get("atoms").unwrap();
    let x1 = get_f_col(atoms1, "x").unwrap();
    let x2 = get_f_col(atoms2, "x").unwrap();

    for i in 0..2 {
        assert!(
            (x1[i] - x2[i]).abs() < 1e-6,
            "Position mismatch at atom {}: {} vs {}",
            i,
            x1[i],
            x2[i]
        );
    }

    // Compare velocities
    let vx1 = get_f_col(atoms1, "vx").unwrap();
    let vx2 = get_f_col(atoms2, "vx").unwrap();
    for i in 0..2 {
        assert!(
            (vx1[i] - vx2[i]).abs() < 1e-6,
            "Velocity mismatch at atom {}: {} vs {}",
            i,
            vx1[i],
            vx2[i]
        );
    }
}
