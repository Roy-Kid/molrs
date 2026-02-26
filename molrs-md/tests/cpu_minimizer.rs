use molrs::core::block::Block;
use molrs::core::forcefield::ForceField;
use molrs::core::frame::Frame;
use molrs::core::types::F;
use molrs_md::{CPU, MD};
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

/// Helper: build a 2-atom Frame with given coordinates and a pairs block.
fn make_lj_frame(x0: F, y0: F, z0: F, x1: F, y1: F, z1: F) -> Frame {
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

/// Helper: build a 2-atom Frame with a bonds block.
fn make_bond_frame(x0: F, y0: F, z0: F, x1: F, y1: F, z1: F) -> Frame {
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
    frame.insert("atoms", atoms);

    let mut bonds = Block::new();
    bonds
        .insert("i", Array1::from_vec(vec![0_u32]).into_dyn())
        .unwrap();
    bonds
        .insert("j", Array1::from_vec(vec![1_u32]).into_dyn())
        .unwrap();
    bonds
        .insert("type", Array1::from_vec(vec!["A-B".to_string()]).into_dyn())
        .unwrap();
    frame.insert("bonds", bonds);

    frame
}

#[test]
fn test_minimize_2atom_lj() {
    // Define forcefield with unified def_type
    let mut ff = ForceField::new("test");
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

    // Build frame: 2 atoms at r=2.0
    let frame = make_lj_frame(0.0, 0.0, 0.0, 2.0, 0.0, 0.0);

    let (frame_out, result) = MD::minimizer()
        .forcefield(&ff)
        .force_tol(1e-3) // relaxed for f32 precision
        .energy_tol(1e-6) // backup convergence criterion
        .max_iter(5000)
        .compile::<CPU>(())
        .expect("compile failed")
        .run(&frame)
        .expect("run failed");

    // Check coordinates in output frame
    let atoms = frame_out.get("atoms").unwrap();
    let x = get_f_col(atoms, "x").unwrap();
    let y = get_f_col(atoms, "y").unwrap();
    let z = get_f_col(atoms, "z").unwrap();

    let dx = x[1] - x[0];
    let dy = y[1] - y[0];
    let dz = z[1] - z[0];
    let r_final = (dx * dx + dy * dy + dz * dz).sqrt();

    // LJ minimum at r = 2^(1/6)
    let r_min: F = 2.0_f32.powf(1.0 / 6.0) as F;
    assert!(
        (r_final - r_min).abs() < 0.01,
        "final distance {} should be near LJ minimum {}",
        r_final,
        r_min
    );

    // Check result (energy is F, compare with tolerance)
    let energy = result.energy as f64;
    assert!(
        (energy - (-1.0)).abs() < 0.01,
        "final energy {} should be near -1.0",
        energy
    );
    assert!(result.converged);
    assert!(result.n_evals > 0);
}

#[test]
fn test_minimize_bond_harmonic() {
    let mut ff = ForceField::new("test");
    ff.def_bondstyle("harmonic")
        .def_type("A-B", &[("k", 300.0), ("r0", 1.5)]);

    let frame = make_bond_frame(0.0, 0.0, 0.0, 2.5, 0.0, 0.0);

    let (frame_out, result) = MD::minimizer()
        .forcefield(&ff)
        .force_tol(1e-4) // relaxed for f32 precision
        .max_iter(5000)
        .compile::<CPU>(())
        .expect("compile failed")
        .run(&frame)
        .expect("run failed");

    let atoms = frame_out.get("atoms").unwrap();
    let x = get_f_col(atoms, "x").unwrap();
    let dx = x[1] - x[0];
    let r_final = dx.abs(); // along x-axis

    assert!(
        (r_final - 1.5).abs() < 0.01,
        "final distance {} should be near r0=1.5",
        r_final
    );

    let energy = result.energy as f64;
    assert!(
        energy.abs() < 0.01,
        "final energy {} should be near 0",
        energy
    );
    assert!(result.converged);
}

#[test]
fn test_minimize_energy_tol() {
    let mut ff = ForceField::new("test");
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

    let frame = make_lj_frame(0.0, 0.0, 0.0, 2.0, 0.0, 0.0);

    // Use energy_tol only (set force_tol to 0 to disable)
    let (_frame_out, result) = MD::minimizer()
        .forcefield(&ff)
        .energy_tol(1e-8)
        .force_tol(0.0)
        .max_iter(5000)
        .compile::<CPU>(())
        .expect("compile failed")
        .run(&frame)
        .expect("run failed");

    let energy = result.energy as f64;
    assert!(
        (energy - (-1.0)).abs() < 0.01,
        "final energy {} should be near -1.0",
        energy
    );
    assert!(result.converged);
}

#[test]
fn test_minimize_not_converged() {
    let mut ff = ForceField::new("test");
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

    // Start far from minimum with very few iterations
    let frame = make_lj_frame(0.0, 0.0, 0.0, 5.0, 0.0, 0.0);

    let (_frame_out, result) = MD::minimizer()
        .forcefield(&ff)
        .force_tol(1e-12)
        .max_iter(2)
        .compile::<CPU>(())
        .expect("compile failed")
        .run(&frame)
        .expect("run failed");

    assert!(!result.converged);
    assert!(result.n_steps <= 2);
}

#[test]
fn test_minimize_max_eval_limit() {
    let mut ff = ForceField::new("test");
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

    let frame = make_lj_frame(0.0, 0.0, 0.0, 2.0, 0.0, 0.0);

    // Allow many iterations but very few evaluations
    let (_frame_out, result) = MD::minimizer()
        .forcefield(&ff)
        .force_tol(1e-12)
        .max_iter(5000)
        .max_eval(5)
        .compile::<CPU>(())
        .expect("compile failed")
        .run(&frame)
        .expect("run failed");

    assert!(
        result.n_evals <= 6,
        "n_evals {} should be near max_eval 5",
        result.n_evals
    );
    // With so few evals, unlikely to converge
    assert!(!result.converged);
}

#[test]
fn test_missing_forcefield_rejected() {
    let result = MD::minimizer().compile::<CPU>(());
    assert!(result.is_err());
}
