//! TIP3P water box: pack → minimize → dynamics
//!
//! Demonstrates the full molrs MD workflow using a simplified TIP3P water model
//! (harmonic bonds/angles only, no LJ or electrostatics).
//!
//! Run: cargo run -p molrs-md --example tip3p
//!
//! # Unit conventions
//!
//! This example uses "LAMMPS real" units throughout:
//! - Energy:   kcal/mol
//! - Distance: Å
//! - Mass:     g/mol (Dalton)
//! - Time:     internal unit ≈ 48.9 fs  (Å·sqrt(g·mol/kcal))
//!
//! kB = 1 in the engine, so temperature is in energy units (kcal/mol).
//! To convert: T(K) = T(kcal/mol) / 0.001987204.

use molrs::core::block::Block;
use molrs::core::forcefield::ForceField;
use molrs::core::frame::Frame;
use molrs::core::types::F;
use molrs_md::{CPU, DumpZarr, DynamicsEngine, FixLangevin, FixNVE, FixThermo, MD};
use molrs_pack::{InsideBoxConstraint, MinDistConstraint, Molpack, Target};
use ndarray::Array1;

// ---------------------------------------------------------------------------
// TIP3P parameters (LAMMPS real units, from https://docs.lammps.org/Howto_tip3p.html)
// FF params stay f64 since ForceField stores them as f64
// ---------------------------------------------------------------------------
const MASS_O: f64 = 15.9994;
const MASS_H: f64 = 1.008;
const BOND_K: f64 = 450.0; // kcal/mol/Å²
const BOND_R0: f64 = 0.9572; // Å
const ANGLE_K: f64 = 55.0; // kcal/mol/rad²
const ANGLE_THETA0: f64 = 104.52; // degrees

// Simulation parameters (F precision)
const N_WATERS: usize = 50;
const BOX_SIDE: f32 = 11.44; // Å — cube side length (50 TIP3P at ~1.0 g/cm³)
const KB: F = 0.001987204; // kcal/(mol·K)
const TARGET_TEMP_K: F = 300.0;
const DT: F = 0.02; // ~1 fs

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -----------------------------------------------------------------------
    // 1. Build a single water molecule template (O, H, H)
    // -----------------------------------------------------------------------
    let water = build_water_template();

    // -----------------------------------------------------------------------
    // 2. Pack N_WATERS copies into a cubic box with Molpack
    // -----------------------------------------------------------------------
    println!(
        "=== Packing {} water molecules into {:.0} Å box ===",
        N_WATERS, BOX_SIDE
    );

    let target = Target::new(water, N_WATERS)
        .with_constraint(InsideBoxConstraint::cube_from_origin(
            BOX_SIDE,
            [0.0, 0.0, 0.0],
        ))
        .with_name("water");

    let mut packer = Molpack::new(None)
        .with_precision(1e-3)
        .with_maxit(20)
        .with_packall(false)
        .add_constraint(MinDistConstraint::new(2.0).with_short_range(1.0, 3.0));

    let packed = packer
        .pack(&[target], 300, Some(42))
        .map_err(|e| format!("Packing failed: {e}"))?;

    println!(
        "Packing: converged={}, iters={}, violation={:.4e}",
        packed.converged, packed.iterations, packed.final_violation
    );

    // -----------------------------------------------------------------------
    // 3. Use packed frame directly — topology and masses already replicated
    // -----------------------------------------------------------------------
    let frame = packed.frame;
    let n_atoms = frame.get("atoms").unwrap().nrows().unwrap_or(0);
    println!("System: {} atoms ({} waters)", n_atoms, n_atoms / 3);

    // -----------------------------------------------------------------------
    // 4. Define TIP3P force field (harmonic bond/angle only)
    // -----------------------------------------------------------------------
    let ff = build_tip3p_ff();

    // -----------------------------------------------------------------------
    // 5. Minimize
    // -----------------------------------------------------------------------
    println!("\n=== Energy minimization ===");

    let mut minimizer = MD::minimizer()
        .forcefield(&ff)
        .force_tol(1e-4)
        .max_iter(5000)
        .dmax(0.1)
        .compile::<CPU>(())?;

    let (minimized, min_result) = minimizer.run(&frame)?;
    println!(
        "Minimized: E={:.4} kcal/mol, converged={}, steps={}",
        min_result.energy as f64, min_result.converged, min_result.n_steps
    );

    // -----------------------------------------------------------------------
    // 6. Run MD: NVE integrator + Langevin thermostat
    // -----------------------------------------------------------------------
    println!(
        "\n=== MD: NVE + Langevin at {:.0} K ===",
        TARGET_TEMP_K as f64
    );

    let target_temp = TARGET_TEMP_K * KB; // 300 K → kcal/mol

    let traj_path = "tip3p_traj.zarr";
    let dump = DumpZarr::zarr(traj_path)
        .every(10)
        .with_positions()
        .with_velocities()
        .with_scalars(&["pe", "ke"])
        .with_box_h()
        .build();

    let mut dynamics = MD::dynamics()
        .forcefield(&ff)
        .dt(DT)
        .fix(FixNVE::new())
        .fix(FixLangevin::new(target_temp, 2.0, 12345))
        .fix(FixThermo::every(100))
        .dump(dump)
        .compile::<CPU>(())?;

    let state = dynamics.init(&minimized)?;

    // Equilibration
    println!("\n--- Equilibration (1000 steps) ---");
    let state = dynamics.run(1000, state)?;

    // Production
    println!("\n--- Production (2000 steps) ---");
    let state = dynamics.run(2000, state)?;

    dynamics.finish()?;

    let temp_k = state.temperature() / KB;
    println!(
        "\nFinal: step={}, PE={:.4}, KE={:.4}, T={:.1} K",
        state.step, state.pe as f64, state.ke as f64, temp_k as f64
    );

    println!("\nTrajectory written to: {}", traj_path);

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: build a single water molecule (O at origin, two H)
// ---------------------------------------------------------------------------
fn build_water_template() -> Frame {
    let half_angle = (ANGLE_THETA0 / 2.0_f64).to_radians();
    let hx = BOND_R0 * half_angle.cos();
    let hy = BOND_R0 * half_angle.sin();

    // O at origin, H1 and H2 symmetric about x-axis
    let mut frame = Frame::new();
    let mut atoms = Block::new();
    atoms
        .insert(
            "x",
            Array1::from_vec(vec![0.0_f32, hx as f32, hx as f32]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "y",
            Array1::from_vec(vec![0.0_f32, hy as f32, -hy as f32]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "z",
            Array1::from_vec(vec![0.0_f32, 0.0_f32, 0.0_f32]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "type",
            Array1::from_vec(vec!["OW".to_string(), "HW".to_string(), "HW".to_string()]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "element",
            Array1::from_vec(vec!["O".to_string(), "H".to_string(), "H".to_string()]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "mass",
            Array1::from_vec(vec![MASS_O, MASS_H, MASS_H]).into_dyn(),
        )
        .unwrap();
    frame.insert("atoms", atoms);

    // Bonds: O-H1 and O-H2
    let mut bonds = Block::new();
    bonds
        .insert("i", Array1::from_vec(vec![0_u32, 0]).into_dyn())
        .unwrap();
    bonds
        .insert("j", Array1::from_vec(vec![1_u32, 2]).into_dyn())
        .unwrap();
    bonds
        .insert(
            "type",
            Array1::from_vec(vec!["OW-HW".to_string(), "OW-HW".to_string()]).into_dyn(),
        )
        .unwrap();
    frame.insert("bonds", bonds);

    // Angles: H1-O-H2
    let mut angles = Block::new();
    angles
        .insert("i", Array1::from_vec(vec![1_u32]).into_dyn())
        .unwrap();
    angles
        .insert("j", Array1::from_vec(vec![0_u32]).into_dyn())
        .unwrap();
    angles
        .insert("k", Array1::from_vec(vec![2_u32]).into_dyn())
        .unwrap();
    angles
        .insert(
            "type",
            Array1::from_vec(vec!["HW-OW-HW".to_string()]).into_dyn(),
        )
        .unwrap();
    frame.insert("angles", angles);

    frame
}

// ---------------------------------------------------------------------------
// Helper: define TIP3P force field (bond + angle only, no LJ/Coulomb)
// ---------------------------------------------------------------------------
fn build_tip3p_ff() -> ForceField {
    let mut ff = ForceField::new("TIP3P-noCoulomb");

    // O-H bond: harmonic
    ff.def_bondstyle("harmonic")
        .def_type("OW-HW", &[("k", BOND_K), ("r0", BOND_R0)]);

    // H-O-H angle: harmonic
    ff.def_anglestyle("harmonic")
        .def_type("HW-OW-HW", &[("k", ANGLE_K), ("theta0", ANGLE_THETA0)]);

    ff
}
