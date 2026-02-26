use criterion::{Criterion, criterion_group, criterion_main};
use molrs::core::block::Block;
use molrs::core::forcefield::ForceField;
use molrs::core::frame::Frame;
use molrs_md::{CPU, DynamicsEngine, FixNVE, MD};
use ndarray::Array1;

fn make_chain_frame(n_atoms: usize) -> Frame {
    let mut frame = Frame::new();

    // Place atoms in a line along x, spacing = 1.2 (near LJ minimum)
    let mut xs = Vec::with_capacity(n_atoms);
    let mut ys = Vec::with_capacity(n_atoms);
    let mut zs = Vec::with_capacity(n_atoms);
    let mut vxs = Vec::with_capacity(n_atoms);
    let mut vys = Vec::with_capacity(n_atoms);
    let mut vzs = Vec::with_capacity(n_atoms);
    let mut masses = Vec::with_capacity(n_atoms);

    for i in 0..n_atoms {
        xs.push(i as f64 * 1.2);
        ys.push(0.0_f64);
        zs.push(0.0_f64);
        vxs.push(if i % 2 == 0 { 0.1 } else { -0.1 });
        vys.push(0.0_f64);
        vzs.push(0.0_f64);
        masses.push(1.0_f64);
    }

    let mut atoms = Block::new();
    atoms.insert("x", Array1::from_vec(xs).into_dyn()).unwrap();
    atoms.insert("y", Array1::from_vec(ys).into_dyn()).unwrap();
    atoms.insert("z", Array1::from_vec(zs).into_dyn()).unwrap();
    atoms
        .insert("vx", Array1::from_vec(vxs).into_dyn())
        .unwrap();
    atoms
        .insert("vy", Array1::from_vec(vys).into_dyn())
        .unwrap();
    atoms
        .insert("vz", Array1::from_vec(vzs).into_dyn())
        .unwrap();
    atoms
        .insert("mass", Array1::from_vec(masses).into_dyn())
        .unwrap();
    frame.insert("atoms", atoms);

    // Pair interactions: nearest neighbors
    let n_pairs = n_atoms - 1;
    let pair_i: Vec<u32> = (0..n_pairs as u32).collect();
    let pair_j: Vec<u32> = (1..=n_pairs as u32).collect();
    let pair_type: Vec<String> = vec!["A".to_string(); n_pairs];

    let mut pairs = Block::new();
    pairs
        .insert("i", Array1::from_vec(pair_i).into_dyn())
        .unwrap();
    pairs
        .insert("j", Array1::from_vec(pair_j).into_dyn())
        .unwrap();
    pairs
        .insert("type", Array1::from_vec(pair_type).into_dyn())
        .unwrap();
    frame.insert("pairs", pairs);

    frame
}

fn make_lj_ff() -> ForceField {
    let mut ff = ForceField::new("bench");
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);
    ff
}

fn bench_dynamics_nve(c: &mut Criterion) {
    let n_atoms = 100;
    let n_steps = 100;
    let frame = make_chain_frame(n_atoms);
    let ff = make_lj_ff();

    c.bench_function(
        &format!("dynamics_nve_{n_atoms}atoms_{n_steps}steps"),
        |b| {
            b.iter(|| {
                let mut engine = MD::dynamics()
                    .forcefield(&ff)
                    .dt(0.001)
                    .fix(FixNVE::new())
                    .compile::<CPU>(())
                    .unwrap();
                let state = engine.init(&frame).unwrap();
                engine.run(n_steps, state).unwrap()
            })
        },
    );
}

criterion_group!(benches, bench_dynamics_nve);
criterion_main!(benches);
