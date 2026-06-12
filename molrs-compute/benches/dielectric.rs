//! Benchmarks for dielectric susceptibility computations.
//!
//! Covers static dielectric constant (scalar + per-axis), Einstein-Helfand,
//! and Green-Kubo spectra across trajectory-length sweeps.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use molrs_compute::dielectric::{
    compute_current_density, compute_dipole_moment, einstein_helfand_spectrum, green_kubo_spectrum,
    static_dielectric_constant, static_dielectric_constant_components,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Duration;

// ── Sweep constants ──────────────────────────────────────────────────────────

const FRAME_COUNTS: &[usize] = &[200, 500, 2_000, 10_000, 50_000];
const VOLUME: f64 = 30_000.0; // Å³
const TEMPERATURE: f64 = 300.0; // K
const EPSILON_INF: f64 = 1.0;
const DT: f64 = 0.001; // ps
const MAX_CORRELATION: usize = 200;

fn configure<M: criterion::measurement::Measurement>(group: &mut criterion::BenchmarkGroup<'_, M>) {
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(15);
}

/// Generate a synthetic dipole trajectory with a known fluctuation amplitude.
///
/// `seed` ensures reproducibility across bench runs.
fn synthetic_dipoles(n_frames: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = Array2::zeros((n_frames, 3));
    for t in 0..n_frames {
        for d in 0..3 {
            m[[t, d]] = rng.random::<f64>() - 0.5; // centred at 0
        }
    }
    m
}

/// Generate a set of charges and positions that produce a fluctuating dipole.
fn synthetic_charges_positions(
    n_atoms: usize,
    n_frames: usize,
    seed: u64,
) -> (Array1<f64>, Vec<Array2<f64>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let charges: Array1<f64> = (0..n_atoms)
        .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
        .collect();
    let positions: Vec<Array2<f64>> = (0..n_frames)
        .map(|_| {
            let mut pos = Array2::zeros((n_atoms, 3));
            for i in 0..n_atoms {
                for d in 0..3 {
                    pos[[i, d]] = rng.random::<f64>() * 30.0;
                }
            }
            pos
        })
        .collect();
    (charges, positions)
}

// ── Static dielectric constant ───────────────────────────────────────────────

fn bench_static_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/static_scalar");
    configure(&mut group);

    for &nf in FRAME_COUNTS {
        let dm = synthetic_dipoles(nf, 42);
        group.throughput(Throughput::Elements(nf as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nf), &nf, |b, _| {
            b.iter(|| {
                std::hint::black_box(
                    static_dielectric_constant(&dm, VOLUME, TEMPERATURE, EPSILON_INF).unwrap(),
                );
            })
        });
    }

    group.finish();
}

fn bench_static_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/static_components");
    configure(&mut group);

    for &nf in FRAME_COUNTS {
        let dm = synthetic_dipoles(nf, 42);
        group.throughput(Throughput::Elements(nf as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nf), &nf, |b, _| {
            b.iter(|| {
                std::hint::black_box(
                    static_dielectric_constant_components(&dm, VOLUME, TEMPERATURE, EPSILON_INF)
                        .unwrap(),
                );
            })
        });
    }

    group.finish();
}

// ── Einstein-Helfand spectrum ────────────────────────────────────────────────

fn bench_einstein_helfand(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/einstein_helfand");
    configure(&mut group);

    for &nf in &FRAME_COUNTS[..4] {
        // Skip the largest for spectrum (4x FFT cost).
        let dm = synthetic_dipoles(nf, 42);
        group.throughput(Throughput::Elements(nf as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nf), &nf, |b, _| {
            b.iter(|| {
                std::hint::black_box(
                    einstein_helfand_spectrum(
                        &dm,
                        DT,
                        VOLUME,
                        TEMPERATURE,
                        EPSILON_INF,
                        MAX_CORRELATION,
                        "hann",
                    )
                    .unwrap(),
                );
            })
        });
    }

    group.finish();
}

// ── Green-Kubo spectrum ──────────────────────────────────────────────────────

fn bench_green_kubo(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/green_kubo");
    configure(&mut group);

    for &nf in &FRAME_COUNTS[..4] {
        let dm = synthetic_dipoles(nf, 42);
        let j = compute_current_density(&dm, DT, VOLUME).unwrap();
        group.throughput(Throughput::Elements(nf as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nf), &nf, |b, _| {
            b.iter(|| {
                std::hint::black_box(
                    green_kubo_spectrum(
                        &j,
                        DT,
                        VOLUME,
                        TEMPERATURE,
                        EPSILON_INF,
                        MAX_CORRELATION,
                        "hann",
                    )
                    .unwrap(),
                );
            })
        });
    }

    group.finish();
}

// ── Dipole moment computation ────────────────────────────────────────────────

fn bench_dipole_moment(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/dipole_moment");
    configure(&mut group);

    let n_atoms_sizes: &[usize] = &[100, 1_000, 10_000, 50_000];

    for &n in n_atoms_sizes {
        let (charges, positions) = synthetic_charges_positions(n, 1, 42);
        let pos = &positions[0];
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                std::hint::black_box(compute_dipole_moment(&charges, pos).unwrap());
            })
        });
    }

    group.finish();
}

// ── Current density ──────────────────────────────────────────────────────────

fn bench_current_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/current_density");
    configure(&mut group);

    for &nf in &FRAME_COUNTS[..4] {
        let dm = synthetic_dipoles(nf, 42);
        group.throughput(Throughput::Elements(nf as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nf), &nf, |b, _| {
            b.iter(|| {
                std::hint::black_box(compute_current_density(&dm, DT, VOLUME).unwrap());
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_static_scalar,
    bench_static_components,
    bench_einstein_helfand,
    bench_green_kubo,
    bench_dipole_moment,
    bench_current_density,
);
