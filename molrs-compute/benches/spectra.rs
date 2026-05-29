//! Benchmarks for vibrational spectra computations.
//!
//! Covers power spectrum (VDOS), IR spectrum, and Raman spectrum
//! across trajectory-length and DOF sweeps.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use molrs_compute::spectra::{ir_spectrum, power_spectrum, raman_spectrum};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

// ── Sweep constants ──────────────────────────────────────────────────────────

const FRAME_COUNTS: &[usize] = &[200, 500, 2_000, 10_000];
const DT_FS: f64 = 0.5; // fs
const RESOLUTION: usize = 100;

fn configure<M: criterion::measurement::Measurement>(group: &mut criterion::BenchmarkGroup<'_, M>) {
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(15);
}

/// Generate synthetic velocity time series with oscillatory content.
fn synthetic_velocities(n_frames: usize, n_dof: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v = Array2::zeros((n_frames, n_dof));
    for d in 0..n_dof {
        let freq = 1.0 + rng.random::<f64>() * 50.0; // 1-51 THz
        let phase = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
        for t in 0..n_frames {
            let time = t as f64 * DT_FS;
            v[[t, d]] = (2.0 * std::f64::consts::PI * freq * 1e-3 * time + phase).sin();
        }
    }
    v
}

/// Generate synthetic dipole moment time series.
fn synthetic_dipoles(n_frames: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = Array2::zeros((n_frames, 3));
    for d in 0..3 {
        let freq = 1.0 + rng.random::<f64>() * 30.0; // THz
        let phase = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
        for t in 0..n_frames {
            let time = t as f64 * DT_FS;
            m[[t, d]] = (2.0 * std::f64::consts::PI * freq * 1e-3 * time + phase).sin();
        }
    }
    m
}

/// Generate synthetic polarizability time series in Voigt notation.
fn synthetic_polarizabilities(n_frames: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut alpha = Array2::zeros((n_frames, 6));
    for c in 0..6 {
        let freq = 1.0 + rng.random::<f64>() * 20.0;
        let phase = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
        for t in 0..n_frames {
            let time = t as f64 * DT_FS;
            alpha[[t, c]] = (2.0 * std::f64::consts::PI * freq * 1e-3 * time + phase).sin();
        }
    }
    alpha
}

// ── Power spectrum (VDOS) ────────────────────────────────────────────────────

fn bench_power_spectrum_frames(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectra/power_spectrum/n_frames");
    configure(&mut group);
    let n_dof = 30; // 10 atoms × 3

    for &nf in FRAME_COUNTS {
        let v = synthetic_velocities(nf, n_dof, 42);
        group.throughput(Throughput::Elements(nf as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nf), &nf, |b, _| {
            b.iter(|| {
                std::hint::black_box(power_spectrum(&v, DT_FS, RESOLUTION).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_power_spectrum_dofs(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectra/power_spectrum/n_dof");
    configure(&mut group);
    let n_frames = 2_000;
    let dof_sizes: &[usize] = &[3, 30, 300, 3_000];

    for &nd in dof_sizes {
        let v = synthetic_velocities(n_frames, nd, 99);
        group.throughput(Throughput::Elements((n_frames * nd) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nd), &nd, |b, _| {
            b.iter(|| {
                std::hint::black_box(power_spectrum(&v, DT_FS, RESOLUTION).unwrap());
            })
        });
    }

    group.finish();
}

// ── IR spectrum ──────────────────────────────────────────────────────────────

fn bench_ir_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectra/ir_spectrum/n_frames");
    configure(&mut group);

    for &nf in FRAME_COUNTS {
        let dm = synthetic_dipoles(nf, 42);
        group.throughput(Throughput::Elements(nf as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nf), &nf, |b, _| {
            b.iter(|| {
                std::hint::black_box(ir_spectrum(&dm, DT_FS, RESOLUTION).unwrap());
            })
        });
    }

    group.finish();
}

// ── Raman spectrum ───────────────────────────────────────────────────────────

fn bench_raman_spectrum_not_averaged(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectra/raman_spectrum/not_averaged");
    configure(&mut group);

    for &nf in &FRAME_COUNTS[..3] {
        // Raman is the heaviest — skip largest.
        let alpha = synthetic_polarizabilities(nf, 42);
        group.throughput(Throughput::Elements(nf as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nf), &nf, |b, _| {
            b.iter(|| {
                std::hint::black_box(
                    raman_spectrum(&alpha, DT_FS, RESOLUTION, 0.0, 300.0, false).unwrap(),
                );
            })
        });
    }

    group.finish();
}

fn bench_raman_spectrum_averaged(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectra/raman_spectrum/averaged");
    configure(&mut group);

    for &nf in &FRAME_COUNTS[..3] {
        let alpha = synthetic_polarizabilities(nf, 42);
        group.throughput(Throughput::Elements(nf as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nf), &nf, |b, _| {
            b.iter(|| {
                std::hint::black_box(
                    raman_spectrum(&alpha, DT_FS, RESOLUTION, 10000.0, 300.0, true).unwrap(),
                );
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_power_spectrum_frames,
    bench_power_spectrum_dofs,
    bench_ir_spectrum,
    bench_raman_spectrum_not_averaged,
    bench_raman_spectrum_averaged,
);
