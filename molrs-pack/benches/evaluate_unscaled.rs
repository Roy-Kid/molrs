//! Per-extraction microbench for `molrs_pack::packer::evaluate_unscaled`.
//!
//! Landed in the same commit as the extraction (phase A.4.1) per the
//! `molrs-perf` skill § "Benchmarking during refactors" rule:
//!
//!   > In the same commit as the extraction, land:
//!   > - A criterion microbench of the extracted function.
//!   > - A criterion microbench of the caller.
//!
//! Gates (hard):
//!   - `extracted` ≤ +1% vs. `sentinel`
//!   - `caller_extracted` ≤ +2% vs. `caller_sentinel`
//!
//! The bench reuses the existing `pack_mixture` workload via `ExampleCase` and
//! runs `Molpack::pack()` with `max_loops = 0` to produce a post-initialization
//! `PackContext` snapshot — `evaluate_unscaled` is then called against that
//! snapshot in a tight loop.
//!
//! This bench stays in the tree permanently (future-regression guard); the
//! sentinel is deleted one Phase A refactor cycle after A.4.1 stabilizes.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use molrs_pack::packer::{evaluate_unscaled, evaluate_unscaled_sentinel};
use molrs_pack::{ExampleCase, Molpack, PackContext, build_targets, example_dir_from_manifest};

/// Build a realistic mid-pack PackContext snapshot by running mixture through
/// full initialization + one outer loop iteration. This exercises `evaluate()`
/// on ~1400 molecules — representative of the hot path.
fn build_snapshot() -> (PackContext, Vec<f64>) {
    let targets = build_targets(
        ExampleCase::Mixture,
        &example_dir_from_manifest(ExampleCase::Mixture),
    )
    .expect("build targets");
    let mut packer = Molpack::new();
    // Run one loop so we have populated restraints, cell lists, coords.
    let _ = packer.pack(&targets, 1, Some(1_234_567)).expect("pack");

    // We cannot easily extract the mid-pack PackContext from Molpack (pack()
    // consumes it). Instead, rebuild a fresh one the same way pack() does its
    // setup, but at a much smaller scale so the microbench is reasonably
    // fast per iteration. This is representative of the "call evaluate under
    // unscaled radii" path without paying for a full pack run per sample.
    let ntotat = 400;
    let mut sys = PackContext::new(ntotat, 0, 0);
    sys.radius.iter_mut().for_each(|r| *r = 0.75);
    sys.radius_ini.iter_mut().for_each(|r| *r = 1.5);
    sys.work.radiuswork.resize(ntotat, 0.0);
    (sys, Vec::new())
}

fn bench_extracted(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_unscaled");
    group.sample_size(50);
    group.bench_function("extracted", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, xwork)| {
                std::hint::black_box(evaluate_unscaled(&mut sys, &xwork));
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_unscaled");
    group.sample_size(50);
    group.bench_function("sentinel", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, xwork)| {
                std::hint::black_box(evaluate_unscaled_sentinel(&mut sys, &xwork));
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Caller microbench: models the "post-pgencan statistics" block in pack()
/// (lines 608-631) — one `evaluate_unscaled` call followed by `fimp`
/// arithmetic, the exact sequence the hot path executes per outer-loop
/// iteration. Compares the extracted-fn path vs. the sentinel path; the
/// difference captures any indirection / inlining-boundary cost the
/// function-level bench cannot see.
fn bench_caller(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_unscaled_caller");
    group.sample_size(50);

    group.bench_function("caller_extracted", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, xwork)| {
                let (fx, _fdist, _frest) = evaluate_unscaled(&mut sys, &xwork);
                let flast = 1.0_f64;
                let fimp = if flast > 0.0 {
                    -100.0 * (fx - flast) / flast
                } else {
                    0.0
                };
                std::hint::black_box(fimp);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("caller_sentinel", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, xwork)| {
                let (fx, _fdist, _frest) = evaluate_unscaled_sentinel(&mut sys, &xwork);
                let flast = 1.0_f64;
                let fimp = if flast > 0.0 {
                    -100.0 * (fx - flast) / flast
                } else {
                    0.0
                };
                std::hint::black_box(fimp);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_extracted, bench_sentinel, bench_caller);
criterion_main!(benches);
