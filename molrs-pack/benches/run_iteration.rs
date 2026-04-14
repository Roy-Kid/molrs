//! Per-extraction microbench for `molrs_pack::packer::run_iteration`.
//!
//! Landed in the same commit as the extraction (phase A.4.3) per the
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
//! Setup: empty-molecule PackContext identical to the unit test in
//! `packer::tests::run_iteration_matches_sentinel_on_empty_context`. With
//! `ntotmol=0` the body's pgencan/evaluate/movebad branches run on empty
//! vectors — this is intentional: the gate measures *function-call boundary
//! cost* (indirection, inlining decisions) on a trivial body, which is exactly
//! what the sentinel sibling controls for. A full-workload bench lives in
//! `benches/pack_end_to_end.rs` (catastrophic-regression alarm, ≤ +10%).
//!
//! This bench stays permanently; the sentinel is deleted one Phase A cycle
//! after A.4.3 stabilizes.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use molrs_pack::gencan::{GencanParams, GencanWorkspace};
use molrs_pack::handler::{Handler, PhaseInfo};
use molrs_pack::initial::SwapState;
use molrs_pack::movebad::MoveBadConfig;
use molrs_pack::packer::{IterOutcome, run_iteration, run_iteration_sentinel};
use molrs_pack::relaxer::RelaxerRunner;
use molrs_pack::{F, PackContext};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Snapshot = (
    PackContext,
    Vec<F>,
    SwapState,
    GencanWorkspace,
    Vec<(usize, Vec<Box<dyn RelaxerRunner>>)>,
    Vec<Box<dyn Handler>>,
    SmallRng,
);

fn build_snapshot() -> Snapshot {
    let ntotat = 4;
    let mut sys = PackContext::new(ntotat, 0, 0);
    sys.radius.fill(0.75);
    sys.radius_ini.fill(1.5);
    sys.work.radiuswork.resize(ntotat, 0.0);
    let x: Vec<F> = Vec::new();
    let swap = SwapState::init(&x, &sys);
    let ws = GencanWorkspace::new();
    let runners: Vec<(usize, Vec<Box<dyn RelaxerRunner>>)> = Vec::new();
    let handlers: Vec<Box<dyn Handler>> = Vec::new();
    let rng = SmallRng::seed_from_u64(1_234_567);
    (sys, x, swap, ws, runners, handlers, rng)
}

fn phase_info() -> PhaseInfo {
    PhaseInfo {
        phase: 0,
        total_phases: 1,
        molecule_type: None,
    }
}

fn movebad_cfg() -> MoveBadConfig<'static> {
    MoveBadConfig {
        movefrac: 0.05,
        maxmove_per_type: &[],
        movebadrandom: false,
        gencan_maxit: 20,
    }
}

fn gencan_params() -> GencanParams {
    GencanParams::default()
}

fn bench_extracted(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_iteration");
    group.sample_size(50);
    let pi = phase_info();
    let mb = movebad_cfg();
    let gp = gencan_params();
    group.bench_function("extracted", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, mut x, mut swap, mut ws, mut runners, mut handlers, mut rng)| {
                let mut flast = 0.0_f64;
                let mut fimp_prev = F::INFINITY;
                let mut radscale = 1.0_f64;
                let out = run_iteration(
                    0,
                    10,
                    true,
                    0,
                    pi,
                    0.01,
                    true,
                    &mb,
                    &gp,
                    &mut sys,
                    &mut x,
                    &mut swap,
                    &mut flast,
                    &mut fimp_prev,
                    &mut radscale,
                    &mut runners,
                    &mut handlers,
                    &mut ws,
                    &mut rng,
                );
                std::hint::black_box(out);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_iteration");
    group.sample_size(50);
    let pi = phase_info();
    let mb = movebad_cfg();
    let gp = gencan_params();
    group.bench_function("sentinel", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, mut x, mut swap, mut ws, mut runners, mut handlers, mut rng)| {
                let mut flast = 0.0_f64;
                let mut fimp_prev = F::INFINITY;
                let mut radscale = 1.0_f64;
                let out = run_iteration_sentinel(
                    0,
                    10,
                    true,
                    0,
                    pi,
                    0.01,
                    true,
                    &mb,
                    &gp,
                    &mut sys,
                    &mut x,
                    &mut swap,
                    &mut flast,
                    &mut fimp_prev,
                    &mut radscale,
                    &mut runners,
                    &mut handlers,
                    &mut ws,
                    &mut rng,
                );
                std::hint::black_box(out);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Caller microbench: models the phase's `for loop_idx in 0..max_loops`
/// scaffold (packer.rs lines 530-557) calling `run_iteration` once and
/// matching on the `IterOutcome` variant. The difference between
/// `caller_extracted` and `caller_sentinel` captures any indirection /
/// inlining-boundary cost the function-level bench cannot see.
fn bench_caller(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_iteration_caller");
    group.sample_size(50);
    let pi = phase_info();
    let mb = movebad_cfg();
    let gp = gencan_params();

    group.bench_function("caller_extracted", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, mut x, mut swap, mut ws, mut runners, mut handlers, mut rng)| {
                let mut flast = 0.0_f64;
                let mut fimp_prev = F::INFINITY;
                let mut radscale = 1.0_f64;
                let mut converged = false;
                for loop_idx in 0..10 {
                    let out = run_iteration(
                        loop_idx,
                        10,
                        true,
                        0,
                        pi,
                        0.01,
                        true,
                        &mb,
                        &gp,
                        &mut sys,
                        &mut x,
                        &mut swap,
                        &mut flast,
                        &mut fimp_prev,
                        &mut radscale,
                        &mut runners,
                        &mut handlers,
                        &mut ws,
                        &mut rng,
                    );
                    match out {
                        IterOutcome::Continue => {}
                        IterOutcome::Converged => {
                            converged = true;
                            break;
                        }
                        IterOutcome::EarlyStop => break,
                    }
                }
                std::hint::black_box(converged);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("caller_sentinel", |b| {
        b.iter_batched(
            build_snapshot,
            |(mut sys, mut x, mut swap, mut ws, mut runners, mut handlers, mut rng)| {
                let mut flast = 0.0_f64;
                let mut fimp_prev = F::INFINITY;
                let mut radscale = 1.0_f64;
                let mut converged = false;
                for loop_idx in 0..10 {
                    let out = run_iteration_sentinel(
                        loop_idx,
                        10,
                        true,
                        0,
                        pi,
                        0.01,
                        true,
                        &mb,
                        &gp,
                        &mut sys,
                        &mut x,
                        &mut swap,
                        &mut flast,
                        &mut fimp_prev,
                        &mut radscale,
                        &mut runners,
                        &mut handlers,
                        &mut ws,
                        &mut rng,
                    );
                    match out {
                        IterOutcome::Continue => {}
                        IterOutcome::Converged => {
                            converged = true;
                            break;
                        }
                        IterOutcome::EarlyStop => break,
                    }
                }
                std::hint::black_box(converged);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_extracted, bench_sentinel, bench_caller);
criterion_main!(benches);
