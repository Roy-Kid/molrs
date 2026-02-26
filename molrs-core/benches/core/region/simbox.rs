use criterion::{Criterion, criterion_group};
use molrs::core::region::simbox::SimBox;
use ndarray::array;

fn bench_shortest_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/region/simbox/shortest_vector");

    let bx = SimBox::cube(10.0, array![0.0_f32, 0.0, 0.0], [true, true, true])
        .expect("invalid box length");
    let a = [1.0_f32, 2.0, 3.0];
    let b = [8.5_f32, 9.0, 1.0];

    group.bench_function("pbc_cube", |bencher| {
        bencher.iter(|| std::hint::black_box(bx.shortest_vector_impl(a, b)));
    });

    let bx_no_pbc = SimBox::cube(10.0, array![0.0_f32, 0.0, 0.0], [false, false, false])
        .expect("invalid box length");

    group.bench_function("no_pbc_cube", |bencher| {
        bencher.iter(|| std::hint::black_box(bx_no_pbc.shortest_vector_impl(a, b)));
    });

    group.finish();
}

fn bench_make_fractional(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/region/simbox/make_fractional");

    let bx = SimBox::cube(10.0, array![0.0_f32, 0.0, 0.0], [true, true, true])
        .expect("invalid box length");
    let r = [3.5_f32, 7.2, 1.8];

    group.bench_function("cube", |bencher| {
        bencher.iter(|| std::hint::black_box(bx.make_fractional_impl(r)));
    });

    group.finish();
}

fn bench_calc_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/region/simbox/calc_distance");

    let bx = SimBox::cube(10.0, array![0.0_f32, 0.0, 0.0], [true, true, true])
        .expect("invalid box length");
    let a = [1.0_f32, 2.0, 3.0];
    let b = [8.5_f32, 9.0, 1.0];

    group.bench_function("pbc_cube", |bencher| {
        bencher.iter(|| std::hint::black_box(bx.calc_distance_impl(a, b)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_shortest_vector,
    bench_make_fractional,
    bench_calc_distance
);
