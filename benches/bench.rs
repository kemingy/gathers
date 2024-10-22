use criterion::criterion_main;
use criterion::{criterion_group, BenchmarkId, Criterion};
use rand::{thread_rng, Rng};

use gathers::distance::{
    native_argmin, native_dot_produce, native_l2_norm, native_squared_euclidean,
};
use gathers::simd::{argmin, dot_product, l2_norm, l2_squared_distance};

pub fn l2_norm_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("l2 norm");
    for dim in [64, 118, 124, 128, 512, 1024].into_iter() {
        let x: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        group.bench_with_input(BenchmarkId::new("native", dim), &x, |b, input| {
            b.iter(|| native_l2_norm(&input))
        });
        group.bench_with_input(BenchmarkId::new("simd", dim), &x, |b, input| {
            b.iter(|| unsafe { l2_norm(&input) })
        });
    }
    group.finish();
}

pub fn argmin_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("argmin");
    for dim in [64, 118, 124, 128, 512, 1024].into_iter() {
        let x: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        group.bench_with_input(BenchmarkId::new("native", dim), &x, |b, input| {
            b.iter(|| native_argmin(&input))
        });
        group.bench_with_input(BenchmarkId::new("simd", dim), &x, |b, input| {
            b.iter(|| unsafe { argmin(&input) })
        });
    }
}

pub fn l2_distance_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("l2 distance");
    for dim in [64, 118, 124, 128, 512, 1024].into_iter() {
        let lhs: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let rhs: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        group.bench_with_input(
            BenchmarkId::new("native", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| native_squared_euclidean(&input.0, &input.1)),
        );
        group.bench_with_input(BenchmarkId::new("simd", dim), &(&lhs, &rhs), |b, input| {
            b.iter(|| unsafe { l2_squared_distance(&input.0, &input.1) })
        });
    }
    group.finish();
}

pub fn ip_distance_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("dot product distance");
    for dim in [64, 118, 124, 128, 512, 1024].into_iter() {
        let lhs: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let rhs: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        group.bench_with_input(
            BenchmarkId::new("native", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| native_dot_produce(&input.0, &input.1)),
        );
        group.bench_with_input(BenchmarkId::new("simd", dim), &(&lhs, &rhs), |b, input| {
            b.iter(|| unsafe { dot_product(&input.0, &input.1) })
        });
    }
    group.finish();
}

criterion_group!(l2_benches, l2_distance_benchmark);
criterion_group!(ip_benches, ip_distance_benchmark);
criterion_group!(norm_benches, l2_norm_benchmark);
criterion_group!(argmin_benches, argmin_benchmark);
criterion_main!(l2_benches, ip_benches, norm_benches, argmin_benches);
