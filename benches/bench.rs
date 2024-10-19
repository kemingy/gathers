use criterion::criterion_main;
use criterion::{criterion_group, BenchmarkId, Criterion};
use rand::{thread_rng, Rng};

use gathers::distance::{native_neg_dot_produce, native_squared_euclidean};
use gathers::simd::{l2_squared_distance, neg_dot_product};

pub fn l2_distance_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("distance");
    for dim in [64, 118, 124, 128, 512, 1024].into_iter() {
        let lhs: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let rhs: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        group.bench_with_input(
            BenchmarkId::new("native l2", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| native_squared_euclidean(&input.0, &input.1)),
        );
        group.bench_with_input(
            BenchmarkId::new("simd l2", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| unsafe { l2_squared_distance(&input.0, &input.1) }),
        );
    }
    group.finish();
}

pub fn ip_distance_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();

    let mut group = c.benchmark_group("distance");
    for dim in [64, 118, 124, 128, 512, 1024].into_iter() {
        let lhs: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let rhs: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        group.bench_with_input(
            BenchmarkId::new("native ip", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| native_neg_dot_produce(&input.0, &input.1)),
        );
        group.bench_with_input(
            BenchmarkId::new("simd ip", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| unsafe { neg_dot_product(&input.0, &input.1) }),
        );
    }
    group.finish();
}

criterion_group!(l2_benches, l2_distance_benchmark);
criterion_group!(ip_benches, ip_distance_benchmark);
criterion_main!(l2_benches, ip_benches);
