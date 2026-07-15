use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use gathers::distance::{
    Distance, l2_norm_native, native_argmin, native_dot_product, native_squared_euclidean,
};
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
use gathers::distance::{
    l2_norm as dispatched_l2_norm, neg_dot_product as dispatched_neg_dot_product,
    squared_euclidean as dispatched_squared_euclidean,
};
use gathers::kmeans::{KMeans, base_assign_parallel};
use gathers::rabitq::{binary_dot_product_native, min_max_residual, min_max_residual_native};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use gathers::simd::{self, argmin, dot_product, l2_norm, l2_squared_distance};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use pulp::x86::V3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn l2_norm_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("norm");
    for dim in [64, 118, 124, 128, 512, 1024].into_iter() {
        let x: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        group.bench_with_input(BenchmarkId::new("native", dim), &x, |b, input| {
            b.iter(|| l2_norm_native(input))
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        group.bench_with_input(BenchmarkId::new("simd", dim), &x, |b, input| {
            b.iter(|| unsafe { l2_norm(input) })
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = V3::try_new() {
            group.bench_with_input(BenchmarkId::new("pulp", dim), &x, |b, input| {
                b.iter(|| simd::pulp::l2_norm(simd, input))
            });
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        group.bench_with_input(BenchmarkId::new("pulp", dim), &x, |b, input| {
            b.iter(|| dispatched_l2_norm(input))
        });
    }
    group.finish();
}

pub fn min_max_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("min_max");
    for dim in [64, 118, 124, 128, 512, 1024].into_iter() {
        let x: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let y: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let residual = vec![0.0; dim];

        group.bench_with_input(
            BenchmarkId::new("native", dim),
            &(&residual, &x, &y),
            |b, input| {
                b.iter(|| {
                    let mut res = input.0.clone();
                    min_max_residual_native(&mut res, input.1, input.2)
                });
            },
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        group.bench_with_input(
            BenchmarkId::new("simd", dim),
            &(&residual, &x, &y),
            |b, input| {
                b.iter(|| {
                    let mut res = input.0.clone();
                    unsafe { simd::min_max_residual(&mut res, input.1, input.2) }
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("pulp", dim),
            &(&residual, &x, &y),
            |b, input| {
                b.iter(|| {
                    let mut res = input.0.clone();
                    min_max_residual(&mut res, input.1, input.2)
                });
            },
        );
    }
}

pub fn argmin_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("argmin");
    for dim in [64, 118, 124, 128, 512, 1024].into_iter() {
        let x: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        group.bench_with_input(BenchmarkId::new("native", dim), &x, |b, input| {
            b.iter(|| native_argmin(input))
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        group.bench_with_input(BenchmarkId::new("simd", dim), &x, |b, input| {
            b.iter(|| unsafe { argmin(input) })
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = V3::try_new() {
            group.bench_with_input(BenchmarkId::new("pulp", dim), &x, |b, input| {
                b.iter(|| simd::pulp::argmin(simd, input))
            });
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        group.bench_with_input(BenchmarkId::new("pulp", dim), &x, |b, input| {
            b.iter(|| gathers::distance::argmin(input))
        });
    }
}

pub fn l2_distance_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("l2 distance");
    for dim in [64, 118, 124, 128, 512, 1024].into_iter() {
        let lhs: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let rhs: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        group.bench_with_input(
            BenchmarkId::new("native", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| native_squared_euclidean(input.0, input.1)),
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        group.bench_with_input(BenchmarkId::new("simd", dim), &(&lhs, &rhs), |b, input| {
            b.iter(|| unsafe { l2_squared_distance(input.0, input.1) })
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = V3::try_new() {
            group.bench_with_input(BenchmarkId::new("pulp", dim), &(&lhs, &rhs), |b, input| {
                b.iter(|| simd::pulp::l2_squared_distance(simd, input.0, input.1))
            });
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        group.bench_with_input(BenchmarkId::new("pulp", dim), &(&lhs, &rhs), |b, input| {
            b.iter(|| dispatched_squared_euclidean(input.0, input.1))
        });
    }
    group.finish();
}

pub fn ip_distance_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("dot product distance");
    for dim in [64, 118, 124, 128, 512, 1024].into_iter() {
        let lhs: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let rhs: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        group.bench_with_input(
            BenchmarkId::new("native", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| native_dot_product(input.0, input.1)),
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        group.bench_with_input(BenchmarkId::new("simd", dim), &(&lhs, &rhs), |b, input| {
            b.iter(|| unsafe { dot_product(input.0, input.1) })
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = V3::try_new() {
            group.bench_with_input(BenchmarkId::new("pulp", dim), &(&lhs, &rhs), |b, input| {
                b.iter(|| simd::pulp::dot_product(simd, input.0, input.1))
            });
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        group.bench_with_input(BenchmarkId::new("pulp", dim), &(&lhs, &rhs), |b, input| {
            b.iter(|| dispatched_neg_dot_product(input.0, input.1))
        });
    }
    group.finish();
}

pub fn binary_ip_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("binary dot product");
    for dim in [1, 2, 4, 8, 16, 36].into_iter() {
        let lhs: Vec<u64> = (0..dim).map(|_| rng.random::<u64>()).collect();
        let rhs: Vec<u64> = (0..dim).map(|_| rng.random::<u64>()).collect();

        group.bench_with_input(
            BenchmarkId::new("native", dim * 64),
            &(&lhs, &rhs),
            |b, input| {
                b.iter(|| binary_dot_product_native(input.0, input.1));
            },
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        group.bench_with_input(
            BenchmarkId::new("simd", dim * 64),
            &(&lhs, &rhs),
            |b, input| {
                b.iter(|| unsafe { simd::binary_dot_product_simd(input.0, input.1) });
            },
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        group.bench_with_input(
            BenchmarkId::new("pulp", dim * 64),
            &(&lhs, &rhs),
            |b, input| {
                b.iter(|| simd::binary_dot_product(input.0, input.1));
            },
        );
    }
}

pub fn assignment_benchmark(c: &mut Criterion) {
    const NUM_VECTORS: usize = 4096;
    const NUM_CENTROIDS: usize = 256;
    const DIM: usize = 128;

    let mut rng = StdRng::seed_from_u64(42);
    let vecs: Vec<f32> = (0..NUM_VECTORS * DIM).map(|_| rng.random()).collect();
    let centroids: Vec<f32> = (0..NUM_CENTROIDS * DIM).map(|_| rng.random()).collect();
    let mut labels = vec![0; NUM_VECTORS];

    let mut group = c.benchmark_group("assignment");
    group.sample_size(20);
    group.throughput(Throughput::Elements((NUM_VECTORS * NUM_CENTROIDS) as u64));
    group.bench_function("parallel_l2_4096x256x128", |b| {
        b.iter(|| {
            base_assign_parallel(
                black_box(&vecs),
                black_box(&centroids),
                DIM,
                Distance::SquaredEuclidean,
                black_box(&mut labels),
            )
        })
    });
    group.finish();
}

pub fn kmeans_benchmark(c: &mut Criterion) {
    const NUM_VECTORS: usize = 10_240;
    const NUM_CENTROIDS: usize = 256;
    const DIM: usize = 128;
    const ITERATIONS: usize = 5;

    let mut rng = StdRng::seed_from_u64(43);
    let source: Vec<f32> = (0..NUM_VECTORS * DIM).map(|_| rng.random()).collect();
    let vecs = gathers::utils::as_continuous_vec(&[source]);
    let kmeans = KMeans::new(
        NUM_CENTROIDS as u32,
        ITERATIONS as u32,
        f32::MIN_POSITIVE,
        Distance::SquaredEuclidean,
        false,
    );

    let mut group = c.benchmark_group("kmeans");
    group.sample_size(20);
    group.throughput(Throughput::Elements(
        (NUM_VECTORS * NUM_CENTROIDS * ITERATIONS) as u64,
    ));
    group.bench_function("fit_10240x256x128_5iter", |b| {
        b.iter(|| kmeans.fit(black_box(vecs.clone()), DIM))
    });
    group.finish();
}

criterion_group!(l2_benches, l2_distance_benchmark);
criterion_group!(ip_benches, ip_distance_benchmark);
criterion_group!(norm_benches, l2_norm_benchmark);
criterion_group!(argmin_benches, argmin_benchmark);
criterion_group!(min_max_benches, min_max_benchmark);
criterion_group!(binary_ip_benches, binary_ip_benchmark);
criterion_group!(assignment_benches, assignment_benchmark);
criterion_group!(kmeans_benches, kmeans_benchmark);
criterion_main!(
    kmeans_benches,
    assignment_benches,
    l2_benches,
    ip_benches,
    norm_benches,
    argmin_benches,
    min_max_benches,
    binary_ip_benches,
);
