use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gathers::distance::{
    argmin, l2_norm, l2_norm_native, native_argmin, native_dot_product, native_squared_euclidean,
    neg_dot_product, squared_euclidean,
};
#[cfg(target_arch = "aarch64")]
use pulp::aarch64::Neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use pulp::x86::V3;
use rabitq::{
    binary_dot_product_native, min_max_residual, min_max_residual_native, simd as rabitq_simd,
    vector_binarize_query,
};
use rand::RngExt;

// Backend names are intentionally independent of instruction-set versions:
// - native: scalar baseline
// - pulp_dispatch: public API with runtime Pulp dispatch
// - pulp_<arch>: kernels expressed through Pulp
// - simd_<arch>: architecture-specific intrinsic kernels

pub fn l2_norm_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("l2_norm");
    for dim in [64, 118, 124, 128, 512, 1024] {
        let x: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        group.bench_with_input(BenchmarkId::new("native", dim), &x, |b, input| {
            b.iter(|| l2_norm_native(input))
        });
        group.bench_with_input(BenchmarkId::new("pulp_dispatch", dim), &x, |b, input| {
            b.iter(|| l2_norm(input))
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = V3::try_new() {
            group.bench_with_input(BenchmarkId::new("pulp_x86", dim), &x, |b, input| {
                b.iter(|| gathers::simd::pulp::l2_norm(simd, input))
            });
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(simd) = Neon::try_new() {
            group.bench_with_input(BenchmarkId::new("pulp_aarch64", dim), &x, |b, input| {
                b.iter(|| gathers::simd::pulp::l2_norm(simd, input))
            });
        }
    }
    group.finish();
}

#[allow(unsafe_code)]
pub fn min_max_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("min_max");
    for dim in [64, 118, 124, 128, 512, 1024] {
        let x: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let y: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let residual = vec![0.0; dim];

        group.bench_with_input(
            BenchmarkId::new("native", dim),
            &(&residual, &x, &y),
            |b, input| {
                let mut res = input.0.clone();
                b.iter(|| min_max_residual_native(&mut res, input.1, input.2));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("pulp_dispatch", dim),
            &(&residual, &x, &y),
            |b, input| {
                let mut res = input.0.clone();
                b.iter(|| min_max_residual(&mut res, input.1, input.2));
            },
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx") {
            group.bench_with_input(
                BenchmarkId::new("simd_x86", dim),
                &(&residual, &x, &y),
                |b, input| {
                    let mut res = input.0.clone();
                    b.iter(|| unsafe {
                        rabitq_simd::x86::legacy::min_max_residual(&mut res, input.1, input.2)
                    });
                },
            );
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = V3::try_new() {
            group.bench_with_input(
                BenchmarkId::new("pulp_x86", dim),
                &(&residual, &x, &y),
                |b, input| {
                    let mut res = input.0.clone();
                    b.iter(|| {
                        rabitq_simd::pulp::min_max_residual(simd, &mut res, input.1, input.2)
                    });
                },
            );
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(simd) = Neon::try_new() {
            group.bench_with_input(
                BenchmarkId::new("pulp_aarch64", dim),
                &(&residual, &x, &y),
                |b, input| {
                    let mut res = input.0.clone();
                    b.iter(|| {
                        rabitq_simd::pulp::min_max_residual(simd, &mut res, input.1, input.2)
                    });
                },
            );
        }
        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(
            BenchmarkId::new("simd_aarch64", dim),
            &(&residual, &x, &y),
            |b, input| {
                let mut res = input.0.clone();
                b.iter(|| {
                    rabitq_simd::aarch64::legacy::min_max_residual(&mut res, input.1, input.2)
                });
            },
        );
    }
    group.finish();
}

fn scalar_quantize_native(
    quantized: &mut [u8],
    vec: &[f32],
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    quantized
        .iter_mut()
        .zip(vec)
        .map(|(quantized, &value)| {
            *quantized = ((value - lower_bound) * multiplier).round() as u8;
            *quantized as u32
        })
        .sum()
}

fn vector_binarize_query_native(vec: &[u8], binary: &mut [u64]) {
    for j in 0..4 {
        for (i, &value) in vec.iter().enumerate() {
            binary[(i + j * vec.len()) / 64] |= (((value >> j) & 1) as u64) << (i % 64);
        }
    }
}

pub fn vector_binarize_query_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();
    let mut group = c.benchmark_group("vector_binarize_query");

    for dim in [64, 256, 1024, 4096] {
        let input = (0..dim)
            .map(|_| rng.random::<u8>() & 0x0f)
            .collect::<Vec<_>>();

        group.bench_with_input(BenchmarkId::new("native", dim), &input, |b, input| {
            let mut binary = vec![0; input.len() / 16];
            b.iter(|| vector_binarize_query_native(input, &mut binary));
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") {
            group.bench_with_input(BenchmarkId::new("pulp_x86", dim), &input, |b, input| {
                let mut binary = vec![0; input.len() / 16];
                b.iter(|| rabitq_simd::x86::vector_binarize_query(input, &mut binary));
            });
        }
        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("pulp_aarch64", dim), &input, |b, input| {
            let mut binary = vec![0; input.len() / 16];
            b.iter(|| rabitq_simd::aarch64::vector_binarize_query(input, &mut binary));
        });
        group.bench_with_input(BenchmarkId::new("dispatch", dim), &input, |b, input| {
            let mut binary = vec![0; input.len() / 16];
            b.iter(|| vector_binarize_query(input, &mut binary));
        });
    }
    group.finish();
}

pub fn scalar_quantize_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();
    let mut group = c.benchmark_group("scalar_quantize");

    for dim in [64, 128, 512, 1024] {
        let input = (0..dim).map(|_| rng.random::<f32>()).collect::<Vec<_>>();
        let lower_bound = 0.0;
        let multiplier = 15.0;

        group.bench_with_input(BenchmarkId::new("native", dim), &input, |b, input| {
            let mut quantized = vec![0; input.len()];
            b.iter(|| scalar_quantize_native(&mut quantized, input, lower_bound, multiplier));
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") {
            group.bench_with_input(BenchmarkId::new("pulp_x86", dim), &input, |b, input| {
                let mut quantized = vec![0; input.len()];
                b.iter(|| {
                    rabitq_simd::x86::scalar_quantize(
                        &mut quantized,
                        input,
                        lower_bound,
                        multiplier,
                    )
                });
            });
        }
        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("pulp_aarch64", dim), &input, |b, input| {
            let mut quantized = vec![0; input.len()];
            b.iter(|| {
                rabitq_simd::aarch64::scalar_quantize(
                    &mut quantized,
                    input,
                    lower_bound,
                    multiplier,
                )
            });
        });
    }
    group.finish();
}

pub fn argmin_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("argmin");
    for dim in [64, 118, 124, 128, 512, 1024] {
        let x: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        group.bench_with_input(BenchmarkId::new("native", dim), &x, |b, input| {
            b.iter(|| native_argmin(input))
        });
        group.bench_with_input(BenchmarkId::new("pulp_dispatch", dim), &x, |b, input| {
            b.iter(|| argmin(input))
        });
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = V3::try_new() {
            group.bench_with_input(BenchmarkId::new("pulp_x86", dim), &x, |b, input| {
                b.iter(|| gathers::simd::pulp::argmin(simd, input))
            });
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(simd) = Neon::try_new() {
            group.bench_with_input(BenchmarkId::new("pulp_aarch64", dim), &x, |b, input| {
                b.iter(|| gathers::simd::pulp::argmin(simd, input))
            });
        }
    }
    group.finish();
}

pub fn l2_distance_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("l2_distance");
    for dim in [64, 118, 124, 128, 512, 1024] {
        let lhs: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let rhs: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        group.bench_with_input(
            BenchmarkId::new("native", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| native_squared_euclidean(input.0, input.1)),
        );
        group.bench_with_input(
            BenchmarkId::new("pulp_dispatch", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| squared_euclidean(input.0, input.1)),
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = V3::try_new() {
            group.bench_with_input(
                BenchmarkId::new("pulp_x86", dim),
                &(&lhs, &rhs),
                |b, input| {
                    b.iter(|| gathers::simd::pulp::l2_squared_distance(simd, input.0, input.1))
                },
            );
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(simd) = Neon::try_new() {
            group.bench_with_input(
                BenchmarkId::new("pulp_aarch64", dim),
                &(&lhs, &rhs),
                |b, input| {
                    b.iter(|| gathers::simd::pulp::l2_squared_distance(simd, input.0, input.1))
                },
            );
        }
    }
    group.finish();
}

pub fn ip_distance_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("dot_product");
    for dim in [64, 118, 124, 128, 512, 1024] {
        let lhs: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let rhs: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        group.bench_with_input(
            BenchmarkId::new("native", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| native_dot_product(input.0, input.1)),
        );
        group.bench_with_input(
            BenchmarkId::new("pulp_dispatch", dim),
            &(&lhs, &rhs),
            |b, input| b.iter(|| neg_dot_product(input.0, input.1)),
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = V3::try_new() {
            group.bench_with_input(
                BenchmarkId::new("pulp_x86", dim),
                &(&lhs, &rhs),
                |b, input| b.iter(|| gathers::simd::pulp::dot_product(simd, input.0, input.1)),
            );
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(simd) = Neon::try_new() {
            group.bench_with_input(
                BenchmarkId::new("pulp_aarch64", dim),
                &(&lhs, &rhs),
                |b, input| b.iter(|| gathers::simd::pulp::dot_product(simd, input.0, input.1)),
            );
        }
    }
    group.finish();
}

#[allow(unsafe_code)]
pub fn binary_ip_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("binary_dot_product");
    for dim in [1, 2, 4, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36] {
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
        if std::is_x86_feature_detected!("avx2") {
            group.bench_with_input(
                BenchmarkId::new("simd_x86", dim * 64),
                &(&lhs, &rhs),
                |b, input| {
                    b.iter(|| unsafe {
                        rabitq_simd::x86::legacy::binary_dot_product(input.0, input.1)
                    });
                },
            );
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") {
            group.bench_with_input(
                BenchmarkId::new("pulp_x86", dim * 64),
                &(&lhs, &rhs),
                |b, input| {
                    b.iter(|| rabitq_simd::x86::binary_dot_product(input.0, input.1));
                },
            );
        }
        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(
            BenchmarkId::new("pulp_aarch64", dim * 64),
            &(&lhs, &rhs),
            |b, input| {
                b.iter(|| rabitq_simd::aarch64::binary_dot_product(input.0, input.1));
            },
        );
    }
    group.finish();
}

criterion_group!(l2_benches, l2_distance_benchmark);
criterion_group!(ip_benches, ip_distance_benchmark);
criterion_group!(norm_benches, l2_norm_benchmark);
criterion_group!(argmin_benches, argmin_benchmark);
criterion_group!(min_max_benches, min_max_benchmark);
criterion_group!(
    vector_binarize_query_benches,
    vector_binarize_query_benchmark
);
criterion_group!(scalar_quantize_benches, scalar_quantize_benchmark);
criterion_group!(binary_ip_benches, binary_ip_benchmark);
criterion_main!(
    l2_benches,
    ip_benches,
    norm_benches,
    argmin_benches,
    min_max_benches,
    vector_binarize_query_benches,
    scalar_quantize_benches,
    binary_ip_benches,
);
