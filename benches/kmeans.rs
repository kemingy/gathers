use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use gathers::distance::Distance;
use gathers::kmeans::{KMeans, base_assign_parallel};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

criterion_group!(assignment_benches, assignment_benchmark);
criterion_group!(kmeans_benches, kmeans_benchmark);
criterion_main!(kmeans_benches, assignment_benches);
