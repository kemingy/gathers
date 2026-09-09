use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use gathers::distance::Distance;
use gathers::kmeans::{KMeans, base_assign, base_assign_parallel, rabitq_assign_parallel};
use gathers::rabitq::RaBitQ;
use rand::Rng;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator, ParallelSlice,
};

fn assignment_benchmark(c: &mut Criterion) {
    const NUM_VECTORS: usize = 4096;
    const NUM_CENTROIDS: usize = 256;
    const DIM: usize = 128;

    let mut rng = rand::rng();
    let vecs: Vec<f32> = (0..NUM_VECTORS * DIM).map(|_| rng.random()).collect();
    let centroids: Vec<f32> = (0..NUM_CENTROIDS * DIM).map(|_| rng.random()).collect();
    let mut labels = vec![0; NUM_VECTORS];
    let rabitq = RaBitQ::new(&centroids, DIM);

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
    group.bench_function("serial_dot_4096x256x128", |b| {
        b.iter(|| {
            base_assign(
                black_box(&vecs),
                black_box(&centroids),
                DIM,
                Distance::NegativeDotProduct,
                black_box(&mut labels),
            )
        })
    });
    group.bench_function("parallel_dot_4096x256x128", |b| {
        b.iter(|| {
            base_assign_parallel(
                black_box(&vecs),
                black_box(&centroids),
                DIM,
                Distance::NegativeDotProduct,
                black_box(&mut labels),
            )
        })
    });
    group.bench_function("rabitq_l2_4096x256x128", |b| {
        b.iter(|| {
            rabitq_assign_parallel(
                black_box(&vecs),
                black_box(&centroids),
                DIM,
                black_box(&mut labels),
            )
        })
    });
    group.bench_function("rabitq_per_query_4096x256x128", |b| {
        b.iter(|| {
            labels
                .par_iter_mut()
                .zip(vecs.par_chunks_exact(DIM))
                .for_each(|(label, query)| {
                    *label = rabitq.retrieve_top_one(black_box(query)) as u32;
                })
        })
    });
    group.bench_function("rabitq_batch_4096x256x128", |b| {
        b.iter(|| rabitq.retrieve_top_one_batch(black_box(&vecs), DIM, black_box(&mut labels)))
    });
    group.finish();
}

fn kmeans_benchmark(c: &mut Criterion) {
    const NUM_VECTORS: usize = 50_176;
    const NUM_CENTROIDS: usize = 256;
    const DIM: usize = 128;
    const ITERATIONS: usize = 5;

    let mut rng = rand::rng();
    let source: Vec<f32> = (0..NUM_VECTORS * DIM).map(|_| rng.random()).collect();
    let vecs = gathers::utils::as_continuous_vec(&[source]);
    let l2_kmeans = KMeans::new(
        NUM_CENTROIDS as u32,
        ITERATIONS as u32,
        f32::MIN_POSITIVE,
        Distance::SquaredEuclidean,
        false,
    );
    let dot_kmeans = KMeans::new(
        NUM_CENTROIDS as u32,
        ITERATIONS as u32,
        f32::MIN_POSITIVE,
        Distance::NegativeDotProduct,
        false,
    );

    let mut group = c.benchmark_group("kmeans");
    group.sample_size(20);
    group.throughput(Throughput::Elements(
        (NUM_VECTORS * NUM_CENTROIDS * ITERATIONS) as u64,
    ));
    group.bench_function("fit_50176x256x128_5iter", |b| {
        b.iter(|| l2_kmeans.fit(black_box(vecs.clone()), DIM))
    });
    group.bench_function("fit_dot_50176x256x128_5iter", |b| {
        b.iter(|| dot_kmeans.fit(black_box(vecs.clone()), DIM))
    });
    group.finish();
}

criterion_group!(assignment_benches, assignment_benchmark);
criterion_group!(kmeans_benches, kmeans_benchmark);
criterion_main!(kmeans_benches, assignment_benches);
