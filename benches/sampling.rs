use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use gathers::sampling::sample_indices;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use seed_rand::seeded_rng;

fn reservoir_indices(rows: usize, samples: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut indices: Vec<_> = (0..samples).collect();
    for index in samples..rows {
        let slot = rng.random_range(0..=index);
        if slot < samples {
            indices[slot] = index;
        }
    }
    indices
}

fn sampling_benchmark(c: &mut Criterion) {
    let seed = seeded_rng().random();
    let mut shapes = [
        1_000, 100_000, 1_000_000, 3_333_333, 5_000_000, 6_369_536, 9_900_000,
    ]
    .map(|samples| (10_000_000, samples))
    .to_vec();
    if let Ok(rows) = usize::try_from(10_000_000_000_u64) {
        shapes.push((rows, 1_000_000));
    }
    let mut group = c.benchmark_group("sorted_sample_indices");
    group.sample_size(10);
    for (rows, samples) in shapes {
        for method in ["reservoir", "rand", "adaptive"] {
            // The 10B case measures bounded sampling, not a full-source reference scan.
            if rows > u32::MAX as usize && method == "reservoir" {
                continue;
            }
            group.bench_function(BenchmarkId::new(method, format!("{rows}/{samples}")), |b| {
                b.iter_batched(
                    || StdRng::seed_from_u64(seed),
                    |mut rng| {
                        let mut indices = match method {
                            "reservoir" => reservoir_indices(rows, samples, &mut rng),
                            "rand" => rand::seq::index::sample(&mut rng, rows, samples).into_vec(),
                            _ => sample_indices(rows, samples, &mut rng),
                        };
                        indices.sort_unstable();
                        indices
                    },
                    BatchSize::PerIteration,
                );
            });
        }
    }
    group.finish();
}

criterion_group!(benches, sampling_benchmark);
criterion_main!(benches);
