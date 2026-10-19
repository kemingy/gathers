//! Centroid aggregation with stable source-row order in both serial and parallel paths.
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

// Index construction and scheduling do not pay off for small coordinate arrays.
const MIN_PARALLEL_VALUES: usize = 1 << 18;

fn try_indices(len: usize) -> Option<Vec<usize>> {
    let mut indices = Vec::new();
    indices.try_reserve_exact(len).ok()?;
    indices.resize(len, 0);
    Some(indices)
}

fn accumulate_serial(
    vecs: &[f32],
    labels: &[u32],
    dim: usize,
    means: &mut [f32],
    counts: &mut [usize],
) {
    for (vector, &label) in vecs.chunks_exact(dim).zip(labels) {
        let label = label as usize;
        counts[label] += 1;
        for (sum, &value) in means[label * dim..(label + 1) * dim].iter_mut().zip(vector) {
            *sum += value;
        }
    }
}

fn accumulate_indexed(
    vecs: &[f32],
    labels: &[u32],
    dim: usize,
    means: &mut [f32],
    counts: &mut [usize],
) -> bool {
    let Some(mut offsets) = counts.len().checked_add(1).and_then(try_indices) else {
        return false;
    };
    let Some(mut rows) = try_indices(labels.len()) else {
        return false;
    };
    for &label in labels {
        counts[label as usize] += 1;
    }
    for (label, &count) in counts.iter().enumerate() {
        offsets[label + 1] = offsets[label] + count;
    }
    // Stable counting partition: each centroid's rows remain in source order. Use usize
    // indices so training/reconstruction does not silently truncate datasets above u32::MAX.
    for (row, &label) in labels.iter().enumerate() {
        let cursor = &mut offsets[label as usize];
        rows[*cursor] = row;
        *cursor += 1;
    }
    // Cursors now contain cell ends; shift once to recover the original cell starts.
    offsets.copy_within(..counts.len(), 1);
    offsets[0] = 0;
    means
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(label, mean)| {
            for &row in &rows[offsets[label]..offsets[label + 1]] {
                for (sum, &value) in mean.iter_mut().zip(&vecs[row * dim..(row + 1) * dim]) {
                    *sum += value;
                }
            }
        });
    true
}

// Inputs and label bounds are validated by the caller. Keep ordinary f32 accumulation unchanged,
// but divide overflowing sums in f64 before casting: a mean of finite inputs is representable.
pub(crate) fn cluster_means(
    vecs: &[f32],
    labels: &[u32],
    dim: usize,
    means: &mut [f32],
) -> Vec<usize> {
    means.fill(0.0);
    let mut counts = vec![0usize; means.len() / dim];
    if vecs.len() < MIN_PARALLEL_VALUES
        || counts.len() < 2
        || rayon::current_num_threads() == 1
        || !accumulate_indexed(vecs, labels, dim, means, &mut counts)
    {
        counts.fill(0);
        accumulate_serial(vecs, labels, dim, means, &mut counts);
    }
    if means.iter().all(|value| value.is_finite()) {
        for (mean, &count) in means.chunks_exact_mut(dim).zip(&counts) {
            if count != 0 {
                let inverse = (count as f32).recip();
                mean.iter_mut().for_each(|value| *value *= inverse);
            }
        }
    } else {
        let mut sums = vec![0.0_f64; means.len()];
        for (vector, &label) in vecs.chunks_exact(dim).zip(labels) {
            let label = label as usize;
            for (sum, &value) in sums[label * dim..(label + 1) * dim].iter_mut().zip(vector) {
                *sum += f64::from(value);
            }
        }
        for ((mean, sum), &count) in means
            .chunks_exact_mut(dim)
            .zip(sums.chunks_exact(dim))
            .zip(&counts)
        {
            if count != 0 {
                for (mean, &sum) in mean.iter_mut().zip(sum) {
                    *mean = (sum / count as f64) as f32;
                }
            } else {
                mean.fill(0.0);
            }
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::{accumulate_indexed, accumulate_serial, cluster_means};

    #[test]
    fn stable_partition_preserves_source_order_and_empty_clusters() {
        for dim in [1, 33, 129] {
            let vectors: Vec<f32> = (0..513 * dim)
                .map(|i| [1e20, 1.0, -1e20, 2.0, -0.5][i % 5])
                .collect();
            let labels: Vec<u32> = (0..513)
                .map(|i| if i % 5 == 0 { 0 } else { (i % 17) as u32 })
                .collect();
            let mut expected = vec![0.0; 19 * dim];
            let mut counts = vec![0; 19];
            accumulate_serial(&vectors, &labels, dim, &mut expected, &mut counts);
            for threads in [1, 4] {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        let mut actual = vec![0.0; expected.len()];
                        let mut actual_counts = vec![0; counts.len()];
                        assert!(accumulate_indexed(
                            &vectors,
                            &labels,
                            dim,
                            &mut actual,
                            &mut actual_counts
                        ));
                        assert_eq!(
                            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                        assert_eq!(actual_counts, counts);
                        assert_eq!(&actual[17 * dim..], vec![0.0; 2 * dim]);
                    });
            }
        }
    }

    #[test]
    fn parallel_means_match_serial_with_overflow_and_skew() {
        let dim = 129;
        let labels: Vec<u32> = (0..2049)
            .map(|i| if i % 5 == 0 { (i % 17) as u32 } else { 0 })
            .collect();
        for magnitude in [1.0, f32::MAX] {
            let vectors: Vec<f32> = (0..labels.len() * dim)
                .map(|i| {
                    if magnitude == f32::MAX {
                        magnitude
                    } else {
                        [1e20, 1.0, -1e20, 2.0][i % 4]
                    }
                })
                .collect();
            let run = |threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        let mut means = vec![0.0; 19 * dim];
                        let counts = cluster_means(&vectors, &labels, dim, &mut means);
                        assert!(means.iter().all(|value| value.is_finite()));
                        (
                            means
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            counts,
                        )
                    })
            };
            assert_eq!(run(1), run(4));
        }
    }
}
