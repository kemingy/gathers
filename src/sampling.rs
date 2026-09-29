//! Down sampling methods.

use aligned_vec::AVec;
use num_traits::Num;
use rand::{Rng, RngExt};

/// Select distinct row indices uniformly without replacement.
///
/// Only indices are retained, so the vectors may live on disk. Sparse samples use rand's
/// adaptive sampler; samples larger than one third of the source use reservoir sampling.
/// Very sparse samples take work and memory proportional to the sample size, including
/// for sources larger than u32. At higher ratios rand may allocate a full u32 index array.
/// Output order is unspecified; sort the indices for disk reads. A seeded RNG reproduces
/// selection within the same build and target. Panics if `n_sample` exceeds `num_rows`.
pub fn sample_indices<R: Rng + ?Sized>(
    num_rows: usize,
    n_sample: usize,
    rng: &mut R,
) -> Vec<usize> {
    assert!(n_sample <= num_rows, "sample exceeds source row count");
    if n_sample == 0 {
        return Vec::new();
    }
    // Sparse sampling avoids O(num_rows) random draws. At dense ratios the reservoir
    // needs fewer draws and avoids rand's full-source shuffle or rejection hash table.
    if n_sample <= num_rows / 3 {
        rand::seq::index::sample(rng, num_rows, n_sample).into_vec()
    } else {
        reservoir_indices(num_rows, n_sample, rng)
    }
}

fn reservoir_indices<R: Rng + ?Sized>(num_rows: usize, n_sample: usize, rng: &mut R) -> Vec<usize> {
    let mut indices = (0..n_sample).collect::<Vec<_>>();
    if n_sample != 0 {
        for i in n_sample..num_rows {
            let j = rng.random_range(0..=i);
            if j < n_sample {
                indices[j] = i;
            }
        }
    }
    indices
}

pub(crate) fn subsample_flat<R: Rng + ?Sized>(
    n_sample: usize,
    vecs: &[f32],
    dim: usize,
    rng: &mut R,
) -> AVec<f32> {
    // Preserve the existing in-memory fit's sample order and RNG consumption.
    let indices = reservoir_indices(vecs.len() / dim, n_sample, rng);
    let mut sample = AVec::with_capacity(64, n_sample * dim);
    for index in indices {
        sample.extend_from_slice(&vecs[index * dim..(index + 1) * dim]);
    }
    sample
}

/// Subsample a given number of vectors from a list of vectors.
pub fn subsample(n_sample: usize, vecs: &[f32], dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    subsample_inner(n_sample, vecs, dim, &mut rng)
}

pub(crate) fn subsample_inner<R: Rng + ?Sized>(
    n_sample: usize,
    vecs: &[f32],
    dim: usize,
    rng: &mut R,
) -> Vec<Vec<f32>> {
    reservoir_sampling_inner(
        n_sample,
        &mut vecs.chunks(dim).map(|chunk| chunk.to_vec()),
        rng,
    )
}

/// Reservoir sampling algorithm.
///
/// Accepts an iterator of vectors and returns a list of vectors of size `n_sample`.
pub fn reservoir_sampling<I, T>(n_sample: usize, iteration: &mut I) -> Vec<Vec<T>>
where
    I: Iterator<Item = Vec<T>>,
    T: Num + Copy,
{
    let mut rng = rand::rng();
    reservoir_sampling_inner(n_sample, iteration, &mut rng)
}

fn reservoir_sampling_inner<I, T, R>(n_sample: usize, iteration: &mut I, rng: &mut R) -> Vec<Vec<T>>
where
    I: Iterator<Item = Vec<T>>,
    T: Num + Copy,
    R: Rng + ?Sized,
{
    let mut res = Vec::with_capacity(n_sample);
    for _ in 0..n_sample {
        res.push(iteration.next().expect("iteration less than n_sample"));
    }

    for (i, vec) in (n_sample..).zip(iteration.by_ref()) {
        let j = rng.random_range(0..=i);
        if j < n_sample {
            res[j] = vec;
        }
    }

    res
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use seed_rand::seeded_rng;

    use super::{reservoir_sampling, sample_indices, subsample_flat, subsample_inner};

    #[test]
    fn flat_sampling_preserves_reservoir_order_and_rng_stream() {
        let seed = seeded_rng().random();
        let values = (0..600 * 3).map(|value| value as f32).collect::<Vec<_>>();
        for count in [1, 256, 600] {
            let mut flat_rng = StdRng::seed_from_u64(seed);
            let mut reference_rng = StdRng::seed_from_u64(seed);
            let actual = subsample_flat(count, &values, 3, &mut flat_rng);
            let expected = subsample_inner(count, &values, 3, &mut reference_rng);
            assert_eq!(&*actual, expected.concat());
            assert_eq!(flat_rng.random::<u64>(), reference_rng.random::<u64>());
        }
    }

    #[test]
    fn index_samples_are_unique_in_bounds_and_reproducible() {
        let seed = seeded_rng().random();
        let mut shapes = vec![
            (0, 0),
            (600, 0),
            (600, 1),
            (600, 200),
            (600, 201),
            (600, 600),
        ];
        if let Ok(rows) = usize::try_from(10_000_000_000_u64) {
            shapes.push((rows, 1_000));
        }
        for (rows, count) in shapes {
            let mut indices = sample_indices(rows, count, &mut StdRng::seed_from_u64(seed));
            assert_eq!(
                indices,
                sample_indices(rows, count, &mut StdRng::seed_from_u64(seed))
            );
            indices.sort_unstable();
            assert_eq!(indices.len(), count);
            assert!(indices.iter().all(|&index| index < rows));
            assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    #[should_panic(expected = "sample exceeds source row count")]
    fn index_sampling_rejects_oversized_samples() {
        sample_indices(5, 6, &mut seeded_rng());
    }

    #[test]
    fn test_reservoir_sampling() {
        let n_sample = 3;
        let data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let mut data_iter = data.into_iter();
        let res = reservoir_sampling(n_sample, &mut data_iter);
        assert_eq!(res.len(), n_sample);
    }
}
