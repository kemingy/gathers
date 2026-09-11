//! Down sampling methods.

use num_traits::Num;
use rand::{Rng, RngExt};

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
mod test {
    use super::reservoir_sampling;

    #[test]
    fn test_reservoir_sampling() {
        let n_sample = 3;
        let data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let mut data_iter = data.into_iter();
        let res = reservoir_sampling(n_sample, &mut data_iter);
        assert_eq!(res.len(), n_sample);
    }
}
