//! Down sampling methods.

use num_traits::Num;
use rand::Rng;

/// Subsample a given number of vectors from a list of vectors.
pub fn subsample(n_sample: usize, vecs: &[f32], dim: usize) -> Vec<Vec<f32>> {
    reservoir_sampling(n_sample, &mut vecs.chunks(dim).map(|chunk| chunk.to_vec()))
}

/// Reservoir sampling algorithm.
///
/// Accepts an iterator of vectors and returns a list of vectors of size `n_sample`.
pub fn reservoir_sampling<I, T>(n_sample: usize, iteration: &mut I) -> Vec<Vec<T>>
where
    I: Iterator<Item = Vec<T>>,
    T: Num + Copy,
{
    let mut res = Vec::with_capacity(n_sample);
    let mut rng = rand::rng();

    for _ in 0..n_sample {
        res.push(iteration.next().expect("iteration less than n_sample"));
    }

    let mut i = n_sample;
    for vec in iteration.by_ref() {
        let j = rng.random_range(0..=i);
        if j < n_sample {
            res[j] = vec;
        }
        i += 1;
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
