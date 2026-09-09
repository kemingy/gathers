//! Clustering algorithms for Rust.
//!
//! ## Examples
//!
//! ```
//! use gathers::kmeans::{KMeans, rabitq_assign};
//! use gathers::utils::as_continuous_vec;
//! # use rand::Rng;
//! # let mut rng = rand::rng();
//! # let vecs = (0..1000).map(|_| (0..32).map(|_| rng.random::<f32>()).collect::<Vec<f32>>()).collect::<Vec<Vec<f32>>>();
//!
//!
//! let kmeans = KMeans::default();
//! let num = vecs.len();
//! let dim = vecs[0].len();
//!
//! // fit
//! let centroids = kmeans.fit(as_continuous_vec(&vecs), dim);
//! // predict
//! let mut labels = vec![0; num];
//! rabitq_assign(&as_continuous_vec(&vecs), &centroids, dim, &mut labels);
//! ```

#![deny(missing_docs)]

pub mod distance;
pub mod kmeans;
pub mod rabitq;
pub mod sampling;
pub mod simd;
pub mod utils;

#[cfg(test)]
pub(crate) mod test_utils {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const TEST_SEED_ENV: &str = "GATHERS_TEST_SEED";

    pub(crate) fn random_test_rng() -> StdRng {
        let seed = std::env::var(TEST_SEED_ENV).map_or_else(
            |_| rand::rng().random(),
            |value| {
                value.parse().unwrap_or_else(|_| {
                    panic!("{TEST_SEED_ENV} must be an unsigned 64-bit integer")
                })
            },
        );
        eprintln!("random seed: {seed}; reproduce with {TEST_SEED_ENV}={seed}");
        StdRng::seed_from_u64(seed)
    }
}
