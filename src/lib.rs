//! Clustering algorithms for Rust.
//!
//! ## Examples
//!
//! ```
//! use gathers::kmeans::{KMeans, rabitq_assign};
//! use gathers::utils::as_continuous_vec;
//! # use rand::Rng;
//! # let mut rng = rand::thread_rng();
//! # let vecs = (0..1000).map(|_| (0..32).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>()).collect::<Vec<Vec<f32>>>();
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
