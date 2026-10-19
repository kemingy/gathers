//! Clustering algorithms for Rust.
//!
//! ## Examples
//!
//! ```
//! use gathers::kmeans::{AssignmentPolicy, KMeans, batch_assign};
//! use gathers::distance::Distance;
//! use gathers::utils::as_continuous_vec;
//! # use rand::RngExt;
//! # let mut rng = rand::rng();
//! # let vecs = (0..1000).map(|_| (0..32).map(|_| rng.random::<f32>()).collect::<Vec<f32>>()).collect::<Vec<Vec<f32>>>();
//!
//!
//! let kmeans = KMeans::default();
//! let num = vecs.len();
//! let dim = vecs[0].len();
//!
//! // fit
//! let centroids = kmeans.fit(as_continuous_vec(&vecs), dim).unwrap();
//! // predict
//! let mut labels = vec![0; num];
//! batch_assign(&as_continuous_vec(&vecs), &centroids, dim, Distance::SquaredEuclidean,
//!     AssignmentPolicy::Auto, Some(42), &mut labels).unwrap();
//! ```

#![deny(missing_docs)]

pub mod distance;
pub mod kmeans;
mod linalg;
/// Dimensionality-reduction transforms for clustering workloads.
pub mod reduction;
/// RaBitQ-based approximate top-1 vector retrieval.
pub use rabitq;
pub mod sampling;
pub mod simd;
pub mod utils;
