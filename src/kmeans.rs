//! K-means clustering implementation.

use crate::distance::{assign, update_centroids, Distance};
use crate::sampling::subsample;
use crate::utils::{as_continuous_vec, centroid_residual, normalize};
use core::panic;

use log::debug;

const MIN_POINTS_PER_CENTROID: usize = 39;
const MAX_POINTS_PER_CENTROID: usize = 256;
// const EPS: f32 = 1.0 / 1024.0;

/// K-means clustering algorithm.
#[derive(Debug)]
pub struct KMeans {
    n_cluster: u32,
    max_iter: u32,
    _tolerance: f32,
    distance: Distance,
    use_residual: bool,
}

impl Default for KMeans {
    fn default() -> Self {
        Self {
            n_cluster: 8,
            max_iter: 25,
            _tolerance: 1e-4,
            distance: Distance::default(),
            use_residual: false,
        }
    }
}

impl KMeans {
    /// Create a new KMeans instance.
    pub fn new(
        n_cluster: u32,
        max_iter: u32,
        tolerance: f32,
        distance: Distance,
        use_residual: bool,
    ) -> Self {
        if n_cluster <= 1 {
            panic!("n_cluster must be greater than 1");
        }
        if max_iter <= 1 {
            panic!("max_iter must be greater than 1");
        }
        if tolerance <= 0.0 {
            panic!("tolerance must be greater than 0.0");
        }
        Self {
            n_cluster,
            max_iter,
            _tolerance: tolerance,
            distance,
            use_residual,
        }
    }

    /// Fit the KMeans configurations to the given vectors and return the centroids.
    pub fn fit(&self, mut vecs: Vec<f32>, dim: usize) -> Vec<f32> {
        let num = vecs.len() / dim;
        debug!("num of points: {}", num);
        if num < self.n_cluster as usize {
            panic!("number of samples must be greater than n_cluster");
        }
        if num < self.n_cluster as usize * MIN_POINTS_PER_CENTROID {
            panic!("too few samples for n_cluster");
        }

        // use residual for more accurate L2 distance computations
        if self.distance == Distance::EuclideanSquared && self.use_residual {
            debug!("use residual");
            centroid_residual(&mut vecs, dim);
        }

        // subsample
        if num > MAX_POINTS_PER_CENTROID * self.n_cluster as usize {
            let n_sample = MAX_POINTS_PER_CENTROID * self.n_cluster as usize;
            debug!("subsample to {} points", n_sample);
            let subsampled = as_continuous_vec(&subsample(n_sample, &vecs, dim));
            vecs.shrink_to(subsampled.len());
            vecs.copy_from_slice(&subsampled);
        }

        let mut centroids = as_continuous_vec(&subsample(self.n_cluster as usize, &vecs, dim));
        if self.distance == Distance::DotProduct {
            centroids.chunks_mut(dim).for_each(normalize);
        }

        let mut labels: Vec<u32> = vec![0; num];
        for i in 0..self.max_iter {
            debug!("iter {}", i);
            assign(&vecs, &centroids, dim, self.distance, &mut labels);
            update_centroids(&vecs, &mut centroids, dim, &labels);
            if self.distance == Distance::DotProduct {
                centroids.chunks_mut(dim).for_each(normalize);
            }
        }

        centroids
    }
}
