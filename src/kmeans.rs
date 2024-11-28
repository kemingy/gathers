//! K-means clustering implementation.

use core::panic;
use std::time::Instant;

use log::debug;
use rand::Rng;

use crate::distance::{argmin, neg_dot_product, squared_euclidean, Distance};
use crate::rabitq::RaBitQ;
use crate::sampling::subsample;
use crate::utils::{as_continuous_vec, centroid_residual, normalize};

const EPS: f32 = 1.0 / 1024.0;
const MIN_POINTS_PER_CENTROID: usize = 39;
const MAX_POINTS_PER_CENTROID: usize = 256;

/// Assign vectors to centroids.
pub fn assign(vecs: &[f32], centroids: &[f32], dim: usize, distance: Distance, labels: &mut [u32]) {
    let mut distances = vec![f32::MAX; centroids.len() / dim];

    match distance {
        Distance::NegativeDotProduct => {
            for (i, vec) in vecs.chunks(dim).enumerate() {
                for (j, centroid) in centroids.chunks(dim).enumerate() {
                    distances[j] = neg_dot_product(vec, centroid);
                    if j == 0 || distances[j] < distances[labels[i] as usize] {
                        labels[i] = j as u32;
                    }
                }
            }
        }
        Distance::SquaredEuclidean => {
            // pre-compute the x**2 & y**2 for L2 distance
            // let squared_x: Vec<f32> = vecs.chunks(dim).map(l2_norm).collect();
            // let squared_y: Vec<f32> = centroids.chunks(dim).map(l2_norm).collect();

            for (i, vec) in vecs.chunks(dim).enumerate() {
                for (j, centroid) in centroids.chunks(dim).enumerate() {
                    distances[j] =
                        // squared_x[i] + squared_y[j] + 2.0 * neg_dot_product(vec, centroid);
                    squared_euclidean(vec, centroid);
                }
                labels[i] = argmin(&distances) as u32;
            }
        }
    }
}

/// Assign vectors to centroids with RaBitQ.
///
/// TODO: support dot product distance
pub fn rabitq_assign(vecs: &[f32], centroids: &[f32], dim: usize, labels: &mut [u32]) {
    let rabitq = RaBitQ::new(centroids, dim);
    for (i, vec) in vecs.chunks(dim).enumerate() {
        labels[i] = rabitq.retrieve_top_one(vec) as u32;
    }
}

/// Update centroids to the mean of assigned vectors.
pub fn update_centroids(vecs: &[f32], centroids: &mut [f32], dim: usize, labels: &[u32]) -> f32 {
    let mut means = vec![0.0; centroids.len()];
    let mut elements = vec![0; centroids.len() / dim];
    for (i, vec) in vecs.chunks(dim).enumerate() {
        let label = labels[i] as usize;
        elements[label] += 1;
        means[label * dim..(label + 1) * dim]
            .iter_mut()
            .zip(vec.iter())
            .for_each(|(m, &v)| *m += v);
    }
    let diff = squared_euclidean(centroids, &means);

    let mut zero_count = 0;
    for i in 0..elements.len() {
        if elements[i] == 0 {
            // need to split another cluster to fill this empty cluster
            zero_count += 1;
            let mut target = 0;
            let mut rng = rand::thread_rng();
            let base = 1.0 / (vecs.len() / dim - labels.len()) as f32;
            loop {
                let p = (elements[target] - 1) as f32 * base;
                if rng.gen::<f32>() < p {
                    break;
                }
                target = (target + 1) % labels.len();
            }
            debug!("split cluster {} to fill empty cluster {}", target, i);
            if i < target {
                let (left, right) = centroids.split_at_mut(target * dim);
                left[i * dim..(i + 1) * dim].copy_from_slice(&right[..dim]);
            } else {
                let (left, right) = centroids.split_at_mut(i * dim);
                right[..dim].copy_from_slice(&left[target * dim..(target + 1) * dim]);
            }
            // small symmetric perturbation
            for j in 0..dim {
                if j % 2 == 0 {
                    centroids[i * dim + j] *= 1.0 + EPS;
                    centroids[target * dim + j] *= 1.0 - EPS;
                } else {
                    centroids[i * dim + j] *= 1.0 - EPS;
                    centroids[target * dim + j] *= 1.0 + EPS;
                }
            }
            // update elements
            elements[i] = elements[target] / 2;
            elements[target] -= elements[i];
        }
        let divider = (elements[i] as f32).recip();
        for j in i * dim..(i + 1) * dim {
            centroids[j] = means[j] * divider;
        }
    }
    if zero_count != 0 {
        debug!("fixed {} empty clusters", zero_count);
    }
    diff
}

/// K-means clustering algorithm.
#[derive(Debug)]
pub struct KMeans {
    n_cluster: u32,
    max_iter: u32,
    tolerance: f32,
    distance: Distance,
    use_residual: bool,
}

impl Default for KMeans {
    fn default() -> Self {
        Self {
            n_cluster: 8,
            max_iter: 25,
            tolerance: 1e-4,
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
            tolerance,
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
        if self.distance == Distance::SquaredEuclidean && self.use_residual {
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
        if self.distance == Distance::NegativeDotProduct {
            centroids.chunks_mut(dim).for_each(normalize);
        }

        let mut labels: Vec<u32> = vec![0; num];
        debug!("start training");
        for i in 0..self.max_iter {
            let start_time = Instant::now();
            assign(&vecs, &centroids, dim, self.distance, &mut labels);
            // rabitq_assign(&vecs, &centroids, dim, &mut labels);
            let diff = update_centroids(&vecs, &mut centroids, dim, &labels);
            if self.distance == Distance::NegativeDotProduct {
                centroids.chunks_mut(dim).for_each(normalize);
            }
            debug!("iter {} takes {} s", i, start_time.elapsed().as_secs_f32());
            if diff < self.tolerance {
                debug!("converged at iter {}", i);
                break;
            }
        }

        centroids
    }
}
