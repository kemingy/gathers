//! K-means clustering implementation.

use core::panic;
use std::time::Instant;

use aligned_vec::AVec;
use log::debug;
use rand::Rng;
use rayon::prelude::*;

use crate::distance::{Distance, argmin, neg_dot_product, squared_euclidean};
use crate::rabitq::RaBitQ;
use crate::sampling::subsample;
use crate::utils::{as_continuous_vec, centroid_residual, normalize};

const EPS: f32 = 1.0 / 1024.0;
const MIN_POINTS_PER_CENTROID: usize = 39;
const MAX_POINTS_PER_CENTROID: usize = 256;
const LARGE_CLUSTER_THRESHOLD: usize = 1 << 28;
const RAYON_BLOCK_SIZE: usize = 64;

/// Assign vectors to centroids in single thread.
pub fn base_assign(
    vecs: &[f32],
    centroids: &[f32],
    dim: usize,
    distance: Distance,
    labels: &mut [u32],
) {
    let mut distances = vec![f32::MAX; centroids.len() / dim];
    match distance {
        Distance::NegativeDotProduct => {
            for (i, vec) in vecs.chunks(dim).enumerate() {
                for (j, centroid) in centroids.chunks(dim).enumerate() {
                    distances[j] = neg_dot_product(vec, centroid);
                }
                labels[i] = argmin(&distances) as u32;
            }
        }
        Distance::SquaredEuclidean => {
            for (i, vec) in vecs.chunks(dim).enumerate() {
                for (j, centroid) in centroids.chunks(dim).enumerate() {
                    distances[j] = squared_euclidean(vec, centroid);
                }
                labels[i] = argmin(&distances) as u32;
            }
        }
    }
}

/// Assign vectors to centroids in multi-threads.
pub fn base_assign_parallel(
    vecs: &[f32],
    centroids: &[f32],
    dim: usize,
    distance: Distance,
    labels: &mut [u32],
) {
    match distance {
        Distance::NegativeDotProduct => {
            let mut distances = vec![f32::MAX; centroids.len() / dim];
            for (i, vec) in vecs.chunks(dim).enumerate() {
                for (j, centroid) in centroids.chunks(dim).enumerate() {
                    distances[j] = neg_dot_product(vec, centroid);
                }
                labels[i] = argmin(&distances) as u32;
            }
        }
        Distance::SquaredEuclidean => {
            // pre-compute the x**2 & y**2 for L2 distance
            // let squared_x: Vec<f32> = vecs.chunks(dim).map(l2_norm).collect();
            // let squared_y: Vec<f32> = centroids.chunks(dim).map(l2_norm).collect();

            labels.copy_from_slice(
                &vecs
                    .par_chunks(dim * RAYON_BLOCK_SIZE)
                    .flat_map(|vec| {
                        let mut par_labels = vec![0; vec.len() / dim];
                        let mut par_distances = vec![f32::MAX; centroids.len() / dim];
                        for (i, v) in vec.chunks(dim).enumerate() {
                            for (j, centroid) in centroids.chunks(dim).enumerate() {
                                par_distances[j] = squared_euclidean(v, centroid);
                            }
                            par_labels[i] = argmin(&par_distances) as u32;
                        }
                        par_labels
                    })
                    .collect::<Vec<_>>(),
            );
        }
    }
}

/// Assign vectors to centroids with RaBitQ in single thread.
pub fn rabitq_assign(vecs: &[f32], centroids: &[f32], dim: usize, labels: &mut [u32]) {
    let start = Instant::now();
    let rabitq = RaBitQ::new(centroids, dim);
    debug!("RaBitQ: build takes {} s", start.elapsed().as_secs_f32());

    for (i, vec) in vecs.chunks(dim).enumerate() {
        labels[i] = rabitq.retrieve_top_one(vec) as u32;
    }

    let (rough, precise) = rabitq.get_metrics();
    debug!(
        "RaBitQ: rough_cmp({}), precise_cmp({}), ratio({})",
        rough,
        precise,
        rough as f32 / precise as f32
    )
}

/// Assign vectors to centroids with RaBitQ in multi-threads.
///
/// TODO: support dot product distance
pub fn rabitq_assign_parallel(vecs: &[f32], centroids: &[f32], dim: usize, labels: &mut [u32]) {
    let rabitq = RaBitQ::new(centroids, dim);

    labels.copy_from_slice(
        &vecs
            .par_chunks(dim * RAYON_BLOCK_SIZE)
            .flat_map(|vec| {
                vec.chunks(dim)
                    .map(|v| rabitq.retrieve_top_one(v) as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    );

    let (rough, precise) = rabitq.get_metrics();
    debug!(
        "RaBitQ: rough_cmp({}), precise_cmp({}), ratio({})",
        rough,
        precise,
        rough as f32 / precise as f32
    )
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
            let mut rng = rand::rng();
            let base = 1.0 / (vecs.len() / dim - labels.len()) as f32;
            loop {
                let p = (elements[target] - 1) as f32 * base;
                if rng.random::<f32>() < p {
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
    use_default_config: bool,
}

impl Default for KMeans {
    fn default() -> Self {
        Self {
            n_cluster: 8,
            max_iter: 25,
            tolerance: 1e-4,
            distance: Distance::default(),
            use_residual: false,
            use_default_config: true,
        }
    }
}

impl KMeans {
    /// Create a new KMeans instance.
    ///
    /// # Arguments
    ///
    /// * `n_cluster` - number of clusters, recommend to be a number in [sqrt(n) * 4, sqrt(n) * 8]
    /// * `max_iter` - max number of iterations
    /// * `tolerance` - convergence tolerance, stop when the diff is less than this value
    /// * `distance` - distance metric
    /// * `use_residual` - use residual for more accurate L2 distance computations, only work for L2
    pub fn new(
        n_cluster: u32,
        max_iter: u32,
        tolerance: f32,
        distance: Distance,
        use_residual: bool,
    ) -> Self {
        if n_cluster < 1 {
            panic!("n_cluster must be greater than 0");
        }
        if max_iter < 1 {
            panic!("max_iter must be greater than 0");
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
            use_default_config: false,
        }
    }

    /// Fit the KMeans configurations to the given vectors and return the centroids.
    pub fn fit(&self, mut vecs: AVec<f32>, dim: usize) -> AVec<f32> {
        let num = vecs.len() / dim;

        // auto-config the `n_cluster` if it's initialized with `default()`
        let n_cluster = match self.use_default_config {
            true => (((num as f32).sqrt() as u32) * 4).min((num / MIN_POINTS_PER_CENTROID) as u32),
            false => self.n_cluster,
        };
        debug!("num of points: {}, num of clusters: {}", num, n_cluster);

        if num < n_cluster as usize {
            panic!("number of samples must be greater than n_cluster");
        }
        if num < n_cluster as usize * MIN_POINTS_PER_CENTROID {
            panic!("too few samples for n_cluster");
        }

        // use residual for more accurate L2 distance computations
        if self.distance == Distance::SquaredEuclidean && self.use_residual {
            debug!("use residual");
            centroid_residual(&mut vecs, dim);
        }

        // subsample
        if num > MAX_POINTS_PER_CENTROID * n_cluster as usize {
            let n_sample = MAX_POINTS_PER_CENTROID * n_cluster as usize;
            debug!("subsample to {} points", n_sample);
            vecs = as_continuous_vec(&subsample(n_sample, &vecs, dim));
        }

        let mut centroids = as_continuous_vec(&subsample(n_cluster as usize, &vecs, dim));
        if self.distance == Distance::NegativeDotProduct {
            centroids.chunks_mut(dim).for_each(normalize);
        }

        let mut labels: Vec<u32> = vec![0; num];
        debug!("start training");
        for i in 0..self.max_iter {
            let start_time = Instant::now();
            if self.distance == Distance::NegativeDotProduct || num * dim <= LARGE_CLUSTER_THRESHOLD
            {
                #[cfg(feature = "perf")]
                base_assign(&vecs, &centroids, dim, self.distance, &mut labels);
                #[cfg(not(feature = "perf"))]
                base_assign_parallel(&vecs, &centroids, dim, self.distance, &mut labels);
            } else {
                #[cfg(feature = "perf")]
                rabitq_assign(&vecs, &centroids, dim, &mut labels);
                #[cfg(not(feature = "perf"))]
                rabitq_assign_parallel(&vecs, &centroids, dim, &mut labels);
            }
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

#[cfg(test)]
mod test {
    use rand::Rng;

    use super::{KMeans, base_assign, rabitq_assign};
    use crate::distance::{Distance, argmin, squared_euclidean};
    use crate::utils::as_continuous_vec;

    #[test]
    fn test_kmeans() {
        let mut rng = rand::rng();
        let dim = 32;
        let n = 1000;
        let km = KMeans::default();
        let rabitq_match_rate = 0.99;

        for r in 0..1 {
            let vecs = (0..n)
                .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect::<Vec<f32>>())
                .collect::<Vec<Vec<f32>>>();
            let centroids = km.fit(as_continuous_vec(&vecs), dim);

            let mut labels = vec![0; n];
            for (i, vec) in vecs.iter().enumerate() {
                let mut distances = vec![f32::MAX; centroids.len() / dim];
                for (j, centroid) in centroids.chunks(dim).enumerate() {
                    distances[j] = squared_euclidean(vec.as_slice(), centroid);
                }
                labels[i] = argmin(&distances) as u32;
            }

            let continue_vecs = as_continuous_vec(&vecs);

            // check the base assignment
            let mut base_labels = vec![0; n];
            base_assign(
                &continue_vecs,
                &centroids,
                dim,
                Distance::SquaredEuclidean,
                &mut base_labels,
            );
            assert_eq!(labels, base_labels);

            // check the rabitq assignment
            let mut rabitq_labels = vec![0; n];
            rabitq_assign(&continue_vecs, &centroids, dim, &mut rabitq_labels);
            let mut match_count = 0;
            for i in 0..n {
                if labels[i] == rabitq_labels[i] {
                    match_count += 1;
                }
            }
            let match_rate = match_count as f32 / n as f32;
            assert!(
                match_rate >= rabitq_match_rate,
                "round: {}, match rate: {}",
                r,
                match_rate
            );
        }
    }
}
