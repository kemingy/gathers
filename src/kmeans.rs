//! K-means clustering implementation.

use core::panic;
use std::time::Instant;

use aligned_vec::AVec;
use log::debug;
use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut};

use crate::distance::{Distance, squared_euclidean};
use crate::rabitq::{RaBitQ, RaBitQWorkspace};
use crate::sampling::subsample;
use crate::utils::{as_continuous_vec, centroid_residual, normalize};

const EPS: f32 = 1.0 / 1024.0;
const MIN_POINTS_PER_CENTROID: usize = 39;
const MAX_POINTS_PER_CENTROID: usize = 256;
const LARGE_CLUSTER_THRESHOLD: usize = 1 << 28;
const RAYON_BLOCK_SIZE: usize = 64;

struct AssignBlock<'a> {
    vecs: &'a [f32],
    centroids: &'a [f32],
    dim: usize,
    distance: Distance,
    labels: &'a mut [u32],
}

impl pulp::WithSimd for AssignBlock<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let Self {
            vecs,
            centroids,
            dim,
            distance,
            labels,
        } = self;

        for (label, vector) in labels.iter_mut().zip(vecs.chunks_exact(dim)) {
            let mut best_distance = f32::MAX;
            let mut best_index = 0;
            for (index, centroid) in centroids.chunks_exact(dim).enumerate() {
                let candidate = match distance {
                    Distance::SquaredEuclidean => {
                        crate::simd::pulp::l2_squared_distance(simd, vector, centroid)
                    }
                    Distance::NegativeDotProduct => {
                        -crate::simd::pulp::dot_product(simd, vector, centroid)
                    }
                };
                if candidate < best_distance {
                    best_distance = candidate;
                    best_index = index;
                }
            }
            *label = best_index as u32;
        }
    }
}

fn validate_assignment_inputs(vecs: &[f32], centroids: &[f32], dim: usize, labels: &[u32]) {
    assert!(dim > 0, "dimension must be greater than zero");
    assert_eq!(vecs.len() % dim, 0, "vectors must be complete");
    assert_eq!(centroids.len() % dim, 0, "centroids must be complete");
    assert_eq!(
        labels.len(),
        vecs.len() / dim,
        "one label is required per vector"
    );
    assert!(!centroids.is_empty(), "at least one centroid is required");
}

/// Assign vectors to centroids in single thread.
pub fn base_assign(
    vecs: &[f32],
    centroids: &[f32],
    dim: usize,
    distance: Distance,
    labels: &mut [u32],
) {
    validate_assignment_inputs(vecs, centroids, dim, labels);
    pulp::Arch::new().dispatch(AssignBlock {
        vecs,
        centroids,
        dim,
        distance,
        labels,
    });
}

/// Assign vectors to centroids in multi-threads.
pub fn base_assign_parallel(
    vecs: &[f32],
    centroids: &[f32],
    dim: usize,
    distance: Distance,
    labels: &mut [u32],
) {
    validate_assignment_inputs(vecs, centroids, dim, labels);
    labels
        .par_chunks_mut(RAYON_BLOCK_SIZE)
        .zip(vecs.par_chunks(dim * RAYON_BLOCK_SIZE))
        .for_each(|(labels, vecs)| {
            pulp::Arch::new().dispatch(AssignBlock {
                vecs,
                centroids,
                dim,
                distance,
                labels,
            });
        });
}

/// Assign vectors to centroids with RaBitQ in single thread.
pub fn rabitq_assign(vecs: &[f32], centroids: &[f32], dim: usize, labels: &mut [u32]) {
    validate_assignment_inputs(vecs, centroids, dim, labels);

    let start = Instant::now();
    let rabitq = RaBitQ::new(centroids, dim);
    debug!("RaBitQ: build takes {} s", start.elapsed().as_secs_f32());

    let mut workspace = RaBitQWorkspace::new(rabitq.dim());
    let mut precise = 0;
    for (label, vector) in labels.iter_mut().zip(vecs.chunks_exact(dim)) {
        let (index, count) = rabitq.retrieve_top_one_with_workspace(vector, &mut workspace);
        *label = index as u32;
        precise += count;
    }
    let rough = u64::try_from(labels.len())
        .expect("label count exceeds u64")
        .checked_mul(u64::try_from(rabitq.len()).expect("centroid count exceeds u64"))
        .expect("comparison count exceeds u64");
    rabitq.update_metrics(rough, precise);

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
    validate_assignment_inputs(vecs, centroids, dim, labels);

    let rabitq = RaBitQ::new(centroids, dim);
    rabitq.retrieve_top_one_batch(vecs, dim, labels);

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
    validate_assignment_inputs(vecs, centroids, dim, labels);
    let num_centroids = centroids.len() / dim;
    assert!(
        labels.len() >= num_centroids,
        "number of vectors must be at least the number of centroids"
    );
    assert!(
        labels.iter().all(|&label| (label as usize) < num_centroids),
        "labels must reference an existing centroid"
    );

    let mut means = vec![0.0; centroids.len()];
    let mut cluster_sizes = vec![0usize; num_centroids];
    for (i, vec) in vecs.chunks(dim).enumerate() {
        let label = labels[i] as usize;
        cluster_sizes[label] += 1;
        means[label * dim..(label + 1) * dim]
            .iter_mut()
            .zip(vec.iter())
            .for_each(|(m, &v)| *m += v);
    }
    let mut empty_cluster_count = 0;
    for i in 0..cluster_sizes.len() {
        if cluster_sizes[i] != 0 {
            let divider = (cluster_sizes[i] as f32).recip();
            means[i * dim..(i + 1) * dim]
                .iter_mut()
                .for_each(|value| *value *= divider);
        }
    }

    for empty_cluster in 0..cluster_sizes.len() {
        if cluster_sizes[empty_cluster] == 0 {
            // need to split another cluster to fill this empty cluster
            empty_cluster_count += 1;
            let mut rng = rand::rng();
            let total_weight: usize = cluster_sizes
                .iter()
                .map(|&size| size.saturating_sub(1))
                .sum();
            let mut donor_sample = rng.random_range(0..total_weight);
            let mut donor_cluster = 0;
            for (candidate, &size) in cluster_sizes.iter().enumerate() {
                let donor_weight = size.saturating_sub(1);
                if donor_sample < donor_weight {
                    donor_cluster = candidate;
                    break;
                }
                donor_sample -= donor_weight;
            }
            debug!("split cluster {donor_cluster} to fill empty cluster {empty_cluster}");
            if empty_cluster < donor_cluster {
                let (left, right) = means.split_at_mut(donor_cluster * dim);
                left[empty_cluster * dim..(empty_cluster + 1) * dim].copy_from_slice(&right[..dim]);
            } else {
                let (left, right) = means.split_at_mut(empty_cluster * dim);
                right[..dim].copy_from_slice(&left[donor_cluster * dim..(donor_cluster + 1) * dim]);
            }
            // small symmetric perturbation
            for j in 0..dim {
                if j % 2 == 0 {
                    means[empty_cluster * dim + j] *= 1.0 + EPS;
                    means[donor_cluster * dim + j] *= 1.0 - EPS;
                } else {
                    means[empty_cluster * dim + j] *= 1.0 - EPS;
                    means[donor_cluster * dim + j] *= 1.0 + EPS;
                }
            }
            // update cluster sizes
            cluster_sizes[empty_cluster] = cluster_sizes[donor_cluster] / 2;
            cluster_sizes[donor_cluster] -= cluster_sizes[empty_cluster];
        }
    }
    let diff = squared_euclidean(centroids, &means);
    centroids.copy_from_slice(&means);
    if empty_cluster_count != 0 {
        debug!("fixed {empty_cluster_count} empty clusters");
    }
    diff
}

/// K-means clustering algorithm.
#[derive(Debug)]
pub struct KMeans {
    num_clusters: u32,
    max_iter: u32,
    tolerance: f32,
    distance: Distance,
    use_residual: bool,
    use_default_config: bool,
}

impl Default for KMeans {
    fn default() -> Self {
        Self {
            num_clusters: 8,
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
    /// * `num_clusters` - number of clusters, recommend to be a number in [sqrt(n) * 4, sqrt(n) * 8]
    /// * `max_iter` - max number of iterations
    /// * `tolerance` - convergence tolerance, stop when the diff is less than this value
    /// * `distance` - distance metric
    /// * `use_residual` - use residual for more accurate L2 distance computations, only work for L2
    pub fn new(
        num_clusters: u32,
        max_iter: u32,
        tolerance: f32,
        distance: Distance,
        use_residual: bool,
    ) -> Self {
        if num_clusters < 1 {
            panic!("num_clusters must be greater than 0");
        }
        if max_iter < 1 {
            panic!("max_iter must be greater than 0");
        }
        if tolerance <= 0.0 {
            panic!("tolerance must be greater than 0.0");
        }
        Self {
            num_clusters,
            max_iter,
            tolerance,
            distance,
            use_residual,
            use_default_config: false,
        }
    }

    /// Fit the KMeans configurations to the given vectors and return the centroids.
    pub fn fit(&self, mut vecs: AVec<f32>, dim: usize) -> AVec<f32> {
        let num_vectors = vecs.len() / dim;

        // auto-config `num_clusters` when initialized with `default()`
        let num_clusters = match self.use_default_config {
            true => (((num_vectors as f32).sqrt() as u32) * 4)
                .min((num_vectors / MIN_POINTS_PER_CENTROID) as u32),
            false => self.num_clusters,
        };
        debug!("num of points: {num_vectors}, num of clusters: {num_clusters}");

        if num_vectors < num_clusters as usize {
            panic!("number of samples must be greater than num_clusters");
        }
        if num_vectors < num_clusters as usize * MIN_POINTS_PER_CENTROID {
            panic!("too few samples for num_clusters");
        }

        // use residual for more accurate L2 distance computations
        if self.distance == Distance::SquaredEuclidean && self.use_residual {
            debug!("use residual");
            centroid_residual(&mut vecs, dim);
        }

        // subsample
        if num_vectors > MAX_POINTS_PER_CENTROID * num_clusters as usize {
            let n_sample = MAX_POINTS_PER_CENTROID * num_clusters as usize;
            debug!("subsample to {n_sample} points");
            vecs = as_continuous_vec(&subsample(n_sample, &vecs, dim));
        }

        let mut centroids = as_continuous_vec(&subsample(num_clusters as usize, &vecs, dim));
        if self.distance == Distance::NegativeDotProduct {
            centroids.chunks_mut(dim).for_each(normalize);
        }

        let training_num = vecs.len() / dim;
        let mut labels: Vec<u32> = vec![0; training_num];
        debug!("start training");
        for i in 0..self.max_iter {
            let start_time = Instant::now();
            if self.distance == Distance::NegativeDotProduct
                || training_num * dim <= LARGE_CLUSTER_THRESHOLD
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
                debug!("converged at iter {i}");
                break;
            }
        }

        centroids
    }
}

#[cfg(test)]
mod test {
    use rand::Rng;

    use super::{KMeans, base_assign, base_assign_parallel, rabitq_assign, update_centroids};
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

            let flattened_vecs = as_continuous_vec(&vecs);

            // check the base assignment
            let mut base_labels = vec![0; n];
            base_assign(
                &flattened_vecs,
                &centroids,
                dim,
                Distance::SquaredEuclidean,
                &mut base_labels,
            );
            assert_eq!(labels, base_labels);

            // check the rabitq assignment
            let mut rabitq_labels = vec![0; n];
            rabitq_assign(&flattened_vecs, &centroids, dim, &mut rabitq_labels);
            let mut match_count = 0;
            for i in 0..n {
                if labels[i] == rabitq_labels[i] {
                    match_count += 1;
                }
            }
            let match_rate = match_count as f32 / n as f32;
            assert!(
                match_rate >= rabitq_match_rate,
                "round: {r}, match rate: {match_rate}"
            );
        }
    }

    #[test]
    fn test_parallel_assignment_matches_single_thread() {
        let mut rng = rand::rng();
        let dim = 32;
        let vecs = (0..257 * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let centroids = (0..17 * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();

        for distance in [Distance::SquaredEuclidean, Distance::NegativeDotProduct] {
            let mut expected = vec![0; 257];
            let mut actual = vec![0; 257];
            base_assign(&vecs, &centroids, dim, distance, &mut expected);
            base_assign_parallel(&vecs, &centroids, dim, distance, &mut actual);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    #[should_panic(expected = "vectors must be complete")]
    fn test_rabitq_assignment_rejects_incomplete_vectors() {
        let mut labels = [0];
        rabitq_assign(&[0.0, 1.0, 2.0], &[0.0, 1.0], 2, &mut labels);
    }

    #[test]
    fn test_update_centroids_returns_distance_to_new_means() {
        let vecs = vec![0.0, 0.0, 2.0, 2.0];
        let labels = vec![0, 0, 1, 1];
        let mut centroids = vec![0.0, 4.0];

        let diff = update_centroids(&vecs, &mut centroids, 1, &labels);

        assert_eq!(centroids, vec![0.0, 2.0]);
        assert_eq!(diff, 4.0);
    }

    #[test]
    #[should_panic(expected = "number of vectors must be at least the number of centroids")]
    fn test_update_centroids_rejects_more_centroids_than_vectors() {
        let mut centroids = vec![0.0, 1.0];
        update_centroids(&[0.0], &mut centroids, 1, &[0]);
    }

    #[test]
    fn test_update_centroids_repairs_empty_cluster() {
        let vecs = vec![2.0, 2.0, 2.0, 2.0];
        let labels = vec![0, 0, 0, 0];
        let mut centroids = vec![0.0, 10.0];

        let diff = update_centroids(&vecs, &mut centroids, 1, &labels);

        assert!(diff.is_finite());
        assert!(centroids.iter().all(|value| (*value - 2.0).abs() < 0.01));
        assert_ne!(centroids[0], centroids[1]);
    }
}
