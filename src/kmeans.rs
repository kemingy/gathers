//! K-means clustering implementation.

mod config;
#[cfg(not(feature = "perf"))]
mod matrix;
mod reduced;

use core::panic;
use std::borrow::Cow;
use std::time::Instant;

use aligned_vec::AVec;
pub use config::{KMeansConfig, ReductionConfig};
use log::debug;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
pub use reduced::{KMeansError, KMeansFit};

use crate::distance::{Distance, squared_euclidean};
use crate::rabitq::{RaBitQ, RaBitQWorkspace};
use crate::reduction::validate_shape;
use crate::sampling::subsample_flat;
use crate::utils::{centroid_residual, normalize_nonzero_row, try_normalize_rows};

const EPS: f32 = 1.0 / 1024.0;
const MIN_POINTS_PER_CENTROID: usize = 39;
const DEFAULT_SAMPLES_PER_CLUSTER: usize = 256;
const LARGE_CLUSTER_THRESHOLD: usize = 1 << 28;
const RAYON_BLOCK_SIZE: usize = 64;

fn normalize_assignment_rows(vecs: &[f32], dim: usize, distance: Distance) -> Cow<'_, [f32]> {
    if distance != Distance::Cosine {
        return Cow::Borrowed(vecs);
    }
    let mut normalized = vecs.to_vec();
    try_normalize_rows(&mut normalized, dim, false)
        .unwrap_or_else(|error| panic!("cannot assign cosine vectors: {error}"));
    Cow::Owned(normalized)
}

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
                    // Cosine inputs are normalized once at the public assignment boundary.
                    Distance::NegativeDotProduct | Distance::Cosine => {
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

fn validate_vectors(vecs: &[f32], dim: usize) {
    assert!(dim > 0, "dimension must be greater than zero");
    assert_eq!(vecs.len() % dim, 0, "vectors must be complete");
}

fn validate_assignment_inputs(vecs: &[f32], centroids: &[f32], dim: usize, labels: &[u32]) {
    validate_vectors(vecs, dim);
    assert_eq!(centroids.len() % dim, 0, "centroids must be complete");
    assert_eq!(
        labels.len(),
        vecs.len() / dim,
        "one label is required per vector"
    );
    assert!(!centroids.is_empty(), "at least one centroid is required");
}

/// Assign vectors to centroids with single-threaded scoring.
///
/// Cosine assignment requires finite nonzero row norms and normalizes copies of the inputs
/// once before dot-product scoring. Input slices are unchanged. Already-normalized callers
/// can use [`Distance::NegativeDotProduct`] to avoid copying and normalization.
/// Scores use the `f32` dot kernel, so near ties may differ from direct `f64` cosine scoring.
pub fn base_assign(
    vecs: &[f32],
    centroids: &[f32],
    dim: usize,
    distance: Distance,
    labels: &mut [u32],
) {
    validate_assignment_inputs(vecs, centroids, dim, labels);
    let vecs = normalize_assignment_rows(vecs, dim, distance);
    let centroids = normalize_assignment_rows(centroids, dim, distance);
    pulp::Arch::new().dispatch(AssignBlock {
        vecs: &vecs,
        centroids: &centroids,
        dim,
        distance,
        labels,
    });
}

/// Assign vectors to centroids in multi-threads.
///
/// Cosine assignment requires finite nonzero row norms and normalizes copies of the inputs
/// once before dot-product scoring. Input slices are unchanged. Already-normalized callers
/// can use [`Distance::NegativeDotProduct`] to avoid copying and normalization.
/// Scores use the `f32` dot kernel, so near ties may differ from direct `f64` cosine scoring.
pub fn base_assign_parallel(
    vecs: &[f32],
    centroids: &[f32],
    dim: usize,
    distance: Distance,
    labels: &mut [u32],
) {
    validate_assignment_inputs(vecs, centroids, dim, labels);
    let vecs = normalize_assignment_rows(vecs, dim, distance);
    let centroids = normalize_assignment_rows(centroids, dim, distance);
    labels
        .par_chunks_mut(RAYON_BLOCK_SIZE)
        .zip(vecs.par_chunks(dim * RAYON_BLOCK_SIZE))
        .for_each(|(labels, vecs)| {
            pulp::Arch::new().dispatch(AssignBlock {
                vecs,
                centroids: &centroids,
                dim,
                distance,
                labels,
            });
        });
}

/// Assign vectors to centroids with RaBitQ in single thread.
pub fn rabitq_assign(vecs: &[f32], centroids: &[f32], dim: usize, labels: &mut [u32]) {
    rabitq_assign_inner(vecs, centroids, dim, labels, &mut rand::rng());
}

fn rabitq_assign_inner<R: Rng + ?Sized>(
    vecs: &[f32],
    centroids: &[f32],
    dim: usize,
    labels: &mut [u32],
    rng: &mut R,
) {
    validate_assignment_inputs(vecs, centroids, dim, labels);

    let start = Instant::now();
    let rabitq = RaBitQ::new_with_rng(centroids, dim, rng);
    debug!("RaBitQ: build takes {} s", start.elapsed().as_secs_f32());

    let mut workspace = RaBitQWorkspace::new(rabitq.dim());
    let mut precise = 0;
    for (label, vector) in labels.iter_mut().zip(vecs.chunks_exact(dim)) {
        let (index, count) = rabitq.retrieve_top_one_with_workspace(vector, &mut workspace);
        *label = index as u32;
        precise += count;
    }
    rabitq.record_queries(labels.len(), precise);

    debug!("RaBitQ: {}", rabitq.metrics());
}

/// Assign vectors to centroids with RaBitQ in multi-threads.
///
/// TODO: support dot product distance
pub fn rabitq_assign_parallel(vecs: &[f32], centroids: &[f32], dim: usize, labels: &mut [u32]) {
    rabitq_assign_parallel_inner(vecs, centroids, dim, labels, &mut rand::rng());
}

fn rabitq_assign_parallel_inner<R: Rng + ?Sized>(
    vecs: &[f32],
    centroids: &[f32],
    dim: usize,
    labels: &mut [u32],
    rng: &mut R,
) {
    validate_assignment_inputs(vecs, centroids, dim, labels);

    let rabitq = RaBitQ::new_with_rng(centroids, dim, rng);
    rabitq.retrieve_top_one_batch(vecs, dim, labels);

    debug!("RaBitQ: {}", rabitq.metrics());
}

/// Update centroids to the mean of assigned vectors.
pub fn update_centroids(vecs: &[f32], centroids: &mut [f32], dim: usize, labels: &[u32]) -> f32 {
    let mut rng = rand::rng();
    update_centroids_inner(vecs, centroids, dim, labels, false, &mut rng)
}

fn update_centroids_inner<R: Rng + ?Sized>(
    vecs: &[f32],
    centroids: &mut [f32],
    dim: usize,
    labels: &[u32],
    unit_directions: bool,
    rng: &mut R,
) -> f32 {
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
    if unit_directions {
        for (mean, previous) in means.chunks_exact_mut(dim).zip(centroids.chunks_exact(dim)) {
            match normalize_nonzero_row(mean) {
                Some(true) => {}
                // With a zero cluster sum, every unit direction has the same dot objective.
                // Retain the previous direction; dot training may also have a zero seed.
                Some(false) => mean.copy_from_slice(previous),
                None => panic!("centroid has a non-finite norm"),
            }
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
#[derive(Debug, Default)]
pub struct KMeans {
    config: KMeansConfig,
}

impl KMeans {
    /// Construct a trainer with automatic defaults and explicit overrides.
    ///
    /// Settings are validated against the input shape during fitting, or ahead of loading
    /// through [`KMeansConfig::resolve`]. Construction does not fit a projection.
    ///
    /// ```
    /// use gathers::distance::Distance;
    /// use gathers::kmeans::{KMeans, KMeansConfig, ReductionConfig};
    /// use gathers::utils::as_continuous_vec;
    /// let model = KMeans::new(KMeansConfig {
    ///     n_clusters: Some(1), distance: Distance::Cosine, seed: Some(42),
    ///     reduction: ReductionConfig::PCA { output_dim: 2, training_samples: None },
    ///     ..Default::default()
    /// });
    /// let centroids = model.fit(as_continuous_vec(&[[1.0, 2.0, 3.0]; 40]), 3)?;
    /// assert_eq!(centroids.len(), 3);
    /// # Ok::<(), gathers::kmeans::KMeansError>(())
    /// ```
    pub fn new(config: KMeansConfig) -> Self {
        Self { config }
    }

    /// Select a training sample, fit K-means, and return original-dimensional centroids.
    ///
    /// Automatic settings use the original row count and dimension before sampling. Inputs must
    /// contain complete finite rows. Sampled cosine rows must be nonzero and are normalized before
    /// PCA fitting. Reduction always returns centroids in the original dimension.
    /// Sampling uses `config.samples_per_cluster` unless `config.training_samples` overrides it.
    pub fn fit(&self, vecs: AVec<f32>, dim: usize) -> Result<AVec<f32>, KMeansError> {
        Ok(self.fit_dispatch(vecs, dim, true)?.centroids)
    }

    /// Train every supplied row without further subsampling, returning centroids and labels.
    ///
    /// Automatic settings otherwise use the supplied sample's shape. For disk-loaded samples,
    /// first call [`KMeansConfig::resolve`] with original source metadata, then construct the
    /// trainer from that resolved config. At least 39 supplied rows per cluster are required.
    /// Labels describe the last training assignment, not reassignment to returned centroids.
    /// Empty-cluster repair can produce a centroid with no matching label. Inputs must be complete
    /// and finite; original cosine rows must be nonzero, while projected zero rows are allowed.
    /// Both `training_samples` and `samples_per_cluster` are ignored; all supplied rows are used.
    pub fn fit_sample(&self, vecs: AVec<f32>, dim: usize) -> Result<KMeansFit, KMeansError> {
        self.fit_dispatch(vecs, dim, false)
    }

    fn fit_dispatch(
        &self,
        vecs: AVec<f32>,
        dim: usize,
        subsample: bool,
    ) -> Result<KMeansFit, KMeansError> {
        if let Some(seed) = self.config.seed {
            self.fit_inner(vecs, dim, subsample, &mut StdRng::seed_from_u64(seed))
        } else {
            self.fit_inner(vecs, dim, subsample, &mut rand::rng())
        }
    }

    fn fit_inner<R: Rng + ?Sized>(
        &self,
        mut vecs: AVec<f32>,
        dim: usize,
        subsample: bool,
        rng: &mut R,
    ) -> Result<KMeansFit, KMeansError> {
        let rows = validate_shape(&vecs, dim)?;
        let config = if subsample {
            self.config
        } else {
            KMeansConfig {
                training_samples: Some(rows),
                ..self.config
            }
        }
        .resolve(rows, dim)?;
        debug!("resolved K-means config: {config:?}");
        let samples = config.training_samples.expect("sample size was resolved");
        if subsample && samples < rows {
            vecs = subsample_flat(samples, &vecs, dim, rng);
        }
        Self { config }.fit_prepared(vecs, dim, rng)
    }

    fn fit_raw<R: Rng + ?Sized>(
        &self,
        mut vecs: AVec<f32>,
        dim: usize,
        rng: &mut R,
    ) -> (AVec<f32>, Vec<u32>) {
        let num_clusters = self.config.n_clusters.expect("cluster count was resolved");
        // Center selected L2 rows so assignment uses smaller coordinates.
        let residual_mean =
            if self.config.distance == Distance::SquaredEuclidean && self.config.use_residual {
                debug!("use residual");
                Some(centroid_residual(&mut vecs, dim))
            } else {
                None
            };

        // Original and projected cosine rows were normalized in fit_prepared/fit_projected.
        let assignment_distance = if self.config.distance == Distance::Cosine {
            Distance::NegativeDotProduct
        } else {
            self.config.distance
        };

        let mut centroids = subsample_flat(num_clusters as usize, &vecs, dim, rng);
        if assignment_distance == Distance::NegativeDotProduct {
            for centroid in centroids.chunks_exact_mut(dim) {
                assert!(
                    normalize_nonzero_row(centroid).is_some(),
                    "centroid has a non-finite norm"
                );
            }
        }

        let training_num = vecs.len() / dim;
        let mut labels: Vec<u32> = vec![0; training_num];
        let use_exact_assignment = assignment_distance == Distance::NegativeDotProduct
            || training_num * dim <= LARGE_CLUSTER_THRESHOLD;
        #[cfg(not(feature = "perf"))]
        let mut matrix_workspace = if use_exact_assignment {
            matrix::MatrixAssignmentWorkspace::try_new(
                &vecs,
                centroids.len() / dim,
                dim,
                assignment_distance,
            )
        } else {
            None
        };
        debug!("start training");
        for i in 0..self.config.max_iter {
            let start_time = Instant::now();
            if use_exact_assignment {
                #[cfg(feature = "perf")]
                base_assign(&vecs, &centroids, dim, assignment_distance, &mut labels);
                #[cfg(not(feature = "perf"))]
                if let Some(workspace) = &mut matrix_workspace {
                    workspace.assign(&vecs, &centroids, dim, &mut labels);
                } else {
                    base_assign_parallel(&vecs, &centroids, dim, assignment_distance, &mut labels);
                }
            } else {
                #[cfg(feature = "perf")]
                rabitq_assign_inner(&vecs, &centroids, dim, &mut labels, rng);
                #[cfg(not(feature = "perf"))]
                rabitq_assign_parallel_inner(&vecs, &centroids, dim, &mut labels, rng);
            }
            let diff = update_centroids_inner(
                &vecs,
                &mut centroids,
                dim,
                &labels,
                assignment_distance == Distance::NegativeDotProduct,
                rng,
            );
            debug!("iter {} takes {} s", i, start_time.elapsed().as_secs_f32());
            if diff < self.config.tolerance {
                debug!("converged at iter {i}");
                break;
            }
        }

        if let Some(mean) = residual_mean {
            for centroid in centroids.chunks_mut(dim) {
                for (value, offset) in centroid.iter_mut().zip(&mean) {
                    *value += *offset;
                }
            }
        }

        (centroids, labels)
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use seed_rand::seeded_rng;

    use super::{
        KMeans, KMeansConfig, base_assign, base_assign_parallel, rabitq_assign,
        rabitq_assign_inner, rabitq_assign_parallel_inner, update_centroids,
        update_centroids_inner,
    };
    use crate::distance::{Distance, argmin, squared_euclidean};
    use crate::rabitq::RaBitQ;
    use crate::utils::as_continuous_vec;

    #[test]
    fn prepared_sample_uses_all_rows_above_the_automatic_cap() {
        let values = (0..600).map(|value| vec![value as f32]).collect::<Vec<_>>();
        let model = KMeans::new(KMeansConfig {
            n_clusters: Some(1),
            max_iter: 1,
            tolerance: 0.01,
            distance: Distance::SquaredEuclidean,
            use_residual: false,
            seed: Some(42),
            ..Default::default()
        });
        assert_eq!(
            model.config.resolve(600, 1).unwrap().training_samples,
            Some(256)
        );
        let fit = model.fit_sample(as_continuous_vec(&values), 1).unwrap();
        let centroids = fit.centroids;
        let labels = fit.labels;
        assert_eq!(&*centroids, &[299.5]);
        assert_eq!(labels, vec![0; 600]);
        let small = &values[..128];
        assert_eq!(
            model.config.resolve(128, 1).unwrap().training_samples,
            Some(128)
        );
        assert_eq!(
            model
                .fit_sample(as_continuous_vec(small), 1)
                .unwrap()
                .centroids,
            model.fit(as_continuous_vec(small), 1).unwrap()
        );
    }

    #[test]
    fn fit_sample_ignores_both_sampling_controls() {
        let rows = (0..600).map(|row| [row as f32]).collect::<Vec<_>>();
        for samples_per_cluster in [0, 128, 512] {
            let model = KMeans::new(KMeansConfig {
                n_clusters: Some(1),
                max_iter: 1,
                samples_per_cluster,
                training_samples: Some(39),
                seed: Some(42),
                ..Default::default()
            });
            let fit = model.fit_sample(as_continuous_vec(&rows), 1).unwrap();
            assert_eq!(&*fit.centroids, &[299.5]);
            assert_eq!(fit.labels.len(), rows.len());
        }
    }

    #[test]
    fn residual_fit_returns_centroids_in_input_coordinates() {
        let rows = (0..40)
            .map(|index| vec![10_000.0 + index as f32, -20_000.0 + 2.0 * index as f32])
            .collect::<Vec<_>>();
        let model = KMeans::new(KMeansConfig {
            n_clusters: Some(1),
            max_iter: 2,
            tolerance: 1e-4,
            distance: Distance::SquaredEuclidean,
            use_residual: true,
            seed: Some(42),
            ..Default::default()
        });

        for centroids in [
            model.fit(as_continuous_vec(&rows), 2).unwrap(),
            model
                .fit_sample(as_continuous_vec(&rows), 2)
                .unwrap()
                .centroids,
        ] {
            assert_eq!(&*centroids, &[10_019.5, -19_961.0]);
        }
    }

    #[test]
    fn cosine_training_normalizes_rows_but_dot_preserves_magnitudes() {
        let vectors = (0..78)
            .map(|row| {
                if row % 2 == 0 {
                    vec![100.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                }
            })
            .collect::<Vec<_>>();
        let fit = |distance| {
            KMeans::new(KMeansConfig {
                n_clusters: Some(1),
                max_iter: 1,
                tolerance: 0.01,
                distance,
                use_residual: false,
                seed: Some(42),
                ..Default::default()
            })
            .fit_sample(as_continuous_vec(&vectors), 2)
            .unwrap()
            .centroids
        };
        let cosine = fit(Distance::Cosine);
        let dot = fit(Distance::NegativeDotProduct);
        let diagonal = 0.5_f32.sqrt();
        assert!((cosine[0] - diagonal).abs() < 1e-5);
        assert!((cosine[1] - diagonal).abs() < 1e-5);
        assert!(dot[0] > 0.999);
        assert!(dot[1] < 0.011);
    }

    #[test]
    fn cosine_training_rejects_zero_vectors() {
        let mut vectors = vec![vec![1.0, 0.0]; 39];
        vectors[38] = vec![0.0, 0.0];
        assert!(
            KMeans::new(KMeansConfig {
                n_clusters: Some(1),
                distance: Distance::Cosine,
                ..Default::default()
            })
            .fit_sample(as_continuous_vec(&vectors), 2)
            .is_err()
        );
    }

    #[test]
    fn antipodal_cosine_cluster_keeps_a_unit_direction() {
        let vectors = (0..40)
            .map(|row| {
                if row % 2 == 0 {
                    vec![1.0, 0.0]
                } else {
                    vec![-1.0, 0.0]
                }
            })
            .collect::<Vec<_>>();
        let centroids = KMeans::new(KMeansConfig {
            n_clusters: Some(1),
            max_iter: 3,
            tolerance: 0.01,
            distance: Distance::Cosine,
            use_residual: false,
            seed: Some(42),
            ..Default::default()
        })
        .fit_sample(as_continuous_vec(&vectors), 2)
        .unwrap()
        .centroids;
        assert!(centroids.iter().all(|value| value.is_finite()));
        assert_eq!(centroids[0].abs(), 1.0);
        assert_eq!(centroids[1], 0.0);
        let mut labels = [0];
        base_assign(&[1.0, 0.0], &centroids, 2, Distance::Cosine, &mut labels);
        assert_eq!(labels, [0]);

        let mut previous = [1.0, 0.0];
        let flat = vectors.iter().flatten().copied().collect::<Vec<_>>();
        let diff = update_centroids_inner(
            &flat,
            &mut previous,
            2,
            &[0; 40],
            true,
            &mut StdRng::seed_from_u64(42),
        );
        assert_eq!(previous, [1.0, 0.0]);
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn dot_training_preserves_zero_seeds_and_zero_means() {
        let vectors = vec![vec![0.0, 0.0]; 40];
        let centroids = KMeans::new(KMeansConfig {
            n_clusters: Some(1),
            max_iter: 3,
            tolerance: 0.01,
            distance: Distance::NegativeDotProduct,
            use_residual: false,
            seed: Some(42),
            ..Default::default()
        })
        .fit_sample(as_continuous_vec(&vectors), 2)
        .unwrap()
        .centroids;
        assert_eq!(&*centroids, &[0.0, 0.0]);
    }

    #[test]
    fn cosine_assignment_accounts_for_centroid_magnitude() {
        let vector = [1.0, 2.0];
        let centroids = [100.0, 0.0, 1.0, 1.0];
        let mut labels = [0];
        base_assign(&vector, &centroids, 2, Distance::Cosine, &mut labels);
        assert_eq!(labels, [1]);
        base_assign_parallel(&vector, &centroids, 2, Distance::Cosine, &mut labels);
        assert_eq!(labels, [1]);
        base_assign(
            &vector,
            &centroids,
            2,
            Distance::NegativeDotProduct,
            &mut labels,
        );
        assert_eq!(labels, [0]);
    }

    #[test]
    fn cosine_assignment_matches_reference_across_scales_and_dimensions() {
        let tiny = f32::from_bits(1);
        for dim in [2, 3, 8, 9, 32, 65, 768] {
            let mut vectors = vec![0.0; 67 * dim];
            for (row, vector) in vectors.chunks_exact_mut(dim).enumerate() {
                let (left, right) = match row % 4 {
                    0 => (f32::MAX / 4.0, f32::MAX / 2.0),
                    1 => (tiny, 2.0 * tiny),
                    2 => (-3.0, 1.0),
                    _ => (1.0, -3.0),
                };
                vector[0] = left;
                vector[1] = right;
            }
            let mut centroids = vec![0.0; 4 * dim];
            for (centroid, (left, right)) in centroids.chunks_exact_mut(dim).zip([
                (100.0, 0.0),
                (tiny, tiny),
                (-f32::MAX, 0.0),
                (tiny, tiny), // Equal scores must retain the first centroid.
            ]) {
                centroid[0] = left;
                centroid[1] = right;
            }
            let original_vectors = vectors.clone();
            let original_centroids = centroids.clone();
            let expected = vectors
                .chunks_exact(dim)
                .map(|vector| {
                    // Independent f64 cosine reference, without normalizing f32 copies.
                    let norm = |row: &[f32]| {
                        row.iter()
                            .map(|&value| f64::from(value).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    };
                    centroids
                        .chunks_exact(dim)
                        .enumerate()
                        .map(|(index, centroid)| {
                            let dot = vector
                                .iter()
                                .zip(centroid)
                                .map(|(&left, &right)| f64::from(left) * f64::from(right))
                                .sum::<f64>();
                            (index as u32, -dot / norm(vector) / norm(centroid))
                        })
                        .min_by(|left, right| left.1.total_cmp(&right.1))
                        .unwrap()
                        .0
                })
                .collect::<Vec<_>>();
            let mut labels = vec![0; expected.len()];
            for assign in [base_assign, base_assign_parallel] {
                assign(&vectors, &centroids, dim, Distance::Cosine, &mut labels);
                assert_eq!(labels, expected, "dimension {dim}");
                assert_eq!(vectors, original_vectors);
                assert_eq!(centroids, original_centroids);
                // Empty query batches are permitted.
                assign(&[], &centroids, dim, Distance::Cosine, &mut []);
            }
        }
    }

    #[test]
    fn cosine_assignment_rejects_invalid_vector_and_centroid_norms() {
        for assign in [base_assign, base_assign_parallel] {
            for invalid in [0.0, f32::NAN, f32::INFINITY] {
                for (vectors, centroids) in
                    [([invalid, 0.0], [1.0, 0.0]), ([1.0, 0.0], [invalid, 0.0])]
                {
                    assert!(
                        std::panic::catch_unwind(|| {
                            assign(&vectors, &centroids, 2, Distance::Cosine, &mut [0]);
                        })
                        .is_err()
                    );
                }
            }
        }
    }

    #[test]
    fn rabitq_assignment_advances_the_supplied_rng() {
        let mut rng = seeded_rng();
        let dim = 65;
        let vecs = (0..37 * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let centroids = &vecs[..33 * dim];
        let seed = rng.random();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        pool.install(|| {
            for assign in [
                rabitq_assign_inner::<StdRng>,
                rabitq_assign_parallel_inner::<StdRng>,
            ] {
                let mut expected_rng = StdRng::seed_from_u64(seed);
                let mut actual_rng = StdRng::seed_from_u64(seed);
                for _ in 0..2 {
                    let index = RaBitQ::new_with_rng(centroids, dim, &mut expected_rng);
                    let mut expected = vec![0; 37];
                    index.retrieve_top_one_batch(&vecs, dim, &mut expected);

                    let mut actual = vec![0; 37];
                    assign(&vecs, centroids, dim, &mut actual, &mut actual_rng);
                    assert_eq!(actual, expected);
                    // Labels can match even with different rotations; also check the RNG stream.
                    assert_eq!(actual_rng.random::<u64>(), expected_rng.random::<u64>());
                }
            }
        });
    }

    #[test]
    fn test_fit_rejects_zero_dimension() {
        assert!(
            KMeans::default()
                .fit(as_continuous_vec(&[vec![1.0]]), 0)
                .is_err()
        );
    }

    #[test]
    fn test_fit_rejects_incomplete_vectors() {
        assert!(
            KMeans::default()
                .fit(as_continuous_vec(&[vec![1.0, 2.0, 3.0]]), 2)
                .is_err()
        );
    }

    #[test]
    fn test_fit_rejects_empty_input() {
        let vecs: Vec<Vec<f32>> = Vec::new();
        assert!(KMeans::default().fit(as_continuous_vec(&vecs), 1).is_err());
    }

    #[test]
    fn test_kmeans() {
        let mut rng = seeded_rng();
        let dim = 32;
        let n = 1000;
        let km = KMeans::default();
        let rabitq_match_rate = 0.99;

        for r in 0..1 {
            let vecs = (0..n)
                .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect::<Vec<f32>>())
                .collect::<Vec<Vec<f32>>>();
            let centroids = km.fit(as_continuous_vec(&vecs), dim).unwrap();

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
        let mut rng = seeded_rng();
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
