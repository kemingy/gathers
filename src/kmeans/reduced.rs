//! Shared raw/projected training and original-space centroid reconstruction.

use std::time::{Duration, Instant};

use aligned_vec::AVec;
use rand::{Rng, RngExt};

use crate::distance::Distance;
use crate::kmeans::{KMeans, ReductionConfig};
use crate::reduction::{PCA, Reduction, ReductionError, SRHT, validate_shape};
use crate::utils::{NormalizeRowsError, try_normalize_rows};

/// Error returned by K-means configuration resolution or training.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KMeansError {
    /// Invalid training settings or insufficient rows.
    #[error("{0}")]
    InvalidConfig(&'static str),
    /// Projection construction, input validation, or transformation failed.
    #[error(transparent)]
    Reduction(#[from] ReductionError),
    /// Cosine inputs or reconstructed centroids could not be normalized.
    #[error(transparent)]
    Normalization(#[from] NormalizeRowsError),
}

/// Centroids, final training labels, and stage timings for a prepared sample.
///
/// Centroids have the original dimension, whether training was raw or projected.
#[derive(Debug)]
pub struct KMeansFit {
    /// Original-dimensional centroids, using original-space means for occupied projected clusters.
    pub centroids: AVec<f32>,
    /// Final training assignment for each supplied row; not reassignment to returned centroids.
    pub labels: Vec<u32>,
    /// Dimension used for training assignment.
    pub training_dim: usize,
    /// Empty final projected clusters reconstructed using an inverse transform; zero for raw fits.
    pub empty_clusters: usize,
    /// Retained variance fraction for PCA; absent for raw and SRHT fits.
    pub preserved_variance: Option<f64>,
    /// Time normalizing original and projected cosine rows.
    pub normalization_time: Duration,
    /// Time fitting PCA or constructing SRHT; zero for raw fits.
    pub projection_fit_time: Duration,
    /// Time projecting the sample, excluding normalization; zero for raw fits.
    pub projection_transform_time: Duration,
    /// Time fitting K-means, excluding normalization and projection.
    pub fit_time: Duration,
    /// Time reconstructing and normalizing original-space projected centroids; zero for raw fits.
    pub reconstruction_time: Duration,
}

impl KMeans {
    pub(crate) fn fit_prepared<R: Rng + ?Sized>(
        &self,
        mut vectors: AVec<f32>,
        dim: usize,
        rng: &mut R,
    ) -> Result<KMeansFit, KMeansError> {
        let start = Instant::now();
        if self.config.distance == Distance::Cosine {
            try_normalize_rows(&mut vectors, dim, false)?;
        }
        let normalization_time = if self.config.distance == Distance::Cosine {
            start.elapsed()
        } else {
            Duration::ZERO
        };
        let start = Instant::now();
        let mut fit = match self.config.reduction {
            ReductionConfig::None => {
                let (centroids, labels) = self.fit_raw(vectors, dim, rng);
                KMeansFit {
                    centroids,
                    labels,
                    training_dim: dim,
                    empty_clusters: 0,
                    preserved_variance: None,
                    normalization_time: Duration::ZERO,
                    projection_fit_time: Duration::ZERO,
                    projection_transform_time: Duration::ZERO,
                    fit_time: start.elapsed(),
                    reconstruction_time: Duration::ZERO,
                }
            }
            ReductionConfig::PCA {
                output_dim,
                training_samples,
            } => {
                let projection = PCA::fit_sample(
                    &vectors,
                    dim,
                    output_dim,
                    training_samples.expect("PCA fitting budget was resolved"),
                    self.config.seed.unwrap_or_else(|| rng.random()),
                )?;
                let projection_fit_time = start.elapsed();
                let mut fit = self.fit_projected(vectors, &projection, rng)?;
                fit.projection_fit_time = projection_fit_time;
                fit.preserved_variance = Some(projection.preserved_variance());
                fit
            }
            ReductionConfig::SRHT { output_dim } => {
                let projection = SRHT::new(
                    dim,
                    output_dim,
                    self.config.seed.unwrap_or_else(|| rng.random()),
                )?;
                let projection_fit_time = start.elapsed();
                let mut fit = self.fit_projected(vectors, &projection, rng)?;
                fit.projection_fit_time = projection_fit_time;
                fit
            }
            ReductionConfig::Auto => unreachable!("reduction was resolved before training"),
        };
        fit.normalization_time += normalization_time;
        Ok(fit)
    }

    fn fit_projected<P: Reduction, R: Rng + ?Sized>(
        &self,
        vectors: AVec<f32>,
        projection: &P,
        rng: &mut R,
    ) -> Result<KMeansFit, KMeansError> {
        let training_dim = projection.output_dim();
        let start = Instant::now();
        let mut projected = projection.transform(&vectors)?;
        let projection_transform_time = start.elapsed();
        validate_shape(&projected, training_dim)?;
        let start = Instant::now();
        let normalization_time = if self.config.distance == Distance::Cosine {
            try_normalize_rows(&mut projected, training_dim, true)?;
            start.elapsed()
        } else {
            Duration::ZERO
        };
        let start = Instant::now();
        let (reduced_centroids, labels) = self.fit_raw(projected, training_dim, rng);
        let fit_time = start.elapsed();
        let start = Instant::now();
        let (centroids, empty_clusters) = reconstruct_original_centroids(
            projection,
            &vectors,
            &reduced_centroids,
            &labels,
            self.config.distance,
        )?;
        Ok(KMeansFit {
            centroids,
            labels,
            training_dim,
            empty_clusters,
            preserved_variance: None,
            normalization_time,
            projection_fit_time: Duration::ZERO,
            projection_transform_time,
            fit_time,
            reconstruction_time: start.elapsed(),
        })
    }
}

fn reconstruct_original_centroids<R: Reduction + ?Sized>(
    projection: &R,
    original_vectors: &[f32],
    reduced_centroids: &[f32],
    labels: &[u32],
    distance: Distance,
) -> Result<(AVec<f32>, usize), KMeansError> {
    let input_dim = projection.input_dim();
    let output_dim = projection.output_dim();
    let num_centroids = reduced_centroids.len() / output_dim;
    debug_assert_eq!(labels.len(), original_vectors.len() / input_dim);
    // Inverse projection loses null-space information. Use it only for empty clusters;
    // occupied centroids are exact original-space means unless the mean direction is zero.
    let mut centroids = projection.inverse_transform(reduced_centroids)?;
    let inverse_rows = validate_shape(&centroids, input_dim)?;
    debug_assert_eq!(inverse_rows, num_centroids);
    let mut counts = vec![0_usize; num_centroids];
    let mut representatives = if distance == Distance::SquaredEuclidean {
        Vec::new()
    } else {
        vec![None; num_centroids]
    };
    for (row, (&label, vector)) in labels
        .iter()
        .zip(original_vectors.chunks_exact(input_dim))
        .enumerate()
    {
        let index = label as usize;
        let centroid = &mut centroids[index * input_dim..(index + 1) * input_dim];
        if counts[index] == 0 {
            centroid.fill(0.0);
        }
        if distance != Distance::SquaredEuclidean
            && representatives[index].is_none()
            && vector.iter().any(|&value| value != 0.0)
        {
            representatives[index] = Some(row);
        }
        counts[index] += 1;
        centroid
            .iter_mut()
            .zip(vector)
            .for_each(|(sum, &value)| *sum += value);
    }
    for (index, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let inverse = (count as f32).recip();
        for value in &mut centroids[index * input_dim..(index + 1) * input_dim] {
            *value *= inverse;
        }
    }
    let empty_clusters = counts.iter().filter(|&&count| count == 0).count();
    if distance != Distance::SquaredEuclidean {
        for (index, centroid) in centroids.chunks_exact_mut(input_dim).enumerate() {
            if centroid.iter().all(|&value| value == 0.0) {
                // A zero cosine mean needs a unit representative. Dot may retain a zero cluster.
                let representative =
                    representatives[index].or_else(|| (distance == Distance::Cosine).then_some(0));
                if let Some(row) = representative {
                    centroid
                        .copy_from_slice(&original_vectors[row * input_dim..(row + 1) * input_dim]);
                }
            }
        }
        try_normalize_rows(
            &mut centroids,
            input_dim,
            distance == Distance::NegativeDotProduct,
        )?;
    }
    Ok((centroids, empty_clusters))
}

#[cfg(test)]
mod tests {
    use super::{KMeansError, reconstruct_original_centroids};
    use crate::distance::Distance;
    use crate::kmeans::{KMeans, KMeansConfig, ReductionConfig};
    use crate::reduction::{PCA, Reduction, SRHT};
    use crate::utils::{NormalizeRowsError, as_continuous_vec};

    #[test]
    fn projected_assignments_reconstruct_original_space_means() {
        let projection = SRHT::new(2, 1, 42).unwrap();
        let original = [-1.0, 100.0, -2.0, 200.0, 1.0, 300.0, 2.0, 400.0];
        let reduced_centroids = [-1.5, 1.5, 7.0];
        let fallback = projection.inverse_transform(&reduced_centroids).unwrap();
        let (centroids, empty) = reconstruct_original_centroids(
            &projection,
            &original,
            &reduced_centroids,
            &[0, 0, 1, 1],
            Distance::SquaredEuclidean,
        )
        .unwrap();
        assert_eq!(empty, 1);
        assert_eq!(&centroids[..4], &[-1.5, 150.0, 1.5, 350.0]);
        assert_eq!(&centroids[4..], &fallback[4..]);
    }

    #[test]
    fn projected_antipodal_cluster_uses_an_assigned_unit_direction() {
        let projection = SRHT::new(2, 1, 42).unwrap();
        let (centroids, empty) = reconstruct_original_centroids(
            &projection,
            &[1.0, 0.0, -1.0, 0.0],
            &[1.0, 0.0],
            &[0, 0],
            Distance::Cosine,
        )
        .unwrap();
        assert_eq!(empty, 1);
        assert_eq!(&*centroids, &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn projected_dot_training_can_preserve_zero_centroids() {
        let projection = SRHT::new(2, 1, 42).unwrap();
        let (centroids, empty) = reconstruct_original_centroids(
            &projection,
            &[0.0; 4],
            &[0.0; 2],
            &[0, 0],
            Distance::NegativeDotProduct,
        )
        .unwrap();
        assert_eq!(empty, 1);
        assert_eq!(&*centroids, &[0.0; 4]);
    }

    #[test]
    fn reduced_fit_returns_original_means_and_one_label_per_sample_row() {
        let rows = (0..78)
            .map(|row| [row as f32, (row % 3) as f32 * 100.0, -10.0])
            .collect::<Vec<_>>();
        for reduction in [
            ReductionConfig::PCA {
                output_dim: 1,
                training_samples: None,
            },
            ReductionConfig::SRHT { output_dim: 1 },
        ] {
            let model = KMeans::new(KMeansConfig {
                n_clusters: Some(2),
                max_iter: 2,
                seed: Some(42),
                reduction,
                ..Default::default()
            });
            let fit = model.fit_sample(as_continuous_vec(&rows), 3).unwrap();
            assert_eq!(fit.labels.len(), rows.len());
            assert_eq!(fit.centroids.len(), 6);
            assert_eq!(fit.training_dim, 1);
            for (cluster, centroid) in fit.centroids.as_chunks::<3>().0.iter().enumerate() {
                let assigned = rows
                    .iter()
                    .zip(&fit.labels)
                    .filter(|(_, label)| **label as usize == cluster)
                    .map(|(row, _)| row)
                    .collect::<Vec<_>>();
                if assigned.is_empty() {
                    continue;
                }
                for coordinate in 0..3 {
                    let mean = assigned
                        .iter()
                        .map(|row| f64::from(row[coordinate]))
                        .sum::<f64>()
                        / assigned.len() as f64;
                    assert!(
                        (f64::from(centroid[coordinate]) - mean).abs() < 1e-4 + 1e-6 * mean.abs()
                    );
                }
            }
        }
    }

    #[test]
    fn cosine_normalizes_before_pca_and_allows_projected_zeros() {
        for distance in [
            Distance::SquaredEuclidean,
            Distance::Cosine,
            Distance::NegativeDotProduct,
        ] {
            let model = KMeans::new(KMeansConfig {
                n_clusters: Some(1),
                max_iter: 1,
                distance,
                seed: Some(42),
                reduction: ReductionConfig::PCA {
                    output_dim: 1,
                    training_samples: None,
                },
                ..Default::default()
            });
            let fit = model
                .fit_sample(as_continuous_vec(&[[2.0, 0.0]; 40]), 2)
                .unwrap();
            assert_eq!(
                &*fit.centroids,
                if distance == Distance::SquaredEuclidean {
                    &[2.0, 0.0]
                } else {
                    &[1.0, 0.0]
                }
            );
            assert_eq!(fit.labels, vec![0; 40]);
        }
        let model = KMeans::new(KMeansConfig {
            n_clusters: Some(1),
            distance: Distance::Cosine,
            reduction: ReductionConfig::PCA {
                output_dim: 1,
                training_samples: None,
            },
            ..Default::default()
        });
        let mut rows = [[1.0, 2.0]; 40];
        rows[3] = [0.0, 0.0];
        assert_eq!(
            model.fit_sample(as_continuous_vec(&rows), 2).unwrap_err(),
            KMeansError::Normalization(NormalizeRowsError::InvalidNorm(3))
        );
    }

    #[test]
    fn cosine_projected_labels_match_pca_fitted_on_normalized_rows() {
        let rows = (0..78)
            .map(|row| {
                [
                    (1 + row % 7) as f32 * if row % 2 == 0 { 100.0 } else { 1.0 },
                    (1 + row % 3) as f32,
                ]
            })
            .collect::<Vec<_>>();
        let mut normalized = as_continuous_vec(&rows);
        crate::utils::try_normalize_rows(&mut normalized, 2, false).unwrap();
        let projection = PCA::fit(&normalized, 2, 1).unwrap();
        let mut projected = projection.transform(&normalized).unwrap();
        crate::utils::try_normalize_rows(&mut projected, 1, true).unwrap();
        let config = KMeansConfig {
            n_clusters: Some(2),
            max_iter: 2,
            distance: Distance::Cosine,
            seed: Some(42),
            reduction: ReductionConfig::PCA {
                output_dim: 1,
                training_samples: None,
            },
            ..Default::default()
        };
        let actual = KMeans::new(config)
            .fit_sample(as_continuous_vec(&rows), 2)
            .unwrap();
        let expected = KMeans::new(KMeansConfig {
            distance: Distance::NegativeDotProduct,
            reduction: ReductionConfig::None,
            ..config
        })
        .fit_sample(projected, 1)
        .unwrap();
        assert_eq!(actual.labels, expected.labels);
    }

    #[test]
    fn source_metadata_keeps_auto_pca_for_a_small_disk_loaded_sample() {
        let config = KMeansConfig {
            n_clusters: Some(1),
            max_iter: 1,
            training_samples: Some(40),
            seed: Some(42),
            ..Default::default()
        }
        .resolve(1_000_000, 197)
        .unwrap();
        let rows = vec![vec![1.0; 197]; 40];
        let fit = KMeans::new(config)
            .fit_sample(as_continuous_vec(&rows), 197)
            .unwrap();
        assert_eq!(fit.training_dim, 128);
        assert_eq!(fit.labels.len(), 40);
        assert_eq!(&*fit.centroids, &vec![1.0; 197]);
        let direct_sample = KMeans::default()
            .fit_sample(as_continuous_vec(&rows), 197)
            .unwrap();
        assert_eq!(direct_sample.training_dim, 197);
    }

    #[test]
    fn fit_samples_but_fit_sample_uses_every_row() {
        let rows = (0..600).map(|row| [row as f32, 1.0]).collect::<Vec<_>>();
        let model = KMeans::new(KMeansConfig {
            n_clusters: Some(1),
            max_iter: 1,
            seed: Some(42),
            reduction: ReductionConfig::PCA {
                output_dim: 1,
                training_samples: None,
            },
            ..Default::default()
        });
        let sampled = model.fit(as_continuous_vec(&rows), 2).unwrap();
        assert_ne!(sampled[0], 299.5);
        let prepared = model.fit_sample(as_continuous_vec(&rows), 2).unwrap();
        assert_eq!(&*prepared.centroids, &[299.5, 1.0]);
        assert_eq!(prepared.labels.len(), 600);
    }
}
