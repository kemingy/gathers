//! Training settings and shape-based automatic planning.

use std::str::FromStr;

use crate::distance::Distance;
use crate::kmeans::{DEFAULT_SAMPLES_PER_CLUSTER, KMeansError, MIN_POINTS_PER_CENTROID};

/// Dimensionality reduction used during training; centroids retain the original dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(
    clippy::upper_case_acronyms,
    reason = "Match the public transform names"
)]
pub enum ReductionConfig {
    /// PCA to 128 dimensions when source rows >= 1,000,000 and input dimension > 196;
    /// otherwise no reduction. This is a workload heuristic, not a recall guarantee.
    #[default]
    Auto,
    /// Train without dimensionality reduction.
    None,
    /// Fit PCA after sampling and cosine normalization.
    PCA {
        /// Number of projected coordinates, positive and smaller than the input dimension.
        output_dim: usize,
        /// Fitting rows; `None` uses min(training rows, 100 * input dimension).
        training_samples: Option<usize>,
    },
    /// Apply a seeded subsampled randomized Hadamard transform.
    SRHT {
        /// Number of projected coordinates, positive and smaller than the input dimension.
        output_dim: usize,
    },
}

impl FromStr for ReductionConfig {
    type Err = KMeansError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "raw" => Ok(Self::None),
            "pca" => Ok(Self::PCA {
                output_dim: 128,
                training_samples: None,
            }),
            "srht" => Ok(Self::SRHT { output_dim: 128 }),
            _ => Err(KMeansError::InvalidConfig(
                "reduction must be auto, raw, pca, or srht",
            )),
        }
    }
}

/// K-means settings, with automatic workload planning and explicit overrides.
///
/// Distance is application-specific and defaults to L2; it is never inferred from shape.
/// Use [`Self::resolve`] with original source metadata before loading an external sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KMeansConfig {
    /// Cluster count; `None` derives floor(rows^0.8 / 16), bounded by 39 rows per cluster.
    pub n_clusters: Option<u32>,
    /// Iteration limit, default 10.
    pub max_iter: u32,
    /// Absolute squared centroid-shift tolerance, default 1e-4.
    pub tolerance: f32,
    /// Training metric, default squared Euclidean.
    pub distance: Distance,
    /// Center L2 training rows to improve numerical accuracy; default false.
    pub use_residual: bool,
    /// Training rows per cluster, default 256. With no explicit total, `fit` retains
    /// min(source rows, samples_per_cluster * clusters); at least 39 are required.
    /// Ignored when `training_samples` is set or when using `fit_sample`.
    pub samples_per_cluster: usize,
    /// Exact training rows retained by `fit`, overriding `samples_per_cluster`.
    /// `None` derives the total from that factor. `fit_sample` always uses all supplied rows.
    pub training_samples: Option<usize>,
    /// Projection policy, default [`ReductionConfig::Auto`].
    pub reduction: ReductionConfig,
    /// Seed for source sampling, initialization, projections, and empty-cluster repair.
    /// `None` uses fresh randomness. Reproducibility requires the same build and worker count.
    pub seed: Option<u64>,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: None,
            max_iter: 10,
            tolerance: 1e-4,
            distance: Distance::default(),
            use_residual: false,
            samples_per_cluster: DEFAULT_SAMPLES_PER_CLUSTER,
            training_samples: None,
            reduction: ReductionConfig::Auto,
            seed: None,
        }
    }
}

impl KMeansConfig {
    /// Validate settings and resolve automatic choices using original source rows and dimension.
    ///
    /// The result has explicit cluster count, training sample size, reduction method, and PCA
    /// fitting budget. It can be passed to `KMeans::new(...).fit_sample(...)` after loading just
    /// those training rows, so automatic choices are not recomputed from the smaller sample.
    /// This inspects only metadata and does not allocate vector buffers. Explicit choices are
    /// never silently reduced to fit memory. Fewer than 39 rows per cluster are rejected.
    pub fn resolve(mut self, rows: usize, dim: usize) -> Result<Self, KMeansError> {
        if dim == 0 || rows == 0 {
            return Err(KMeansError::InvalidConfig(
                "rows and dimension must be positive",
            ));
        }
        if self.max_iter == 0 || !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(KMeansError::InvalidConfig(
                "iterations and finite tolerance must be positive",
            ));
        }
        let clusters = self.n_clusters.unwrap_or_else(|| {
            (((rows as f64).powf(0.8) / 16.0).floor().max(1.0) as u32)
                .min((rows / MIN_POINTS_PER_CENTROID).min(u32::MAX as usize) as u32)
        });
        if clusters == 0 || rows / (clusters as usize) < MIN_POINTS_PER_CENTROID {
            return Err(KMeansError::InvalidConfig(
                "at least 39 rows per cluster are required",
            ));
        }
        if self.training_samples.is_none() && self.samples_per_cluster < MIN_POINTS_PER_CENTROID {
            return Err(KMeansError::InvalidConfig(
                "samples per cluster must be at least 39",
            ));
        }
        let samples = self.training_samples.unwrap_or_else(|| {
            rows.min((clusters as usize).saturating_mul(self.samples_per_cluster))
        });
        if samples == 0 || samples > rows {
            return Err(KMeansError::InvalidConfig(
                "training samples must be positive and no larger than the source",
            ));
        }
        if samples / (clusters as usize) < MIN_POINTS_PER_CENTROID {
            return Err(KMeansError::InvalidConfig(
                "training samples require at least 39 rows per cluster",
            ));
        }
        self.n_clusters = Some(clusters);
        self.training_samples = Some(samples);
        if self.reduction == ReductionConfig::Auto {
            self.reduction = if rows >= 1_000_000 && dim > 196 {
                ReductionConfig::PCA {
                    output_dim: 128,
                    training_samples: None,
                }
            } else {
                ReductionConfig::None
            };
        }
        let output_dim = match self.reduction {
            ReductionConfig::PCA {
                output_dim,
                training_samples,
            } => {
                let fitting_rows =
                    training_samples.unwrap_or_else(|| samples.min(dim.saturating_mul(100)));
                if !(2..=samples).contains(&fitting_rows) {
                    return Err(KMeansError::InvalidConfig(
                        "PCA fitting samples must be between 2 and training samples",
                    ));
                }
                self.reduction = ReductionConfig::PCA {
                    output_dim,
                    training_samples: Some(fitting_rows),
                };
                Some(output_dim)
            }
            ReductionConfig::SRHT { output_dim } => Some(output_dim),
            ReductionConfig::None => None,
            ReductionConfig::Auto => unreachable!("auto reduction was resolved above"),
        };
        if output_dim.is_some_and(|output_dim| output_dim == 0 || output_dim >= dim) {
            return Err(KMeansError::InvalidConfig(
                "reduced dimension must be positive and smaller than input dimension",
            ));
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{KMeansConfig, ReductionConfig};

    #[test]
    fn auto_pca_boundaries_and_explicit_overrides() {
        for (rows, dim, pca) in [
            (999_999, 197, false),
            (1_000_000, 196, false),
            (1_000_000, 197, true),
            (1_000_001, 768, true),
            (2_000_000, 128, false),
        ] {
            let config = KMeansConfig::default().resolve(rows, dim).unwrap();
            assert_eq!(
                matches!(
                    config.reduction,
                    ReductionConfig::PCA {
                        output_dim: 128,
                        ..
                    }
                ),
                pca
            );
            assert_eq!(config.resolve(rows, dim).unwrap(), config);
        }
        for reduction in [
            ReductionConfig::None,
            ReductionConfig::SRHT { output_dim: 64 },
            ReductionConfig::PCA {
                output_dim: 32,
                training_samples: Some(100),
            },
        ] {
            let config = KMeansConfig {
                n_clusters: Some(1),
                training_samples: Some(200),
                reduction,
                ..Default::default()
            }
            .resolve(1_000_000, 197)
            .unwrap();
            assert_eq!(config.n_clusters, Some(1));
            assert_eq!(config.training_samples, Some(200));
            assert_eq!(config.reduction, reduction);
        }
    }

    #[test]
    fn automatic_cluster_and_fitting_budgets_are_bounded() {
        let config = KMeansConfig::default().resolve(10_000_000, 768).unwrap();
        assert_eq!(config.n_clusters, Some(24_881));
        assert_eq!(config.training_samples, Some(6_369_536));
        assert_eq!(
            config.reduction,
            ReductionConfig::PCA {
                output_dim: 128,
                training_samples: Some(76_800),
            }
        );
        for rows in [39, 40, 256, 600, 1_000_000] {
            let config = KMeansConfig::default().resolve(rows, 3).unwrap();
            let clusters = config.n_clusters.unwrap() as usize;
            let samples = config.training_samples.unwrap();
            assert!(samples <= rows && samples / clusters >= 39 && samples <= 256 * clusters);
        }
        assert!(KMeansConfig::default().resolve(38, 3).is_err());
        assert!(
            KMeansConfig {
                n_clusters: Some(0),
                ..Default::default()
            }
            .resolve(100, 3)
            .is_err()
        );
        assert!(
            KMeansConfig {
                tolerance: f32::NAN,
                ..Default::default()
            }
            .resolve(100, 3)
            .is_err()
        );
    }

    #[test]
    fn sampling_factor_is_bounded_and_explicit_total_takes_precedence() {
        let base = KMeansConfig {
            n_clusters: Some(2),
            ..Default::default()
        };
        for (samples_per_cluster, expected) in [
            (39, 78),
            (128, 256),
            (256, 512),
            (512, 1024),
            (usize::MAX, 3000),
        ] {
            let config = KMeansConfig {
                samples_per_cluster,
                ..base
            }
            .resolve(3000, 3)
            .unwrap();
            assert_eq!(config.training_samples, Some(expected));
        }
        for samples_per_cluster in [0, 38, 128, 512] {
            let config = KMeansConfig {
                samples_per_cluster,
                ..base
            };
            if samples_per_cluster < 39 {
                assert!(config.resolve(3000, 3).is_err());
            }
            let explicit = KMeansConfig {
                training_samples: Some(100),
                ..config
            }
            .resolve(3000, 3)
            .unwrap();
            assert_eq!(explicit.training_samples, Some(100));
        }
        let default = KMeansConfig::default().resolve(10_000_000, 768).unwrap();
        let reduced = KMeansConfig {
            samples_per_cluster: 128,
            ..Default::default()
        }
        .resolve(10_000_000, 768)
        .unwrap();
        assert_eq!(reduced.n_clusters, default.n_clusters);
        assert_eq!(
            reduced.training_samples.unwrap() * 2,
            default.training_samples.unwrap()
        );
        // PCA's independent dimension-based fitting cap is unchanged at this scale.
        assert_eq!(reduced.reduction, default.reduction);
        let small_pca = KMeansConfig {
            samples_per_cluster: 39,
            reduction: ReductionConfig::PCA {
                output_dim: 1,
                training_samples: None,
            },
            ..base
        }
        .resolve(3000, 3)
        .unwrap();
        assert_eq!(
            small_pca.reduction,
            ReductionConfig::PCA {
                output_dim: 1,
                training_samples: Some(78),
            }
        );
    }
}
