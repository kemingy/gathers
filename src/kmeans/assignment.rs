//! Public fixed-centroid assignment using the shared guarded kernels and probe policy.

use std::borrow::Cow;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::distance::Distance;
use crate::kmeans::hierarchy::{GROUPED_ROWS, Hierarchy, ProbeOptions, ProbeWorkspace};
use crate::kmeans::matrix::MatrixAssignmentWorkspace;
use crate::kmeans::{AssignmentPolicy, KMeansError, base_assign, base_assign_parallel};
use crate::reduction::validate_shape;
use crate::utils::try_normalize_rows;

/// Reusable assignment to fixed, flat row-major centroids.
///
/// Construction validates centroids and builds an eligible automatic hierarchy once.
/// Each nonempty batch calibrates 512 uniformly sampled pilots (or all rows for smaller
/// batches) with `min(p100, ceil(1.10 * p99))`. Approximate probing has no recall guarantee.
/// Exact assignment uses guarded GEMM or direct SIMD according to workload size.
///
/// Centroids are borrowed for L2/dot and copied once for cosine normalization. Query
/// normalization uses bounded reusable scratch; input slices are never modified. New
/// query batches recompute matrix norms, so changing queries cannot reuse stale norms.
/// The `perf` feature retains serial exact assignment.
///
/// ```
/// use gathers::distance::Distance;
/// use gathers::kmeans::{AssignmentPolicy, CentroidAssigner};
///
/// let centroids = [0.0, 0.0, 1.0, 1.0];
/// let mut assigner = CentroidAssigner::new(
///     &centroids, 2, Distance::SquaredEuclidean, AssignmentPolicy::Auto, Some(42),
/// )?;
/// let mut labels = [0; 2];
/// assigner.assign(&[0.1, 0.2, 0.9, 0.8], &mut labels)?;
/// assert_eq!(labels, [0, 1]);
/// # Ok::<(), gathers::kmeans::KMeansError>(())
/// ```
pub struct CentroidAssigner<'a> {
    centroids: Cow<'a, [f32]>,
    dim: usize,
    distance: Distance,
    hierarchy: Option<Hierarchy>,
    pilots: ProbeWorkspace,
    rng: StdRng,
    normalized: Vec<f32>,
}

impl<'a> CentroidAssigner<'a> {
    /// Validate nonempty finite centroids and prepare the requested policy.
    ///
    /// Cosine centroids must be nonzero. A seed reproduces hierarchy construction and
    /// successive pilot samples for the same batch sequence, build and worker count.
    /// Numerical hierarchy-build failures select exact assignment.
    pub fn new(
        centroids: &'a [f32],
        dim: usize,
        distance: Distance,
        assignment: AssignmentPolicy,
        seed: Option<u64>,
    ) -> Result<Self, KMeansError> {
        let count = validate_shape(centroids, dim)?;
        if count > u32::MAX as usize {
            return Err(KMeansError::InvalidConfig(
                "centroid count exceeds u32 labels",
            ));
        }
        let centroids = if distance == Distance::Cosine {
            let mut values = centroids.to_vec();
            try_normalize_rows(&mut values, dim, false)?;
            Cow::Owned(values)
        } else {
            Cow::Borrowed(centroids)
        };
        let mut rng =
            seed.map_or_else(|| StdRng::from_rng(&mut rand::rng()), StdRng::seed_from_u64);
        let hierarchy = if !cfg!(feature = "perf")
            && assignment == AssignmentPolicy::Auto
            && Hierarchy::supports(count)
        {
            Hierarchy::build(&centroids, dim, Some(rng.random()))
        } else {
            None
        };
        Ok(Self {
            centroids,
            dim,
            distance,
            hierarchy,
            pilots: ProbeWorkspace::default(),
            rng,
            normalized: Vec::new(),
        })
    }

    /// Assign complete finite query rows, writing one original centroid index per row.
    ///
    /// Empty queries with empty labels are accepted. Cosine queries must be nonzero.
    /// Shape, finite-value and cosine-norm errors are checked before changing labels.
    /// Finite extreme scores can still overflow; rescale such inputs for meaningful
    /// rankings. Exact scoring follows the existing direct `f32` reference and tie rules.
    pub fn assign(&mut self, vectors: &[f32], labels: &mut [u32]) -> Result<(), KMeansError> {
        if !vectors.len().is_multiple_of(self.dim) || labels.len() != vectors.len() / self.dim {
            return Err(KMeansError::InvalidConfig(
                "complete queries require one label per row",
            ));
        }
        if !vectors.is_empty() {
            validate_shape(vectors, self.dim)?;
        }
        if self.distance == Distance::Cosine
            && let Some(row) = vectors
                .chunks_exact(self.dim)
                .position(|row| row.iter().all(|&value| value == 0.0))
        {
            return Err(crate::utils::NormalizeRowsError::InvalidNorm(row).into());
        }
        if labels.is_empty() {
            return Ok(());
        }
        let budget = self.hierarchy.as_ref().and_then(|hierarchy| {
            self.pilots
                .calibrate(hierarchy, vectors, self.distance, &mut self.rng)
        });
        let distance = if self.distance == Distance::Cosine {
            Distance::NegativeDotProduct
        } else {
            self.distance
        };
        let adaptive = self.hierarchy.as_ref().zip(budget);
        if self.distance == Distance::Cosine && adaptive.is_some() {
            // At most one scratch buffer per source task, with at most as many tasks
            // as workers. Nested GEMM/normalization can steal other source tasks, so
            // partition explicitly instead of allocating scratch per small group.
            let rows_per_task = labels
                .len()
                .div_ceil(GROUPED_ROWS)
                .div_ceil(rayon::current_num_threads())
                * GROUPED_ROWS;
            vectors
                .par_chunks(rows_per_task * self.dim)
                .zip(labels.par_chunks_mut(rows_per_task))
                .try_for_each_init(Vec::new, |normalized, (vectors, labels)| {
                    for (vectors, labels) in vectors
                        .chunks(GROUPED_ROWS * self.dim)
                        .zip(labels.chunks_mut(GROUPED_ROWS))
                    {
                        normalized.clear();
                        normalized.extend_from_slice(vectors);
                        try_normalize_rows(normalized, self.dim, false)?;
                        assign_batch(
                            normalized,
                            &self.centroids,
                            self.dim,
                            distance,
                            labels,
                            adaptive,
                        );
                    }
                    Ok::<(), KMeansError>(())
                })?;
        } else if self.distance == Distance::Cosine {
            // Calibrate on the full batch once; normalization must not cap pilot coverage
            // to the first chunk or allocate a second full-dataset query buffer.
            for (vectors, labels) in vectors
                .chunks(GROUPED_ROWS * self.dim)
                .zip(labels.chunks_mut(GROUPED_ROWS))
            {
                self.normalized.clear();
                self.normalized.extend_from_slice(vectors);
                try_normalize_rows(&mut self.normalized, self.dim, false)?;
                assign_batch(
                    &self.normalized,
                    &self.centroids,
                    self.dim,
                    distance,
                    labels,
                    adaptive,
                );
            }
        } else {
            assign_batch(
                vectors,
                &self.centroids,
                self.dim,
                distance,
                labels,
                adaptive,
            );
        }
        Ok(())
    }
}

/// Assign a batch with the same automatic/exact policy as K-means training.
///
/// Inputs are flat row-major slices, `dim` must be positive, and `labels` must contain
/// one slot per query. Cosine requires finite nonzero rows; L2/dot require finite rows.
/// Use [`CentroidAssigner`] to retain a fixed-centroid hierarchy across batch calls.
/// Exact refers to assignment in the supplied space; no projection is applied here.
pub fn batch_assign(
    vectors: &[f32],
    centroids: &[f32],
    dim: usize,
    distance: Distance,
    assignment: AssignmentPolicy,
    seed: Option<u64>,
    labels: &mut [u32],
) -> Result<(), KMeansError> {
    CentroidAssigner::new(centroids, dim, distance, assignment, seed)?.assign(vectors, labels)
}

fn assign_batch(
    vectors: &[f32],
    centroids: &[f32],
    dim: usize,
    distance: Distance,
    labels: &mut [u32],
    adaptive: Option<(&Hierarchy, usize)>,
) {
    if let Some((hierarchy, budget)) = adaptive
        && hierarchy.probe_checked(
            vectors,
            labels,
            ProbeOptions {
                distance,
                budget,
                incumbent_centroids: None,
            },
        )
    {
        return;
    }
    let mut matrix = if cfg!(feature = "perf") {
        None
    } else {
        MatrixAssignmentWorkspace::try_new(vectors, centroids.len() / dim, dim, distance)
    };
    assign_exact(vectors, centroids, dim, distance, labels, matrix.as_mut());
}

pub(crate) fn assign_exact(
    vectors: &[f32],
    centroids: &[f32],
    dim: usize,
    distance: Distance,
    labels: &mut [u32],
    matrix: Option<&mut MatrixAssignmentWorkspace>,
) {
    if cfg!(feature = "perf") {
        base_assign(vectors, centroids, dim, distance, labels);
    } else if let Some(matrix) = matrix {
        matrix.assign(vectors, centroids, dim, labels);
    } else {
        base_assign_parallel(vectors, centroids, dim, distance, labels);
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use seed_rand::seeded_rng;

    use super::{CentroidAssigner, batch_assign};
    use crate::distance::Distance;
    use crate::kmeans::{AssignmentPolicy, base_assign_parallel};

    #[test]
    fn exact_and_small_auto_match_direct_across_changed_query_batches() {
        let mut rng = seeded_rng();
        for dim in [1, 33, 128, 129] {
            let centers: Vec<_> = (0..257 * dim)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            for distance in [
                Distance::SquaredEuclidean,
                Distance::Cosine,
                Distance::NegativeDotProduct,
            ] {
                for policy in [AssignmentPolicy::Auto, AssignmentPolicy::Exact] {
                    let mut assigner =
                        CentroidAssigner::new(&centers, dim, distance, policy, Some(42)).unwrap();
                    for rows in [129, 17, 0] {
                        let vectors: Vec<_> = (0..rows * dim)
                            .map(|_| rng.random_range(-1.0..1.0))
                            .collect();
                        let original = vectors.clone();
                        let mut expected = vec![0; rows];
                        base_assign_parallel(&vectors, &centers, dim, distance, &mut expected);
                        let mut actual = vec![u32::MAX; rows];
                        assigner.assign(&vectors, &mut actual).unwrap();
                        assert_eq!(actual, expected);
                        assert_eq!(vectors, original);
                    }
                }
                let mut actual = [u32::MAX];
                batch_assign(
                    &centers[..dim],
                    &centers,
                    dim,
                    distance,
                    AssignmentPolicy::Exact,
                    Some(42),
                    &mut actual,
                )
                .unwrap();
                let mut expected = [0];
                base_assign_parallel(&centers[..dim], &centers, dim, distance, &mut expected);
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    #[cfg(not(feature = "perf"))]
    fn eligible_batches_reuse_hierarchy_and_match_reference_probing() {
        use crate::kmeans::hierarchy::ProbeOptions;
        use crate::kmeans::hierarchy::policy::{DEFAULT_PERCENTILE, budget};
        use crate::utils::try_normalize_rows;

        let mut rng = seeded_rng();
        let dim = 128;
        let centers: Vec<_> = (0..3136 * dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        for distance in [
            Distance::SquaredEuclidean,
            Distance::Cosine,
            Distance::NegativeDotProduct,
        ] {
            let mut first =
                CentroidAssigner::new(&centers, dim, distance, AssignmentPolicy::Auto, Some(42))
                    .unwrap();
            let mut second =
                CentroidAssigner::new(&centers, dim, distance, AssignmentPolicy::Auto, Some(42))
                    .unwrap();
            let original_hierarchy = std::ptr::from_ref(first.hierarchy.as_ref().unwrap());
            for rows in [65, 513] {
                let mut vectors: Vec<_> = (0..rows * dim)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect();
                let original = vectors.clone();
                let mut actual = vec![u32::MAX; rows];
                let mut repeated = actual.clone();
                first.assign(&vectors, &mut actual).unwrap();
                second.assign(&vectors, &mut repeated).unwrap();
                assert_eq!(actual, repeated);
                assert_eq!(vectors, original);
                let distance = if distance == Distance::Cosine {
                    try_normalize_rows(&mut vectors, dim, false).unwrap();
                    Distance::NegativeDotProduct
                } else {
                    distance
                };
                let mut expected = vec![u32::MAX; rows];
                let hierarchy = first.hierarchy.as_ref().unwrap();
                hierarchy.probe_parallel(
                    &vectors,
                    &mut expected,
                    &mut [],
                    ProbeOptions {
                        distance,
                        budget: budget(&first.pilots.required, DEFAULT_PERCENTILE),
                        incumbent_centroids: None,
                    },
                );
                assert_eq!(actual, expected);
                assert_eq!(std::ptr::from_ref(hierarchy), original_hierarchy);
                assert_eq!(first.pilots.required.len(), rows.min(512));
            }
        }
    }

    #[test]
    #[cfg(not(feature = "perf"))]
    fn adaptive_cosine_handles_multiple_normalization_groups_and_tail() {
        use crate::kmeans::hierarchy::ProbeOptions;
        use crate::kmeans::hierarchy::policy::{DEFAULT_PERCENTILE, budget};
        use crate::utils::try_normalize_rows;

        let centers: Vec<_> = (0..3136)
            .flat_map(|index| {
                let angle = index as f32 * std::f32::consts::TAU / 3136.0;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let mut assigner = CentroidAssigner::new(
            &centers,
            2,
            Distance::Cosine,
            AssignmentPolicy::Auto,
            Some(42),
        )
        .unwrap();
        let queries = [2.0, 1.0, -1.0, 3.0];
        let rows = super::GROUPED_ROWS + 1;
        let vectors: Vec<_> = (0..rows)
            .flat_map(|row| queries[row % 2 * 2..row % 2 * 2 + 2].iter().copied())
            .collect();
        let mut labels = vec![u32::MAX; rows];
        assigner.assign(&vectors, &mut labels).unwrap();
        let mut normalized = queries;
        try_normalize_rows(&mut normalized, 2, false).unwrap();
        let mut expected = [u32::MAX; 2];
        assigner.hierarchy.as_ref().unwrap().probe_parallel(
            &normalized,
            &mut expected,
            &mut [],
            ProbeOptions {
                distance: Distance::NegativeDotProduct,
                budget: budget(&assigner.pilots.required, DEFAULT_PERCENTILE),
                incumbent_centroids: None,
            },
        );
        assert!(
            labels
                .iter()
                .enumerate()
                .all(|(row, &label)| label == expected[row % 2])
        );
        assert!(assigner.normalized.is_empty());
    }

    #[test]
    fn invalid_queries_leave_labels_unchanged_and_cosine_batches_are_bounded() {
        for (centers, dim) in [
            (&[][..], 1),
            (&[1.0][..], 0),
            (&[1.0][..], 2),
            (&[f32::NAN][..], 1),
        ] {
            assert!(
                CentroidAssigner::new(
                    centers,
                    dim,
                    Distance::SquaredEuclidean,
                    AssignmentPolicy::Auto,
                    Some(42)
                )
                .is_err()
            );
        }
        let mut assigner = CentroidAssigner::new(
            &[-2.0, 3.0],
            1,
            Distance::Cosine,
            AssignmentPolicy::Exact,
            None,
        )
        .unwrap();
        let mut labels = [7, 7];
        for vectors in [
            &[1.0][..],
            &[1.0, 0.0][..],
            &[1.0, f32::NAN][..],
            &[f32::INFINITY, 1.0][..],
        ] {
            assert!(assigner.assign(vectors, &mut labels).is_err());
            assert_eq!(labels, [7, 7]);
        }
        let vectors: Vec<_> = (0..super::GROUPED_ROWS + 1)
            .map(|row| if row % 2 == 0 { -100.0 } else { 0.5 })
            .collect();
        let mut labels = vec![u32::MAX; vectors.len()];
        assigner.assign(&vectors, &mut labels).unwrap();
        assert!(
            labels
                .iter()
                .enumerate()
                .all(|(row, &label)| label == (row % 2) as u32)
        );
        assert_eq!(assigner.normalized.len(), 1);
        assert!(assigner.normalized.capacity() <= super::GROUPED_ROWS);
        // A bad row after the first processing chunk must still fail before any write.
        let mut invalid = vectors;
        invalid[super::GROUPED_ROWS] = 0.0;
        labels.fill(7);
        assert!(assigner.assign(&invalid, &mut labels).is_err());
        assert!(labels.iter().all(|&label| label == 7));
    }

    #[test]
    fn numerical_build_failure_falls_back_to_exact_for_public_assignment() {
        let centers = vec![f32::MAX; 3136];
        let mut assigner = CentroidAssigner::new(
            &centers,
            1,
            Distance::SquaredEuclidean,
            AssignmentPolicy::Auto,
            Some(42),
        )
        .unwrap();
        assert!(assigner.hierarchy.is_none());
        let mut labels = [u32::MAX];
        assigner.assign(&[1.0], &mut labels).unwrap();
        assert_eq!(labels, [0]);
    }
}
