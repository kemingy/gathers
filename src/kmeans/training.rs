//! Per-fit state for exact and pilot-calibrated Lloyd assignment.

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use crate::distance::Distance;
use crate::kmeans::AssignmentPolicy;
use crate::kmeans::assignment::assign_exact;
use crate::kmeans::hierarchy::{Hierarchy, ProbeOptions, ProbeWorkspace};
use crate::kmeans::matrix::MatrixAssignmentWorkspace;

pub(crate) struct TrainingWorkspace {
    matrix: Option<MatrixAssignmentWorkspace>,
    rng: Option<StdRng>,
    pilots: ProbeWorkspace,
}

impl TrainingWorkspace {
    pub(crate) fn new<R: Rng + ?Sized>(
        vectors: &[f32],
        centroids: &[f32],
        dim: usize,
        distance: Distance,
        policy: AssignmentPolicy,
        rng: &mut R,
    ) -> Self {
        let adaptive = !cfg!(feature = "perf")
            && policy == AssignmentPolicy::Auto
            && Hierarchy::supports(centroids.len() / dim);
        Self {
            matrix: if cfg!(feature = "perf") {
                None
            } else {
                MatrixAssignmentWorkspace::try_new(vectors, centroids.len() / dim, dim, distance)
            },
            // Keep routing/pilot randomness separate from initialization and repair.
            // Ineligible and exact workloads consume no extra training RNG draws.
            rng: adaptive.then(|| StdRng::seed_from_u64(rng.random())),
            pilots: ProbeWorkspace::default(),
        }
    }

    /// Return whether this round used approximate probing. Numerical failures and
    /// convergence confirmation use the reusable exact workspace.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn assign(
        &mut self,
        vectors: &[f32],
        centroids: &[f32],
        dim: usize,
        distance: Distance,
        labels: &mut [u32],
        has_incumbent: bool,
        force_exact: bool,
    ) -> bool {
        if !force_exact && self.probe(vectors, centroids, dim, distance, labels, has_incumbent) {
            return true;
        }
        assign_exact(
            vectors,
            centroids,
            dim,
            distance,
            labels,
            self.matrix.as_mut(),
        );
        false
    }

    fn probe(
        &mut self,
        vectors: &[f32],
        centroids: &[f32],
        dim: usize,
        distance: Distance,
        labels: &mut [u32],
        has_incumbent: bool,
    ) -> bool {
        let Some(rng) = &mut self.rng else {
            return false;
        };
        let Some(hierarchy) = Hierarchy::build(centroids, dim, Some(rng.random())) else {
            return false;
        };
        let Some(probes) = self.pilots.calibrate(&hierarchy, vectors, distance, rng) else {
            return false;
        };
        log::debug!(
            "adaptive training: {} pilots, {probes} probes",
            self.pilots.required.len()
        );
        hierarchy.probe_checked(
            vectors,
            labels,
            ProbeOptions {
                distance,
                budget: probes,
                incumbent_centroids: has_incumbent.then_some(centroids),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use super::TrainingWorkspace;
    use crate::distance::Distance;
    #[cfg(not(feature = "perf"))]
    use crate::distance::{neg_dot_product, squared_euclidean};
    use crate::kmeans::{AssignmentPolicy, base_assign_parallel};

    #[test]
    fn exact_and_ineligible_assignment_preserve_labels_and_rng() {
        let vectors: Vec<_> = (0..129 * 33).map(|i| (i % 17) as f32 * 0.125).collect();
        for (policy, count) in [
            (AssignmentPolicy::Auto, 31),
            (AssignmentPolicy::Exact, 3136),
        ] {
            let centers: Vec<_> = (0..count * 129).map(|i| (i % 23) as f32 * 0.0625).collect();
            let mut rng = StdRng::seed_from_u64(42);
            let mut expected_rng = StdRng::seed_from_u64(42);
            let mut workspace = TrainingWorkspace::new(
                &vectors,
                &centers,
                129,
                Distance::SquaredEuclidean,
                policy,
                &mut rng,
            );
            let mut actual = vec![0; 33];
            assert!(!workspace.assign(
                &vectors,
                &centers,
                129,
                Distance::SquaredEuclidean,
                &mut actual,
                false,
                false
            ));
            let mut expected = vec![0; 33];
            base_assign_parallel(
                &vectors,
                &centers,
                129,
                Distance::SquaredEuclidean,
                &mut expected,
            );
            assert_eq!(actual, expected);
            assert_eq!(rng.random::<u64>(), expected_rng.random::<u64>());
            assert!(workspace.rng.is_none());
        }
    }

    #[test]
    #[cfg(not(feature = "perf"))]
    fn adaptive_rounds_recalibrate_current_centroids_and_confirm_exactly() {
        for distance in [Distance::SquaredEuclidean, Distance::NegativeDotProduct] {
            let mut centers: Vec<_> = (0..3136)
                .flat_map(|i| {
                    let angle = i as f32 * std::f32::consts::TAU / 3136.0;
                    [angle.cos(), angle.sin()]
                })
                .collect();
            let vectors: Vec<_> = centers
                .as_chunks::<2>()
                .0
                .iter()
                .step_by(3)
                .take(1024)
                .flatten()
                .copied()
                .collect();
            let mut workspace = TrainingWorkspace::new(
                &vectors,
                &centers,
                2,
                distance,
                AssignmentPolicy::Auto,
                &mut StdRng::seed_from_u64(42),
            );
            let mut labels = vec![0; vectors.len() / 2];
            assert!(workspace.assign(&vectors, &centers, 2, distance, &mut labels, false, false));
            let first_pilots = workspace.pilots.pilots.clone();
            centers.rotate_left(2 * 251);
            let score = |v: &[f32], c: &[f32]| match distance {
                Distance::SquaredEuclidean => squared_euclidean(v, c),
                _ => neg_dot_product(v, c),
            };
            let incumbents: Vec<_> = vectors
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&labels)
                .map(|(v, &label)| score(v, &centers[label as usize * 2..label as usize * 2 + 2]))
                .collect();
            workspace.pilots.required.fill(usize::MAX);
            assert!(workspace.assign(&vectors, &centers, 2, distance, &mut labels, true, false));
            assert_ne!(workspace.pilots.pilots, first_pilots);
            assert!(
                workspace
                    .pilots
                    .required
                    .iter()
                    .all(|&n| (1..=16).contains(&n))
            );
            for ((v, &label), incumbent) in vectors
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&labels)
                .zip(incumbents)
            {
                assert!(
                    score(v, &centers[label as usize * 2..label as usize * 2 + 2]) <= incumbent
                );
            }
            let pilots_before_confirmation = workspace.pilots.pilots.clone();
            assert!(!workspace.assign(&vectors, &centers, 2, distance, &mut labels, true, true));
            let mut expected = vec![0; labels.len()];
            base_assign_parallel(&vectors, &centers, 2, distance, &mut expected);
            assert_eq!(labels, expected);
            assert_eq!(workspace.pilots.pilots, pilots_before_confirmation);
        }
    }

    #[test]
    fn numerical_hierarchy_failure_uses_exact_assignment() {
        let centers = vec![f32::MAX; 3136];
        let mut workspace = TrainingWorkspace::new(
            &[1.0],
            &centers,
            1,
            Distance::SquaredEuclidean,
            AssignmentPolicy::Auto,
            &mut StdRng::seed_from_u64(42),
        );
        let mut labels = [u32::MAX];
        assert!(!workspace.assign(
            &[1.0],
            &centers,
            1,
            Distance::SquaredEuclidean,
            &mut labels,
            false,
            false
        ));
        assert_eq!(labels, [0]);
    }
}
