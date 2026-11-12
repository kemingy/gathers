//! Full-batch Lloyd training after shared preparation and initialization.

use std::time::Instant;

use aligned_vec::AVec;
use log::debug;
use rand::Rng;

use crate::distance::Distance;
use crate::kmeans::{KMeansError, TrainingFit, TrainingParameters, update_centroids_inner};
#[cfg(feature = "perf")]
use crate::kmeans::{base_assign, rabitq_assign_inner};
#[cfg(not(feature = "perf"))]
use crate::kmeans::{
    base_assign_parallel, matrix::MatrixAssignmentWorkspace, rabitq_assign_parallel_inner,
};

const LARGE_CLUSTER_THRESHOLD: usize = 1 << 28;

/// Lloyd's update policy; sampling and projection belong to the shared pipeline.
pub(crate) struct Lloyd {
    max_iter: u32,
    tolerance: f32,
}

impl Lloyd {
    pub(crate) fn new(max_iter: u32, tolerance: f32) -> Self {
        Self {
            max_iter,
            tolerance,
        }
    }

    pub(crate) fn fit<R: Rng + ?Sized>(
        &self,
        vecs: &[f32],
        dim: usize,
        mut centroids: AVec<f32>,
        assignment_distance: Distance,
        rng: &mut R,
    ) -> Result<TrainingFit, KMeansError> {
        let training_num = vecs.len() / dim;
        let mut labels: Vec<u32> = vec![0; training_num];
        let use_exact_assignment = assignment_distance == Distance::NegativeDotProduct
            || vecs.len() <= LARGE_CLUSTER_THRESHOLD;
        #[cfg(not(feature = "perf"))]
        let mut matrix_workspace = if use_exact_assignment {
            MatrixAssignmentWorkspace::try_new(
                vecs,
                centroids.len() / dim,
                dim,
                assignment_distance,
            )
        } else {
            None
        };
        debug!("start training");
        for i in 0..self.max_iter {
            let start_time = Instant::now();
            if use_exact_assignment {
                #[cfg(feature = "perf")]
                base_assign(vecs, &centroids, dim, assignment_distance, &mut labels);
                #[cfg(not(feature = "perf"))]
                if let Some(workspace) = &mut matrix_workspace {
                    workspace.assign(vecs, &centroids, dim, &mut labels);
                } else {
                    base_assign_parallel(vecs, &centroids, dim, assignment_distance, &mut labels);
                }
            } else {
                #[cfg(feature = "perf")]
                rabitq_assign_inner(vecs, &centroids, dim, &mut labels, rng);
                #[cfg(not(feature = "perf"))]
                rabitq_assign_parallel_inner(vecs, &centroids, dim, &mut labels, rng);
            }
            let diff = update_centroids_inner(
                vecs,
                &mut centroids,
                dim,
                &labels,
                assignment_distance == Distance::NegativeDotProduct,
                rng,
            )?;
            debug!("iter {} takes {} s", i, start_time.elapsed().as_secs_f32());
            if diff < f64::from(self.tolerance) {
                debug!("converged at iter {i}");
                break;
            }
        }

        Ok(TrainingFit {
            centroids,
            labels,
            parameters: TrainingParameters::Lloyd {
                max_iter: self.max_iter,
                tolerance: self.tolerance,
            },
        })
    }
}
