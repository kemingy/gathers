use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator, ParallelSlice,
    ParallelSliceMut,
};

use super::base_assign;
use crate::distance::{Distance, squared_euclidean};

// These crossover and block-size choices are tuned on Apple Silicon. Other targets use the
// same safe defaults until platform-specific benchmarks justify changing them.
const MATRIX_ASSIGNMENT_THRESHOLD: usize = 1 << 19;
// Matrix setup is proportionally more expensive at dimensions below 64.
const SMALL_DIM_MATRIX_ASSIGNMENT_THRESHOLD: usize = 1 << 22;
// Bound the reusable score matrix to 128 MiB.
const MAX_MATRIX_ASSIGNMENT_ELEMENTS: usize = (128 << 20) / size_of::<f32>();
// Additional engineering margin for the norm, dot-product, and final addition errors.
const L2_ERROR_BOUND_SAFETY_FACTOR: f32 = 8.0;

pub(super) struct MatrixAssignmentWorkspace {
    distance: Distance,
    vector_norms: Vec<f32>,
    centroid_norms: Vec<f32>,
    dot_products: Vec<f32>,
}

impl MatrixAssignmentWorkspace {
    pub(super) fn try_new(
        vecs: &[f32],
        num_centroids: usize,
        dim: usize,
        distance: Distance,
    ) -> Option<Self> {
        let num_vectors = vecs.len() / dim;
        let comparison_count = num_vectors.checked_mul(num_centroids)?;
        let assignment_threshold = if dim < 64 {
            SMALL_DIM_MATRIX_ASSIGNMENT_THRESHOLD
        } else {
            MATRIX_ASSIGNMENT_THRESHOLD
        };
        if dim < 32
            || num_centroids < 2
            || !(assignment_threshold..=MAX_MATRIX_ASSIGNMENT_ELEMENTS).contains(&comparison_count)
        {
            return None;
        }

        let vector_norms = if distance == Distance::SquaredEuclidean {
            vecs.par_chunks_exact(dim)
                .map(|vector| vector.iter().map(|value| value * value).sum())
                .collect()
        } else {
            Vec::new()
        };
        Some(Self {
            distance,
            vector_norms,
            centroid_norms: if distance == Distance::SquaredEuclidean {
                vec![0.0; num_centroids]
            } else {
                Vec::new()
            },
            dot_products: vec![0.0; comparison_count],
        })
    }

    pub(super) fn assign(
        &mut self,
        vecs: &[f32],
        centroids: &[f32],
        dim: usize,
        labels: &mut [u32],
    ) {
        let num_centroids = centroids.len() / dim;
        if self.distance == Distance::SquaredEuclidean {
            debug_assert_eq!(self.vector_norms.len(), vecs.len() / dim);
            debug_assert_eq!(self.centroid_norms.len(), num_centroids);
        }
        debug_assert_eq!(self.dot_products.len(), labels.len() * num_centroids);

        // Process 256 input rows per GEMM task; tuned on Apple Silicon with
        // benchmark datasets averaging 196 training vectors per centroid.
        const MATMUL_BLOCK_SIZE: usize = 256;
        let scale = match self.distance {
            Distance::SquaredEuclidean => -2.0,
            Distance::NegativeDotProduct => -1.0,
        };
        self.dot_products
            .par_chunks_mut(MATMUL_BLOCK_SIZE * num_centroids)
            .zip(vecs.par_chunks(MATMUL_BLOCK_SIZE * dim))
            .for_each(|(dot_products, vectors)| {
                let block_rows = vectors.len() / dim;
                let vectors = MatRef::from_row_major_slice(vectors, block_rows, dim);
                let centroids = MatRef::from_row_major_slice(centroids, num_centroids, dim);
                let scores =
                    MatMut::from_row_major_slice_mut(dot_products, block_rows, num_centroids);
                matmul(
                    scores,
                    Accum::Replace,
                    vectors,
                    centroids.transpose(),
                    scale,
                    Par::Seq,
                );
            });

        if self.distance == Distance::NegativeDotProduct {
            labels
                .par_iter_mut()
                .zip(self.dot_products.par_chunks_exact(num_centroids))
                .for_each(|(label, scores)| {
                    let mut best_score = f32::MAX;
                    let mut best_index = 0;
                    for (index, &score) in scores.iter().enumerate() {
                        if score < best_score {
                            best_score = score;
                            best_index = index;
                        }
                    }
                    *label = best_index as u32;
                });
            return;
        }

        self.centroid_norms
            .par_iter_mut()
            .zip(centroids.par_chunks_exact(dim))
            .for_each(|(norm, centroid)| {
                *norm = centroid.iter().map(|value| value * value).sum();
            });
        let max_centroid_norm = self.centroid_norms.iter().copied().fold(0.0_f32, f32::max);
        // Higham, Accuracy and Stability of Numerical Algorithms, Chapters 2-3,
        // bounds an n-term floating-point reduction with gamma_n = n*u/(1-n*u):
        // https://doi.org/10.1137/1.9780898718027. `f32::EPSILON` is about 2*u,
        // so using it here is already conservative.
        let dimension_error = dim as f32 * f32::EPSILON;
        let gamma = dimension_error / (1.0 - dimension_error);

        labels
            .par_iter_mut()
            .zip(&self.vector_norms)
            .zip(vecs.par_chunks_exact(dim))
            .zip(self.dot_products.par_chunks_exact(num_centroids))
            .for_each(|(((label, &vector_norm), vector), scores)| {
                let mut best_distance = f32::MAX;
                let mut second_best_distance = f32::MAX;
                let mut best_index = 0;
                for (index, (&score, &centroid_norm)) in
                    scores.iter().zip(&self.centroid_norms).enumerate()
                {
                    let distance = score + vector_norm + centroid_norm;
                    if distance < best_distance {
                        second_best_distance = best_distance;
                        best_distance = distance;
                        best_index = index;
                    } else if distance < second_best_distance {
                        second_best_distance = distance;
                    }
                }

                // The norm identity can suffer cancellation. The safety factor is an
                // engineering margin, not part of Higham's gamma_n bound.
                let uncertainty = L2_ERROR_BOUND_SAFETY_FACTOR
                    * gamma
                    * (vector_norm.abs() + max_centroid_norm.abs());
                // Two estimates can each err in opposite directions by `uncertainty`.
                let ambiguity = 2.0 * uncertainty;
                if !best_distance.is_finite() {
                    let mut exact_label = [0];
                    base_assign(
                        vector,
                        centroids,
                        dim,
                        Distance::SquaredEuclidean,
                        &mut exact_label,
                    );
                    *label = exact_label[0];
                } else if best_distance < 0.0 || second_best_distance - best_distance <= ambiguity {
                    let cutoff = best_distance + ambiguity;
                    let mut exact_best_distance = f32::MAX;
                    let mut exact_best_index = best_index;
                    for (index, ((&score, &centroid_norm), centroid)) in scores
                        .iter()
                        .zip(&self.centroid_norms)
                        .zip(centroids.chunks_exact(dim))
                        .enumerate()
                    {
                        let approximate_distance = score + vector_norm + centroid_norm;
                        if approximate_distance <= cutoff {
                            let exact_distance = squared_euclidean(vector, centroid);
                            if exact_distance < exact_best_distance {
                                exact_best_distance = exact_distance;
                                exact_best_index = index;
                            }
                        }
                    }
                    *label = exact_best_index as u32;
                } else {
                    *label = best_index as u32;
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::MatrixAssignmentWorkspace;
    use crate::distance::Distance;
    use crate::kmeans::base_assign;

    fn random_test_rng() -> StdRng {
        let seed = rand::rng().random();
        eprintln!("random seed: {seed}");
        StdRng::seed_from_u64(seed)
    }

    #[test]
    fn test_matrix_assignment_matches_direct_assignment() {
        let mut rng = random_test_rng();
        let dim = 64;
        let num_vectors = 8192;
        let num_centroids = 64;
        let vecs = (0..num_vectors * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let centroids = (0..num_centroids * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let mut expected = vec![0; num_vectors];
        let mut actual = vec![0; num_vectors];

        base_assign(
            &vecs,
            &centroids,
            dim,
            Distance::SquaredEuclidean,
            &mut expected,
        );
        let mut workspace = MatrixAssignmentWorkspace::try_new(
            &vecs,
            num_centroids,
            dim,
            Distance::SquaredEuclidean,
        )
        .unwrap();
        workspace.assign(&vecs, &centroids, dim, &mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_matrix_dot_product_matches_direct_assignment() {
        let mut rng = random_test_rng();
        let dim = 64;
        let num_vectors = 8192;
        let num_centroids = 64;
        let vecs = (0..num_vectors * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let centroids = (0..num_centroids * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let mut expected = vec![0; num_vectors];
        let mut actual = vec![0; num_vectors];

        base_assign(
            &vecs,
            &centroids,
            dim,
            Distance::NegativeDotProduct,
            &mut expected,
        );
        let mut workspace = MatrixAssignmentWorkspace::try_new(
            &vecs,
            num_centroids,
            dim,
            Distance::NegativeDotProduct,
        )
        .unwrap();
        workspace.assign(&vecs, &centroids, dim, &mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_matrix_assignment_handles_large_common_offset() {
        let dim = 64;
        let num_vectors = 8192;
        let num_centroids = 64;
        let mut centroids = vec![1_000_000.0; num_centroids * dim];
        for (index, centroid) in centroids.chunks_exact_mut(dim).enumerate() {
            centroid[0] += index as f32 * 2.0;
        }
        let mut vecs = Vec::with_capacity(num_vectors * dim);
        for index in 0..num_vectors {
            let centroid = &centroids[(index % num_centroids) * dim..][..dim];
            vecs.extend_from_slice(centroid);
            *vecs.last_mut().unwrap() += 0.125;
        }
        let mut labels = vec![0; num_vectors];

        let mut workspace = MatrixAssignmentWorkspace::try_new(
            &vecs,
            num_centroids,
            dim,
            Distance::SquaredEuclidean,
        )
        .unwrap();
        workspace.assign(&vecs, &centroids, dim, &mut labels);

        for (index, &label) in labels.iter().enumerate() {
            assert_eq!(label as usize, index % num_centroids);
        }
    }
}
