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
// Higham's gamma_n error bound is finite only while n * epsilon is below one.
const MAX_ERROR_BOUND_DIMENSION: usize = (1.0 / f32::EPSILON) as usize;
const MATMUL_BLOCK_SIZE: usize = 256;
// Additional engineering margins on top of using `f32::EPSILON`, which is already twice
// Higham's unit roundoff. L2 needs more margin because it also adds two computed norms and can
// suffer cancellation; dot-product scores only contain the reduction error.
const L2_ERROR_BOUND_SAFETY_FACTOR: f32 = 8.0;
const DOT_ERROR_BOUND_SAFETY_FACTOR: f32 = 1.0;

fn try_zeroed(len: usize) -> Option<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).ok()?;
    values.resize(len, 0.0);
    Some(values)
}

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
        if !(32..MAX_ERROR_BOUND_DIMENSION).contains(&dim)
            || num_centroids < 2
            || MATMUL_BLOCK_SIZE.checked_mul(num_centroids).is_none()
            || !(assignment_threshold..=MAX_MATRIX_ASSIGNMENT_ELEMENTS).contains(&comparison_count)
        {
            return None;
        }

        let mut vector_norms = try_zeroed(num_vectors)?;
        vector_norms
            .par_iter_mut()
            .zip(vecs.par_chunks_exact(dim))
            .for_each(|(norm, vector)| {
                *norm = vector.iter().map(|value| value * value).sum();
            });
        Some(Self {
            distance,
            vector_norms,
            centroid_norms: try_zeroed(num_centroids)?,
            dot_products: try_zeroed(comparison_count)?,
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
        debug_assert_eq!(self.vector_norms.len(), vecs.len() / dim);
        debug_assert_eq!(self.centroid_norms.len(), num_centroids);
        debug_assert_eq!(self.dot_products.len(), labels.len() * num_centroids);

        // Process 256 input rows per GEMM task; tuned on Apple Silicon with
        // benchmark datasets averaging 196 training vectors per centroid.
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

        self.centroid_norms
            .par_iter_mut()
            .zip(centroids.par_chunks_exact(dim))
            .for_each(|(norm, centroid)| {
                *norm = centroid.iter().map(|value| value * value).sum();
            });
        let centroid_norms_are_finite = self.centroid_norms.iter().all(|norm| norm.is_finite());
        let max_centroid_norm = self.centroid_norms.iter().copied().fold(0.0_f32, f32::max);
        // Higham, Accuracy and Stability of Numerical Algorithms, Chapters 2-3,
        // bounds an n-term floating-point reduction with gamma_n = n*u/(1-n*u):
        // https://doi.org/10.1137/1.9780898718027. `f32::EPSILON` is about 2*u,
        // so using it here is already conservative.
        let dimension_error = dim as f32 * f32::EPSILON;
        let gamma = dimension_error / (1.0 - dimension_error);

        if self.distance == Distance::NegativeDotProduct {
            let max_centroid_norm = max_centroid_norm.sqrt();
            labels
                .par_iter_mut()
                .zip(&self.vector_norms)
                .zip(vecs.par_chunks_exact(dim))
                .zip(self.dot_products.par_chunks_exact(num_centroids))
                .for_each(|(((label, &vector_norm), vector), scores)| {
                    let mut best_score = f32::MAX;
                    let mut second_best_score = f32::MAX;
                    let mut best_index = 0;
                    for (index, &score) in scores.iter().enumerate() {
                        if score < best_score {
                            second_best_score = best_score;
                            best_score = score;
                            best_index = index;
                        } else if score < second_best_score {
                            second_best_score = score;
                        }
                    }

                    // Cauchy-Schwarz bounds the magnitude of the exact dot product by
                    // ||x||₂ ||c||₂. Scores whose error intervals overlap are recomputed
                    // with the direct implementation to preserve its assignment semantics.
                    let uncertainty = DOT_ERROR_BOUND_SAFETY_FACTOR
                        * gamma
                        * vector_norm.sqrt()
                        * max_centroid_norm;
                    let ambiguity = 2.0 * uncertainty;
                    if !centroid_norms_are_finite
                        || !vector_norm.is_finite()
                        || !uncertainty.is_finite()
                        || !best_score.is_finite()
                        || second_best_score - best_score <= ambiguity
                    {
                        let mut direct = [0];
                        base_assign(
                            vector,
                            centroids,
                            dim,
                            Distance::NegativeDotProduct,
                            &mut direct,
                        );
                        *label = direct[0];
                    } else {
                        *label = best_index as u32;
                    }
                });
            return;
        }

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
                if !centroid_norms_are_finite
                    || !vector_norm.is_finite()
                    || !uncertainty.is_finite()
                    || !best_distance.is_finite()
                {
                    let mut direct = [0];
                    base_assign(
                        vector,
                        centroids,
                        dim,
                        Distance::SquaredEuclidean,
                        &mut direct,
                    );
                    *label = direct[0];
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
    use crate::test_utils::random_test_rng;
    use crate::utils::normalize;

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
    fn test_matrix_dot_product_falls_back_for_near_ties() {
        let mut rng = StdRng::seed_from_u64(7);
        let dim = 64;
        let num_vectors = 8192;
        let num_centroids = 64;
        let mut direction = (0..dim)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect::<Vec<_>>();
        normalize(&mut direction);
        let mut centroids = Vec::with_capacity(num_centroids * dim);
        for _ in 0..num_centroids {
            let mut centroid = direction.clone();
            centroid.iter_mut().for_each(|value| {
                *value += (rng.random::<f32>() * 2.0 - 1.0) * 1e-6;
            });
            normalize(&mut centroid);
            centroids.extend(centroid);
        }
        let mut vecs = Vec::with_capacity(num_vectors * dim);
        for _ in 0..num_vectors {
            let mut vector = (0..dim)
                .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                .collect::<Vec<_>>();
            normalize(&mut vector);
            vecs.extend(vector);
        }
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

    #[test]
    fn test_matrix_assignment_handles_overflowed_norms() {
        let dim = 64;
        let num_vectors = 8192;
        let num_centroids = 64;
        let mut centroids = vec![1e20; num_centroids * dim];
        for (index, centroid) in centroids.chunks_exact_mut(dim).enumerate() {
            centroid[0] += index as f32 * 1e14;
        }
        let mut vecs = Vec::with_capacity(num_vectors * dim);
        for index in 0..num_vectors {
            vecs.extend_from_slice(&centroids[(index % num_centroids) * dim..][..dim]);
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
