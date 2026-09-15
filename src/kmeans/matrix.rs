use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::distance::{Distance, squared_euclidean};
use crate::kmeans::{base_assign, base_assign_parallel};

// These crossover and block-size choices are tuned on Apple Silicon. Other targets use the
// same safe defaults until platform-specific benchmarks justify changing them.
const MATRIX_ASSIGNMENT_THRESHOLD: usize = 1 << 19;
// Matrix setup is proportionally more expensive at dimensions below 64.
const SMALL_DIM_MATRIX_ASSIGNMENT_THRESHOLD: usize = 1 << 22;
// Bound explicit score tiles across Rayon workers to 128 MiB. Their size is
// vector_block_rows * centroid_block_rows; input vectors and centroids are borrowed.
const MAX_MATRIX_SCRATCH_ELEMENTS: usize = (128 << 20) / size_of::<f32>();
// Higham's gamma_n error bound is finite only while n * epsilon is below one.
const MAX_ERROR_BOUND_DIMENSION: usize = (1.0 / f32::EPSILON) as usize;
const PREFERRED_MATMUL_BLOCK_SIZE: usize = 512;
const PREFERRED_CENTROID_BLOCK_SIZE: usize = 1024;
const MIN_MATMUL_BLOCK_SIZE: usize = 64;
// Additional engineering margins on top of using `f32::EPSILON`, which is already twice
// Higham's unit roundoff. L2 needs more margin because it also adds two computed norms and can
// suffer cancellation; dot-product scores only contain the reduction error.
const L2_ERROR_BOUND_SAFETY_FACTOR: f32 = 8.0;
const DOT_ERROR_BOUND_SAFETY_FACTOR: f32 = 1.0;
const MIN_SUBNORMAL: f32 = f32::from_bits(1);

fn try_zeroed(len: usize) -> Option<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).ok()?;
    values.resize(len, 0.0);
    Some(values)
}

fn matmul_block_size(num_centroids: usize, num_threads: usize) -> Option<usize> {
    let centroid_block_size = num_centroids.min(PREFERRED_CENTROID_BLOCK_SIZE);
    let scratch_denominator = centroid_block_size.checked_mul(num_threads)?;
    let max_block_size = MAX_MATRIX_SCRATCH_ELEMENTS.checked_div(scratch_denominator)?;
    (max_block_size >= MIN_MATMUL_BLOCK_SIZE)
        .then(|| PREFERRED_MATMUL_BLOCK_SIZE.min(1 << max_block_size.ilog2()))
}

fn stable_l2_norm(values: &[f32]) -> f32 {
    let mut scale = 0.0_f32;
    for &value in values {
        if !value.is_finite() {
            // Preserve NaN or infinity so assignment detects a non-finite bound and falls back to
            // the direct implementation.
            return value.abs();
        }
        scale = scale.max(value.abs());
    }
    if scale == 0.0 {
        return 0.0;
    }

    let scaled_sum = values
        .iter()
        .map(|&value| {
            let scaled = value / scale;
            scaled * scaled
        })
        .sum::<f32>();
    scale * scaled_sum.sqrt()
}

#[derive(Clone, Copy)]
struct BestTwo {
    best_score: f32,
    second_best_score: f32,
    best_index: usize,
    exact_best_distance: f32,
    exact_best_index: usize,
}

impl BestTwo {
    fn new() -> Self {
        Self {
            best_score: f32::MAX,
            second_best_score: f32::MAX,
            best_index: 0,
            exact_best_distance: f32::MAX,
            exact_best_index: 0,
        }
    }

    fn update(&mut self, score: f32, index: usize) {
        if score < self.best_score {
            self.second_best_score = self.best_score;
            self.best_score = score;
            self.best_index = index;
        } else if score < self.second_best_score {
            self.second_best_score = score;
        }
    }

    fn update_exact(&mut self, distance: f32, index: usize) {
        if distance < self.exact_best_distance {
            self.exact_best_distance = distance;
            self.exact_best_index = index;
        }
    }
}

impl Distance {
    fn matrix_scale(self) -> f32 {
        match self {
            Self::SquaredEuclidean => -2.0,
            Self::NegativeDotProduct => -1.0,
        }
    }

    fn matrix_norm(self, values: &[f32]) -> f32 {
        match self {
            Self::SquaredEuclidean => values.iter().map(|value| value * value).sum(),
            Self::NegativeDotProduct => stable_l2_norm(values),
        }
    }

    fn matrix_uncertainty(
        self,
        gamma: f32,
        vector_norm: f32,
        max_centroid_norm: f32,
        underflow_error: f32,
    ) -> f32 {
        match self {
            // The norm identity can suffer cancellation. The safety factor is an
            // engineering margin, not part of Higham's gamma_n bound.
            Self::SquaredEuclidean => {
                L2_ERROR_BOUND_SAFETY_FACTOR
                    * (gamma * (vector_norm.abs() + max_centroid_norm.abs()) + underflow_error)
            }
            // Cauchy-Schwarz bounds the magnitude of the exact dot product by
            // ||x||₂ ||c||₂.
            Self::NegativeDotProduct => {
                DOT_ERROR_BOUND_SAFETY_FACTOR
                    * (gamma * vector_norm * max_centroid_norm + underflow_error)
            }
        }
    }
}

pub(crate) struct MatrixAssignmentWorkspace {
    distance: Distance,
    vector_norms: Vec<f32>,
    centroid_norms: Vec<f32>,
}

impl MatrixAssignmentWorkspace {
    pub(crate) fn try_new(
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
            || comparison_count < assignment_threshold
        {
            return None;
        }
        matmul_block_size(num_centroids, rayon::current_num_threads())?;

        let mut vector_norms = try_zeroed(num_vectors)?;
        vector_norms
            .par_iter_mut()
            .zip(vecs.par_chunks_exact(dim))
            .for_each(|(norm, vector)| {
                *norm = distance.matrix_norm(vector);
            });
        Some(Self {
            distance,
            vector_norms,
            centroid_norms: try_zeroed(num_centroids)?,
        })
    }

    pub(crate) fn assign(
        &mut self,
        vecs: &[f32],
        centroids: &[f32],
        dim: usize,
        labels: &mut [u32],
    ) {
        let num_centroids = centroids.len() / dim;
        debug_assert_eq!(self.vector_norms.len(), vecs.len() / dim);
        debug_assert_eq!(self.centroid_norms.len(), num_centroids);

        self.centroid_norms
            .par_iter_mut()
            .zip(centroids.par_chunks_exact(dim))
            .for_each(|(norm, centroid)| {
                *norm = self.distance.matrix_norm(centroid);
            });
        let centroid_norms_are_finite = self.centroid_norms.iter().all(|norm| norm.is_finite());
        let max_centroid_norm = self.centroid_norms.iter().copied().fold(0.0_f32, f32::max);
        // Higham, Accuracy and Stability of Numerical Algorithms, Chapters 2-3,
        // bounds an n-term floating-point reduction with gamma_n = n*u/(1-n*u):
        // https://doi.org/10.1137/1.9780898718027. `f32::EPSILON` is about 2*u,
        // so using it here is already conservative.
        let dimension_error = dim as f32 * f32::EPSILON;
        let gamma = dimension_error / (1.0 - dimension_error);
        // The relative gamma_n model excludes underflow. Allow one minimum-subnormal rounding
        // unit per reduction term so tiny, finite inputs still trigger the ambiguity fallback.
        let underflow_error = dim as f32 * MIN_SUBNORMAL;

        let Some(block_size) = matmul_block_size(num_centroids, rayon::current_num_threads())
        else {
            base_assign_parallel(vecs, centroids, dim, self.distance, labels);
            return;
        };
        let context = ScoreContext {
            distance: self.distance,
            centroids,
            centroid_norms_are_finite,
            max_centroid_norm,
            gamma,
            underflow_error,
        };
        let centroid_block_size = num_centroids.min(PREFERRED_CENTROID_BLOCK_SIZE);
        let scratch_len = block_size * centroid_block_size;
        let scale = self.distance.matrix_scale();
        labels
            .par_chunks_mut(block_size)
            .zip(self.vector_norms.par_chunks(block_size))
            .zip(vecs.par_chunks(block_size * dim))
            .for_each_init(
                || {
                    let scores = try_zeroed(scratch_len)?;
                    let mut best = Vec::new();
                    best.try_reserve_exact(block_size).ok()?;
                    best.resize(block_size, BestTwo::new());
                    Some((scores, best))
                },
                |scratch, ((labels, vector_norms), vectors)| {
                    let Some((scores, best)) = scratch else {
                        base_assign(vectors, centroids, dim, self.distance, labels);
                        return;
                    };
                    let block_rows = labels.len();
                    let best = &mut best[..block_rows];
                    best.fill(BestTwo::new());
                    for (tile_index, (centroid_tile, centroid_norm_tile)) in centroids
                        .chunks(centroid_block_size * dim)
                        .zip(self.centroid_norms.chunks(centroid_block_size))
                        .enumerate()
                    {
                        let tile_start = tile_index * centroid_block_size;
                        let tile_rows = centroid_norm_tile.len();
                        let scores = &mut scores[..block_rows * tile_rows];
                        matmul(
                            MatMut::from_row_major_slice_mut(scores, block_rows, tile_rows),
                            Accum::Replace,
                            MatRef::from_row_major_slice(vectors, block_rows, dim),
                            MatRef::from_row_major_slice(centroid_tile, tile_rows, dim).transpose(),
                            scale,
                            Par::Seq,
                        );
                        best.iter_mut()
                            .zip(vector_norms)
                            .zip(vectors.chunks_exact(dim))
                            .zip(scores.chunks_exact(tile_rows))
                            .for_each(|(((best, &vector_norm), vector), scores)| {
                                let ambiguity = context.ambiguity(vector_norm);
                                scores
                                    .iter()
                                    .zip(centroid_norm_tile)
                                    .zip(centroid_tile.chunks_exact(dim))
                                    .enumerate()
                                    .for_each(|(index, ((&score, &centroid_norm), centroid))| {
                                        let score = match self.distance {
                                            Distance::NegativeDotProduct => score,
                                            Distance::SquaredEuclidean => {
                                                score + vector_norm + centroid_norm
                                            }
                                        };
                                        let index = tile_start + index;
                                        best.update(score, index);
                                        // The approximate best can only decrease as more tiles are
                                        // visited, so this cutoff contains the final cutoff. Refine
                                        // candidates now instead of retaining earlier score tiles.
                                        if self.distance == Distance::SquaredEuclidean
                                            && ambiguity.is_finite()
                                            && score <= best.best_score + ambiguity
                                        {
                                            best.update_exact(
                                                squared_euclidean(vector, centroid),
                                                index,
                                            );
                                        }
                                    });
                            });
                    }
                    labels
                        .iter_mut()
                        .zip(vector_norms)
                        .zip(vectors.chunks_exact(dim))
                        .zip(best)
                        .for_each(|(((label, &vector_norm), vector), best)| {
                            *label = context.assign(vector, vector_norm, *best);
                        });
                },
            );
    }
}

struct ScoreContext<'a> {
    distance: Distance,
    centroids: &'a [f32],
    centroid_norms_are_finite: bool,
    max_centroid_norm: f32,
    gamma: f32,
    underflow_error: f32,
}

impl ScoreContext<'_> {
    fn ambiguity(&self, vector_norm: f32) -> f32 {
        // Two estimates can each err in opposite directions by this uncertainty.
        2.0 * self.distance.matrix_uncertainty(
            self.gamma,
            vector_norm,
            self.max_centroid_norm,
            self.underflow_error,
        )
    }

    fn assign(&self, vector: &[f32], vector_norm: f32, best: BestTwo) -> u32 {
        let ambiguity = self.ambiguity(vector_norm);
        let invalid = !self.centroid_norms_are_finite
            || !vector_norm.is_finite()
            || !ambiguity.is_finite()
            || !best.best_score.is_finite();
        let use_direct = invalid
            || (self.distance == Distance::NegativeDotProduct
                && best.second_best_score - best.best_score <= ambiguity);
        if use_direct {
            let mut direct = [0];
            base_assign(
                vector,
                self.centroids,
                vector.len(),
                self.distance,
                &mut direct,
            );
            direct[0]
        } else if self.distance == Distance::SquaredEuclidean
            && (best.best_score < 0.0 || best.second_best_score - best.best_score <= ambiguity)
        {
            best.exact_best_index as u32
        } else {
            best.best_index as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use rayon::ThreadPoolBuilder;
    use seed_rand::seeded_rng;

    use super::{MatrixAssignmentWorkspace, PREFERRED_CENTROID_BLOCK_SIZE, matmul_block_size};
    use crate::distance::Distance;
    use crate::kmeans::base_assign;
    use crate::utils::normalize;

    #[test]
    fn test_matrix_assignment_adapts_block_size_to_worker_scratch_budget() {
        assert_eq!(matmul_block_size(4096, 16), Some(512));
        assert_eq!(matmul_block_size(131_073, 16), Some(512));
        assert_eq!(matmul_block_size(4096, 128), Some(256));
        assert_eq!(matmul_block_size(4096, 256), Some(128));
        assert_eq!(matmul_block_size(4096, 512), Some(64));
        assert_eq!(matmul_block_size(4096, 513), None);
    }

    #[test]
    fn test_matrix_assignment_bounds_streaming_scratch_instead_of_total_scores() {
        let dim = 64;
        let num_vectors = 8193;
        let vectors = vec![0.0; num_vectors * dim];
        ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                assert!(
                    MatrixAssignmentWorkspace::try_new(
                        &vectors,
                        4096,
                        dim,
                        Distance::SquaredEuclidean,
                    )
                    .is_some(),
                    "streaming should accept more than 128 MiB of total scores",
                );
                assert!(
                    MatrixAssignmentWorkspace::try_new(
                        &vectors,
                        131_073,
                        dim,
                        Distance::SquaredEuclidean,
                    )
                    .is_some(),
                    "centroid tiling should make scratch independent of the total centroid count",
                );
            });
    }

    #[test]
    fn test_matrix_assignment_matches_direct_across_centroid_tiles() {
        let mut rng = seeded_rng();
        let dim = 64;
        let num_vectors = 513;
        let num_centroids = PREFERRED_CENTROID_BLOCK_SIZE + 1;
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
    fn test_matrix_dot_product_matches_direct_across_centroid_tiles() {
        let mut rng = seeded_rng();
        let dim = 64;
        let num_vectors = 513;
        let num_centroids = PREFERRED_CENTROID_BLOCK_SIZE + 1;
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
    fn test_matrix_assignment_matches_direct_assignment() {
        let mut rng = seeded_rng();
        let dim = 64;
        // One row beyond the preferred block boundary exercises the final partial block.
        let num_vectors = 8193;
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
        let mut rng = seeded_rng();
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
    fn test_matrix_assignment_reuses_workspace_with_updated_centroids() {
        let mut rng = seeded_rng();
        let dim = 64;
        let num_vectors = 8192;
        let num_centroids = 64;
        let vecs = (0..num_vectors * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();

        for distance in [Distance::SquaredEuclidean, Distance::NegativeDotProduct] {
            let mut centroids = (0..num_centroids * dim)
                .map(|_| rng.random::<f32>())
                .collect::<Vec<_>>();
            let mut actual = vec![0; num_vectors];
            let mut workspace =
                MatrixAssignmentWorkspace::try_new(&vecs, num_centroids, dim, distance).unwrap();
            workspace.assign(&vecs, &centroids, dim, &mut actual);

            centroids.iter_mut().for_each(|value| *value = 1.0 - *value);
            let mut expected = vec![0; num_vectors];
            base_assign(&vecs, &centroids, dim, distance, &mut expected);
            workspace.assign(&vecs, &centroids, dim, &mut actual);

            assert_eq!(actual, expected);
        }
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
        let num_vectors = 513;
        let num_centroids = PREFERRED_CENTROID_BLOCK_SIZE + 1;
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

    #[test]
    fn test_matrix_dot_product_handles_subnormal_scores() {
        let dim = 64;
        let num_vectors = 8192;
        let num_centroids = 64;
        let tiny = f32::from_bits(1).sqrt();
        let mut centroids = Vec::with_capacity(num_centroids * dim);
        for index in 0..num_centroids {
            centroids.extend((0..dim).map(|coordinate| {
                if index & (1 << (coordinate % 6)) == 0 {
                    tiny
                } else {
                    -tiny
                }
            }));
        }
        let mut vecs = Vec::with_capacity(num_vectors * dim);
        for index in 0..num_vectors {
            vecs.extend_from_slice(&centroids[(index % num_centroids) * dim..][..dim]);
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
    fn test_matrix_dot_product_handles_asymmetric_scales() {
        let mut rng = StdRng::seed_from_u64(0);
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
            centroid.iter_mut().for_each(|value| *value *= 1e15);
            centroids.extend(centroid);
        }
        let mut vecs = Vec::with_capacity(num_vectors * dim);
        for _ in 0..num_vectors {
            let mut vector = (0..dim)
                .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                .collect::<Vec<_>>();
            normalize(&mut vector);
            vector.iter_mut().for_each(|value| *value *= 1e-25);
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
}
