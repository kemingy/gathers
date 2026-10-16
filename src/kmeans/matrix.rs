use faer::{Accum, MatMut, MatRef, Par};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::distance::{Distance, neg_dot_product, squared_euclidean};
use crate::kmeans::{base_assign, base_assign_parallel};
use crate::linalg::matmul;

// These crossover and block-size choices are tuned on Apple Silicon. Other targets use the
// same safe defaults until platform-specific benchmarks justify changing them.
const MATRIX_ASSIGNMENT_THRESHOLD: usize = 1 << 19;
// Matrix setup is proportionally more expensive at dimensions below 64.
const SMALL_DIM_MATRIX_ASSIGNMENT_THRESHOLD: usize = 1 << 22;
// Apple Silicon SGEMM amortizes setup earlier for medium batches at PCA128 and above.
// Retain the previous cutoff for tiny queries and lower dimensions.
const ACCELERATE_MATRIX_ASSIGNMENT_THRESHOLD: usize = 1 << 15;
const MIN_ACCELERATE_BATCH_ROWS: usize = 16;
const MIN_ACCELERATE_BATCH_VALUES: usize = 4096;
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

fn matrix_assignment_threshold(num_vectors: usize, dim: usize) -> usize {
    if dim < 64 {
        SMALL_DIM_MATRIX_ASSIGNMENT_THRESHOLD
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && dim >= 128
        && num_vectors >= MIN_ACCELERATE_BATCH_ROWS
        && num_vectors.saturating_mul(dim) >= MIN_ACCELERATE_BATCH_VALUES
    {
        ACCELERATE_MATRIX_ASSIGNMENT_THRESHOLD
    } else {
        MATRIX_ASSIGNMENT_THRESHOLD
    }
}

fn try_zeroed(len: usize) -> Option<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).ok()?;
    values.resize(len, 0.0);
    Some(values)
}

pub(crate) fn matmul_block_size(
    num_vectors: usize,
    num_centroids: usize,
    num_threads: usize,
) -> Option<usize> {
    // Reject a zero worker count explicitly; `div_ceil` below would divide by zero.
    if num_threads == 0 {
        return None;
    }
    let centroid_block_size = num_centroids.min(PREFERRED_CENTROID_BLOCK_SIZE);
    let scratch_denominator = centroid_block_size.checked_mul(num_threads)?;
    let max_block_size = MAX_MATRIX_SCRATCH_ELEMENTS.checked_div(scratch_denominator)?;
    if max_block_size < MIN_MATMUL_BLOCK_SIZE {
        return None;
    }
    // Give every worker one exact-fit tile: ceil(rows / workers) clamped to 64-512 rows.
    // Measured on Apple Silicon, exact-fit tiles beat power-of-two rounding: rounding up can
    // leave workers idle on mid-size batches while saving little GEMM packing cost.
    let parallel_block_size = num_vectors
        .div_ceil(num_threads)
        .clamp(MIN_MATMUL_BLOCK_SIZE, PREFERRED_MATMUL_BLOCK_SIZE);
    Some(
        PREFERRED_MATMUL_BLOCK_SIZE
            .min(1 << max_block_size.ilog2())
            .min(parallel_block_size),
    )
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
            best_score: f32::INFINITY,
            second_best_score: f32::INFINITY,
            best_index: 0,
            exact_best_distance: f32::INFINITY,
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

fn matrix_scale(distance: Distance) -> f32 {
    match distance {
        Distance::SquaredEuclidean => -2.0,
        Distance::NegativeDotProduct | Distance::Cosine => -1.0,
    }
}

fn matrix_norm(distance: Distance, values: &[f32]) -> f32 {
    match distance {
        Distance::SquaredEuclidean => values.iter().map(|value| value * value).sum(),
        Distance::NegativeDotProduct | Distance::Cosine => stable_l2_norm(values),
    }
}

fn matrix_uncertainty(
    distance: Distance,
    gamma: f32,
    vector_norm: f32,
    max_centroid_norm: f32,
    underflow_error: f32,
) -> f32 {
    match distance {
        // The norm identity can suffer cancellation. The safety factor is an
        // engineering margin, not part of Higham's gamma_n bound.
        Distance::SquaredEuclidean => {
            L2_ERROR_BOUND_SAFETY_FACTOR
                * (gamma * (vector_norm.abs() + max_centroid_norm.abs()) + underflow_error)
        }
        // Cauchy-Schwarz bounds the magnitude of the exact dot product by
        // ||x||₂ ||c||₂.
        Distance::NegativeDotProduct | Distance::Cosine => {
            DOT_ERROR_BOUND_SAFETY_FACTOR
                * (gamma * vector_norm * max_centroid_norm + underflow_error)
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
        Self::try_new_with_threshold(
            vecs,
            num_centroids,
            dim,
            distance,
            matrix_assignment_threshold(vecs.len().checked_div(dim)?, dim),
        )
    }

    // Explicit inputs let the real-data benchmark compare policies without global overrides.
    pub(crate) fn try_new_with_threshold(
        vecs: &[f32],
        num_centroids: usize,
        dim: usize,
        distance: Distance,
        assignment_threshold: usize,
    ) -> Option<Self> {
        // Cosine fitting normalizes rows and uses the dot-product workspace instead.
        if distance == Distance::Cosine {
            return None;
        }
        let num_vectors = vecs.len() / dim;
        let comparison_count = num_vectors.checked_mul(num_centroids)?;
        if !(32..MAX_ERROR_BOUND_DIMENSION).contains(&dim)
            || num_centroids < 2
            || comparison_count < assignment_threshold
        {
            return None;
        }
        matmul_block_size(num_vectors, num_centroids, rayon::current_num_threads())?;

        let mut vector_norms = try_zeroed(num_vectors)?;
        vector_norms
            .par_iter_mut()
            .zip(vecs.par_chunks_exact(dim))
            .for_each(|(norm, vector)| {
                *norm = matrix_norm(distance, vector);
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
        let block_size = matmul_block_size(
            labels.len(),
            centroids.len() / dim,
            rayon::current_num_threads(),
        );
        self.assign_with_kernel(vecs, centroids, dim, labels, block_size, matmul);
    }

    pub(crate) fn assign_with_kernel(
        &mut self,
        vecs: &[f32],
        centroids: &[f32],
        dim: usize,
        labels: &mut [u32],
        block_size: Option<usize>,
        multiply: impl Fn(MatMut<'_, f32>, Accum, MatRef<'_, f32>, MatRef<'_, f32>, f32, Par) + Sync,
    ) {
        let num_centroids = centroids.len() / dim;
        debug_assert_eq!(self.vector_norms.len(), vecs.len() / dim);
        debug_assert_eq!(self.centroid_norms.len(), num_centroids);

        self.centroid_norms
            .par_iter_mut()
            .zip(centroids.par_chunks_exact(dim))
            .for_each(|(norm, centroid)| {
                *norm = matrix_norm(self.distance, centroid);
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

        let Some(block_size) = block_size else {
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
        let scale = matrix_scale(self.distance);
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
                        multiply(
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
                                            Distance::NegativeDotProduct | Distance::Cosine => {
                                                score
                                            }
                                            Distance::SquaredEuclidean => {
                                                score + vector_norm + centroid_norm
                                            }
                                        };
                                        let index = tile_start + index;
                                        best.update(score, index);
                                        // The approximate best can only decrease as more tiles are
                                        // visited, so this cutoff contains the final cutoff. Refine
                                        // candidates now instead of retaining earlier score tiles.
                                        if ambiguity.is_finite()
                                            && score <= best.best_score + ambiguity
                                        {
                                            let distance = match self.distance {
                                                Distance::SquaredEuclidean => {
                                                    squared_euclidean(vector, centroid)
                                                }
                                                Distance::NegativeDotProduct | Distance::Cosine => {
                                                    neg_dot_product(vector, centroid)
                                                }
                                            };
                                            best.update_exact(distance, index);
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
        2.0 * matrix_uncertainty(
            self.distance,
            self.gamma,
            vector_norm,
            self.max_centroid_norm,
            self.underflow_error,
        )
    }

    fn assign(&self, vector: &[f32], vector_norm: f32, best: BestTwo) -> u32 {
        let ambiguity = self.ambiguity(vector_norm);
        // Gamma scaling can leave a finite error bound even when unscaled dot products
        // overflow. GEMM and the direct kernel can then overflow differently; retain the
        // full scan because the finite-arithmetic error model cannot prune those scores.
        let dot_magnitude_overflows = self.distance != Distance::SquaredEuclidean
            && !(vector_norm * self.max_centroid_norm + ambiguity).is_finite();
        let invalid = dot_magnitude_overflows
            || !self.centroid_norms_are_finite
            || !vector_norm.is_finite()
            || !ambiguity.is_finite()
            || !best.best_score.is_finite();
        if invalid {
            let mut direct = [0];
            base_assign(
                vector,
                self.centroids,
                vector.len(),
                self.distance,
                &mut direct,
            );
            direct[0]
        } else if (self.distance == Distance::SquaredEuclidean && best.best_score < 0.0)
            || best.second_best_score - best.best_score <= ambiguity
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
    fn medium_batch_crossover_preserves_tiny_queries_and_reference_labels() {
        let accelerate = cfg!(all(target_os = "macos", target_arch = "aarch64"));
        // Boundaries for the minimum query count, query values, and comparison count.
        for (dim, rows, count, eligible) in [
            (768, 15, 2048, false),
            (768, 16, 2048, accelerate),
            (128, 31, 1024, false),
            (128, 32, 1023, false),
            (128, 32, 1024, accelerate),
            (128, 33, 1024, accelerate),
            (129, 32, 1024, accelerate),
            (64, 64, 1024, false),
        ] {
            let vectors: Vec<f32> = (0..rows * dim)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.125)
                .collect();
            let centroids: Vec<f32> = (0..count * dim)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.125)
                .collect();
            for distance in [Distance::SquaredEuclidean, Distance::NegativeDotProduct] {
                let workspace = MatrixAssignmentWorkspace::try_new(&vectors, count, dim, distance);
                assert_eq!(workspace.is_some(), eligible, "{rows}x{count}x{dim}");
                if let Some(mut workspace) = workspace {
                    let mut actual = vec![0; rows];
                    let mut expected = vec![0; rows];
                    workspace.assign(&vectors, &centroids, dim, &mut actual);
                    base_assign(&vectors, &centroids, dim, distance, &mut expected);
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn test_matrix_assignment_adapts_block_size_to_worker_scratch_budget() {
        assert_eq!(matmul_block_size(1_000_000, 4096, 16), Some(512));
        assert_eq!(matmul_block_size(1_000_000, 131_073, 16), Some(512));
        assert_eq!(matmul_block_size(1_000_000, 4096, 128), Some(256));
        assert_eq!(matmul_block_size(1_000_000, 4096, 256), Some(128));
        assert_eq!(matmul_block_size(1_000_000, 4096, 512), Some(64));
        assert_eq!(matmul_block_size(1_000_000, 4096, 513), None);
    }

    #[test]
    fn test_matrix_blocks_adapt_to_batch_parallelism() {
        // Native query batches have 1,000 rows. Tiles give every worker one exact-fit
        // chunk, clamped to the 64-512 row range, without power-of-two rounding.
        for (threads, expected) in [(1, 512), (8, 125), (12, 84), (16, 64)] {
            assert_eq!(matmul_block_size(1000, 4096, threads), Some(expected));
        }
        // Mid-size batches keep exact-fit tiles instead of rounding up to 512 rows.
        assert_eq!(matmul_block_size(5000, 4096, 16), Some(313));
        assert_eq!(matmul_block_size(6144, 4096, 16), Some(384));
        assert_eq!(matmul_block_size(65_536, 4096, 16), Some(512));
        assert_eq!(matmul_block_size(1000, 4096, 0), None);
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
        let dim = 65;
        let num_vectors = 513;
        let num_centroids = PREFERRED_CENTROID_BLOCK_SIZE + 1;
        let vecs = (0..num_vectors * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let centroids = (0..num_centroids * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let mut expected = vec![0; num_vectors];

        base_assign(
            &vecs,
            &centroids,
            dim,
            Distance::SquaredEuclidean,
            &mut expected,
        );
        for threads in [1, 4, 8, 16] {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut actual = vec![0; num_vectors];
                    let mut workspace = MatrixAssignmentWorkspace::try_new(
                        &vecs,
                        num_centroids,
                        dim,
                        Distance::SquaredEuclidean,
                    )
                    .unwrap();
                    workspace.assign(&vecs, &centroids, dim, &mut actual);

                    assert_eq!(actual, expected);
                });
        }
    }

    #[test]
    fn test_matrix_dot_product_matches_direct_across_centroid_tiles() {
        let mut rng = seeded_rng();
        let dim = 65;
        let num_vectors = 513;
        let num_centroids = PREFERRED_CENTROID_BLOCK_SIZE + 1;
        let vecs = (0..num_vectors * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let centroids = (0..num_centroids * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let mut expected = vec![0; num_vectors];

        base_assign(
            &vecs,
            &centroids,
            dim,
            Distance::NegativeDotProduct,
            &mut expected,
        );
        for threads in [1, 4, 8, 16] {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut actual = vec![0; num_vectors];
                    let mut workspace = MatrixAssignmentWorkspace::try_new(
                        &vecs,
                        num_centroids,
                        dim,
                        Distance::NegativeDotProduct,
                    )
                    .unwrap();
                    workspace.assign(&vecs, &centroids, dim, &mut actual);

                    assert_eq!(actual, expected);
                });
        }
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
    fn test_matrix_dot_product_refines_near_ties() {
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
    fn dot_refinement_preserves_late_winners_and_first_ties_across_tiles() {
        let dim = 65;
        let rows = 3;
        let count = PREFERRED_CENTROID_BLOCK_SIZE + 2;
        let mut vectors = vec![0.0; rows * dim];
        for vector in vectors.chunks_exact_mut(dim) {
            vector[0] = 1.0;
        }
        let mut centroids = vec![0.0; count * dim];
        centroids[0] = 1.0;
        // The true winner appears in the second tile and ties its next neighbor.
        for index in [
            PREFERRED_CENTROID_BLOCK_SIZE,
            PREFERRED_CENTROID_BLOCK_SIZE + 1,
        ] {
            centroids[index * dim] = 1.0 + 4.0 * f32::EPSILON;
        }
        let mut expected = vec![0; rows];
        base_assign(
            &vectors,
            &centroids,
            dim,
            Distance::NegativeDotProduct,
            &mut expected,
        );
        assert_eq!(expected, vec![PREFERRED_CENTROID_BLOCK_SIZE as u32; rows]);
        let mut workspace = MatrixAssignmentWorkspace::try_new_with_threshold(
            &vectors,
            count,
            dim,
            Distance::NegativeDotProduct,
            0,
        )
        .unwrap();
        let mut actual = vec![0; rows];
        workspace.assign_with_kernel(
            &vectors,
            &centroids,
            dim,
            &mut actual,
            Some(64),
            |mut dst, accum, lhs, rhs, alpha, parallelism| {
                crate::linalg::matmul(dst.as_mut(), accum, lhs, rhs, alpha, parallelism);
                // Simulate admissible GEMM rounding that makes centroid zero look best.
                // The perturbation is smaller than the existing dimension error bound.
                for row in 0..dst.nrows() {
                    for col in 0..dst.ncols() {
                        dst[(row, col)] = dst[(row, col)].round();
                    }
                }
            },
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn dot_refinement_retains_full_scan_when_bound_overflows() {
        let dim = 65;
        let mut vector = vec![0.0; dim];
        vector[0] = 1e30;
        let mut centroids = vec![0.0; 2 * dim];
        centroids[0] = -1e30;
        centroids[dim] = 1e-30;
        let mut expected = [0];
        base_assign(
            &vector,
            &centroids,
            dim,
            Distance::NegativeDotProduct,
            &mut expected,
        );
        assert_eq!(expected, [1]);
        let mut workspace = MatrixAssignmentWorkspace::try_new_with_threshold(
            &vector,
            2,
            dim,
            Distance::NegativeDotProduct,
            0,
        )
        .unwrap();
        let mut actual = [0];
        workspace.assign(&vector, &centroids, dim, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn dot_refinement_retains_full_scan_for_overflow_hidden_by_gemm() {
        let dim = 65;
        let mut vector = vec![0.0; dim];
        vector[0] = 1e20;
        let mut centroids = vec![0.0; 3 * dim];
        centroids[0] = 1e20;
        centroids[dim] = 1e-20;
        centroids[2 * dim] = 1e-20;
        let mut expected = [0];
        base_assign(
            &vector,
            &centroids,
            dim,
            Distance::NegativeDotProduct,
            &mut expected,
        );
        assert_eq!(expected, [0]);
        let mut workspace = MatrixAssignmentWorkspace::try_new_with_threshold(
            &vector,
            3,
            dim,
            Distance::NegativeDotProduct,
            0,
        )
        .unwrap();
        let mut actual = [0];
        workspace.assign_with_kernel(
            &vector,
            &centroids,
            dim,
            &mut actual,
            Some(64),
            |mut dst, _, _, _, _, _| {
                // An overflowed centroid may be NaN in GEMM but -infinity in the
                // direct kernel. The finite tied scores must still invoke the full scan.
                dst[(0, 0)] = f32::NAN;
                dst[(0, 1)] = -1.0;
                dst[(0, 2)] = -1.0;
            },
        );
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
