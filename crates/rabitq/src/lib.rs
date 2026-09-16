//! A minimal RaBitQ implementation for top-1 retrieval.

use core::f32;
use std::sync::atomic::{AtomicU64, Ordering};

use faer::{Col, Mat, MatRef, Row};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

mod fastscan;
pub mod rotator;

use fastscan::BinaryVectors;
use rotator::FhtKacRotator;

pub mod simd;
pub use simd::{
    asymmetric_binary_dot_product, binary_dot_product_native, min_max_residual,
    min_max_residual_native, scalar_quantize, vector_binarize_query,
};

#[inline]
fn squared_euclidean(lhs: &[f32], rhs: &[f32]) -> f32 {
    struct Impl<'a> {
        lhs: &'a [f32],
        rhs: &'a [f32],
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = f32;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            simd::pulp::l2_squared_distance(simd, self.lhs, self.rhs)
        }
    }

    pulp::Arch::new().dispatch(Impl { lhs, rhs })
}

const DEFAULT_X_DOT_PRODUCT: f32 = 0.8;
const EPSILON: f32 = 1.9;
pub(crate) const THETA_LOG_DIM: usize = 4;
const SCALAR: f32 = 1.0 / ((1 << THETA_LOG_DIM) as f32 - 1.0);
const QUERY_BLOCK_SIZE: usize = 4;

/// Factor struct to store the metadata for centroids.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Factor {
    /// ip
    pub factor_ip: f32,
    /// ppc
    pub factor_ppc: f32,
    /// error bound
    pub error_bound: f32,
    /// (x - c) ** 2
    pub center_distance_square: f32,
}

impl From<Vec<f32>> for Factor {
    fn from(f32s: Vec<f32>) -> Self {
        assert_eq!(f32s.len(), 4);
        Self {
            factor_ip: f32s[0],
            factor_ppc: f32s[1],
            error_bound: f32s[2],
            center_distance_square: f32s[3],
        }
    }
}

/// Convert the vector to binary format and store in a u64 vector.
#[inline]
pub fn vector_binarize_u64(vec: &[f32]) -> Vec<u64> {
    let mut binary = vec![0u64; vec.len().div_ceil(64)];
    for (i, &v) in vec.iter().enumerate() {
        if v > 0.0 {
            binary[i / 64] |= 1 << (i % 64);
        }
    }
    binary
}

/// Convert the vector to +1/-1 format.
#[inline]
pub fn vector_binarize_one(vec: &[f32]) -> Col<f32> {
    Col::from_fn(vec.len(), |i| if vec[i] > 0.0 { 1.0 } else { -1.0 })
}

/// Project the vector to the orthogonal matrix.
#[inline]
pub fn project(vec: &[f32], orthogonal: &MatRef<f32>) -> Col<f32> {
    struct Impl<'a> {
        vec: &'a [f32],
        orthogonal: &'a MatRef<'a, f32>,
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = Col<f32>;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { vec, orthogonal } = self;
            Col::from_fn(orthogonal.ncols(), |i| {
                simd::pulp::dot_product(
                    simd,
                    vec,
                    orthogonal
                        .col(i)
                        .try_as_col_major()
                        .expect("col major")
                        .as_slice(),
                )
            })
        }
    }

    pulp::Arch::new().dispatch(Impl { vec, orthogonal })
}

#[derive(Debug, Default)]
struct Metrics {
    pub rough: AtomicU64,
    pub precise: AtomicU64,
}

/// Reusable scratch space for RaBitQ queries.
pub struct RaBitQWorkspace {
    query: Vec<f32>,
    projected: Vec<f32>,
    quantized: Vec<u8>,
    binary: Vec<u64>,
    residual: Vec<f32>,
    fastscan: Option<fastscan::Workspace>,
}

impl RaBitQWorkspace {
    /// Create workspace for the padded index dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            query: vec![0.0; dim],
            projected: vec![0.0; dim],
            quantized: vec![0; dim],
            binary: vec![0; (dim * THETA_LOG_DIM).div_ceil(64)],
            residual: vec![0.0; dim],
            fastscan: None,
        }
    }
}

impl Metrics {
    pub fn update(&self, rough: u64, precise: u64) {
        self.rough.fetch_add(rough, Ordering::Relaxed);
        self.precise.fetch_add(precise, Ordering::Relaxed);
    }

    pub fn fetch(&self) -> (u64, u64) {
        (
            self.rough.load(Ordering::Relaxed),
            self.precise.load(Ordering::Relaxed),
        )
    }
}

/// RaBitQ struct for top-1 retrieval.
pub struct RaBitQ {
    centroids: Mat<f32>,
    mean: Row<f32>,
    rotator: FhtKacRotator,
    factors: Vec<Factor>,
    binary_vec: BinaryVectors,
    sorted_to_original: Vec<usize>,
    input_dim: usize,
    dim: usize,
    metrics: Metrics,
}

struct QueryFactors {
    center_distance_squared: f32,
    center_distance: f32,
    lower_bound: f32,
    scalar_sum: u32,
    delta: f32,
}

#[inline(always)]
fn rough_distance(
    binary_dot_product: u32,
    factor_ip: f32,
    factor_ppc: f32,
    error_bound: f32,
    center_distance_square: f32,
    query: &QueryFactors,
) -> f32 {
    center_distance_square
        + query.center_distance_squared
        + query.lower_bound * factor_ppc
        + (2.0 * binary_dot_product as f32 - query.scalar_sum as f32) * factor_ip * query.delta
        - error_bound * query.center_distance
}

fn compute_rough_distances(
    factors: &[Factor],
    start: usize,
    scores: &[u32],
    query: &QueryFactors,
    output: &mut [f32],
) {
    let factors = &factors[start..start + scores.len()];
    for index in 0..scores.len() {
        let factor = factors[index];
        output[index] = rough_distance(
            scores[index],
            factor.factor_ip,
            factor.factor_ppc,
            factor.error_bound,
            factor.center_distance_square,
            query,
        );
    }
}

impl RaBitQ {
    /// Return the padded index dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the number of indexed centroids.
    pub fn len(&self) -> usize {
        self.sorted_to_original.len()
    }

    /// Return whether the index contains no centroids.
    pub fn is_empty(&self) -> bool {
        self.sorted_to_original.is_empty()
    }

    /// Create a new RaBitQ instance.
    pub fn new(centroids: &[f32], dim: usize) -> Self {
        assert!(dim > 0, "dimension must be greater than zero");
        assert_eq!(centroids.len() % dim, 0, "centroids must be complete");
        assert!(!centroids.is_empty(), "at least one centroid is required");

        // init
        let num_centroids = centroids.len() / dim;
        let dim_pad = dim.div_ceil(64) * 64;
        let centroids_mat = Mat::from_fn(num_centroids, dim_pad, |i, j| match j < dim {
            true => centroids[i * dim + j],
            false => 0.0,
        });
        let dim_sqrt = (dim_pad as f32).sqrt();

        let mut rng = rand::rng();
        let rotator = FhtKacRotator::new(dim, dim_pad, &mut rng);
        let mut projected_data = vec![0.0; num_centroids * dim_pad];
        for (centroid, projected) in centroids
            .chunks_exact(dim)
            .zip(projected_data.chunks_exact_mut(dim_pad))
        {
            rotator.rotate(centroid, projected);
        }
        let projected = Mat::from_fn(num_centroids, dim_pad, |i, j| {
            projected_data[i * dim_pad + j]
        });
        let mut factors = vec![Factor::default(); num_centroids];
        let mut xc_distances = vec![0.0; num_centroids];
        let mut x_dot_product = vec![0.0; num_centroids];
        let mut binary_vec = Vec::with_capacity(num_centroids);
        let mut signed_vec = Vec::with_capacity(num_centroids);
        let mut mean = Row::zeros(dim_pad);
        for v in projected.row_iter() {
            mean += v;
        }
        mean.iter_mut().for_each(|v| *v /= num_centroids as f32);

        // factors
        for (i, p) in projected.row_iter().enumerate() {
            let xc = p - &mean;
            xc_distances[i] = xc.norm_l2();
            factors[i].center_distance_square = xc_distances[i].powi(2);
            let xc_slice = xc.try_as_row_major().expect("row major").as_slice();
            binary_vec.push(vector_binarize_u64(xc_slice));
            signed_vec.push(vector_binarize_one(xc_slice));
            let norm = xc_distances[i] * dim_sqrt;
            x_dot_product[i] = match norm.is_normal() {
                true => (&xc * &signed_vec[i]) / norm,
                false => DEFAULT_X_DOT_PRODUCT,
            };
        }

        let error_base = 2.0 * EPSILON / (dim_pad as f32 - 1.0).sqrt();
        for i in 0..num_centroids {
            let xc_over_ip = xc_distances[i] / x_dot_product[i];
            let factor = &mut factors[i];
            factor.error_bound =
                error_base * (xc_over_ip * xc_over_ip - factor.center_distance_square).sqrt();
            factor.factor_ip = -2.0 / dim_sqrt * xc_over_ip;
            factor.factor_ppc = factor.factor_ip * signed_vec[i].sum();
        }

        // sort by distances
        let mut sorted_to_original = xc_distances.iter().enumerate().collect::<Vec<_>>();
        sorted_to_original.sort_by(|&x, &y| x.1.partial_cmp(y.1).unwrap());
        let sorted_to_original = sorted_to_original
            .into_iter()
            .map(|(original_index, _)| original_index)
            .collect::<Vec<_>>();
        let sorted_binary_vec = sorted_to_original
            .iter()
            .flat_map(|&original_index| binary_vec[original_index].clone())
            .collect::<Vec<_>>();
        let binary_vec = BinaryVectors::new(sorted_binary_vec, num_centroids, dim_pad);
        let factors: Vec<Factor> = sorted_to_original
            .iter()
            .map(|&original_index| factors[original_index])
            .collect();
        let centroids_col_based = Mat::from_fn(num_centroids, dim_pad, |sorted_index, j| {
            *centroids_mat.get(sorted_to_original[sorted_index], j)
        })
        .transpose()
        .to_owned();

        RaBitQ {
            centroids: centroids_col_based,
            rotator,
            mean,
            binary_vec,
            factors,
            sorted_to_original,
            input_dim: dim,
            dim: dim_pad,
            metrics: Metrics::default(),
        }
    }

    /// Retrieve the top-1 index.
    pub fn retrieve_top_one(&self, query: &[f32]) -> usize {
        let mut workspace = RaBitQWorkspace::new(self.dim);
        let (index, precise) = self.retrieve_top_one_with_workspace(query, &mut workspace);
        self.metrics
            .update(self.sorted_to_original.len() as u64, precise);
        index
    }

    /// Retrieve the nearest centroid for a batch of queries in parallel.
    ///
    /// Workspaces are initialized by Rayon tasks and reused while each task
    /// processes its queries.
    pub fn retrieve_top_one_batch(&self, queries: &[f32], dim: usize, labels: &mut [u32]) {
        assert!(dim > 0, "dimension must be greater than zero");
        assert_eq!(
            dim, self.input_dim,
            "query dimension must match the vector dimension"
        );
        assert_eq!(queries.len() % dim, 0, "queries must be complete");
        assert_eq!(labels.len(), queries.len() / dim);

        let precise = match &self.binary_vec {
            BinaryVectors::FastScan(fastscan) => {
                self.retrieve_top_one_fastscan_batch(fastscan, queries, labels)
            }
            BinaryVectors::Row(_) => self.retrieve_top_one_parallel(queries, labels),
        };
        let rough = u64::try_from(labels.len())
            .expect("label count exceeds u64")
            .checked_mul(u64::try_from(self.len()).expect("centroid count exceeds u64"))
            .expect("comparison count exceeds u64");
        self.metrics.update(rough, precise);
    }

    /// Retrieve the top-1 index with reusable query workspace.
    ///
    /// Returns the index and the number of precise distance comparisons.
    pub fn retrieve_top_one_with_workspace(
        &self,
        query: &[f32],
        workspace: &mut RaBitQWorkspace,
    ) -> (usize, u64) {
        assert_eq!(
            query.len(),
            self.input_dim,
            "query dimension must match the vector dimension"
        );
        let query_factors = self.prepare_query(query, workspace);
        match &self.binary_vec {
            BinaryVectors::Row(binary_vec) => {
                workspace.binary.fill(0);
                vector_binarize_query(&workspace.quantized, &mut workspace.binary);
                let rough_scores = binary_vec
                    .chunks_exact(workspace.binary.len() / THETA_LOG_DIM)
                    .map(|binary| asymmetric_binary_dot_product(binary, &workspace.binary));
                self.select_top_one(rough_scores, &workspace.query, &query_factors)
            }
            BinaryVectors::FastScan(fastscan) => {
                let fastscan_workspace = workspace
                    .fastscan
                    .get_or_insert_with(|| fastscan::Workspace::new(self.dim));
                fastscan_workspace.prepare(&workspace.quantized);
                self.select_top_one_fastscan(
                    fastscan,
                    fastscan_workspace,
                    &workspace.query,
                    &query_factors,
                )
            }
        }
    }

    fn retrieve_top_one_parallel(&self, queries: &[f32], labels: &mut [u32]) -> u64 {
        labels
            .par_iter_mut()
            .zip(queries.par_chunks_exact(self.input_dim))
            .map_init(
                || RaBitQWorkspace::new(self.dim),
                |workspace, (label, query)| {
                    let (index, precise) = self.retrieve_top_one_with_workspace(query, workspace);
                    *label = index as u32;
                    precise
                },
            )
            .sum()
    }

    fn retrieve_top_one_fastscan_batch(
        &self,
        fastscan: &fastscan::FastScan,
        queries: &[f32],
        labels: &mut [u32],
    ) -> u64 {
        let full_query_count = labels.len() / QUERY_BLOCK_SIZE * QUERY_BLOCK_SIZE;
        let (full_labels, tail_labels) = labels.split_at_mut(full_query_count);
        let (full_queries, tail_queries) = queries.split_at(full_query_count * self.input_dim);

        let full_precise = full_labels
            .par_chunks_exact_mut(QUERY_BLOCK_SIZE)
            .zip(full_queries.par_chunks_exact(self.input_dim * QUERY_BLOCK_SIZE))
            .map_init(
                || std::array::from_fn(|_| RaBitQWorkspace::new(self.dim)),
                |workspaces, (labels, queries)| {
                    self.retrieve_top_one_fastscan_block(fastscan, queries, labels, workspaces)
                },
            )
            .sum::<u64>();

        full_precise + self.retrieve_top_one_tail(tail_queries, tail_labels)
    }

    fn retrieve_top_one_tail(&self, queries: &[f32], labels: &mut [u32]) -> u64 {
        if labels.is_empty() {
            return 0;
        }

        let mut workspace = RaBitQWorkspace::new(self.dim);
        labels
            .iter_mut()
            .zip(queries.chunks_exact(self.input_dim))
            .map(|(label, query)| {
                let (index, precise) = self.retrieve_top_one_with_workspace(query, &mut workspace);
                *label = index as u32;
                precise
            })
            .sum()
    }

    fn prepare_query(&self, query: &[f32], workspace: &mut RaBitQWorkspace) -> QueryFactors {
        workspace.query.fill(0.0);
        workspace.query[..query.len()].copy_from_slice(query);
        self.rotator.rotate(query, &mut workspace.projected);
        let mean_slice = self.mean.try_as_row_major().expect("row major").as_slice();
        let query_center_distance_squared = squared_euclidean(&workspace.projected, mean_slice);

        let (lower_bound, upper_bound) =
            min_max_residual(&mut workspace.residual, &workspace.projected, mean_slice);
        let delta = (upper_bound - lower_bound) * SCALAR;
        let one_over_delta = delta.recip();
        let scalar_sum = scalar_quantize(
            &mut workspace.quantized,
            &workspace.residual,
            lower_bound,
            one_over_delta,
        );
        let query_center_distance = query_center_distance_squared.sqrt();
        QueryFactors {
            center_distance_squared: query_center_distance_squared,
            center_distance: query_center_distance,
            lower_bound,
            scalar_sum,
            delta,
        }
    }

    fn retrieve_top_one_fastscan_block(
        &self,
        fastscan: &fastscan::FastScan,
        queries: &[f32],
        labels: &mut [u32],
        workspaces: &mut [RaBitQWorkspace; QUERY_BLOCK_SIZE],
    ) -> u64 {
        assert_eq!(labels.len(), QUERY_BLOCK_SIZE);
        assert_eq!(queries.len(), QUERY_BLOCK_SIZE * self.input_dim);

        let query_factors: [QueryFactors; QUERY_BLOCK_SIZE] = std::array::from_fn(|index| {
            let start = index * self.input_dim;
            let query = &queries[start..start + self.input_dim];
            let workspace = &mut workspaces[index];
            let factors = self.prepare_query(query, workspace);
            workspace
                .fastscan
                .get_or_insert_with(|| fastscan::Workspace::new(self.dim))
                .prepare(&workspace.quantized);
            factors
        });

        let mut thresholds = [f32::MAX; QUERY_BLOCK_SIZE];
        let mut min_indices = [0; QUERY_BLOCK_SIZE];
        let mut precise = [0_u64; QUERY_BLOCK_SIZE];
        let mut scores = [[0; fastscan::BATCH_SIZE]; QUERY_BLOCK_SIZE];
        let mut rough_distances = [[0.0; fastscan::BATCH_SIZE]; QUERY_BLOCK_SIZE];
        let luts = std::array::from_fn(|query| {
            workspaces[query]
                .fastscan
                .as_ref()
                .expect("FastScan workspace initialized above")
                .lut()
        });
        let mut position = 0;
        for binary in fastscan.batches() {
            fastscan.accumulate_many(binary, &luts, &mut scores);
            let candidates = fastscan::BATCH_SIZE.min(self.len() - position);
            for query in 0..QUERY_BLOCK_SIZE {
                compute_rough_distances(
                    &self.factors,
                    position,
                    &scores[query][..candidates],
                    &query_factors[query],
                    &mut rough_distances[query][..candidates],
                );
            }

            for query in 0..QUERY_BLOCK_SIZE {
                for (candidate, &rough_distance) in
                    rough_distances[query][..candidates].iter().enumerate()
                {
                    if rough_distance < thresholds[query] {
                        precise[query] += 1;
                        let sorted_index = position + candidate;
                        let accurate = squared_euclidean(
                            self.centroids
                                .col(sorted_index)
                                .try_as_col_major()
                                .expect("col major")
                                .as_slice(),
                            &workspaces[query].query,
                        );
                        if accurate < thresholds[query] {
                            thresholds[query] = accurate;
                            min_indices[query] = self.sorted_to_original[sorted_index];
                        }
                    }
                }
            }
            position += candidates;
        }

        for (label, index) in labels.iter_mut().zip(min_indices) {
            *label = index as u32;
        }
        precise.iter().sum()
    }

    fn select_top_one_fastscan(
        &self,
        fastscan: &fastscan::FastScan,
        workspace: &mut fastscan::Workspace,
        query: &[f32],
        query_factors: &QueryFactors,
    ) -> (usize, u64) {
        let mut threshold = f32::MAX;
        let mut min_index = 0;
        let mut precise = 0;
        let mut position = 0;
        let mut rough_distances = [0.0; fastscan::BATCH_SIZE];
        for binary in fastscan.batches() {
            fastscan.accumulate(binary, workspace);
            let count = fastscan::BATCH_SIZE.min(self.len() - position);
            compute_rough_distances(
                &self.factors,
                position,
                &workspace.scores()[..count],
                query_factors,
                &mut rough_distances[..count],
            );
            for &rough in &rough_distances[..count] {
                let original_index = self.sorted_to_original[position];
                if rough < threshold {
                    precise += 1;
                    let accurate = squared_euclidean(
                        self.centroids
                            .col(position)
                            .try_as_col_major()
                            .expect("col major")
                            .as_slice(),
                        query,
                    );
                    if accurate < threshold {
                        threshold = accurate;
                        min_index = original_index;
                    }
                }
                position += 1;
            }
        }
        (min_index, precise)
    }

    fn select_top_one(
        &self,
        rough_scores: impl Iterator<Item = u32>,
        query: &[f32],
        query_factors: &QueryFactors,
    ) -> (usize, u64) {
        let mut threshold = f32::MAX;
        let mut min_index = 0;
        let mut precise = 0;
        for ((position, &original_index), binary_dot_product) in
            self.sorted_to_original.iter().enumerate().zip(rough_scores)
        {
            let factor = self.factors[position];
            let rough = rough_distance(
                binary_dot_product,
                factor.factor_ip,
                factor.factor_ppc,
                factor.error_bound,
                factor.center_distance_square,
                query_factors,
            );
            if rough < threshold {
                precise += 1;
                let accurate = squared_euclidean(
                    self.centroids
                        .col(position)
                        .try_as_col_major()
                        .expect("col major")
                        .as_slice(),
                    query,
                );
                if accurate < threshold {
                    threshold = accurate;
                    min_index = original_index;
                }
            }
        }
        (min_index, precise)
    }

    /// Add rough and precise comparison counts to the index metrics.
    pub fn update_metrics(&self, rough: u64, precise: u64) {
        self.metrics.update(rough, precise);
    }

    /// Get the rough/precise metrics.
    pub fn get_metrics(&self) -> (u64, u64) {
        self.metrics.fetch()
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use seed_rand::seeded_rng;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
    use super::THETA_LOG_DIM;
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    use super::binary_dot_product_native;
    use super::{
        BinaryVectors, Factor, QueryFactors, RaBitQ, RaBitQWorkspace, SCALAR,
        compute_rough_distances, min_max_residual, min_max_residual_native, squared_euclidean,
    };
    use crate::simd;

    #[test]
    #[should_panic(expected = "centroids must be complete")]
    fn test_new_rejects_incomplete_centroids() {
        RaBitQ::new(&[0.0, 1.0, 2.0], 2);
    }

    #[test]
    fn batched_rough_distances_match_individual_formula_across_batch_tails() {
        let factors = (0..40)
            .map(|index| Factor {
                factor_ip: -0.1 - index as f32 * 0.001,
                factor_ppc: index as f32 * 0.25 - 4.0,
                error_bound: 0.01 + index as f32 * 0.0001,
                center_distance_square: 2.0 + index as f32 * 0.5,
            })
            .collect::<Vec<_>>();
        let query = QueryFactors {
            center_distance_squared: 3.25,
            center_distance: 1.8,
            lower_bound: -0.2,
            scalar_sum: 317,
            delta: 0.015,
        };

        for count in [1, 3, 31, 32] {
            let start = 2;
            let scores = (0..count)
                .map(|index| (index * 37 + 11) as u32)
                .collect::<Vec<_>>();
            let mut actual = vec![0.0; count];
            compute_rough_distances(&factors, start, &scores, &query, &mut actual);

            for (index, &actual) in actual.iter().enumerate() {
                let factor = factors[start + index];
                let expected = factor.center_distance_square
                    + query.center_distance_squared
                    + query.lower_bound * factor.factor_ppc
                    + (2.0 * scores[index] as f32 - query.scalar_sum as f32)
                        * factor.factor_ip
                        * query.delta
                    - factor.error_bound * query.center_distance;
                let tolerance = 1e-6 * expected.abs().max(1.0);
                assert!((actual - expected).abs() <= tolerance);
            }
        }
    }

    #[test]
    #[allow(unsafe_code)]
    fn test_binary_dot_product() {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        if !crate::simd::x86::Avx2::is_available() {
            return;
        }
        let mut rng = seeded_rng();

        for _ in 0..100 {
            for dim in [1, 2, 4, 8, 10] {
                let x = (0..dim).map(|_| rng.random::<u64>()).collect::<Vec<_>>();
                let y = (0..dim).map(|_| rng.random::<u64>()).collect::<Vec<_>>();

                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    assert_eq!(
                        binary_dot_product_native(&x, &y),
                        simd::x86::binary_dot_product(&x, &y),
                    );
                    assert_eq!(binary_dot_product_native(&x, &y), unsafe {
                        simd::x86::legacy::binary_dot_product(&x, &y)
                    });
                }

                #[cfg(target_arch = "aarch64")]
                assert_eq!(
                    simd::native::binary_dot_product(&x, &y),
                    simd::aarch64::binary_dot_product(&x, &y),
                );
            }
        }
    }

    #[test]
    fn test_query_binarize() {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        if !crate::simd::x86::Avx2::is_available() {
            return;
        }
        let mut rng = seeded_rng();

        for _ in 0..100 {
            for dim in [64, 128, 256, 320, 1024] {
                let x = (0..dim).map(|_| rng.random::<u8>()).collect::<Vec<u8>>();
                let mut binary = vec![0u64; (dim * THETA_LOG_DIM).div_ceil(64)];
                simd::native::vector_binarize_query(&x, &mut binary);

                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    let mut binary_simd = vec![0u64; binary.len()];
                    simd::x86::vector_binarize_query(&x, &mut binary_simd);
                    assert_eq!(binary, binary_simd);
                }

                #[cfg(target_arch = "aarch64")]
                {
                    let mut binary_simd = vec![0u64; binary.len()];
                    simd::aarch64::vector_binarize_query(&x, &mut binary_simd);
                    assert_eq!(binary, binary_simd);
                }
            }
        }
    }

    #[test]
    fn test_scalar_quantize() {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        if !simd::x86::Avx2::is_available() {
            return;
        }

        let mut rng = seeded_rng();
        for _ in 0..100 {
            for dim in [1, 15, 16, 17, 64, 128, 256, 320, 1024] {
                let x = (0..dim).map(|_| rng.random::<f32>()).collect::<Vec<f32>>();
                let y = (0..dim).map(|_| rng.random::<f32>()).collect::<Vec<f32>>();
                let mut quantized = vec![0u8; dim];
                let mut residual = vec![0.0; dim];
                let (upper_bound, lower_bound) = min_max_residual_native(&mut residual, &x, &y);
                let multiplier = ((upper_bound - lower_bound) * SCALAR).recip();
                let sum = simd::native::scalar_quantize(
                    &mut quantized,
                    &residual,
                    lower_bound,
                    multiplier,
                );
                let mut quantized_simd = vec![0u8; dim];
                let sum_simd =
                    simd::scalar_quantize(&mut quantized_simd, &residual, lower_bound, multiplier);
                assert_eq!(quantized, quantized_simd);
                assert_eq!(sum, sum_simd);
            }
        }
    }

    #[test]
    #[allow(unsafe_code)]
    fn test_min_max_residual() {
        let mut rng = seeded_rng();
        for _ in 0..100 {
            for dim in [32, 64, 124, 128, 132].into_iter() {
                let x = (0..dim).map(|_| rng.random::<f32>()).collect::<Vec<f32>>();
                let y = (0..dim).map(|_| rng.random::<f32>()).collect::<Vec<f32>>();
                let mut res = vec![0.0; dim];
                let (min, max) = min_max_residual_native(&mut res, &x, &y);

                let mut res_pulp = vec![0.0; dim];
                let (min_pulp, max_pulp) = min_max_residual(&mut res_pulp, &x, &y);

                assert_eq!(min, min_pulp);
                assert_eq!(max, max_pulp);
                assert_eq!(res, res_pulp);

                #[cfg(target_arch = "aarch64")]
                {
                    let mut res_simd = vec![0.0; dim];
                    let (min_simd, max_simd) =
                        simd::aarch64::legacy::min_max_residual(&mut res_simd, &x, &y);
                    assert_eq!(min, min_simd);
                    assert_eq!(max, max_simd);
                    assert_eq!(res, res_simd);
                }

                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if !is_x86_feature_detected!("avx2") {
                        continue;
                    }
                    let mut res_simd = vec![0.0; dim];
                    let (min_simd, max_simd) =
                        unsafe { simd::x86::legacy::min_max_residual(&mut res_simd, &x, &y) };

                    assert_eq!(min, min_simd);
                    assert_eq!(max, max_simd);
                    assert_eq!(res, res_simd);
                }
            }
        }
    }

    #[test]
    fn test_batch_retrieval_matches_individual_retrieval() {
        let mut rng = seeded_rng();
        let dim = 64;
        let centroids = (0..16 * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let queries = (0..128 * dim)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let rabitq = RaBitQ::new(&centroids, dim);

        let expected = queries
            .chunks_exact(dim)
            .map(|query| rabitq.retrieve_top_one(query) as u32)
            .collect::<Vec<_>>();
        let mut actual = vec![0; expected.len()];
        rabitq.retrieve_top_one_batch(&queries, dim, &mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_row_retrieval_does_not_allocate_fastscan_workspace() {
        let dim = 64;
        let centroids = vec![0.0; 16 * dim];
        let rabitq = RaBitQ::new(&centroids, dim);
        let mut workspace = RaBitQWorkspace::new(rabitq.dim());

        assert!(matches!(rabitq.binary_vec, BinaryVectors::Row(_)));
        assert!(workspace.fastscan.is_none());
        rabitq.retrieve_top_one_with_workspace(&centroids[..dim], &mut workspace);
        assert!(workspace.fastscan.is_none());
    }

    #[test]
    fn test_fastscan_retrieval_matches_brute_force_across_partial_batch() {
        let mut rng = seeded_rng();
        let dim = 64;
        let num_centroids = 257;
        let centroids = (0..num_centroids * dim)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect::<Vec<_>>();
        let rabitq = RaBitQ::new(&centroids, dim);
        if !matches!(rabitq.binary_vec, BinaryVectors::FastScan(_)) {
            return;
        }
        assert_ne!(
            rabitq.sorted_to_original,
            (0..num_centroids).collect::<Vec<_>>()
        );

        let mut workspace = RaBitQWorkspace::new(rabitq.dim());
        for query in centroids.chunks_exact(dim) {
            let expected = centroids
                .chunks_exact(dim)
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    squared_euclidean(left, query).total_cmp(&squared_euclidean(right, query))
                })
                .map(|(index, _)| index)
                .unwrap();
            let (actual, _) = rabitq.retrieve_top_one_with_workspace(query, &mut workspace);
            assert_eq!(actual, expected);
        }
        assert!(workspace.fastscan.is_some());
    }

    #[test]
    fn test_fastscan_query_blocks_match_individual_retrieval_across_tails() {
        let mut rng = seeded_rng();
        let dim = 64;
        let num_centroids = 257;
        let centroids = (0..num_centroids * dim)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect::<Vec<_>>();
        let rabitq = RaBitQ::new(&centroids, dim);
        if !matches!(rabitq.binary_vec, BinaryVectors::FastScan(_)) {
            return;
        }

        for count in [1, 3, 4, 5, 8, 9] {
            let queries = (0..count * dim)
                .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                .collect::<Vec<_>>();
            let mut workspace = RaBitQWorkspace::new(rabitq.dim());
            let expected = queries
                .chunks_exact(dim)
                .map(|query| rabitq.retrieve_top_one_with_workspace(query, &mut workspace))
                .collect::<Vec<_>>();
            let before = rabitq.get_metrics();
            let mut actual = vec![0; count];
            rabitq.retrieve_top_one_batch(&queries, dim, &mut actual);
            let after = rabitq.get_metrics();

            assert_eq!(
                actual,
                expected
                    .iter()
                    .map(|&(index, _)| index as u32)
                    .collect::<Vec<_>>()
            );
            assert_eq!(after.0 - before.0, (count * num_centroids) as u64);
            assert_eq!(
                after.1 - before.1,
                expected.iter().map(|&(_, precise)| precise).sum::<u64>()
            );
        }
    }

    #[test]
    fn test_retrieval_matches_brute_force_after_centroid_sorting() {
        for dim in [64, 65] {
            let values = [-30.0, -2.0, 0.0, 1.0, 8.0, 40.0];
            let mut centroids = vec![0.0; values.len() * dim];
            for (centroid, &value) in centroids.chunks_exact_mut(dim).zip(&values) {
                centroid[0] = value;
            }
            let rabitq = RaBitQ::new(&centroids, dim);

            assert_ne!(
                rabitq.sorted_to_original,
                (0..values.len()).collect::<Vec<_>>(),
                "dimension {dim}",
            );
            for query in centroids.chunks_exact(dim) {
                let expected = centroids
                    .chunks_exact(dim)
                    .enumerate()
                    .min_by(|(_, left), (_, right)| {
                        squared_euclidean(left, query).total_cmp(&squared_euclidean(right, query))
                    })
                    .map(|(index, _)| index)
                    .unwrap();
                assert_eq!(rabitq.retrieve_top_one(query), expected, "dimension {dim}");
            }
        }
    }
}
