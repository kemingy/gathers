//! A minimal RaBitQ implementation for top-1 retrieval.

use core::f32;
use std::sync::atomic::{AtomicU64, Ordering};

use faer::{Col, Mat, MatRef, Row};
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator, ParallelSlice,
};

pub mod simd;

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

impl Factor {
    #[allow(dead_code)]
    fn into_vec(self) -> Vec<f32> {
        vec![
            self.factor_ip,
            self.factor_ppc,
            self.error_bound,
            self.center_distance_square,
        ]
    }
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

#[inline]
fn project_into(vec: &[f32], orthogonal: &MatRef<f32>, output: &mut [f32]) {
    struct Impl<'a, 'b> {
        vec: &'a [f32],
        orthogonal: &'a MatRef<'b, f32>,
        output: &'a mut [f32],
    }

    impl pulp::WithSimd for Impl<'_, '_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self {
                vec,
                orthogonal,
                output,
            } = self;
            for (i, value) in output.iter_mut().enumerate() {
                *value = simd::pulp::dot_product(
                    simd,
                    vec,
                    orthogonal
                        .col(i)
                        .try_as_col_major()
                        .expect("col major")
                        .as_slice(),
                );
            }
        }
    }

    assert_eq!(output.len(), orthogonal.ncols());
    pulp::Arch::new().dispatch(Impl {
        vec,
        orthogonal,
        output,
    });
}

/// Get the min/max value of the residual of two vectors.
#[inline]
pub fn min_max_residual_native(res: &mut [f32], x: &[f32], y: &[f32]) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for i in 0..res.len() {
        res[i] = x[i] - y[i];
        if res[i] < min {
            min = res[i];
        }
        if res[i] > max {
            max = res[i];
        }
    }
    (min, max)
}

/// Interface of `min_max_residual`: get the min/max value of the residual of two vectors.
#[inline]
pub fn min_max_residual(res: &mut [f32], x: &[f32], y: &[f32]) -> (f32, f32) {
    struct Impl<'a> {
        res: &'a mut [f32],
        x: &'a [f32],
        y: &'a [f32],
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = (f32, f32);

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { res, x, y } = self;
            simd::pulp::min_max_residual(simd, res, x, y)
        }
    }

    pulp::Arch::new().dispatch(Impl { res, x, y })
}

// Quantize the query residual vector.
#[inline]
fn scalar_quantize_native(
    quantized: &mut [u8],
    vec: &[f32],
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    let mut sum = 0u32;
    for i in 0..quantized.len() {
        let q = ((vec[i] - lower_bound) * multiplier).round() as u8;
        quantized[i] = q;
        sum += q as u32;
    }
    sum
}

/// Interface of `scalar_quantize`: scale vector to u8.
#[inline]
pub fn scalar_quantize(
    quantized: &mut [u8],
    vec: &[f32],
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if simd::Avx2::is_available() {
            simd::scalar_quantize(quantized, vec, lower_bound, multiplier)
        } else {
            scalar_quantize_native(quantized, vec, lower_bound, multiplier)
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        scalar_quantize_native(quantized, vec, lower_bound, multiplier)
    }
}

/// Convert the vector to binary format (one value to multiple bits) and store in a u64 vector.
#[inline]
fn vector_binarize_query_native(vec: &[u8], binary: &mut [u64]) {
    let length = vec.len();
    for j in 0..THETA_LOG_DIM {
        for i in 0..length {
            binary[(i + j * length) / 64] |= (((vec[i] >> j) & 1) as u64) << (i % 64);
        }
    }
}

/// Interface of `vector_binarize_query`
#[inline]
pub fn vector_binarize_query(vec: &[u8], binary: &mut [u64]) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if simd::Avx2::is_available() {
            simd::vector_binarize_query(vec, binary);
        } else {
            vector_binarize_query_native(vec, binary);
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        vector_binarize_query_native(vec, binary);
    }
}

/// Calculate the dot product of two binary vectors.
#[inline]
pub fn binary_dot_product_native(x: &[u64], y: &[u64]) -> u32 {
    let mut res = 0;
    for i in 0..x.len() {
        res += (x[i] & y[i]).count_ones();
    }
    res
}

/// Calculate the dot product of two binary vectors with different lengths.
///
/// The length of `y` should be `x.len() * THETA_LOG_DIM`.
#[inline]
pub fn asymmetric_binary_dot_product(x: &[u64], y: &[u64]) -> u32 {
    let mut res = 0;
    let length = x.len();
    assert_eq!(y.len(), length * THETA_LOG_DIM);
    for i in 0..THETA_LOG_DIM {
        let y_slice = &y[i * length..(i + 1) * length];
        res += {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if simd::Avx2::is_available() {
                    unsafe { simd::binary_dot_product_simd(x, y_slice) << i }
                    // crate::simd::binary_dot_product(x, y_slice) << i
                } else {
                    binary_dot_product_native(x, y_slice) << i
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
            {
                binary_dot_product_native(x, y_slice) << i
            }
        };
    }
    res
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
    orthogonal: Mat<f32>,
    factors: Vec<Factor>,
    binary_vec: Vec<u64>,
    sorted_to_original: Vec<usize>,
    input_dim: usize,
    dim: usize,
    metrics: Metrics,
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

        // orthogonal matrix
        let mut rng = rand::rng();
        let random: Mat<f32> = Mat::from_fn(dim_pad, dim_pad, |_, _| rng.sample(StandardNormal));
        let orthogonal = random.qr().compute_Q();

        let projected = &centroids_mat * &orthogonal;
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
        let binary_vec = sorted_to_original
            .iter()
            .flat_map(|&original_index| binary_vec[original_index].clone())
            .collect();
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
            orthogonal,
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

        let (rough, precise) = labels
            .par_iter_mut()
            .zip(queries.par_chunks_exact(dim))
            .map_init(
                || RaBitQWorkspace::new(self.dim),
                |workspace, (label, query)| {
                    let (index, precise) = self.retrieve_top_one_with_workspace(query, workspace);
                    *label = index as u32;
                    (self.len() as u64, precise)
                },
            )
            .reduce(
                || (0, 0),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
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
        workspace.query.fill(0.0);
        workspace.query[..query.len()].copy_from_slice(query);
        workspace.binary.fill(0);

        project_into(
            &workspace.query,
            &self.orthogonal.as_ref(),
            &mut workspace.projected,
        );
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
        vector_binarize_query(&workspace.quantized, &mut workspace.binary);

        let mut threshold = f32::MAX;
        let mut min_index = 0;
        let mut precise = 0;
        let query_center_distance = query_center_distance_squared.sqrt();
        let offset = workspace.binary.len() / THETA_LOG_DIM;
        for (position, &original_index) in self.sorted_to_original.iter().enumerate() {
            let factor = &self.factors[position];
            let rough = factor.center_distance_square
                + query_center_distance_squared
                + lower_bound * factor.factor_ppc
                + (2.0
                    * asymmetric_binary_dot_product(
                        &self.binary_vec[position * offset..(position + 1) * offset],
                        &workspace.binary,
                    ) as f32
                    - scalar_sum as f32)
                    * factor.factor_ip
                    * delta
                - factor.error_bound * query_center_distance;
            if rough < threshold {
                precise += 1;
                let accurate = squared_euclidean(
                    self.centroids
                        .col(position)
                        .try_as_col_major()
                        .expect("col major")
                        .as_slice(),
                    &workspace.query,
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
mod test {
    use rand::Rng;
    use seed_rand::seeded_rng;

    use super::squared_euclidean;
    use super::{RaBitQ, min_max_residual, min_max_residual_native};
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    use super::{
        SCALAR, THETA_LOG_DIM, binary_dot_product_native, scalar_quantize_native,
        vector_binarize_query_native,
    };
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    use crate::simd;

    #[test]
    #[should_panic(expected = "centroids must be complete")]
    fn test_new_rejects_incomplete_centroids() {
        RaBitQ::new(&[0.0, 1.0, 2.0], 2);
    }

    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn test_binary_dot_product() {
        if !crate::simd::Avx2::is_available() {
            return;
        }
        let mut rng = seeded_rng();

        for _ in 0..100 {
            for dim in [1, 2, 4, 8, 10].into_iter() {
                let x = (0..dim).map(|_| rng.random::<u64>()).collect::<Vec<u64>>();
                let y = (0..dim).map(|_| rng.random::<u64>()).collect::<Vec<u64>>();

                assert_eq!(
                    binary_dot_product_native(&x, &y),
                    simd::binary_dot_product(&x, &y),
                );
                assert_eq!(binary_dot_product_native(&x, &y), unsafe {
                    simd::binary_dot_product_simd(&x, &y)
                },);
            }
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn test_query_binarize() {
        if !crate::simd::Avx2::is_available() {
            return;
        }
        let mut rng = seeded_rng();

        for _ in 0..100 {
            for dim in [64, 128, 256, 320, 1024].into_iter() {
                let x = (0..dim).map(|_| rng.random::<u8>()).collect::<Vec<u8>>();
                let mut binary = vec![0u64; (dim * THETA_LOG_DIM).div_ceil(64)];
                vector_binarize_query_native(&x, &mut binary);
                let mut binary_simd = vec![0u64; (dim * THETA_LOG_DIM).div_ceil(64)];
                simd::vector_binarize_query(&x, &mut binary_simd);
                assert_eq!(binary, binary_simd);
            }
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn test_scalar_quantize() {
        if !crate::simd::Avx2::is_available() {
            return;
        }
        let mut rng = seeded_rng();
        for _ in 0..100 {
            for dim in [64, 128, 256, 320, 1024].into_iter() {
                let x = (0..dim).map(|_| rng.random::<f32>()).collect::<Vec<f32>>();
                let y = (0..dim).map(|_| rng.random::<f32>()).collect::<Vec<f32>>();
                let mut quantized = vec![0u8; dim];
                let mut residual = vec![0.0; dim];
                let (upper_bound, lower_bound) = min_max_residual_native(&mut residual, &x, &y);
                let multiplier = ((upper_bound - lower_bound) * SCALAR).recip();
                let sum =
                    scalar_quantize_native(&mut quantized, &residual, lower_bound, multiplier);
                let mut quantized_simd = vec![0u8; dim];
                let sum_simd =
                    simd::scalar_quantize(&mut quantized_simd, &residual, lower_bound, multiplier);
                assert_eq!(quantized, quantized_simd);
                assert_eq!(sum, sum_simd);
            }
        }
    }

    #[test]
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

                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if !is_x86_feature_detected!("avx2") {
                        continue;
                    }
                    let mut res_simd = vec![0.0; dim];
                    let (min_simd, max_simd) =
                        unsafe { simd::min_max_residual(&mut res_simd, &x, &y) };

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
    fn test_retrieval_matches_brute_force_after_centroid_sorting() {
        let dim = 64;
        let values = [-30.0, -2.0, 0.0, 1.0, 8.0, 40.0];
        let mut centroids = vec![0.0; values.len() * dim];
        for (centroid, &value) in centroids.chunks_exact_mut(dim).zip(&values) {
            centroid[0] = value;
        }
        let rabitq = RaBitQ::new(&centroids, dim);

        assert_ne!(
            rabitq.sorted_to_original,
            (0..values.len()).collect::<Vec<_>>()
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
            assert_eq!(rabitq.retrieve_top_one(query), expected);
        }
    }
}
