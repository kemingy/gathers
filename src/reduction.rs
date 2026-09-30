//! Linear dimensionality reduction for dense row-major vectors.
//!
//! [`Pca`] learns a data-dependent projection that maximizes preserved variance. [`Srht`] is a
//! training-free subsampled randomized Hadamard transform. Both expose approximate inverse
//! transforms so reduced-space centroids can be materialized in the original vector space.

use aligned_vec::{AVec, avec};
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par, Side};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::sampling::sample_indices;

/// Errors returned when constructing or applying a reduction transform.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReductionError {
    /// The input dimension is zero or the flat input does not contain complete rows.
    #[error("input length {len} is not a non-empty multiple of dimension {dim}")]
    InvalidInputShape {
        /// Number of scalar values in the flat input.
        len: usize,
        /// Requested row dimension.
        dim: usize,
    },
    /// The requested output dimension is zero or exceeds the transform's input dimension.
    #[error("output dimension {output_dim} must be between 1 and input dimension {input_dim}")]
    InvalidOutputDimension {
        /// Transform input dimension.
        input_dim: usize,
        /// Requested output dimension.
        output_dim: usize,
    },
    /// PCA requires at least two training rows.
    #[error("PCA requires at least two rows, got {rows}")]
    TooFewRows {
        /// Number of supplied rows.
        rows: usize,
    },
    /// The input contains a NaN or infinity.
    #[error("input contains a non-finite coordinate")]
    NonFiniteInput,
    /// The covariance eigendecomposition did not converge.
    #[error("PCA covariance eigendecomposition failed")]
    DecompositionFailed,
}

fn validate_shape(values: &[f32], dim: usize) -> Result<usize, ReductionError> {
    if dim == 0 || values.is_empty() || !values.len().is_multiple_of(dim) {
        return Err(ReductionError::InvalidInputShape {
            len: values.len(),
            dim,
        });
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(ReductionError::NonFiniteInput);
    }
    Ok(values.len() / dim)
}

fn validate_output_dimension(input_dim: usize, output_dim: usize) -> Result<(), ReductionError> {
    if output_dim == 0 || output_dim > input_dim {
        return Err(ReductionError::InvalidOutputDimension {
            input_dim,
            output_dim,
        });
    }
    Ok(())
}

/// Principal component projection learned from dense `f32` vectors.
///
/// Training centers the data using `f64` means, forms an `f32` covariance matrix, and retains the
/// eigenvectors with the largest eigenvalues. Coordinates are not standardized or whitened, so
/// the transform preserves the covariance-PCA geometry used by squared-Euclidean clustering.
#[derive(Debug, Clone)]
pub struct Pca {
    input_dim: usize,
    output_dim: usize,
    mean: Vec<f32>,
    // Principal directions in descending variance order, row-major output_dim x input_dim.
    components: Vec<f32>,
    explained_variance: Vec<f32>,
    total_variance: f64,
}

impl Pca {
    /// Fit PCA to flat row-major training vectors.
    pub fn fit(
        vectors: &[f32],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, ReductionError> {
        validate_output_dimension(input_dim, output_dim)?;
        let rows = validate_shape(vectors, input_dim)?;
        if rows < 2 {
            return Err(ReductionError::TooFewRows { rows });
        }

        let mut mean64 = vec![0.0_f64; input_dim];
        for row in vectors.chunks_exact(input_dim) {
            for (mean, &value) in mean64.iter_mut().zip(row) {
                *mean += f64::from(value);
            }
        }
        for value in &mut mean64 {
            *value /= rows as f64;
        }
        let mean = mean64.iter().map(|&value| value as f32).collect::<Vec<_>>();

        let mut centered: AVec<f32> = AVec::new(64);
        centered.resize(vectors.len(), 0.0_f32);
        centered
            .par_chunks_mut(input_dim)
            .zip(vectors.par_chunks(input_dim))
            .for_each(|(output, input)| {
                for ((output, &input), &mean) in output.iter_mut().zip(input).zip(&mean) {
                    *output = input - mean;
                }
            });

        let mut covariance = vec![0.0_f32; input_dim * input_dim];
        let centered = MatRef::from_row_major_slice(&centered, rows, input_dim);
        matmul(
            MatMut::from_row_major_slice_mut(&mut covariance, input_dim, input_dim),
            Accum::Replace,
            centered.transpose(),
            centered,
            1.0 / (rows - 1) as f32,
            Par::rayon(0),
        );
        let eigen = MatRef::from_row_major_slice(&covariance, input_dim, input_dim)
            .self_adjoint_eigen(Side::Lower)
            .map_err(|_| ReductionError::DecompositionFailed)?;
        let eigenvalues = eigen.S().column_vector();
        let eigenvectors = eigen.U();
        let total_variance = (0..input_dim)
            .map(|index| f64::from((*eigenvalues.get(index)).max(0.0)))
            .sum();
        let mut explained_variance = Vec::with_capacity(output_dim);
        let mut components = Vec::with_capacity(output_dim * input_dim);
        for component in 0..output_dim {
            let source = input_dim - 1 - component;
            explained_variance.push((*eigenvalues.get(source)).max(0.0));
            for coordinate in 0..input_dim {
                components.push(*eigenvectors.get(coordinate, source));
            }
        }
        Ok(Self {
            input_dim,
            output_dim,
            mean,
            components,
            explained_variance,
            total_variance,
        })
    }

    /// Input vector dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Projected vector dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Coordinate-wise training mean subtracted before projection.
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Principal directions in descending variance order, as row-major
    /// `output_dim × input_dim` values.
    pub fn components(&self) -> &[f32] {
        &self.components
    }

    /// Variance captured by each retained component, in descending order.
    pub fn explained_variance(&self) -> &[f32] {
        &self.explained_variance
    }

    /// Fraction of total training variance captured by the retained components.
    pub fn preserved_variance(&self) -> f64 {
        if self.total_variance == 0.0 {
            1.0
        } else {
            self.explained_variance
                .iter()
                .map(|&value| f64::from(value))
                .sum::<f64>()
                / self.total_variance
        }
    }

    /// Project flat row-major vectors, allocating an aligned output buffer.
    pub fn transform(&self, vectors: &[f32]) -> Result<AVec<f32>, ReductionError> {
        let rows = validate_shape(vectors, self.input_dim)?;
        let mut centered: AVec<f32> = AVec::new(64);
        centered.resize(vectors.len(), 0.0_f32);
        centered
            .par_chunks_mut(self.input_dim)
            .zip(vectors.par_chunks(self.input_dim))
            .for_each(|(output, input)| {
                for ((output, &input), &mean) in output.iter_mut().zip(input).zip(&self.mean) {
                    *output = input - mean;
                }
            });
        let mut output = avec!(0.0_f32; rows * self.output_dim);
        matmul(
            MatMut::from_row_major_slice_mut(&mut output, rows, self.output_dim),
            Accum::Replace,
            MatRef::from_row_major_slice(&centered, rows, self.input_dim),
            MatRef::from_row_major_slice(&self.components, self.output_dim, self.input_dim)
                .transpose(),
            1.0,
            Par::rayon(0),
        );
        Ok(output)
    }

    /// Approximately reconstruct flat row-major projected vectors in the input space.
    pub fn inverse_transform(&self, vectors: &[f32]) -> Result<AVec<f32>, ReductionError> {
        let rows = validate_shape(vectors, self.output_dim)?;
        let mut output = avec!(0.0_f32; rows * self.input_dim);
        matmul(
            MatMut::from_row_major_slice_mut(&mut output, rows, self.input_dim),
            Accum::Replace,
            MatRef::from_row_major_slice(vectors, rows, self.output_dim),
            MatRef::from_row_major_slice(&self.components, self.output_dim, self.input_dim),
            1.0,
            Par::rayon(0),
        );
        output.par_chunks_mut(self.input_dim).for_each(|row| {
            for (value, &mean) in row.iter_mut().zip(&self.mean) {
                *value += mean;
            }
        });
        Ok(output)
    }
}

/// Subsampled randomized Hadamard transform for dense vectors.
///
/// The transform applies deterministic random signs, pads to the next power of two, performs an
/// unnormalized fast Walsh-Hadamard transform, and retains uniformly sampled coordinates scaled
/// by `1 / sqrt(output_dim)`. This preserves squared distances in expectation without fitting.
#[derive(Debug, Clone)]
pub struct Srht {
    input_dim: usize,
    output_dim: usize,
    padded_dim: usize,
    signs: Vec<f32>,
    indices: Vec<usize>,
}

impl Srht {
    /// Construct a deterministic SRHT from dimensions and a random seed.
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Result<Self, ReductionError> {
        validate_output_dimension(input_dim, output_dim)?;
        let padded_dim = input_dim.next_power_of_two();
        let mut rng = StdRng::seed_from_u64(seed);
        let signs = (0..input_dim)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let mut indices = sample_indices(padded_dim, output_dim, &mut rng);
        indices.sort_unstable();
        Ok(Self {
            input_dim,
            output_dim,
            padded_dim,
            signs,
            indices,
        })
    }

    /// Input vector dimension before zero-padding.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Number of sampled Hadamard coordinates.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Padded power-of-two dimension used by the Hadamard transform.
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Selected Hadamard-coordinate indices in ascending order.
    pub fn sampled_indices(&self) -> &[usize] {
        &self.indices
    }

    /// Project flat row-major vectors, allocating an aligned output buffer.
    pub fn transform(&self, vectors: &[f32]) -> Result<AVec<f32>, ReductionError> {
        let rows = validate_shape(vectors, self.input_dim)?;
        let mut output = avec!(0.0_f32; rows * self.output_dim);
        let scale = 1.0 / (self.output_dim as f32).sqrt();
        output
            .par_chunks_mut(self.output_dim)
            .zip(vectors.par_chunks(self.input_dim))
            .for_each_init(
                || vec![0.0_f32; self.padded_dim],
                |scratch, (output, input)| {
                    scratch.fill(0.0);
                    for ((value, &input), &sign) in scratch.iter_mut().zip(input).zip(&self.signs) {
                        *value = input * sign;
                    }
                    hadamard_in_place(scratch);
                    for (value, &index) in output.iter_mut().zip(&self.indices) {
                        *value = scratch[index] * scale;
                    }
                },
            );
        Ok(output)
    }

    /// Reconstruct the minimum-norm input-space approximation of projected vectors.
    pub fn inverse_transform(&self, vectors: &[f32]) -> Result<AVec<f32>, ReductionError> {
        let rows = validate_shape(vectors, self.output_dim)?;
        let mut output = avec!(0.0_f32; rows * self.input_dim);
        let scale = (self.output_dim as f32).sqrt() / self.padded_dim as f32;
        output
            .par_chunks_mut(self.input_dim)
            .zip(vectors.par_chunks(self.output_dim))
            .for_each_init(
                || vec![0.0_f32; self.padded_dim],
                |scratch, (output, input)| {
                    scratch.fill(0.0);
                    for (&value, &index) in input.iter().zip(&self.indices) {
                        scratch[index] = value;
                    }
                    hadamard_in_place(scratch);
                    for ((output, &value), &sign) in
                        output.iter_mut().zip(&*scratch).zip(&self.signs)
                    {
                        *output = value * sign * scale;
                    }
                },
            );
        Ok(output)
    }
}

fn hadamard_in_place(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    let mut width = 1;
    while width < values.len() {
        for block in values.chunks_exact_mut(width * 2) {
            let (left, right) = block.split_at_mut(width);
            for (left, right) in left.iter_mut().zip(right) {
                let a = *left;
                let b = *right;
                *left = a + b;
                *right = a - b;
            }
        }
        width *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::{Pca, ReductionError, Srht};

    fn squared_norm(values: &[f32]) -> f32 {
        values.iter().map(|value| value * value).sum()
    }

    #[test]
    fn pca_orders_components_and_round_trips_full_rank_data() {
        let vectors = [
            -4.0, -1.0, //
            -2.0, 1.0, //
            2.0, -1.0, //
            4.0, 1.0,
        ];
        let pca = Pca::fit(&vectors, 2, 2).unwrap();
        assert!(pca.explained_variance()[0] > pca.explained_variance()[1]);
        assert!((pca.preserved_variance() - 1.0).abs() < 1e-6);
        let projected = pca.transform(&vectors).unwrap();
        let reconstructed = pca.inverse_transform(&projected).unwrap();
        for (&actual, &expected) in reconstructed.iter().zip(&vectors) {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }
    }

    #[test]
    fn srht_full_padded_transform_preserves_norm_and_round_trips() {
        let vectors = [1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0, 8.0];
        let srht = Srht::new(4, 4, 42).unwrap();
        let projected = srht.transform(&vectors).unwrap();
        for (input, output) in vectors
            .as_chunks::<4>()
            .0
            .iter()
            .zip(projected.as_chunks::<4>().0)
        {
            assert!((squared_norm(input) - squared_norm(output)).abs() < 1e-4);
        }
        let reconstructed = srht.inverse_transform(&projected).unwrap();
        for (&actual, &expected) in reconstructed.iter().zip(&vectors) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        assert_eq!(srht.transform(&vectors).unwrap(), projected);
    }

    #[test]
    fn transforms_validate_shapes_and_dimensions() {
        assert!(matches!(
            Srht::new(3, 4, 1),
            Err(ReductionError::InvalidOutputDimension {
                input_dim: 3,
                output_dim: 4,
            })
        ));
        assert!(matches!(
            Pca::fit(&[1.0, 2.0], 2, 1),
            Err(ReductionError::TooFewRows { rows: 1 })
        ));
        let srht = Srht::new(3, 2, 1).unwrap();
        assert!(matches!(
            srht.transform(&[1.0, f32::NAN, 3.0]),
            Err(ReductionError::NonFiniteInput)
        ));
    }

    #[test]
    fn reduction_error_messages() {
        for (error, message) in [
            (
                ReductionError::InvalidInputShape { len: 5, dim: 2 },
                "input length 5 is not a non-empty multiple of dimension 2",
            ),
            (
                ReductionError::InvalidOutputDimension {
                    input_dim: 3,
                    output_dim: 4,
                },
                "output dimension 4 must be between 1 and input dimension 3",
            ),
            (
                ReductionError::TooFewRows { rows: 1 },
                "PCA requires at least two rows, got 1",
            ),
            (
                ReductionError::NonFiniteInput,
                "input contains a non-finite coordinate",
            ),
            (
                ReductionError::DecompositionFailed,
                "PCA covariance eigendecomposition failed",
            ),
        ] {
            assert_eq!(error.to_string(), message);
        }
    }
}
