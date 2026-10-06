//! Dimensionality reduction for dense row-major vectors.
//!
//! [`PCA`] learns a data-dependent projection that maximizes preserved variance. [`SRHT`] is a
//! training-free subsampled randomized Hadamard transform. Both expose approximate inverse
//! transforms so reduced-space centroids can be materialized in the original vector space.
//! Import [`Reduction`] to use the shared transform methods.

use aligned_vec::AVec;

mod pca;
mod srht;

pub use pca::PCA;
pub use srht::SRHT;

/// A dimensionality-reduction transform for flat row-major `f32` vectors.
///
/// Construction remains specific to each method: [`PCA::fit`] learns a projection, while
/// [`SRHT::new`] constructs one from dimensions and a seed.
/// The built-in transforms return [`ReductionError::NumericalOverflow`] when arithmetic
/// produces non-finite intermediates or outputs, even if their inputs were finite.
pub trait Reduction {
    /// Number of coordinates in each original input row.
    fn input_dim(&self) -> usize;

    /// Number of coordinates in each projected row.
    fn output_dim(&self) -> usize;

    /// Project nonempty, complete, finite input rows into an aligned output buffer.
    ///
    /// Input rows have [`Self::input_dim`] coordinates; output rows have [`Self::output_dim`]
    /// coordinates. A nonzero input can project to zero.
    fn transform(&self, vectors: &[f32]) -> Result<AVec<f32>, ReductionError>;

    /// Approximately reconstruct nonempty, complete, finite projected rows.
    ///
    /// Output rows have [`Self::input_dim`] coordinates. Discarded information cannot generally
    /// be recovered; this operation is not necessarily an exact inverse or pseudoinverse.
    fn inverse_transform(&self, vectors: &[f32]) -> Result<AVec<f32>, ReductionError>;
}

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
    /// The requested PCA fitting sample is larger than the supplied data.
    #[error("PCA training sample {requested} exceeds available rows {available}")]
    InvalidSampleSize {
        /// Requested fitting rows.
        requested: usize,
        /// Available input rows.
        available: usize,
    },
    /// The input contains a NaN or infinity.
    #[error("input contains a non-finite coordinate")]
    NonFiniteInput,
    /// Reduction arithmetic produced a non-finite intermediate or output from finite inputs.
    #[error("reduction arithmetic overflowed; rescale the input vectors")]
    NumericalOverflow,
    /// A requested buffer exceeds the addressable allocation size.
    #[error("reduction buffer size exceeds the addressable allocation size")]
    SizeOverflow,
    /// The covariance eigendecomposition did not converge.
    #[error("PCA covariance eigendecomposition failed")]
    DecompositionFailed,
}

pub(crate) fn validate_shape(values: &[f32], dim: usize) -> Result<usize, ReductionError> {
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

pub(crate) fn check_finite_result(values: &[f32]) -> Result<(), ReductionError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ReductionError::NumericalOverflow)
    }
}

pub(crate) fn checked_buffer_len(rows: usize, dim: usize) -> Result<usize, ReductionError> {
    // Include alignment rounding for both the explicit 64-byte and default AVec layouts.
    let alignment = aligned_vec::CACHELINE_ALIGN.max(64);
    rows.checked_mul(dim)
        .filter(|&len| len <= ((isize::MAX as usize) - (alignment - 1)) / size_of::<f32>())
        .ok_or(ReductionError::SizeOverflow)
}

pub(crate) fn validate_output_dimension(
    input_dim: usize,
    output_dim: usize,
) -> Result<(), ReductionError> {
    if output_dim == 0 || output_dim > input_dim {
        return Err(ReductionError::InvalidOutputDimension {
            input_dim,
            output_dim,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PCA, Reduction, ReductionError, SRHT, checked_buffer_len};

    #[test]
    fn buffer_sizes_reject_integer_and_byte_capacity_overflow() {
        let alignment = aligned_vec::CACHELINE_ALIGN.max(64);
        let limit = ((isize::MAX as usize) - (alignment - 1)) / size_of::<f32>();
        assert_eq!(checked_buffer_len(1, limit), Ok(limit));
        assert_eq!(
            checked_buffer_len(1, limit + 1),
            Err(ReductionError::SizeOverflow)
        );
        assert_eq!(
            checked_buffer_len(usize::MAX, 2),
            Err(ReductionError::SizeOverflow)
        );
        for dim in [usize::MAX, 1usize << (usize::BITS - 1)] {
            assert!(matches!(
                SRHT::new(dim, 1, 42),
                Err(ReductionError::SizeOverflow)
            ));
        }
    }

    #[test]
    fn transforms_validate_shapes_and_dimensions() {
        assert!(matches!(
            SRHT::new(3, 4, 1),
            Err(ReductionError::InvalidOutputDimension {
                input_dim: 3,
                output_dim: 4,
            })
        ));
        assert!(matches!(
            PCA::fit(&[1.0, 2.0], 2, 1),
            Err(ReductionError::TooFewRows { rows: 1 })
        ));
        let srht = SRHT::new(3, 2, 1).unwrap();
        let pca = PCA::fit(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
        for model in [&srht as &dyn Reduction, &pca] {
            assert_eq!(model.input_dim(), 3);
            assert_eq!(model.output_dim(), 2);
            assert!(matches!(
                model.transform(&[1.0, f32::NAN, 3.0]),
                Err(ReductionError::NonFiniteInput)
            ));
            let projected = model.transform(&[1.0, 2.0, 3.0]).unwrap();
            assert_eq!(projected.len(), 2);
            assert_eq!(model.inverse_transform(&projected).unwrap().len(), 3);
        }
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
                ReductionError::InvalidSampleSize {
                    requested: 41,
                    available: 40,
                },
                "PCA training sample 41 exceeds available rows 40",
            ),
            (
                ReductionError::NonFiniteInput,
                "input contains a non-finite coordinate",
            ),
            (
                ReductionError::NumericalOverflow,
                "reduction arithmetic overflowed; rescale the input vectors",
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
