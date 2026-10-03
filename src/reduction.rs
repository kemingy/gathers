//! Linear dimensionality reduction for dense row-major vectors.
//!
//! [`PCA`] learns a data-dependent projection that maximizes preserved variance. [`SRHT`] is a
//! training-free subsampled randomized Hadamard transform. Both expose approximate inverse
//! transforms so reduced-space centroids can be materialized in the original vector space.

mod pca;
mod srht;

pub use pca::PCA;
pub use srht::SRHT;

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

#[cfg(test)]
mod tests {
    use super::{PCA, ReductionError, SRHT};

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
