//! Principal component analysis for dense vectors.

use aligned_vec::{AVec, avec};
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par, Side};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::reduction::{ReductionError, validate_output_dimension, validate_shape};

/// Principal component projection learned from dense `f32` vectors.
///
/// Training centers the data using `f64` means, forms an `f32` covariance matrix, and retains the
/// eigenvectors with the largest eigenvalues. Coordinates are not standardized or whitened, so
/// the transform preserves the covariance-PCA geometry used by squared-Euclidean clustering.
#[derive(Debug, Clone)]
pub struct PCA {
    input_dim: usize,
    output_dim: usize,
    mean: Vec<f32>,
    // Principal directions in descending variance order, row-major output_dim x input_dim.
    components: Vec<f32>,
    explained_variance: Vec<f32>,
    total_variance: f64,
}

impl PCA {
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
    ///
    /// Centering or discarding components can map a nonzero input row to zero.
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

#[cfg(test)]
mod tests {
    use super::PCA;

    #[test]
    fn pca_orders_components_and_round_trips_full_rank_data() {
        let vectors = [
            -4.0, -1.0, //
            -2.0, 1.0, //
            2.0, -1.0, //
            4.0, 1.0,
        ];
        let pca = PCA::fit(&vectors, 2, 2).unwrap();
        assert!(pca.explained_variance()[0] > pca.explained_variance()[1]);
        assert!((pca.preserved_variance() - 1.0).abs() < 1e-6);
        let projected = pca.transform(&vectors).unwrap();
        let reconstructed = pca.inverse_transform(&projected).unwrap();
        for (&actual, &expected) in reconstructed.iter().zip(&vectors) {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }
    }
}
