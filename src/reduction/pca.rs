//! Principal component analysis for dense vectors.

use aligned_vec::{AVec, avec};
use faer::{Accum, MatMut, MatRef, Par, Side};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::gemm::{matmul, symmetric_rank_k_lower};
use crate::reduction::{
    Reduction, ReductionError, check_finite_result, checked_buffer_len, validate_output_dimension,
    validate_shape,
};
use crate::sampling::sample_indices;

/// Principal component projection learned from dense `f32` vectors.
///
/// Training centers the data using `f64` means, forms an `f32` covariance matrix, and retains the
/// eigenvectors with the largest eigenvalues. Coordinates are not standardized or whitened, so
/// the transform preserves the covariance-PCA geometry used by squared-Euclidean clustering.
/// Finite inputs can still overflow `f32` arithmetic; fitting and transforms return
/// [`ReductionError::NumericalOverflow`] rather than accepting non-finite intermediates or outputs.
#[derive(Debug, Clone)]
pub struct PCA {
    input_dim: usize,
    output_dim: usize,
    // The f64 mean used for centering; rounding the mean to f32 before subtraction can
    // exceed the variance of large-offset data and bias the covariance.
    mean: Vec<f64>,
    // Principal directions in descending variance order, row-major output_dim x input_dim.
    components: Vec<f32>,
    preserved_variance: f64,
}

// Bound centered scratch independently of corpus size. One full row is the minimum.
const CENTERED_SCRATCH_BYTES: usize = 16 << 20;

fn block_rows(dim: usize) -> usize {
    (CENTERED_SCRATCH_BYTES / size_of::<f32>() / dim).max(1)
}

fn center_rows(
    input: &[f32],
    mean: &[f64],
    dim: usize,
    output: &mut [f32],
) -> Result<(), ReductionError> {
    output
        .par_chunks_mut(dim)
        .zip(input.par_chunks(dim))
        .try_for_each(|(output, input)| {
            for ((output, &input), &mean) in output.iter_mut().zip(input).zip(mean) {
                *output = (f64::from(input) - mean) as f32;
            }
            check_finite_result(output)
        })
}

impl PCA {
    /// Fit PCA on a uniformly selected, bounded sample of the supplied rows.
    ///
    /// Requires at least two fitting rows and no more than the available rows. The seed controls
    /// selection; selected rows retain their source order. No copy is made when all rows are used.
    pub fn fit_sample(
        vectors: &[f32],
        input_dim: usize,
        output_dim: usize,
        training_rows: usize,
        seed: u64,
    ) -> Result<Self, ReductionError> {
        validate_output_dimension(input_dim, output_dim)?;
        let rows = validate_shape(vectors, input_dim)?;
        if training_rows < 2 {
            return Err(ReductionError::TooFewRows {
                rows: training_rows,
            });
        }
        if training_rows > rows {
            return Err(ReductionError::InvalidSampleSize {
                requested: training_rows,
                available: rows,
            });
        }
        if training_rows == rows {
            return Self::fit(vectors, input_dim, output_dim);
        }
        let mut indices = sample_indices(
            rows,
            training_rows,
            &mut StdRng::seed_from_u64(seed ^ 0x5043_415f_5341_4d50),
        );
        indices.sort_unstable();
        let mut sample = Vec::with_capacity(training_rows * input_dim);
        for index in indices {
            sample.extend_from_slice(&vectors[index * input_dim..(index + 1) * input_dim]);
        }
        Self::fit(&sample, input_dim, output_dim)
    }

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
        let covariance_len = checked_buffer_len(input_dim, input_dim)?;
        if covariance_len > isize::MAX as usize / size_of::<f64>() {
            return Err(ReductionError::SizeOverflow);
        }

        let mut mean = vec![0.0_f64; input_dim];
        for row in vectors.chunks_exact(input_dim) {
            for (mean, &value) in mean.iter_mut().zip(row) {
                *mean += f64::from(value);
            }
        }
        for value in &mut mean {
            *value /= rows as f64;
        }

        let block_rows = block_rows(input_dim).min(rows);
        let mut centered = avec!(0.0_f32; checked_buffer_len(block_rows, input_dim)?);
        let mut covariance = vec![0.0_f32; covariance_len];
        // Sum block covariances in f64: f32 accumulation across thousands of blocks
        // would otherwise introduce an additional error proportional to corpus size.
        let mut sums = vec![0.0_f64; covariance_len];
        let scale = 1.0 / (rows - 1) as f32;
        for block in vectors.chunks(centered.len()) {
            let centered = &mut centered[..block.len()];
            center_rows(block, &mean, input_dim, centered)?;
            let matrix = MatRef::from_row_major_slice(centered, block.len() / input_dim, input_dim);
            symmetric_rank_k_lower(
                MatMut::from_row_major_slice_mut(&mut covariance, input_dim, input_dim),
                matrix,
                scale,
                Par::rayon(0),
            );
            // The eigensolver reads Side::Lower. Avoid mirroring, validating, or accumulating
            // the unused upper triangle on every block; retain f64 sums for computed entries.
            for (row, (sums, values)) in sums
                .chunks_mut(input_dim)
                .zip(covariance.chunks(input_dim))
                .enumerate()
            {
                let values = &values[..=row];
                check_finite_result(values)?;
                for (sum, &value) in sums[..=row].iter_mut().zip(values) {
                    *sum += f64::from(value);
                }
            }
        }
        for (value, sum) in covariance.iter_mut().zip(sums) {
            *value = sum as f32;
        }
        check_finite_result(&covariance)?;
        let eigen = MatRef::from_row_major_slice(&covariance, input_dim, input_dim)
            .self_adjoint_eigen(Side::Lower)
            .map_err(|_| ReductionError::DecompositionFailed)?;
        let eigenvalues = eigen.S().column_vector();
        let eigenvectors = eigen.U();
        // Check before max(0.0), which would mask a NaN eigenvalue as zero variance.
        if !(0..input_dim).all(|index| eigenvalues.get(index).is_finite()) {
            return Err(ReductionError::DecompositionFailed);
        }
        let total_variance = (0..input_dim)
            .map(|index| f64::from((*eigenvalues.get(index)).max(0.0)))
            .sum::<f64>();
        let preserved_variance = if total_variance == 0.0 {
            1.0
        } else {
            (input_dim - output_dim..input_dim)
                .rev()
                .map(|index| f64::from((*eigenvalues.get(index)).max(0.0)))
                .sum::<f64>()
                / total_variance
        };
        let mut components = Vec::with_capacity(output_dim * input_dim);
        for component in 0..output_dim {
            let source = input_dim - 1 - component;
            for coordinate in 0..input_dim {
                let value = *eigenvectors.get(coordinate, source);
                if !value.is_finite() {
                    return Err(ReductionError::DecompositionFailed);
                }
                components.push(value);
            }
        }
        Ok(Self {
            input_dim,
            output_dim,
            mean,
            components,
            preserved_variance,
        })
    }

    /// Fraction of total training variance captured by the retained components.
    pub fn preserved_variance(&self) -> f64 {
        self.preserved_variance
    }
}

impl Reduction for PCA {
    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Project flat row-major vectors, allocating an aligned output buffer.
    ///
    /// Centering or discarding components can map a nonzero input row to zero.
    fn transform(&self, vectors: &[f32]) -> Result<AVec<f32>, ReductionError> {
        let rows = validate_shape(vectors, self.input_dim)?;
        let block_rows = block_rows(self.input_dim).min(rows);
        let mut centered = avec!(0.0_f32; checked_buffer_len(block_rows, self.input_dim)?);
        let mut output = avec!(0.0_f32; checked_buffer_len(rows, self.output_dim)?);
        for (input, output) in vectors
            .chunks(centered.len())
            .zip(output.chunks_mut(block_rows * self.output_dim))
        {
            let rows = input.len() / self.input_dim;
            let centered = &mut centered[..input.len()];
            center_rows(input, &self.mean, self.input_dim, centered)?;
            matmul(
                MatMut::from_row_major_slice_mut(output, rows, self.output_dim),
                Accum::Replace,
                MatRef::from_row_major_slice(centered, rows, self.input_dim),
                MatRef::from_row_major_slice(&self.components, self.output_dim, self.input_dim)
                    .transpose(),
                1.0,
                Par::rayon(0),
            );
        }
        check_finite_result(&output)?;
        Ok(output)
    }

    /// Approximately reconstruct flat row-major projected vectors in the input space.
    fn inverse_transform(&self, vectors: &[f32]) -> Result<AVec<f32>, ReductionError> {
        let rows = validate_shape(vectors, self.output_dim)?;
        let mut output = avec!(0.0_f32; checked_buffer_len(rows, self.input_dim)?);
        matmul(
            MatMut::from_row_major_slice_mut(&mut output, rows, self.input_dim),
            Accum::Replace,
            MatRef::from_row_major_slice(vectors, rows, self.output_dim),
            MatRef::from_row_major_slice(&self.components, self.output_dim, self.input_dim),
            1.0,
            Par::rayon(0),
        );
        output.par_chunks_mut(self.input_dim).try_for_each(|row| {
            for (value, &mean) in row.iter_mut().zip(&self.mean) {
                *value = (f64::from(*value) + mean) as f32;
            }
            check_finite_result(row)
        })?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::PCA;
    use crate::reduction::{Reduction, ReductionError};

    #[test]
    fn pca_block_tails_preserve_covariance_and_projection() {
        use super::block_rows;
        // One more than the actual scratch capacity exercises the fitting and transform tails.
        let dim = 3;
        let rows = block_rows(dim) + 1;
        let mut vectors = Vec::with_capacity(rows * dim);
        for row in 0..rows - 1 {
            vectors.extend_from_slice(&[if row % 2 == 0 { 1.0 } else { -1.0 }, 0.0, 0.0]);
        }
        vectors.extend_from_slice(&[0.0, 0.0, 10_000.0]);
        let pca = PCA::fit(&vectors, dim, 1).unwrap();
        // The last row, in a separate block, supplies nearly all of the leading variance.
        let z_variance = 1e8 / rows as f64;
        let x_variance = 1.0 - 1.0 / (rows as f64 * (rows - 1) as f64);
        let expected = z_variance / (z_variance + x_variance);
        assert!((pca.preserved_variance() - expected).abs() < 1e-5);
        let projected = pca.transform(&vectors).unwrap();
        let tail = pca.transform(&vectors[(rows - 1) * dim..]).unwrap();
        assert!((projected[rows - 1] - tail[0]).abs() < 1e-3);
        assert!((tail[0].abs() - 10_000.0 * (1.0 - 1.0 / rows as f32)).abs() < 0.01);
        vectors[(rows - 1) * dim + 2] = 1e25;
        assert_eq!(pca.transform(&vectors).unwrap().len(), rows);
        assert!(matches!(
            PCA::fit(&vectors, dim, 1),
            Err(ReductionError::NumericalOverflow)
        ));
    }

    #[test]
    fn pca_rejects_centering_and_covariance_overflow() {
        // The first residual overflows f32 even though the mean is computed in f64.
        // The second fixture has finite residuals but their squares overflow the covariance.
        for vectors in [&[f32::MAX, -f32::MAX, -f32::MAX][..], &[1e20, -1e20][..]] {
            assert!(matches!(
                PCA::fit(vectors, 1, 1),
                Err(ReductionError::NumericalOverflow)
            ));
        }
    }

    #[test]
    fn pca_rejects_transform_overflow() {
        let constant = PCA::fit(&[-f32::MAX; 2], 1, 1).unwrap();
        assert_eq!(&*constant.transform(&[-f32::MAX]).unwrap(), &[0.0]);
        assert_eq!(
            constant.transform(&[f32::MAX]),
            Err(ReductionError::NumericalOverflow)
        );
        let diagonal = PCA::fit(&[-1.0, -1.0, 1.0, 1.0], 2, 1).unwrap();
        // Centering remains finite; the retained coordinate exceeds f32::MAX.
        assert_eq!(
            diagonal.transform(&[f32::MAX; 2]),
            Err(ReductionError::NumericalOverflow)
        );
    }

    #[test]
    fn pca_rejects_inverse_transform_overflow() {
        let constant = PCA::fit(&[f32::MAX; 2], 1, 1).unwrap();
        assert_eq!(&*constant.inverse_transform(&[0.0]).unwrap(), &[f32::MAX]);
        assert_eq!(
            constant.inverse_transform(&[f32::MAX]),
            Err(ReductionError::NumericalOverflow)
        );
        let diagonal = PCA::fit(&[-1.0, -1.0, 1.0, 1.0], 2, 2).unwrap();
        // One original coordinate overflows in matrix multiplication, before restoring the mean.
        assert_eq!(
            diagonal.inverse_transform(&[f32::MAX; 2]),
            Err(ReductionError::NumericalOverflow)
        );
    }

    #[test]
    fn pca_fitting_sample_is_bounded_reproducible_and_validated() {
        let vectors = (0..40)
            .flat_map(|row| [row as f32, (row % 3) as f32])
            .collect::<Vec<_>>();
        let sampled = PCA::fit_sample(&vectors, 2, 1, 12, 42).unwrap();
        let repeated = PCA::fit_sample(&vectors, 2, 1, 12, 42).unwrap();
        assert_eq!(
            sampled.transform(&vectors).unwrap(),
            repeated.transform(&vectors).unwrap()
        );
        let all = PCA::fit_sample(&vectors, 2, 1, 40, 42).unwrap();
        assert_eq!(
            all.transform(&vectors).unwrap(),
            PCA::fit(&vectors, 2, 1)
                .unwrap()
                .transform(&vectors)
                .unwrap()
        );
        assert!(matches!(
            PCA::fit_sample(&vectors, 2, 1, 1, 42),
            Err(crate::reduction::ReductionError::TooFewRows { rows: 1 })
        ));
        assert!(matches!(
            PCA::fit_sample(&vectors, 2, 1, 41, 42),
            Err(crate::reduction::ReductionError::InvalidSampleSize {
                requested: 41,
                available: 40
            })
        ));
    }

    #[test]
    fn pca_orders_components_and_round_trips_full_rank_data() {
        let vectors = [
            -4.0, -1.0, //
            -2.0, 1.0, //
            2.0, -1.0, //
            4.0, 1.0,
        ];
        let pca = PCA::fit(&vectors, 2, 2).unwrap();
        assert!((pca.preserved_variance() - 1.0).abs() < 1e-6);
        let projected = pca.transform(&vectors).unwrap();
        let variance = |coordinate: usize| {
            projected
                .as_chunks::<2>()
                .0
                .iter()
                .map(|row| f64::from(row[coordinate]).powi(2))
                .sum::<f64>()
        };
        assert!(variance(0) > variance(1));
        let reconstructed = pca.inverse_transform(&projected).unwrap();
        for (&actual, &expected) in reconstructed.iter().zip(&vectors) {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }
    }

    #[test]
    fn pca_centers_large_offsets_with_the_f64_mean() {
        // f32 rounds the 100_000_004.0 mean to 100_000_000.0; f32 centering would produce
        // residuals [0, 8] and double the variance to 64.
        let vectors = [100_000_000.0, 100_000_008.0];
        let pca = PCA::fit(&vectors, 1, 1).unwrap();
        let projected = pca.transform(&vectors).unwrap();
        let variance = projected.iter().map(|value| value * value).sum::<f32>();
        assert!((variance - 32.0).abs() < 1e-3);
        assert!((projected[0].abs() - 4.0).abs() < 1e-4);
        assert!((projected[1].abs() - 4.0).abs() < 1e-4);
        let reconstructed = pca.inverse_transform(&projected).unwrap();
        assert_eq!(&*reconstructed, &vectors);

        // The retained fraction also checks covariance centering, independently of transform.
        // The centered coordinate sums of squares are 64 and 36, with zero cross-covariance.
        let vectors = [
            100_000_000.0,
            -3.0,
            100_000_008.0,
            -3.0,
            100_000_008.0,
            3.0,
            100_000_000.0,
            3.0,
        ];
        let pca = PCA::fit(&vectors, 2, 1).unwrap();
        assert!((pca.preserved_variance() - 0.64).abs() < 1e-6);
    }
}
