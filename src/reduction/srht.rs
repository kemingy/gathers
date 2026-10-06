//! Subsampled randomized Hadamard transforms for dense vectors.

use aligned_vec::{AVec, avec};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::reduction::{
    Reduction, ReductionError, check_finite_result, validate_output_dimension, validate_shape,
};
use crate::sampling::sample_indices;

/// Subsampled randomized Hadamard transform for dense vectors.
///
/// The transform applies deterministic random signs, pads to the next power of two, performs an
/// unnormalized fast Walsh-Hadamard transform, and retains uniformly sampled coordinates scaled
/// by `1 / sqrt(output_dim)`. This preserves squared distances in expectation without fitting.
/// Transforms return [`ReductionError::NumericalOverflow`] if unscaled Hadamard intermediates
/// overflow `f32`, even when the final scaling would make the exact result representable.
#[derive(Debug, Clone)]
pub struct SRHT {
    input_dim: usize,
    output_dim: usize,
    padded_dim: usize,
    signs: Vec<f32>,
    indices: Vec<usize>,
}

impl SRHT {
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
}

impl Reduction for SRHT {
    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Project flat row-major vectors, allocating an aligned output buffer.
    ///
    /// A nonzero input row in the projection's null space maps to zero.
    fn transform(&self, vectors: &[f32]) -> Result<AVec<f32>, ReductionError> {
        let rows = validate_shape(vectors, self.input_dim)?;
        let mut output = avec!(0.0_f32; rows * self.output_dim);
        let scale = 1.0 / (self.output_dim as f32).sqrt();
        output
            .par_chunks_mut(self.output_dim)
            .zip(vectors.par_chunks(self.input_dim))
            .try_for_each_init(
                || vec![0.0_f32; self.padded_dim],
                |scratch, (output, input)| {
                    scratch.fill(0.0);
                    for ((value, &input), &sign) in scratch.iter_mut().zip(input).zip(&self.signs) {
                        *value = input * sign;
                    }
                    hadamard_in_place(scratch);
                    check_finite_result(scratch)?;
                    for (value, &index) in output.iter_mut().zip(&self.indices) {
                        *value = scratch[index] * scale;
                    }
                    // Scale <= 1, so finite Hadamard values cannot overflow here.
                    Ok(())
                },
            )?;
        Ok(output)
    }

    /// Apply the inverse Hadamard transform in padded space and discard the padding.
    ///
    /// Without padding, this is the minimum-norm inverse of the sampled projection. When the
    /// input dimension is not a power of two, truncation removes reconstructed coordinates and
    /// projecting the result again may differ from the supplied projected vectors.
    fn inverse_transform(&self, vectors: &[f32]) -> Result<AVec<f32>, ReductionError> {
        let rows = validate_shape(vectors, self.output_dim)?;
        let mut output = avec!(0.0_f32; rows * self.input_dim);
        let scale = (self.output_dim as f32).sqrt() / self.padded_dim as f32;
        output
            .par_chunks_mut(self.input_dim)
            .zip(vectors.par_chunks(self.output_dim))
            .try_for_each_init(
                || vec![0.0_f32; self.padded_dim],
                |scratch, (output, input)| {
                    scratch.fill(0.0);
                    for (&value, &index) in input.iter().zip(&self.indices) {
                        scratch[index] = value;
                    }
                    hadamard_in_place(scratch);
                    check_finite_result(scratch)?;
                    for ((output, &value), &sign) in
                        output.iter_mut().zip(&*scratch).zip(&self.signs)
                    {
                        *output = value * sign * scale;
                    }
                    // Signs are +/-1 and scale <= 1; no additional overflow check is needed.
                    Ok(())
                },
            )?;
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
    use super::SRHT;
    use crate::reduction::{Reduction, ReductionError};

    fn squared_norm(values: &[f32]) -> f32 {
        values.iter().map(|value| value * value).sum()
    }

    #[test]
    fn srht_rejects_forward_and_inverse_overflow_before_scaling() {
        let full = SRHT::new(2, 2, 42).unwrap();
        for value in [f32::MAX * 0.6, f32::MAX] {
            // With 0.6 * MAX the scaled result would fit, but the unscaled sum overflows.
            assert_eq!(
                full.transform(&[value; 2]),
                Err(ReductionError::NumericalOverflow)
            );
        }
        let sampled = SRHT::new(4, 2, 42).unwrap();
        // The exact inverse is bounded by MAX / sqrt(2), but intermediate sums overflow.
        assert_eq!(
            sampled.inverse_transform(&[f32::MAX; 2]),
            Err(ReductionError::NumericalOverflow)
        );
        // Large values are permitted when intermediate arithmetic still fits.
        let value = f32::MAX * 0.25;
        assert!(
            full.transform(&[value; 2])
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
        assert!(
            sampled
                .inverse_transform(&[value; 2])
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn srht_checks_overflow_in_unsampled_padded_coordinates() {
        for seed in 0..8 {
            let model = SRHT::new(3, 1, seed).unwrap();
            // After random signs, all three coordinates are positive. Their first-stage sums
            // fit, but the DC coordinate overflows in the second stage, sampled or not.
            let input = model
                .signs
                .iter()
                .map(|sign| sign * (f32::MAX * 0.4))
                .collect::<Vec<_>>();
            assert_eq!(
                model.transform(&input),
                Err(ReductionError::NumericalOverflow)
            );
        }
    }

    #[test]
    fn srht_full_padded_transform_preserves_norm_and_round_trips() {
        let vectors = [1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0, 8.0];
        let srht = SRHT::new(4, 4, 42).unwrap();
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
    fn padded_inverse_discards_reconstructed_padding() {
        let srht = SRHT::new(3, 1, 42).unwrap();
        let reconstructed = srht.inverse_transform(&[1.0]).unwrap();
        assert_eq!(reconstructed.len(), 3);
        // The inverse reconstructs four coordinates of magnitude 1/4, then drops the fourth.
        // A true original-space pseudoinverse would reproject to 1 rather than 3/4.
        assert_eq!(&*srht.transform(&reconstructed).unwrap(), &[0.75]);
    }
}
