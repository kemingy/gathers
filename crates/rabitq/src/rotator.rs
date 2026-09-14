//! Portable random rotation based on four rounds of FFHT and Kac's walk.

use pulp::{Arch, Simd, WithSimd};
use rand::{Rng, RngExt};

const NUM_ROUNDS: usize = 4;

/// A random orthogonal transform stored as four bits per padded dimension.
pub struct FhtKacRotator {
    input_dim: usize,
    padded_dim: usize,
    trunc_dim: usize,
    fht_scale: f32,
    flips: Vec<u64>,
    arch: Arch,
}

struct Rotate<'a> {
    rotator: &'a FhtKacRotator,
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for Rotate<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        self.rotator.rotate_simd(simd, self.input, self.output);
    }
}

impl FhtKacRotator {
    /// Sample four sign sequences for an input and padded dimension.
    pub fn new<R: Rng + ?Sized>(input_dim: usize, padded_dim: usize, rng: &mut R) -> Self {
        assert!(input_dim > 0);
        assert!(padded_dim >= input_dim);
        assert_eq!(padded_dim % 64, 0);

        let words_per_round = padded_dim / 64;
        let flips = (0..NUM_ROUNDS * words_per_round)
            .map(|_| rng.random::<u64>())
            .collect();
        let trunc_dim = 1usize << input_dim.ilog2();
        Self {
            input_dim,
            padded_dim,
            trunc_dim,
            fht_scale: (trunc_dim as f32).sqrt().recip(),
            flips,
            arch: Arch::new(),
        }
    }

    /// Rotate one input into the padded output using runtime-dispatched portable SIMD.
    pub fn rotate(&self, input: &[f32], output: &mut [f32]) {
        self.arch.dispatch(Rotate {
            rotator: self,
            input,
            output,
        });
    }

    fn rotate_simd<S: Simd>(&self, simd: S, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.input_dim);
        assert_eq!(output.len(), self.padded_dim);

        simd.vectorize(|| {
            output.fill(0.0);
            output[..self.input_dim].copy_from_slice(input);

            if self.trunc_dim == self.padded_dim {
                for round in 0..NUM_ROUNDS {
                    self.flip_sign(round, output);
                    fast_hadamard_transform_simd(simd, output);
                    rescale_simd(simd, output, self.fht_scale);
                }
                return;
            }

            let suffix_start = self.padded_dim - self.trunc_dim;
            for round in 0..NUM_ROUNDS {
                self.flip_sign(round, output);
                let block = if round.is_multiple_of(2) {
                    &mut output[..self.trunc_dim]
                } else {
                    &mut output[suffix_start..]
                };
                fast_hadamard_transform_simd(simd, block);
                rescale_simd(simd, block, self.fht_scale);
                kacs_walk_simd(simd, output);
            }
            rescale_simd(simd, output, 0.25);
        });
    }

    /// Rotate one input with the scalar correctness reference.
    pub fn rotate_scalar(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.input_dim);
        assert_eq!(output.len(), self.padded_dim);

        output.fill(0.0);
        output[..self.input_dim].copy_from_slice(input);

        if self.trunc_dim == self.padded_dim {
            for round in 0..NUM_ROUNDS {
                self.flip_sign(round, output);
                fast_hadamard_transform_scalar(output);
                rescale_scalar(output, self.fht_scale);
            }
            return;
        }

        let suffix_start = self.padded_dim - self.trunc_dim;
        for round in 0..NUM_ROUNDS {
            self.flip_sign(round, output);
            let block = if round.is_multiple_of(2) {
                &mut output[..self.trunc_dim]
            } else {
                &mut output[suffix_start..]
            };
            fast_hadamard_transform_scalar(block);
            rescale_scalar(block, self.fht_scale);
            kacs_walk_scalar(output);
        }
        rescale_scalar(output, 0.25);
    }

    fn flip_sign(&self, round: usize, data: &mut [f32]) {
        let words_per_round = self.padded_dim / 64;
        let flips = &self.flips[round * words_per_round..(round + 1) * words_per_round];
        for (values, &flip) in data.as_chunks_mut::<64>().0.iter_mut().zip(flips) {
            for (bit, value) in values.iter_mut().enumerate() {
                let sign = ((flip >> bit) as u32 & 1) << 31;
                *value = f32::from_bits(value.to_bits() ^ sign);
            }
        }
    }
}

fn fast_hadamard_transform_scalar(data: &mut [f32]) {
    assert!(data.len().is_power_of_two());
    let mut half = 1;
    while half < data.len() {
        for block in data.chunks_exact_mut(half * 2) {
            let (left, right) = block.split_at_mut(half);
            for (left, right) in left.iter_mut().zip(right) {
                let x = *left;
                let y = *right;
                *left = x + y;
                *right = x - y;
            }
        }
        half *= 2;
    }
}

fn kacs_walk_scalar(data: &mut [f32]) {
    assert!(data.len().is_multiple_of(2));
    let (left, right) = data.split_at_mut(data.len() / 2);
    for (left, right) in left.iter_mut().zip(right) {
        let x = *left;
        let y = *right;
        *left = x + y;
        *right = x - y;
    }
}

fn rescale_scalar(data: &mut [f32], scale: f32) {
    data.iter_mut().for_each(|value| *value *= scale);
}

fn fast_hadamard_transform_simd<S: Simd>(simd: S, data: &mut [f32]) {
    assert!(data.len().is_power_of_two());
    let mut half = 1;
    while half < data.len() {
        for block in data.chunks_exact_mut(half * 2) {
            let (left, right) = block.split_at_mut(half);
            if half < S::F32_LANES {
                butterfly_scalar(left, right);
                continue;
            }

            let (left, left_tail) = S::as_mut_simd_f32s(left);
            let (right, right_tail) = S::as_mut_simd_f32s(right);
            for (left, right) in left.iter_mut().zip(right) {
                let x = *left;
                let y = *right;
                *left = simd.add_f32s(x, y);
                *right = simd.sub_f32s(x, y);
            }
            butterfly_scalar(left_tail, right_tail);
        }
        half *= 2;
    }
}

fn kacs_walk_simd<S: Simd>(simd: S, data: &mut [f32]) {
    assert!(data.len().is_multiple_of(2));
    let (left, right) = data.split_at_mut(data.len() / 2);
    let (left, left_tail) = S::as_mut_simd_f32s(left);
    let (right, right_tail) = S::as_mut_simd_f32s(right);
    for (left, right) in left.iter_mut().zip(right) {
        let x = *left;
        let y = *right;
        *left = simd.add_f32s(x, y);
        *right = simd.sub_f32s(x, y);
    }
    butterfly_scalar(left_tail, right_tail);
}

fn rescale_simd<S: Simd>(simd: S, data: &mut [f32], scale: f32) {
    let scale_vector = simd.splat_f32s(scale);
    let (data, tail) = S::as_mut_simd_f32s(data);
    for value in data {
        *value = simd.mul_f32s(*value, scale_vector);
    }
    tail.iter_mut().for_each(|value| *value *= scale);
}

fn butterfly_scalar(left: &mut [f32], right: &mut [f32]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter_mut().zip(right) {
        let x = *left;
        let y = *right;
        *left = x + y;
        *right = x - y;
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use seed_rand::seeded_rng;

    use super::FhtKacRotator;

    fn squared_norm(values: &[f32]) -> f32 {
        values.iter().map(|value| value * value).sum()
    }

    fn dot_product(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum()
    }

    #[test]
    fn rotation_preserves_norm_for_power_of_two_and_arbitrary_dimensions() {
        let mut rng = seeded_rng();
        for dim in [1usize, 63, 64, 65, 127, 128, 960, 1024, 1088] {
            let padded_dim = dim.div_ceil(64) * 64;
            let rotator = FhtKacRotator::new(dim, padded_dim, &mut rng);
            let input = (0..dim)
                .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                .collect::<Vec<_>>();
            let other = (0..dim)
                .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                .collect::<Vec<_>>();
            let mut output = vec![0.0; padded_dim];
            let mut other_output = vec![0.0; padded_dim];
            rotator.rotate(&input, &mut output);
            rotator.rotate(&other, &mut other_output);

            let expected = squared_norm(&input);
            let actual = squared_norm(&output);
            let tolerance = 1e-4 * expected.max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "dimension {dim}: expected {expected}, got {actual}",
            );

            let expected = dot_product(&input, &other);
            let actual = dot_product(&output, &other_output);
            let tolerance = 1e-4 * squared_norm(&input).sqrt() * squared_norm(&other).sqrt();
            assert!(
                (actual - expected).abs() <= tolerance.max(1e-5),
                "dimension {dim}: expected inner product {expected}, got {actual}",
            );
        }
    }

    #[test]
    fn repeated_rotation_is_deterministic() {
        let mut rng = seeded_rng();
        let rotator = FhtKacRotator::new(960, 960, &mut rng);
        let input = (0..960).map(|_| rng.random::<f32>()).collect::<Vec<_>>();
        let mut first = vec![0.0; 960];
        let mut second = vec![0.0; 960];
        rotator.rotate(&input, &mut first);
        rotator.rotate(&input, &mut second);
        assert_eq!(first, second);
    }

    #[test]
    fn portable_simd_matches_scalar_rotation() {
        let mut rng = seeded_rng();
        for dim in [1usize, 63, 64, 65, 127, 128, 960, 1024, 1088] {
            let padded_dim = dim.div_ceil(64) * 64;
            let rotator = FhtKacRotator::new(dim, padded_dim, &mut rng);
            let input = (0..dim)
                .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                .collect::<Vec<_>>();
            let mut scalar = vec![0.0; padded_dim];
            let mut portable_simd = vec![0.0; padded_dim];
            rotator.rotate_scalar(&input, &mut scalar);
            rotator.rotate(&input, &mut portable_simd);
            assert_eq!(scalar, portable_simd, "dimension {dim}");
        }
    }
}
