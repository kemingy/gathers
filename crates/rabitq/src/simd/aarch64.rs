//! AArch64 kernels implemented with Pulp's Neon intrinsics.

use pulp::aarch64::Neon;

pub mod legacy;

/// Convert four-bit quantized values to bit-sliced binary vectors.
#[inline]
pub fn vector_binarize_query(vec: &[u8], binary: &mut [u64]) {
    assert_eq!(vec.len() % 64, 0);
    assert_eq!(binary.len(), vec.len() / 16);

    let simd = Neon::try_new().expect("Neon is part of the AArch64 baseline");
    simd.vectorize(|| {
        let weights = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let weights = unsafe { simd.neon.vld1q_u8(weights.as_ptr()) };

        macro_rules! bit_mask {
            ($values:expr, 0) => {{
                let bits = simd.neon.vandq_u8($values, simd.neon.vdupq_n_u8(1));
                let weighted = simd.neon.vmulq_u8(bits, weights);
                simd.neon.vaddlv_u8(simd.neon.vget_low_u8(weighted)) as u64
                    | (simd.neon.vaddlv_u8(simd.neon.vget_high_u8(weighted)) as u64) << 8
            }};
            ($values:expr, $shift:literal) => {{
                let bits = simd.neon.vandq_u8(
                    simd.neon.vshrq_n_u8::<$shift>($values),
                    simd.neon.vdupq_n_u8(1),
                );
                let weighted = simd.neon.vmulq_u8(bits, weights);
                simd.neon.vaddlv_u8(simd.neon.vget_low_u8(weighted)) as u64
                    | (simd.neon.vaddlv_u8(simd.neon.vget_high_u8(weighted)) as u64) << 8
            }};
        }

        let chunks = vec.len() / 64;
        for (chunk_index, chunk) in vec.chunks_exact(64).enumerate() {
            let values = [
                unsafe { simd.neon.vld1q_u8(chunk.as_ptr()) },
                unsafe { simd.neon.vld1q_u8(chunk.as_ptr().add(16)) },
                unsafe { simd.neon.vld1q_u8(chunk.as_ptr().add(32)) },
                unsafe { simd.neon.vld1q_u8(chunk.as_ptr().add(48)) },
            ];
            binary[chunk_index] |= bit_mask!(values[0], 0)
                | bit_mask!(values[1], 0) << 16
                | bit_mask!(values[2], 0) << 32
                | bit_mask!(values[3], 0) << 48;
            binary[chunk_index + chunks] |= bit_mask!(values[0], 1)
                | bit_mask!(values[1], 1) << 16
                | bit_mask!(values[2], 1) << 32
                | bit_mask!(values[3], 1) << 48;
            binary[chunk_index + chunks * 2] |= bit_mask!(values[0], 2)
                | bit_mask!(values[1], 2) << 16
                | bit_mask!(values[2], 2) << 32
                | bit_mask!(values[3], 2) << 48;
            binary[chunk_index + chunks * 3] |= bit_mask!(values[0], 3)
                | bit_mask!(values[1], 3) << 16
                | bit_mask!(values[2], 3) << 32
                | bit_mask!(values[3], 3) << 48;
        }
    });
}

/// Compute the u8 scalar quantization of an f32 vector.
#[inline]
pub fn scalar_quantize(
    quantized: &mut [u8],
    vec: &[f32],
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    assert_eq!(quantized.len(), vec.len());

    scalar_quantize_neon(
        Neon::try_new().expect("Neon is part of the AArch64 baseline"),
        quantized,
        vec,
        lower_bound,
        multiplier,
    )
}

fn scalar_quantize_neon(
    simd: Neon,
    quantized: &mut [u8],
    vec: &[f32],
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    let chunks = vec.len() / 16;
    let lower = simd.neon.vdupq_n_f32(lower_bound);
    let multiplier_vector = simd.neon.vdupq_n_f32(multiplier);
    let mut sum = simd.neon.vdupq_n_u32(0);

    for i in 0..chunks {
        let offset = i * 16;
        let quantize = |lane_offset| {
            let values = unsafe { simd.neon.vld1q_f32(vec.as_ptr().add(offset + lane_offset)) };
            simd.neon.vcvtaq_u32_f32(
                simd.neon
                    .vmulq_f32(simd.neon.vsubq_f32(values, lower), multiplier_vector),
            )
        };
        let values0 = quantize(0);
        let values1 = quantize(4);
        let values2 = quantize(8);
        let values3 = quantize(12);
        sum = simd.neon.vaddq_u32(sum, values0);
        sum = simd.neon.vaddq_u32(sum, values1);
        sum = simd.neon.vaddq_u32(sum, values2);
        sum = simd.neon.vaddq_u32(sum, values3);

        let values0 = simd
            .neon
            .vcombine_u16(simd.neon.vqmovn_u32(values0), simd.neon.vqmovn_u32(values1));
        let values1 = simd
            .neon
            .vcombine_u16(simd.neon.vqmovn_u32(values2), simd.neon.vqmovn_u32(values3));
        let values = simd
            .neon
            .vcombine_u8(simd.neon.vqmovn_u16(values0), simd.neon.vqmovn_u16(values1));
        unsafe {
            simd.neon
                .vst1q_u8(quantized.as_mut_ptr().add(offset), values)
        };
    }

    let mut total = simd.neon.vaddvq_u32(sum);
    for (output, &value) in quantized[chunks * 16..].iter_mut().zip(&vec[chunks * 16..]) {
        *output = ((value - lower_bound) * multiplier).round() as u8;
        total += *output as u32;
    }
    total
}

/// Compute the binary dot product of two vectors.
#[inline]
pub fn binary_dot_product(lhs: &[u64], rhs: &[u64]) -> u32 {
    assert_eq!(lhs.len(), rhs.len());

    if lhs.len() < 2 {
        return lhs
            .first()
            .zip(rhs.first())
            .map_or(0, |(&lhs, &rhs)| (lhs & rhs).count_ones());
    }

    // Horizontal reduction has less setup overhead for short inputs. At 1024 bits, four
    // independent widened accumulators begin to hide the reduction latency with throughput.
    if lhs.len() < 16 {
        let simd = Neon::try_new().expect("Neon is part of the AArch64 baseline");
        let vectors = lhs.len() / 2;
        let mut sum = 0;
        for i in 0..vectors {
            unsafe {
                let lhs = simd.neon.vld1q_u8(lhs.as_ptr().add(i * 2).cast());
                let rhs = simd.neon.vld1q_u8(rhs.as_ptr().add(i * 2).cast());
                sum += simd
                    .neon
                    .vaddvq_u8(simd.neon.vcntq_u8(simd.neon.vandq_u8(lhs, rhs)))
                    as u32;
            }
        }
        sum + lhs[vectors * 2..]
            .iter()
            .zip(&rhs[vectors * 2..])
            .map(|(&lhs, &rhs)| (lhs & rhs).count_ones())
            .sum::<u32>()
    } else {
        binary_dot_product_neon_throughput(
            Neon::try_new().expect("Neon is part of the AArch64 baseline"),
            lhs,
            rhs,
        )
    }
}

fn binary_dot_product_neon_throughput(simd: Neon, lhs: &[u64], rhs: &[u64]) -> u32 {
    macro_rules! count {
        ($offset:expr) => {{
            let lhs = unsafe { simd.neon.vld1q_u8(lhs.as_ptr().add($offset).cast()) };
            let rhs = unsafe { simd.neon.vld1q_u8(rhs.as_ptr().add($offset).cast()) };
            simd.neon
                .vpaddlq_u8(simd.neon.vcntq_u8(simd.neon.vandq_u8(lhs, rhs)))
        }};
    }

    let vectors = lhs.len() / 2;
    let groups = vectors / 4;
    let mut sum0 = simd.neon.vdupq_n_u32(0);
    let mut sum1 = simd.neon.vdupq_n_u32(0);
    let mut sum2 = simd.neon.vdupq_n_u32(0);
    let mut sum3 = simd.neon.vdupq_n_u32(0);

    for i in 0..groups {
        let offset = i * 8;
        sum0 = simd.neon.vpadalq_u16(sum0, count!(offset));
        sum1 = simd.neon.vpadalq_u16(sum1, count!(offset + 2));
        sum2 = simd.neon.vpadalq_u16(sum2, count!(offset + 4));
        sum3 = simd.neon.vpadalq_u16(sum3, count!(offset + 6));
    }

    for i in groups * 4..vectors {
        sum0 = simd.neon.vpadalq_u16(sum0, count!(i * 2));
    }

    let sum = simd.neon.vaddq_u32(
        simd.neon.vaddq_u32(sum0, sum1),
        simd.neon.vaddq_u32(sum2, sum3),
    );
    simd.neon.vaddvq_u32(sum)
        + lhs[vectors * 2..]
            .iter()
            .zip(&rhs[vectors * 2..])
            .map(|(&lhs, &rhs)| (lhs & rhs).count_ones())
            .sum::<u32>()
}
