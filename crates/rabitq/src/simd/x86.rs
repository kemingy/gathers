//! x86 kernels implemented with `pulp` intrinsics.

use core::iter;

use crate::THETA_LOG_DIM;

#[allow(unsafe_code)]
pub mod legacy;

::pulp::simd_type!(
    #[allow(clippy::too_many_arguments)]
    pub(crate) struct Avx2 {
        sse2: "sse2",
        avx: "avx",
        avx2: "avx2",
    }
);

/// Compute the u8 scalar quantization of a f32 vector.
///
/// This function doesn't need `bias` because it *round* the f32 to u32 instead of *floor*.
///
/// # Panics
///
/// This function panics if the `sse2`, `avx` and `avx2` target features are not available.
#[inline]
pub fn scalar_quantize(
    quantized: &mut [u8],
    vec: &[f32],
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    use ::pulp;

    let simd = Avx2::try_new().unwrap();
    let avx = simd.avx;
    let avx2 = simd.avx2;

    simd.vectorize(
        #[inline(always)]
        || {
            let (quantize, quantize_tail) = pulp::as_arrays_mut::<8, _>(quantized);

            let lower = avx._mm256_set1_ps(lower_bound);
            let scalar = avx._mm256_set1_ps(multiplier);
            let mut sum256 = avx._mm256_setzero_si256();
            let mask = avx._mm256_setr_epi8(
                0, 4, 8, 12, -1, -1, -1, -1, //
                -1, -1, -1, -1, -1, -1, -1, -1, //
                0, 4, 8, 12, -1, -1, -1, -1, //
                -1, -1, -1, -1, -1, -1, -1, -1,
            );
            let (vec, vec_tail) = pulp::as_arrays::<8, _>(vec);
            let mut quantize8xi32;

            for (q, &v) in iter::zip(quantize, vec) {
                let v = pulp::cast(v);
                // `avx._mm256_cvtps_epi32` is *round* instead of *floor*, so we don't need the bias here
                quantize8xi32 =
                    avx._mm256_cvtps_epi32(avx._mm256_mul_ps(avx._mm256_sub_ps(v, lower), scalar));
                sum256 = avx2._mm256_add_epi32(sum256, quantize8xi32);
                // extract the lower 8 bits of each 32-bit integer and save them to [0..32] and [128..160]
                let shuffled = avx2._mm256_shuffle_epi8(quantize8xi32, mask);
                *q = pulp::cast::<u64, _>(
                    (avx2._mm256_extract_epi32::<0>(shuffled) as u64)
                        | ((avx2._mm256_extract_epi32::<4>(shuffled) as u64) << 32),
                );
            }

            // Compute the sum of the quantized values
            // add [4..7] to [0..3]
            let mut combined =
                avx2._mm256_add_epi32(sum256, avx._mm256_permute2f128_si256::<1>(sum256, sum256));
            // combine [0..3] to [0..1]
            combined = avx2._mm256_hadd_epi32(combined, combined);
            // combine [0..1] to [0]
            combined = avx2._mm256_hadd_epi32(combined, combined);
            let mut sum = avx2._mm256_cvtsi256_si32(combined) as u32;

            for (q, &v) in iter::zip(quantize_tail, vec_tail) {
                *q = ((v - lower_bound) * multiplier).round() as u8;
                sum += *q as u32;
            }

            sum
        },
    )
}

/// Convert an [u8] to 4x binary vector stored as u64.
///
/// # Panics
///
/// This function panics if the `sse2`, `avx` and `avx2` target features are not available.
#[inline]
pub fn vector_binarize_query(vec: &[u8], binary: &mut [u64]) {
    use ::pulp;

    let simd = Avx2::try_new().unwrap();
    let avx2 = simd.avx2;

    simd.vectorize(
        #[inline(always)]
        || {
            assert_eq!(THETA_LOG_DIM, 4);
            assert_eq!(vec.len() % 64, 0);
            assert_eq!(binary.len() % 4, 0);
            assert_eq!(vec.len() / 64, binary.len() / 4);

            let vec = pulp::as_arrays::<64, _>(vec).0;

            let n4 = binary.len() / 4;
            let (binary0, binary) = binary.split_at_mut(n4);
            let (binary1, binary) = binary.split_at_mut(n4);
            let (binary2, binary) = binary.split_at_mut(n4);
            let (binary3, binary) = binary.split_at_mut(n4);
            _ = binary;

            for (((b0, b1), (b2, b3)), &v) in iter::zip(
                iter::zip(iter::zip(binary0, binary1), iter::zip(binary2, binary3)),
                vec,
            ) {
                let mut v: [_; 2] = pulp::cast(v);

                // only the lower 4 bits are useful due to the 4-bit scalar quantization
                v[0] = avx2._mm256_slli_epi32::<4>(v[0]);
                v[1] = avx2._mm256_slli_epi32::<4>(v[1]);

                for b in [b3, b2, b1, b0] {
                    // extract the MSB of each u8
                    let mask0 = (avx2._mm256_movemask_epi8(v[0]) as u32) as u64;
                    let mask1 = (avx2._mm256_movemask_epi8(v[1]) as u32) as u64;

                    *b |= mask0 | (mask1 << 32);

                    // move the next bit to the MSB
                    v[0] = avx2._mm256_slli_epi32::<1>(v[0]);
                    v[1] = avx2._mm256_slli_epi32::<1>(v[1]);
                }
            }
        },
    )
}

/// Compute the binary dot product of two vectors.
///
/// Refer to: <https://github.com/komrad36/popcount>
///
/// # Panics
///
/// This function panics if the `sse2`, `avx` and `avx2` target features are not available.
#[inline]
pub fn binary_dot_product(lhs: &[u64], rhs: &[u64]) -> u32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use ::pulp;

    let simd @ Avx2 { avx, avx2, sse2 } = Avx2::try_new().unwrap();

    simd.vectorize(
        #[inline(always)]
        || {
            let (lhs, lhs_tail) = pulp::as_arrays::<4, _>(lhs);
            let (rhs, rhs_tail) = pulp::as_arrays::<4, _>(rhs);

            let mut sum = 0;
            for (&x, &y) in iter::zip(lhs_tail, rhs_tail) {
                sum += (x & y).count_ones();
            }
            if lhs.is_empty() {
                return sum;
            }

            let mut sum256 = avx._mm256_setzero_si256();

            #[inline(always)]
            fn mm256_popcnt_epi64(simd: Avx2, x: __m256i) -> __m256i {
                let Avx2 { avx, avx2, .. } = simd;

                let lookup_table = avx._mm256_setr_epi8(
                    0, 1, 1, 2, 1, 2, 2, 3, // 0-7
                    1, 2, 2, 3, 2, 3, 3, 4, // 8-15
                    0, 1, 1, 2, 1, 2, 2, 3, // 16-23
                    1, 2, 2, 3, 2, 3, 3, 4, // 24-31
                );
                let mask = avx._mm256_set1_epi8(15);
                let zero = avx._mm256_setzero_si256();

                let mut low = avx2._mm256_and_si256(x, mask);
                let mut high = avx2._mm256_and_si256(avx2._mm256_srli_epi64::<4>(x), mask);
                low = avx2._mm256_shuffle_epi8(lookup_table, low);
                high = avx2._mm256_shuffle_epi8(lookup_table, high);
                avx2._mm256_sad_epu8(avx2._mm256_add_epi8(low, high), zero)
            }

            for (&x, &y) in iter::zip(lhs, rhs) {
                let x256 = pulp::cast(x);
                let y256 = pulp::cast(y);
                let and = avx2._mm256_and_si256(x256, y256);
                sum256 = avx2._mm256_add_epi64(sum256, mm256_popcnt_epi64(simd, and));
            }
            let xa = sse2._mm_add_epi64(
                avx._mm256_castsi256_si128(sum256),
                avx2._mm256_extracti128_si256::<1>(sum256),
            );
            // this assumes the sum is less than 2^31, which should be true for most cases
            sum += sse2._mm_cvtsi128_si32(sse2._mm_add_epi64(xa, sse2._mm_shuffle_epi32::<78>(xa)))
                as u32;

            sum
        },
    )
}
