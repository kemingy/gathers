//! x86 kernels implemented with `pulp` intrinsics.

use core::iter;

use crate::THETA_LOG_DIM;
use crate::fastscan::BATCH_SIZE;

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

/// Accumulate asymmetric binary dot products for 32 packed vectors.
pub(crate) fn fastscan_accumulate(
    simd: Avx2,
    codes: &[u8],
    lut: &[u8],
    result: &mut [u32; BATCH_SIZE],
) {
    use ::pulp;

    assert_eq!(codes.len(), lut.len());
    let Avx2 { avx, avx2, .. } = simd;

    simd.vectorize(
        #[inline(always)]
        || {
            let low_mask = avx._mm256_set1_epi8(0x0f);
            result.fill(0);

            // Each iteration combines two four-coordinate groups. As on NEON, bound each
            // u16 segment to 1,024 groups before widening into the u32 result.
            for (codes, lut) in codes.chunks(16 * 1_024).zip(lut.chunks(16 * 1_024)) {
                let mut lower_sum = avx._mm256_setzero_si256();
                let mut upper_sum = avx._mm256_setzero_si256();
                let (codes, code_tail) = codes.as_chunks::<32>();
                let (luts, lut_tail) = lut.as_chunks::<32>();
                assert!(code_tail.is_empty());
                assert!(lut_tail.is_empty());

                for (codes, lut) in codes.iter().zip(luts) {
                    let codes = pulp::cast(*codes);
                    let lut = pulp::cast(*lut);
                    let lower =
                        avx2._mm256_shuffle_epi8(lut, avx2._mm256_and_si256(codes, low_mask));
                    let upper = avx2._mm256_shuffle_epi8(
                        lut,
                        avx2._mm256_and_si256(avx2._mm256_srli_epi16::<4>(codes), low_mask),
                    );

                    let lower0 = avx2._mm256_cvtepu8_epi16(avx._mm256_castsi256_si128(lower));
                    let lower1 =
                        avx2._mm256_cvtepu8_epi16(avx2._mm256_extracti128_si256::<1>(lower));
                    lower_sum =
                        avx2._mm256_add_epi16(lower_sum, avx2._mm256_add_epi16(lower0, lower1));
                    let upper0 = avx2._mm256_cvtepu8_epi16(avx._mm256_castsi256_si128(upper));
                    let upper1 =
                        avx2._mm256_cvtepu8_epi16(avx2._mm256_extracti128_si256::<1>(upper));
                    upper_sum =
                        avx2._mm256_add_epi16(upper_sum, avx2._mm256_add_epi16(upper0, upper1));
                }

                let lower: [u16; 16] = pulp::cast(lower_sum);
                let upper: [u16; 16] = pulp::cast(upper_sum);
                for (lane, (&lower, &upper)) in lower.iter().zip(&upper).enumerate() {
                    result[lane] += u32::from(lower);
                    result[lane + 16] += u32::from(upper);
                }
            }
        },
    );
}

/// Accumulate asymmetric binary dot products for several queries while sharing code loads.
pub(crate) fn fastscan_accumulate_many<const N: usize>(
    simd: Avx2,
    codes: &[u8],
    luts: &[&[u8]; N],
    results: &mut [[u32; BATCH_SIZE]; N],
) {
    use ::pulp;

    assert!(luts.iter().all(|lut| lut.len() == codes.len()));
    let Avx2 {
        avx, avx2, sse2, ..
    } = simd;

    simd.vectorize(
        #[inline(always)]
        || {
            let low_mask = avx._mm256_set1_epi8(0x0f);
            results.fill([0; BATCH_SIZE]);

            for segment_start in (0..codes.len()).step_by(16 * 1_024) {
                let segment_end = (segment_start + 16 * 1_024).min(codes.len());
                let (code_chunks, code_tail) = codes[segment_start..segment_end].as_chunks::<32>();
                assert!(code_tail.is_empty());
                let lut_chunks: [&[[u8; 32]]; N] = std::array::from_fn(|query| {
                    let (chunks, tail) = luts[query][segment_start..segment_end].as_chunks::<32>();
                    assert!(tail.is_empty());
                    chunks
                });
                let zero = avx._mm256_setzero_si256();
                let zero128 = sse2._mm_setzero_si128();
                let mut sums = [[zero; 2]; N];

                let (code_pairs, code_pair_tail) = code_chunks.as_chunks::<2>();
                assert!(code_pair_tail.is_empty());
                for (pair, code_pair) in code_pairs.iter().enumerate() {
                    let mut byte_sums = [[zero128; 2]; N];
                    for (offset, codes) in code_pair.iter().enumerate() {
                        let codes = pulp::cast(*codes);
                        let lower_codes = avx2._mm256_and_si256(codes, low_mask);
                        let upper_codes =
                            avx2._mm256_and_si256(avx2._mm256_srli_epi16::<4>(codes), low_mask);
                        for query in 0..N {
                            let lut = pulp::cast(lut_chunks[query][pair * 2 + offset]);
                            let lower = avx2._mm256_shuffle_epi8(lut, lower_codes);
                            let upper = avx2._mm256_shuffle_epi8(lut, upper_codes);
                            let lower = sse2._mm_add_epi8(
                                avx._mm256_castsi256_si128(lower),
                                avx2._mm256_extracti128_si256::<1>(lower),
                            );
                            let upper = sse2._mm_add_epi8(
                                avx._mm256_castsi256_si128(upper),
                                avx2._mm256_extracti128_si256::<1>(upper),
                            );
                            byte_sums[query][0] = sse2._mm_add_epi8(byte_sums[query][0], lower);
                            byte_sums[query][1] = sse2._mm_add_epi8(byte_sums[query][1], upper);
                        }
                    }
                    // A shuffle result sums four 4-bit query values. Each code chunk
                    // already combines two 128-bit lanes; folding two chunks is bounded
                    // by 4 * 4 * 15 = 240, so the byte additions cannot overflow.
                    for query in 0..N {
                        sums[query][0] = avx2._mm256_add_epi16(
                            sums[query][0],
                            avx2._mm256_cvtepu8_epi16(byte_sums[query][0]),
                        );
                        sums[query][1] = avx2._mm256_add_epi16(
                            sums[query][1],
                            avx2._mm256_cvtepu8_epi16(byte_sums[query][1]),
                        );
                    }
                }

                for query in 0..N {
                    let lower: [u16; 16] = pulp::cast(sums[query][0]);
                    let upper: [u16; 16] = pulp::cast(sums[query][1]);
                    for (lane, (&lower, &upper)) in lower.iter().zip(&upper).enumerate() {
                        results[query][lane] += u32::from(lower);
                        results[query][lane + 16] += u32::from(upper);
                    }
                }
            }
        },
    );
}

/// Accumulate several queries with AVX-512 while sharing code loads.
///
/// # Safety
///
/// The caller must prove that AVX-512F, AVX-512BW, AVX-512VBMI, and
/// AVX-512VNNI are available.
#[allow(unsafe_code)]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vnni")]
pub(crate) unsafe fn fastscan_accumulate_many_avx512<const N: usize>(
    codes: &[u8],
    luts: &[&[u8]; N],
    results: &mut [[u32; BATCH_SIZE]; N],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // `vpshufb` evaluates four coordinate groups independently in its 128-bit lanes.
    // These indices regroup the four scores for each centroid so VNNI can sum them.
    const TRANSPOSE_INDICES: [u8; 64] = [
        0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53,
        6, 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63,
    ];
    assert!(luts.iter().all(|lut| lut.len() == codes.len()));
    let low_mask = _mm512_set1_epi8(0x0f);
    // SAFETY: the constant contains exactly 64 initialized bytes.
    let transpose = unsafe { _mm512_loadu_si512(TRANSPOSE_INDICES.as_ptr().cast()) };
    let ones = _mm512_set1_epi8(1);
    results.fill([0; BATCH_SIZE]);

    for segment_start in (0..codes.len()).step_by(16 * 1_024) {
        let segment_end = (segment_start + 16 * 1_024).min(codes.len());
        let (code_chunks, code_tail) = codes[segment_start..segment_end].as_chunks::<64>();
        assert!(code_tail.is_empty());
        let lut_chunks: [&[[u8; 64]]; N] = std::array::from_fn(|query| {
            let (chunks, tail) = luts[query][segment_start..segment_end].as_chunks::<64>();
            assert!(tail.is_empty());
            chunks
        });
        let mut sums = [[_mm512_setzero_si512(); 2]; N];

        for (group, codes) in code_chunks.iter().enumerate() {
            // SAFETY: `codes` is an exact 64-byte chunk.
            let codes = unsafe { _mm512_loadu_si512(codes.as_ptr().cast()) };
            let lower_codes = _mm512_and_si512(codes, low_mask);
            let upper_codes = _mm512_and_si512(_mm512_srli_epi16::<4>(codes), low_mask);
            for query in 0..N {
                // SAFETY: every LUT chunk contains exactly 64 initialized bytes.
                let lut = unsafe { _mm512_loadu_si512(lut_chunks[query][group].as_ptr().cast()) };
                let lower =
                    _mm512_permutexvar_epi8(transpose, _mm512_shuffle_epi8(lut, lower_codes));
                let upper =
                    _mm512_permutexvar_epi8(transpose, _mm512_shuffle_epi8(lut, upper_codes));
                sums[query][0] = _mm512_dpbusd_epi32(sums[query][0], lower, ones);
                sums[query][1] = _mm512_dpbusd_epi32(sums[query][1], upper, ones);
            }
        }

        for query in 0..N {
            // SAFETY: each result contains 32 initialized u32 values. The two loads and
            // stores cover its lower and upper 16-element halves without overlap.
            unsafe {
                let lower = results[query].as_mut_ptr();
                let upper = lower.add(16);
                _mm512_storeu_si512(
                    lower.cast(),
                    _mm512_add_epi32(_mm512_loadu_si512(lower.cast()), sums[query][0]),
                );
                _mm512_storeu_si512(
                    upper.cast(),
                    _mm512_add_epi32(_mm512_loadu_si512(upper.cast()), sums[query][1]),
                );
            }
        }
    }
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
