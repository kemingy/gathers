//! Accelerate with SIMD.

use crate::rabitq::THETA_LOG_DIM;

/// Compute the squared Euclidean distance between two vectors.
/// Code refer to https://github.com/nmslib/hnswlib/blob/master/hnswlib/space_l2.h
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "fma,avx")]
#[inline]
pub unsafe fn l2_squared_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    assert_eq!(lhs.len(), rhs.len());
    let mut lhs_ptr = lhs.as_ptr();
    let mut rhs_ptr = rhs.as_ptr();
    let (mut diff, mut vx, mut vy): (__m256, __m256, __m256);
    let mut sum = _mm256_setzero_ps();

    for _ in 0..(lhs.len() / 16) {
        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_fmadd_ps(diff, diff, sum);

        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    for _ in 0..(lhs.len() & 0b1111) / 8 {
        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    #[inline(always)]
    unsafe fn reduce_f32_256(accumulate: __m256) -> f32 {
        // add [4..7] to [0..3]
        let mut combined = _mm256_add_ps(
            accumulate,
            _mm256_permute2f128_ps(accumulate, accumulate, 1),
        );
        // add [0..3] to [0..1]
        combined = _mm256_hadd_ps(combined, combined);
        // add [0..1] to [0]
        combined = _mm256_hadd_ps(combined, combined);
        _mm256_cvtss_f32(combined)
    }

    let mut res = reduce_f32_256(sum);
    for _ in 0..(lhs.len() & 0b111) {
        let residual = *lhs_ptr - *rhs_ptr;
        res += residual * residual;
        lhs_ptr = lhs_ptr.add(1);
        rhs_ptr = rhs_ptr.add(1);
    }
    res
}

/// Compute the negative dot product distance between two vectors.
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "fma,avx")]
#[inline]
pub unsafe fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    assert_eq!(lhs.len(), rhs.len());
    let mut lhs_ptr = lhs.as_ptr();
    let mut rhs_ptr = rhs.as_ptr();
    let mut sum = _mm256_setzero_ps();
    let (mut vx, mut vy): (__m256, __m256);

    for _ in 0..(lhs.len() / 16) {
        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        sum = _mm256_fmadd_ps(vx, vy, sum);

        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        sum = _mm256_fmadd_ps(vx, vy, sum);
    }

    for _ in 0..(lhs.len() & 0b1111) / 8 {
        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        sum = _mm256_fmadd_ps(vx, vy, sum);
    }

    #[inline(always)]
    unsafe fn reduce_f32_256(accumulate: __m256) -> f32 {
        // add [4..7] to [0..3]
        let mut combined = _mm256_add_ps(
            accumulate,
            _mm256_permute2f128_ps(accumulate, accumulate, 1),
        );
        // add [0..3] to [0..1]
        combined = _mm256_hadd_ps(combined, combined);
        // add [0..1] to [0]
        combined = _mm256_hadd_ps(combined, combined);
        _mm256_cvtss_f32(combined)
    }

    let mut res = reduce_f32_256(sum);
    for _ in 0..(lhs.len() & 0b111) {
        res += *lhs_ptr * *rhs_ptr;
        lhs_ptr = lhs_ptr.add(1);
        rhs_ptr = rhs_ptr.add(1);
    }

    res
}

/// Compute the L2 norm of the vector.
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "fma,avx")]
#[inline]
pub unsafe fn l2_norm(vec: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut vec_ptr = vec.as_ptr();
    let mut f32x8: __m256;
    let mut sum = _mm256_setzero_ps();

    for _ in 0..(vec.len() / 16) {
        f32x8 = _mm256_loadu_ps(vec_ptr);
        vec_ptr = vec_ptr.add(8);
        sum = _mm256_fmadd_ps(f32x8, f32x8, sum);

        f32x8 = _mm256_loadu_ps(vec_ptr);
        vec_ptr = vec_ptr.add(8);
        sum = _mm256_fmadd_ps(f32x8, f32x8, sum);
    }

    for _ in 0..(vec.len() & 0b1111) / 8 {
        f32x8 = _mm256_loadu_ps(vec_ptr);
        vec_ptr = vec_ptr.add(8);
        sum = _mm256_fmadd_ps(f32x8, f32x8, sum);
    }

    #[inline(always)]
    unsafe fn reduce_f32_256(accumulate: __m256) -> f32 {
        // add [4..7] to [0..3]
        let mut combined = _mm256_add_ps(
            accumulate,
            _mm256_permute2f128_ps(accumulate, accumulate, 1),
        );
        // add [0..3] to [0..1]
        combined = _mm256_hadd_ps(combined, combined);
        // add [0..1] to [0]
        combined = _mm256_hadd_ps(combined, combined);
        _mm256_cvtss_f32(combined)
    }

    let mut res = reduce_f32_256(sum);
    for _ in 0..(vec.len() & 0b111) {
        res += *vec_ptr * *vec_ptr;
        vec_ptr = vec_ptr.add(1);
    }

    res.sqrt()
}

/// Find the index of the minimum value in the vector.
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "fma,avx")]
#[inline]
pub unsafe fn argmin(vec: &[f32]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut index = 0;
    let mut minimal = f32::MAX;
    let mut comp = _mm256_set1_ps(minimal);
    let mut vec_ptr = vec.as_ptr();
    let (mut y1, mut y2, mut y3, mut y4, mut mask): (__m256, __m256, __m256, __m256, __m256);
    let mut i = 0;

    for _ in 0..(vec.len() / 32) {
        y1 = _mm256_loadu_ps(vec_ptr);
        y2 = _mm256_loadu_ps(vec_ptr.add(8));
        y3 = _mm256_loadu_ps(vec_ptr.add(16));
        y4 = _mm256_loadu_ps(vec_ptr.add(24));
        vec_ptr = vec_ptr.add(32);

        y1 = _mm256_min_ps(y1, y2);
        y3 = _mm256_min_ps(y3, y4);
        y1 = _mm256_min_ps(y1, y3);
        mask = _mm256_cmp_ps(comp, y1, _CMP_GT_OS);
        if 0 == _mm256_testz_ps(mask, mask) {
            for (j, &val) in vec.iter().enumerate().skip(i).take(32) {
                if minimal > val {
                    minimal = val;
                    index = j;
                }
            }
            comp = _mm256_set1_ps(minimal);
        }
        i += 32;
    }

    for (j, &val) in vec.iter().enumerate().skip(i) {
        if minimal > val {
            minimal = val;
            index = j;
        }
    }

    index
}

/// Compute the min and max value of a vector.
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn min_max_residual(res: &mut [f32], x: &[f32], y: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut min_32x8 = _mm256_set1_ps(f32::MAX);
    let mut max_32x8 = _mm256_set1_ps(f32::MIN);
    let mut x_ptr = x.as_ptr();
    let mut y_ptr = y.as_ptr();
    let mut res_ptr = res.as_mut_ptr();
    let mut f32x8 = [0.0f32; 8];
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let length = res.len();
    let rest = length & 0b111;
    let (mut x256, mut y256, mut res256);

    for _ in 0..(length / 8) {
        x256 = _mm256_loadu_ps(x_ptr);
        y256 = _mm256_loadu_ps(y_ptr);
        res256 = _mm256_sub_ps(x256, y256);
        _mm256_storeu_ps(res_ptr, res256);
        x_ptr = x_ptr.add(8);
        y_ptr = y_ptr.add(8);
        res_ptr = res_ptr.add(8);
        min_32x8 = _mm256_min_ps(min_32x8, res256);
        max_32x8 = _mm256_max_ps(max_32x8, res256);
    }
    _mm256_storeu_ps(f32x8.as_mut_ptr(), min_32x8);
    for &x in f32x8.iter() {
        if x < min {
            min = x;
        }
    }
    _mm256_storeu_ps(f32x8.as_mut_ptr(), max_32x8);
    for &x in f32x8.iter() {
        if x > max {
            max = x;
        }
    }

    for _ in 0..rest {
        *res_ptr = *x_ptr - *y_ptr;
        if *res_ptr < min {
            min = *res_ptr;
        }
        if *res_ptr > max {
            max = *res_ptr;
        }
        res_ptr = res_ptr.add(1);
        x_ptr = x_ptr.add(1);
        y_ptr = y_ptr.add(1);
    }

    (min, max)
}

/// Compute the u8 scalar quantization of a f32 vector.
///
/// This function doesn't need `bias` because it *round* the f32 to u32 instead of *floor*.
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx,avx2")]
#[inline]
pub unsafe fn scalar_quantize(
    quantized: &mut [u8],
    vec: &[f32],
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut quantize_ptr = quantized.as_mut_ptr() as *mut u64;

    let lower = _mm256_set1_ps(lower_bound);
    let scalar = _mm256_set1_ps(multiplier);
    let mut sum256 = _mm256_setzero_si256();
    let mask = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, //
        -1, -1, -1, -1, -1, -1, -1, -1, //
        0, 4, 8, 12, -1, -1, -1, -1, //
        -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let length = vec.len();
    let rest = length & 0b111;
    let mut vec_ptr = vec.as_ptr();
    let mut quantize8xi32;

    for _ in 0..(length / 8) {
        let v = _mm256_loadu_ps(vec_ptr);
        // `_mm256_cvtps_epi32` is *round* instead of *floor*, so we don't need the bias here
        quantize8xi32 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(v, lower), scalar));
        sum256 = _mm256_add_epi32(sum256, quantize8xi32);
        // extract the lower 8 bits of each 32-bit integer and save them to [0..32] and [128..160]
        let shuffled = _mm256_shuffle_epi8(quantize8xi32, mask);
        quantize_ptr.write(
            (_mm256_extract_epi32(shuffled, 0) as u64)
                | ((_mm256_extract_epi32(shuffled, 4) as u64) << 32),
        );
        quantize_ptr = quantize_ptr.add(1);
        vec_ptr = vec_ptr.add(8);
    }

    // Compute the sum of the quantized values
    // add [4..7] to [0..3]
    let mut combined = _mm256_add_epi32(sum256, _mm256_permute2f128_si256(sum256, sum256, 1));
    // combine [0..3] to [0..1]
    combined = _mm256_hadd_epi32(combined, combined);
    // combine [0..1] to [0]
    combined = _mm256_hadd_epi32(combined, combined);
    let mut sum = _mm256_cvtsi256_si32(combined) as u32;

    for i in 0..rest {
        // this should be safe as it's a scalar quantization
        let q = ((*vec_ptr - lower_bound) * multiplier)
            .round()
            .to_int_unchecked::<u8>();
        quantized[length - rest + i] = q;
        sum += q as u32;
        vec_ptr = vec_ptr.add(1);
    }

    sum
}

/// Convert an [u8] to 4x binary vector stored as u64.
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx,avx2")]
#[inline]
pub unsafe fn vector_binarize_query(vec: &[u8], binary: &mut [u64]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let length = vec.len();
    let mut ptr = vec.as_ptr() as *const __m256i;

    for i in (0..length).step_by(32) {
        // since it's not guaranteed that the vec is fully-aligned
        let mut v = _mm256_loadu_si256(ptr);
        ptr = ptr.add(1);
        // only the lower 4 bits are useful due to the 4-bit scalar quantization
        v = _mm256_slli_epi32(v, 4);
        for j in 0..THETA_LOG_DIM {
            // extract the MSB of each u8
            let mask = (_mm256_movemask_epi8(v) as u32) as u64;
            // (opposite version) let shift = if (i / 32) % 2 == 0 { 32 } else { 0 };
            let shift = i & 32;
            binary[(3 - j) * (length >> 6) + (i >> 6)] |= mask << shift;
            v = _mm256_slli_epi32(v, 1);
        }
    }
}

/// Compute the binary dot product of two vectors.
///
/// Refer to: https://github.com/komrad36/popcount
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX2 intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "sse2,avx,avx2")]
#[inline]
pub unsafe fn binary_dot_product(lhs: &[u64], rhs: &[u64]) -> u32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut sum = 0;
    let length = lhs.len() / 4;
    if length == 0 {
        for i in 0..lhs.len() {
            sum += (lhs[i] & rhs[i]).count_ones();
        }
        return sum;
    }
    let rest = lhs.len() & 0b11;
    for i in 0..rest {
        sum += (lhs[4 * length + i] & rhs[4 * length + i]).count_ones();
    }

    #[inline]
    unsafe fn mm256_popcnt_epi64(x: __m256i) -> __m256i {
        let lookup_table = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, // 0-7
            1, 2, 2, 3, 2, 3, 3, 4, // 8-15
            0, 1, 1, 2, 1, 2, 2, 3, // 16-23
            1, 2, 2, 3, 2, 3, 3, 4, // 24-31
        );
        let mask = _mm256_set1_epi8(15);
        let zero = _mm256_setzero_si256();

        let mut low = _mm256_and_si256(x, mask);
        let mut high = _mm256_and_si256(_mm256_srli_epi64(x, 4), mask);
        low = _mm256_shuffle_epi8(lookup_table, low);
        high = _mm256_shuffle_epi8(lookup_table, high);
        _mm256_sad_epu8(_mm256_add_epi8(low, high), zero)
    }

    let mut sum256 = _mm256_setzero_si256();
    let mut x_ptr = lhs.as_ptr() as *const __m256i;
    let mut y_ptr = rhs.as_ptr() as *const __m256i;

    for _ in 0..length {
        let x256 = _mm256_loadu_si256(x_ptr);
        let y256 = _mm256_loadu_si256(y_ptr);
        let and = _mm256_and_si256(x256, y256);
        sum256 = _mm256_add_epi64(sum256, mm256_popcnt_epi64(and));
        x_ptr = x_ptr.add(1);
        y_ptr = y_ptr.add(1);
    }

    let xa = _mm_add_epi64(
        _mm256_castsi256_si128(sum256),
        _mm256_extracti128_si256(sum256, 1),
    );
    sum += _mm_cvtsi128_si64(_mm_add_epi64(xa, _mm_shuffle_epi32(xa, 78))) as u32;

    sum
}
