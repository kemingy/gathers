//! Legacy x86 `std::arch` kernels retained for benchmark comparisons.

/// Compute residuals and return their minimum and maximum.
///
/// # Safety
///
/// The caller must ensure that AVX is available.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn min_max_residual(res: &mut [f32], x: &[f32], y: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    assert_eq!(res.len(), x.len());
    assert_eq!(res.len(), y.len());

    let (mut min0, mut min1) = (_mm256_set1_ps(f32::MAX), _mm256_set1_ps(f32::MAX));
    let (mut max0, mut max1) = (_mm256_set1_ps(f32::MIN), _mm256_set1_ps(f32::MIN));
    let mut x_ptr = x.as_ptr();
    let mut y_ptr = y.as_ptr();
    let mut res_ptr = res.as_mut_ptr();
    let mut lanes = [0.0f32; 8];
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let length = res.len();

    unsafe {
        for _ in 0..(length / 16) {
            let x0 = _mm256_loadu_ps(x_ptr);
            let y0 = _mm256_loadu_ps(y_ptr);
            let residual0 = _mm256_sub_ps(x0, y0);
            _mm256_storeu_ps(res_ptr, residual0);
            x_ptr = x_ptr.add(8);
            y_ptr = y_ptr.add(8);
            res_ptr = res_ptr.add(8);
            min0 = _mm256_min_ps(min0, residual0);
            max0 = _mm256_max_ps(max0, residual0);

            let x1 = _mm256_loadu_ps(x_ptr);
            let y1 = _mm256_loadu_ps(y_ptr);
            let residual1 = _mm256_sub_ps(x1, y1);
            _mm256_storeu_ps(res_ptr, residual1);
            x_ptr = x_ptr.add(8);
            y_ptr = y_ptr.add(8);
            res_ptr = res_ptr.add(8);
            min1 = _mm256_min_ps(min1, residual1);
            max1 = _mm256_max_ps(max1, residual1);
        }

        for _ in 0..(length & 0b1111) / 8 {
            let x0 = _mm256_loadu_ps(x_ptr);
            let y0 = _mm256_loadu_ps(y_ptr);
            let residual0 = _mm256_sub_ps(x0, y0);
            _mm256_storeu_ps(res_ptr, residual0);
            x_ptr = x_ptr.add(8);
            y_ptr = y_ptr.add(8);
            res_ptr = res_ptr.add(8);
            min0 = _mm256_min_ps(min0, residual0);
            max0 = _mm256_max_ps(max0, residual0);
        }

        _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_min_ps(min0, min1));
        for &value in &lanes {
            min = min.min(value);
        }
        _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_max_ps(max0, max1));
        for &value in &lanes {
            max = max.max(value);
        }

        for _ in 0..(length & 0b111) {
            *res_ptr = *x_ptr - *y_ptr;
            min = min.min(*res_ptr);
            max = max.max(*res_ptr);
            res_ptr = res_ptr.add(1);
            x_ptr = x_ptr.add(1);
            y_ptr = y_ptr.add(1);
        }
    }

    (min, max)
}

/// Compute the binary dot product of two vectors.
///
/// # Safety
///
/// The caller must ensure that SSE2, AVX, and AVX2 are available.
#[target_feature(enable = "sse2,avx,avx2")]
#[inline]
pub unsafe fn binary_dot_product(lhs: &[u64], rhs: &[u64]) -> u32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    assert_eq!(lhs.len(), rhs.len());

    #[inline(always)]
    unsafe fn mm256_popcnt_epi64(x: __m256i) -> __m256i {
        unsafe {
            let lookup_table = _mm256_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3,
                2, 3, 3, 4,
            );
            let mask = _mm256_set1_epi8(15);
            let low = _mm256_shuffle_epi8(lookup_table, _mm256_and_si256(x, mask));
            let high = _mm256_shuffle_epi8(
                lookup_table,
                _mm256_and_si256(_mm256_srli_epi64(x, 4), mask),
            );
            _mm256_sad_epu8(_mm256_add_epi8(low, high), _mm256_setzero_si256())
        }
    }

    let chunks = lhs.len() / 4;
    let mut sum = lhs[chunks * 4..]
        .iter()
        .zip(&rhs[chunks * 4..])
        .map(|(&x, &y)| (x & y).count_ones())
        .sum();
    if chunks == 0 {
        return sum;
    }

    unsafe {
        let mut sum256 = _mm256_setzero_si256();
        let mut x_ptr = lhs.as_ptr().cast::<__m256i>();
        let mut y_ptr = rhs.as_ptr().cast::<__m256i>();
        for _ in 0..chunks {
            let and = _mm256_and_si256(_mm256_loadu_si256(x_ptr), _mm256_loadu_si256(y_ptr));
            sum256 = _mm256_add_epi64(sum256, mm256_popcnt_epi64(and));
            x_ptr = x_ptr.add(1);
            y_ptr = y_ptr.add(1);
        }

        let halves = _mm_add_epi64(
            _mm256_castsi256_si128(sum256),
            _mm256_extracti128_si256(sum256, 1),
        );
        sum += _mm_cvtsi128_si32(_mm_add_epi64(halves, _mm_shuffle_epi32(halves, 78))) as u32;
    }
    sum
}
