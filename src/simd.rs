//! Accelerate with SIMD.

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
pub unsafe fn neg_dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
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

    -res
}
