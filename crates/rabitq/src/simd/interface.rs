//! Runtime-dispatched SIMD interfaces used by RaBitQ.

use ::pulp as pulp_crate;

use crate::THETA_LOG_DIM;
#[cfg(not(target_arch = "aarch64"))]
use crate::simd::native;
use crate::simd::pulp as kernels;

/// Compute residuals and return their minimum and maximum.
#[inline]
pub fn min_max_residual(res: &mut [f32], x: &[f32], y: &[f32]) -> (f32, f32) {
    struct Impl<'a> {
        res: &'a mut [f32],
        x: &'a [f32],
        y: &'a [f32],
    }

    impl pulp_crate::WithSimd for Impl<'_> {
        type Output = (f32, f32);

        #[inline(always)]
        fn with_simd<S: pulp_crate::Simd>(self, simd: S) -> Self::Output {
            let Self { res, x, y } = self;
            kernels::min_max_residual(simd, res, x, y)
        }
    }

    pulp_crate::Arch::new().dispatch(Impl { res, x, y })
}

/// Scale a vector to `u8` values.
#[inline]
pub fn scalar_quantize(
    quantized: &mut [u8],
    vec: &[f32],
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if crate::simd::x86::Avx2::is_available() {
        return crate::simd::x86::scalar_quantize(quantized, vec, lower_bound, multiplier);
    }

    #[cfg(target_arch = "aarch64")]
    return crate::simd::aarch64::scalar_quantize(quantized, vec, lower_bound, multiplier);

    #[cfg(not(target_arch = "aarch64"))]
    {
        native::scalar_quantize(quantized, vec, lower_bound, multiplier)
    }
}

/// Convert quantized values to bit-sliced binary vectors.
#[inline]
pub fn vector_binarize_query(vec: &[u8], binary: &mut [u64]) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if crate::simd::x86::Avx2::is_available() {
        crate::simd::x86::vector_binarize_query(vec, binary);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    return crate::simd::aarch64::vector_binarize_query(vec, binary);

    #[cfg(not(target_arch = "aarch64"))]
    native::vector_binarize_query(vec, binary);
}

/// Calculate the weighted dot product of bit-sliced binary vectors.
///
/// The length of `y` must be `x.len() * THETA_LOG_DIM`.
#[inline]
pub fn asymmetric_binary_dot_product(x: &[u64], y: &[u64]) -> u32 {
    let length = x.len();
    assert_eq!(y.len(), length * THETA_LOG_DIM);

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if crate::simd::x86::Avx2::is_available() {
        return (0..THETA_LOG_DIM)
            .map(|i| {
                let y = &y[i * length..(i + 1) * length];
                crate::simd::x86::binary_dot_product(x, y) << i
            })
            .sum();
    }

    #[cfg(target_arch = "aarch64")]
    return (0..THETA_LOG_DIM)
        .map(|i| {
            let y = &y[i * length..(i + 1) * length];
            crate::simd::aarch64::binary_dot_product(x, y) << i
        })
        .sum();

    #[cfg(not(target_arch = "aarch64"))]
    (0..THETA_LOG_DIM)
        .map(|i| {
            let y = &y[i * length..(i + 1) * length];
            native::binary_dot_product(x, y) << i
        })
        .sum()
}
