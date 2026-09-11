//! Scalar fallback kernels.

/// Compute residuals and return their minimum and maximum.
#[inline]
pub fn min_max_residual(res: &mut [f32], x: &[f32], y: &[f32]) -> (f32, f32) {
    assert_eq!(res.len(), x.len());
    assert_eq!(res.len(), y.len());

    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for ((res, &x), &y) in res.iter_mut().zip(x).zip(y) {
        *res = x - y;
        min = min.min(*res);
        max = max.max(*res);
    }
    (min, max)
}

/// Scale a vector to `u8` values.
#[inline]
#[cfg(any(test, not(target_arch = "aarch64")))]
pub(crate) fn scalar_quantize(
    quantized: &mut [u8],
    vec: &[f32],
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    assert_eq!(quantized.len(), vec.len());

    quantized
        .iter_mut()
        .zip(vec)
        .map(|(quantized, &value)| {
            *quantized = ((value - lower_bound) * multiplier).round() as u8;
            *quantized as u32
        })
        .sum()
}

/// Convert quantized values to bit-sliced binary vectors.
#[inline]
pub(crate) fn vector_binarize_query(vec: &[u8], binary: &mut [u64]) {
    use crate::THETA_LOG_DIM;

    for j in 0..THETA_LOG_DIM {
        for (i, &value) in vec.iter().enumerate() {
            binary[(i + j * vec.len()) / 64] |= (((value >> j) & 1) as u64) << (i % 64);
        }
    }
}

/// Calculate the dot product of two binary vectors.
#[inline]
pub fn binary_dot_product(x: &[u64], y: &[u64]) -> u32 {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).map(|(&x, &y)| (x & y).count_ones()).sum()
}
