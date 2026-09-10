//! RaBitQ SIMD implementation with `pulp`.

use core::iter;

use ::pulp::Simd;

/// Compute the squared Euclidean distance between two vectors.
#[inline]
pub fn l2_squared_distance<S: Simd>(simd: S, lhs: &[f32], rhs: &[f32]) -> f32 {
    simd.vectorize(
        #[inline(always)]
        || {
            assert_eq!(lhs.len(), rhs.len());
            let mut sum = simd.splat_f32s(0.0);
            let (lhs, lhs_tail) = S::as_simd_f32s(lhs);
            let (rhs, rhs_tail) = S::as_simd_f32s(rhs);
            for (&lhs, &rhs) in iter::zip(lhs, rhs) {
                let diff = simd.sub_f32s(lhs, rhs);
                sum = simd.mul_add_f32s(diff, diff, sum);
            }
            let mut total = simd.reduce_sum_f32s(sum);
            for (&lhs, &rhs) in iter::zip(lhs_tail, rhs_tail) {
                total += (lhs - rhs).powi(2);
            }
            total
        },
    )
}

/// Compute the dot product of two vectors.
#[inline]
pub fn dot_product<S: Simd>(simd: S, lhs: &[f32], rhs: &[f32]) -> f32 {
    simd.vectorize(
        #[inline(always)]
        || {
            assert_eq!(lhs.len(), rhs.len());
            let mut sum = simd.splat_f32s(0.0);
            let (lhs, lhs_tail) = S::as_simd_f32s(lhs);
            let (rhs, rhs_tail) = S::as_simd_f32s(rhs);
            for (&lhs, &rhs) in iter::zip(lhs, rhs) {
                sum = simd.mul_add_f32s(lhs, rhs, sum);
            }
            simd.reduce_sum_f32s(sum)
                + iter::zip(lhs_tail, rhs_tail)
                    .map(|(&lhs, &rhs)| lhs * rhs)
                    .sum::<f32>()
        },
    )
}

/// Compute residuals and return their minimum and maximum.
#[inline]
pub fn min_max_residual<S: Simd>(simd: S, res: &mut [f32], x: &[f32], y: &[f32]) -> (f32, f32) {
    simd.vectorize(
        #[inline(always)]
        || {
            assert_eq!(res.len(), x.len());
            assert_eq!(res.len(), y.len());
            let (res, res_tail) = S::as_mut_simd_f32s(res);
            let (x, x_tail) = S::as_simd_f32s(x);
            let (y, y_tail) = S::as_simd_f32s(y);
            let mut min = simd.splat_f32s(f32::MAX);
            let mut max = simd.splat_f32s(f32::MIN);
            for ((res, &x), &y) in iter::zip(iter::zip(res, x), y) {
                *res = simd.sub_f32s(x, y);
                min = simd.min_f32s(min, *res);
                max = simd.max_f32s(max, *res);
            }
            let mut min = simd.reduce_min_f32s(min);
            let mut max = simd.reduce_max_f32s(max);
            for ((res, &x), &y) in iter::zip(iter::zip(res_tail, x_tail), y_tail) {
                *res = x - y;
                min = min.min(*res);
                max = max.max(*res);
            }
            (min, max)
        },
    )
}
