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
            let (res, res_vector_tail) = ::pulp::as_arrays_mut::<4, _>(res);
            let (x, x_vector_tail) = ::pulp::as_arrays::<4, _>(x);
            let (y, y_vector_tail) = ::pulp::as_arrays::<4, _>(y);
            let mut min = [simd.splat_f32s(f32::MAX); 4];
            let mut max = [simd.splat_f32s(f32::MIN); 4];

            for ((res, &x), &y) in iter::zip(iter::zip(res, x), y) {
                for lane in 0..4 {
                    res[lane] = simd.sub_f32s(x[lane], y[lane]);
                    min[lane] = simd.min_f32s(min[lane], res[lane]);
                    max[lane] = simd.max_f32s(max[lane], res[lane]);
                }
            }
            for ((res, &x), &y) in
                iter::zip(iter::zip(res_vector_tail, x_vector_tail), y_vector_tail)
            {
                *res = simd.sub_f32s(x, y);
                min[0] = simd.min_f32s(min[0], *res);
                max[0] = simd.max_f32s(max[0], *res);
            }

            min[0] = simd.min_f32s(min[0], min[1]);
            min[2] = simd.min_f32s(min[2], min[3]);
            max[0] = simd.max_f32s(max[0], max[1]);
            max[2] = simd.max_f32s(max[2], max[3]);
            let mut min = simd.reduce_min_f32s(simd.min_f32s(min[0], min[2]));
            let mut max = simd.reduce_max_f32s(simd.max_f32s(max[0], max[2]));
            for ((res, &x), &y) in iter::zip(iter::zip(res_tail, x_tail), y_tail) {
                *res = x - y;
                min = min.min(*res);
                max = max.max(*res);
            }
            (min, max)
        },
    )
}
