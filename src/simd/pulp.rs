//! SIMD implementation with `pulp`

use core::iter;

use pulp::{Simd, as_arrays};

#[inline(always)]
fn abs2_add<S: Simd>(simd: S, x: S::f32s, acc: S::f32s) -> S::f32s {
    simd.mul_add_f32s(x, x, acc)
}

/// Compute the squared Euclidean distance between two vectors.
///
/// Code refers to <https://github.com/nmslib/hnswlib/blob/master/hnswlib/space_l2.h>
#[inline]
pub fn l2_squared_distance<S: Simd>(simd: S, lhs: &[f32], rhs: &[f32]) -> f32 {
    simd.vectorize(
        #[inline(always)]
        || {
            assert_eq!(lhs.len(), rhs.len());

            let (lhs, lhs_tail) = S::as_simd_f32s(lhs);
            let (rhs, rhs_tail) = S::as_simd_f32s(rhs);

            let (lhs2, lhs1) = as_arrays::<2, _>(lhs);
            let (rhs2, rhs1) = as_arrays::<2, _>(rhs);

            let mut sum0 = simd.splat_f32s(0.0);
            let mut sum1 = simd.splat_f32s(0.0);

            for (&[l0, l1], &[r0, r1]) in iter::zip(lhs2, rhs2) {
                sum0 = abs2_add(simd, simd.sub_f32s(l0, r0), sum0);
                sum1 = abs2_add(simd, simd.sub_f32s(l1, r1), sum1);
            }

            for (&l0, &r0) in iter::zip(lhs1, rhs1) {
                sum0 = abs2_add(simd, simd.sub_f32s(l0, r0), sum0);
            }
            {
                let l0 = simd.partial_load_f32s(lhs_tail);
                let r0 = simd.partial_load_f32s(rhs_tail);

                sum0 = abs2_add(simd, simd.sub_f32s(l0, r0), sum0);
            }

            simd.reduce_sum_f32s(simd.add_f32s(sum0, sum1))
        },
    )
}

/// Compute the negative dot product distance between two vectors.
#[inline]
pub fn dot_product<S: Simd>(simd: S, lhs: &[f32], rhs: &[f32]) -> f32 {
    simd.vectorize(
        #[inline(always)]
        || {
            assert_eq!(lhs.len(), rhs.len());

            let (lhs, lhs_tail) = S::as_simd_f32s(lhs);
            let (rhs, rhs_tail) = S::as_simd_f32s(rhs);

            let (lhs2, lhs1) = as_arrays::<2, _>(lhs);
            let (rhs2, rhs1) = as_arrays::<2, _>(rhs);

            let mut sum0 = simd.splat_f32s(0.0);
            let mut sum1 = simd.splat_f32s(0.0);

            for (&[l0, l1], &[r0, r1]) in iter::zip(lhs2, rhs2) {
                sum0 = simd.mul_add_f32s(l0, r0, sum0);
                sum1 = simd.mul_add_f32s(l1, r1, sum1);
            }

            for (&l0, &r0) in iter::zip(lhs1, rhs1) {
                sum0 = simd.mul_add_f32s(l0, r0, sum0);
            }
            {
                let l0 = simd.partial_load_f32s(lhs_tail);
                let r0 = simd.partial_load_f32s(rhs_tail);

                sum0 = simd.mul_add_f32s(l0, r0, sum0);
            }

            simd.reduce_sum_f32s(simd.add_f32s(sum0, sum1))
        },
    )
}

/// Compute the L2 norm of the vector.
///
/// Doesn't handle overflow/underflow for inputs outside of the [MIN_EXP/2, MAX_EXP/2] range.
#[inline]
pub fn l2_norm<S: Simd>(simd: S, vec: &[f32]) -> f32 {
    simd.vectorize(
        #[inline(always)]
        || {
            let (vec, vec_tail) = S::as_simd_f32s(vec);
            let (vec2, vec1) = as_arrays::<2, _>(vec);

            let mut sum0 = simd.splat_f32s(0.0);
            let mut sum1 = simd.splat_f32s(0.0);

            for &[v0, v1] in vec2 {
                sum0 = abs2_add(simd, v0, sum0);
                sum1 = abs2_add(simd, v1, sum1);
            }

            for &v0 in vec1 {
                sum0 = abs2_add(simd, v0, sum0);
            }
            {
                let v0 = simd.partial_load_f32s(vec_tail);
                sum0 = abs2_add(simd, v0, sum0);
            }

            simd.reduce_sum_f32s(simd.add_f32s(sum0, sum1)).sqrt()
        },
    )
}

/// Find the index of the minimum value in the vector.
#[inline]
pub fn argmin<S: Simd>(simd: S, vec: &[f32]) -> usize {
    simd.vectorize(
        #[inline(always)]
        || {
            let (vec, vec_tail) = S::as_simd_f32s(vec);
            let (vec4, vec1) = as_arrays::<4, _>(vec);

            let inc = simd.splat_u32s(S::U32_LANES as u32);
            let infty = simd.splat_f32s(f32::INFINITY);

            let mut min0 = infty;
            let mut min1 = infty;
            let mut min2 = infty;
            let mut min3 = infty;

            let mut min_idx0 = simd.splat_u32s(0);
            let mut min_idx1 = simd.splat_u32s(0);
            let mut min_idx2 = simd.splat_u32s(0);
            let mut min_idx3 = simd.splat_u32s(0);

            let mut idx = pulp::cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            for &[v0, v1, v2, v3] in vec4 {
                let less = simd.less_than_f32s(v0, min0);
                min0 = simd.select_f32s_m32s(less, v0, min0);
                min_idx0 = simd.select_u32s_m32s(less, idx, min_idx0);
                idx = simd.add_u32s(idx, inc);

                let less = simd.less_than_f32s(v1, min1);
                min1 = simd.select_f32s_m32s(less, v1, min1);
                min_idx1 = simd.select_u32s_m32s(less, idx, min_idx1);
                idx = simd.add_u32s(idx, inc);

                let less = simd.less_than_f32s(v2, min2);
                min2 = simd.select_f32s_m32s(less, v2, min2);
                min_idx2 = simd.select_u32s_m32s(less, idx, min_idx2);
                idx = simd.add_u32s(idx, inc);

                let less = simd.less_than_f32s(v3, min3);
                min3 = simd.select_f32s_m32s(less, v3, min3);
                min_idx3 = simd.select_u32s_m32s(less, idx, min_idx3);
                idx = simd.add_u32s(idx, inc);
            }

            for &v0 in vec1 {
                let less = simd.less_than_f32s(v0, min0);
                min0 = simd.select_f32s_m32s(less, v0, min0);
                min_idx0 = simd.select_u32s_m32s(less, idx, min_idx0);

                idx = simd.add_u32s(idx, inc);
            }

            {
                let m = simd.mask_between_m32s(0, vec_tail.len() as u32).mask();

                let v0 = simd.partial_load_f32s(vec_tail);
                let v0 = simd.select_f32s_m32s(m, v0, infty);

                let less = simd.less_than_f32s(v0, min0);
                min0 = simd.select_f32s_m32s(less, v0, min0);
                min_idx0 = simd.select_u32s_m32s(less, idx, min_idx0);
            }

            let less = simd.less_than_f32s(min0, min2);
            min0 = simd.select_f32s_m32s(less, min0, min2);
            min_idx0 = simd.select_u32s_m32s(less, min_idx0, min_idx2);

            let less = simd.less_than_f32s(min1, min3);
            min1 = simd.select_f32s_m32s(less, min1, min3);
            min_idx1 = simd.select_u32s_m32s(less, min_idx1, min_idx3);

            let less = simd.less_than_f32s(min0, min1);
            min0 = simd.select_f32s_m32s(less, min0, min1);
            min_idx0 = simd.select_u32s_m32s(less, min_idx0, min_idx1);

            let min = simd.reduce_min_f32s(min0);
            let is_min = simd.equal_f32s(min0, simd.splat_f32s(min));
            let pos = simd.first_true_m32s(is_min);
            let min_idx = bytemuck::cast_slice::<S::u32s, u32>(core::slice::from_ref(&min_idx0));

            if pos < S::U32_LANES {
                min_idx[pos] as usize
            } else {
                0
            }
        },
    )
}
