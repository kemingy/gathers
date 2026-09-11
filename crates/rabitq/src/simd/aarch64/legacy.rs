//! Direct Neon kernels retained for comparison with Pulp in benchmarks.

use core::arch::aarch64::*;

/// Compute residuals and return their minimum and maximum.
#[inline]
pub fn min_max_residual(res: &mut [f32], x: &[f32], y: &[f32]) -> (f32, f32) {
    assert_eq!(res.len(), x.len());
    assert_eq!(res.len(), y.len());

    // Neon is part of the AArch64 baseline.
    unsafe { min_max_residual_neon(res, x, y) }
}

#[target_feature(enable = "neon")]
unsafe fn min_max_residual_neon(res: &mut [f32], x: &[f32], y: &[f32]) -> (f32, f32) {
    let groups = res.len() / 16;
    let mut min0 = vdupq_n_f32(f32::MAX);
    let mut min1 = min0;
    let mut min2 = min0;
    let mut min3 = min0;
    let mut max0 = vdupq_n_f32(f32::MIN);
    let mut max1 = max0;
    let mut max2 = max0;
    let mut max3 = max0;

    macro_rules! residual {
        ($offset:expr, $min:ident, $max:ident) => {{
            let x = unsafe { vld1q_f32(x.as_ptr().add($offset)) };
            let y = unsafe { vld1q_f32(y.as_ptr().add($offset)) };
            let residual = vsubq_f32(x, y);
            unsafe { vst1q_f32(res.as_mut_ptr().add($offset), residual) };
            $min = vminq_f32($min, residual);
            $max = vmaxq_f32($max, residual);
        }};
    }

    for i in 0..groups {
        let offset = i * 16;
        residual!(offset, min0, max0);
        residual!(offset + 4, min1, max1);
        residual!(offset + 8, min2, max2);
        residual!(offset + 12, min3, max3);
    }

    let mut offset = groups * 16;
    while offset + 4 <= res.len() {
        residual!(offset, min0, max0);
        offset += 4;
    }

    let min = vminq_f32(vminq_f32(min0, min1), vminq_f32(min2, min3));
    let max = vmaxq_f32(vmaxq_f32(max0, max1), vmaxq_f32(max2, max3));
    let mut min = vminvq_f32(min);
    let mut max = vmaxvq_f32(max);

    for ((residual, &x), &y) in res[offset..].iter_mut().zip(&x[offset..]).zip(&y[offset..]) {
        *residual = x - y;
        min = min.min(*residual);
        max = max.max(*residual);
    }
    (min, max)
}
