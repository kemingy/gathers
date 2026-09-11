//! Accelerate distance computations with SIMD.

pub mod pulp;

#[cfg(test)]
#[allow(unsafe_code)]
/// Legacy AVX entry points used only by correctness tests.
pub mod legacy {
    /// Compute squared Euclidean distance using the legacy AVX dispatch entry point.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn l2_squared_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma"));
        unsafe { l2_squared_distance_unchecked(lhs, rhs) }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "fma,avx")]
    unsafe fn l2_squared_distance_unchecked(lhs: &[f32], rhs: &[f32]) -> f32 {
        struct Impl<'a> {
            lhs: &'a [f32],
            rhs: &'a [f32],
        }
        impl ::pulp::WithSimd for Impl<'_> {
            type Output = f32;

            #[inline(always)]
            fn with_simd<S: ::pulp::Simd>(self, simd: S) -> Self::Output {
                crate::simd::pulp::l2_squared_distance(simd, self.lhs, self.rhs)
            }
        }
        ::pulp::Arch::new().dispatch(Impl { lhs, rhs })
    }

    /// Compute a dot product using the legacy AVX dispatch entry point.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma"));
        unsafe { dot_product_unchecked(lhs, rhs) }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "fma,avx")]
    unsafe fn dot_product_unchecked(lhs: &[f32], rhs: &[f32]) -> f32 {
        struct Impl<'a> {
            lhs: &'a [f32],
            rhs: &'a [f32],
        }
        impl ::pulp::WithSimd for Impl<'_> {
            type Output = f32;

            #[inline(always)]
            fn with_simd<S: ::pulp::Simd>(self, simd: S) -> Self::Output {
                crate::simd::pulp::dot_product(simd, self.lhs, self.rhs)
            }
        }
        ::pulp::Arch::new().dispatch(Impl { lhs, rhs })
    }

    /// Compute an L2 norm using the legacy AVX dispatch entry point.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn l2_norm(vec: &[f32]) -> f32 {
        assert!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma"));
        unsafe { l2_norm_unchecked(vec) }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "fma,avx")]
    unsafe fn l2_norm_unchecked(vec: &[f32]) -> f32 {
        struct Impl<'a>(&'a [f32]);
        impl ::pulp::WithSimd for Impl<'_> {
            type Output = f32;

            #[inline(always)]
            fn with_simd<S: ::pulp::Simd>(self, simd: S) -> Self::Output {
                crate::simd::pulp::l2_norm(simd, self.0)
            }
        }
        ::pulp::Arch::new().dispatch(Impl(vec))
    }

    /// Return the minimum index using the legacy AVX dispatch entry point.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn argmin(vec: &[f32]) -> usize {
        assert!(is_x86_feature_detected!("avx"));
        unsafe { argmin_unchecked(vec) }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx")]
    unsafe fn argmin_unchecked(vec: &[f32]) -> usize {
        struct Impl<'a>(&'a [f32]);
        impl ::pulp::WithSimd for Impl<'_> {
            type Output = usize;

            #[inline(always)]
            fn with_simd<S: ::pulp::Simd>(self, simd: S) -> Self::Output {
                crate::simd::pulp::argmin(simd, self.0)
            }
        }
        ::pulp::Arch::new().dispatch(Impl(vec))
    }
}
