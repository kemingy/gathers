//! Accelerate distance computations with SIMD.

pub mod pulp;

/// Runtime detection for AVX2 support.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub struct Avx2;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
impl Avx2 {
    /// Return whether AVX2 is available on the current CPU.
    #[inline]
    pub fn is_available() -> bool {
        std::is_x86_feature_detected!("avx2")
    }
}

/// Compute the squared Euclidean distance between two vectors.
///
/// # Safety
///
/// The caller must ensure that AVX is available.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "fma,avx")]
#[inline]
pub unsafe fn l2_squared_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    struct Impl<'a> {
        lhs: &'a [f32],
        rhs: &'a [f32],
    }
    impl ::pulp::WithSimd for Impl<'_> {
        type Output = f32;
        #[inline(always)]
        fn with_simd<S: ::pulp::Simd>(self, simd: S) -> Self::Output {
            pulp::l2_squared_distance(simd, self.lhs, self.rhs)
        }
    }
    ::pulp::Arch::new().dispatch(Impl { lhs, rhs })
}

/// Compute the dot product of two vectors.
///
/// # Safety
///
/// The caller must ensure that AVX is available.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "fma,avx")]
#[inline]
pub unsafe fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    struct Impl<'a> {
        lhs: &'a [f32],
        rhs: &'a [f32],
    }
    impl ::pulp::WithSimd for Impl<'_> {
        type Output = f32;
        #[inline(always)]
        fn with_simd<S: ::pulp::Simd>(self, simd: S) -> Self::Output {
            pulp::dot_product(simd, self.lhs, self.rhs)
        }
    }
    ::pulp::Arch::new().dispatch(Impl { lhs, rhs })
}

/// Compute the L2 norm of a vector.
///
/// # Safety
///
/// The caller must ensure that AVX is available.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "fma,avx")]
#[inline]
pub unsafe fn l2_norm(vec: &[f32]) -> f32 {
    struct Impl<'a>(&'a [f32]);
    impl ::pulp::WithSimd for Impl<'_> {
        type Output = f32;
        #[inline(always)]
        fn with_simd<S: ::pulp::Simd>(self, simd: S) -> Self::Output {
            pulp::l2_norm(simd, self.0)
        }
    }
    ::pulp::Arch::new().dispatch(Impl(vec))
}

/// Return the index of the minimum value.
///
/// # Safety
///
/// The caller must ensure that AVX is available.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn argmin(vec: &[f32]) -> usize {
    struct Impl<'a>(&'a [f32]);
    impl ::pulp::WithSimd for Impl<'_> {
        type Output = usize;
        #[inline(always)]
        fn with_simd<S: ::pulp::Simd>(self, simd: S) -> Self::Output {
            pulp::argmin(simd, self.0)
        }
    }
    ::pulp::Arch::new().dispatch(Impl(vec))
}
