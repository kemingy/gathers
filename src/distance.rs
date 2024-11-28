//! Compute the distance between vectors.

use core::f32;

/// Distance metrics.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub enum Distance {
    /// L2 distance
    #[default]
    SquaredEuclidean,
    /// Dot Product distance
    NegativeDotProduct,
}

/// Native implementation of l2 norm.
pub fn native_l2_norm(vec: &[f32]) -> f32 {
    vec.iter().fold(0.0, |acc, &x| acc + x * x).sqrt()
}

/// Compute the L2 norm of the vector.
#[inline]
pub fn l2_norm(vec: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { crate::simd::l2_norm(vec) }
        } else {
            native_l2_norm(vec)
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        native_l2_norm(vec)
    }
}

/// Native implementation of squared euclidean distance.
#[inline]
pub fn native_squared_euclidean(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&l, &r)| (l - r) * (l - r))
        .sum()
}

/// Compute the squared Euclidean distance between two vectors.
#[inline]
pub fn squared_euclidean(lhs: &[f32], rhs: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { crate::simd::l2_squared_distance(lhs, rhs) }
        } else {
            native_squared_euclidean(lhs, rhs)
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        native_squared_euclidean(lhs, rhs)
    }
}

/// Native implementation of negative dot product.
#[inline]
pub fn native_dot_produce(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&l, &r)| l * r)
        .sum::<f32>()
}

/// Compute the negative dot product between two vectors.
#[inline]
pub fn neg_dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { -crate::simd::dot_product(lhs, rhs) }
        } else {
            -native_dot_produce(lhs, rhs)
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        -native_dot_produce(lhs, rhs)
    }
}

/// Native implementation of argmin.
#[inline]
pub fn native_argmin(vec: &[f32]) -> usize {
    let mut minimal = f32::MAX;
    let mut index = 0;
    for (i, &val) in vec.iter().enumerate() {
        if val < minimal {
            minimal = val;
            index = i;
        }
    }
    index
}

/// Find the index of the minimum value in the vector.
#[inline]
pub fn argmin(vec: &[f32]) -> usize {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { crate::simd::argmin(vec) }
        } else {
            native_argmin(vec)
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        native_argmin(vec)
    }
}

#[cfg(test)]
mod test {
    use rand::{thread_rng, Rng};

    use super::{
        argmin, l2_norm, native_argmin, native_dot_produce, native_l2_norm,
        native_squared_euclidean, neg_dot_product, squared_euclidean,
    };

    #[test]
    fn test_l2_squared_distance() {
        let mut rng = thread_rng();

        for dim in [4, 12, 64, 70, 78].into_iter() {
            let lhs = (0..dim).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
            let rhs = (0..dim).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
            let diff = squared_euclidean(&lhs, &rhs) - native_squared_euclidean(&lhs, &rhs);
            assert!(diff.abs() < 1e-5, "diff: {} for dim: {}", diff, dim);
        }
    }

    #[test]
    fn test_dot_product_distance() {
        let mut rng = thread_rng();

        for dim in [4, 12, 64, 70, 78].into_iter() {
            let lhs = (0..dim).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
            let rhs = (0..dim).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
            let diff = neg_dot_product(&lhs, &rhs) + native_dot_produce(&lhs, &rhs);
            assert!(diff.abs() < 1e-5, "diff: {} for dim: {}", diff, dim);
        }
    }

    #[test]
    fn test_l2_norm() {
        let mut rng = thread_rng();
        for dim in [4, 12, 64, 70, 78].into_iter() {
            let vec = (0..dim).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
            let diff = l2_norm(&vec) - native_l2_norm(&vec);
            assert!(diff.abs() < 1e-5, "diff: {} for dim: {}", diff, dim);
        }
    }

    #[test]
    fn test_argmin() {
        let mut rng = thread_rng();
        for dim in [12, 32, 128, 140].into_iter() {
            let vec = (0..dim).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
            assert_eq!(argmin(&vec), native_argmin(&vec));
        }
    }
}
