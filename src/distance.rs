//! Compute the distance between vectors.

use log::debug;
use rand::{thread_rng, Rng};

const EPS: f32 = 1.0 / 1024.0;

/// Distance metrics.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub enum Distance {
    /// L2 distance
    #[default]
    SquaredEuclidean,
    /// Dot Product distance
    NegativeDotProduct,
}

/// Native implementation of squared euclidean distance.
pub fn native_squared_euclidean(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&l, &r)| (l - r) * (l - r))
        .sum()
}

fn squared_euclidean(lhs: &[f32], rhs: &[f32]) -> f32 {
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
pub fn native_neg_dot_produce(lhs: &[f32], rhs: &[f32]) -> f32 {
    -lhs.iter()
        .zip(rhs.iter())
        .map(|(&l, &r)| l * r)
        .sum::<f32>()
}

fn neg_dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { crate::simd::neg_dot_product(lhs, rhs) }
        } else {
            native_neg_dot_produce(lhs, rhs)
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        native_neg_dot_produce(lhs, rhs)
    }
}

/// Assign vectors to centroids.
pub fn assign(vecs: &[f32], centroids: &[f32], dim: usize, distance: Distance, labels: &mut [u32]) {
    let mut distances = vec![f32::MAX; centroids.len()];
    for (i, vec) in vecs.chunks(dim).enumerate() {
        for (j, centroid) in centroids.chunks(dim).enumerate() {
            distances[j] = match distance {
                Distance::SquaredEuclidean => squared_euclidean(vec, centroid),
                Distance::NegativeDotProduct => neg_dot_product(vec, centroid),
            };
            if j == 0 || distances[j] < distances[labels[i] as usize] {
                labels[i] = j as u32;
            }
        }
    }
}

/// Update centroids to the mean of assigned vectors.
pub fn update_centroids(vecs: &[f32], centroids: &mut [f32], dim: usize, labels: &[u32]) {
    let mut means = vec![0.0; centroids.len()];
    let mut elements = vec![0; centroids.len() / dim];
    for (i, vec) in vecs.chunks(dim).enumerate() {
        let label = labels[i] as usize;
        elements[label] += 1;
        means[label * dim..(label + 1) * dim]
            .iter_mut()
            .zip(vec.iter())
            .for_each(|(m, &v)| *m += v);
    }

    for i in 0..elements.len() {
        if elements[i] == 0 {
            // need to split another cluster to fill this empty cluster
            let mut target = 0;
            let mut rng = thread_rng();
            let base = 1.0 / (vecs.len() / dim - labels.len()) as f32;
            loop {
                let p = (elements[target] - 1) as f32 * base;
                if rng.gen::<f32>() < p {
                    break;
                }
                target = (target + 1) % labels.len();
            }
            debug!("split cluster {} to fill empty cluster {}", target, i);
            if i < target {
                let (left, right) = centroids.split_at_mut(target * dim);
                left[i * dim..(i + 1) * dim].copy_from_slice(&right[..dim]);
            } else {
                let (left, right) = centroids.split_at_mut(i * dim);
                right[..dim].copy_from_slice(&left[target * dim..(target + 1) * dim]);
            }
            // small symmetric perturbation
            for j in 0..dim {
                if j % 2 == 0 {
                    centroids[i * dim + j] *= 1.0 + EPS;
                    centroids[target * dim + j] *= 1.0 - EPS;
                } else {
                    centroids[i * dim + j] *= 1.0 - EPS;
                    centroids[target * dim + j] *= 1.0 + EPS;
                }
            }
            // update elements
            elements[i] = elements[target] / 2;
            elements[target] -= elements[i];
        }
        let divider = (elements[i] as f32).recip();
        for j in i * dim..(i + 1) * dim {
            centroids[j] = means[j] * divider;
        }
    }
}

#[cfg(test)]
mod test {
    use super::{native_neg_dot_produce, native_squared_euclidean};
    use crate::simd::{l2_squared_distance, neg_dot_product};
    use rand::{thread_rng, Rng};

    #[test]
    fn test_l2_squared_distance() {
        let mut rng = thread_rng();

        for dim in [4, 12, 64, 70, 78].into_iter() {
            let lhs = (0..dim).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
            let rhs = (0..dim).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
            let diff =
                unsafe { l2_squared_distance(&lhs, &rhs) } - native_squared_euclidean(&lhs, &rhs);
            assert!(diff.abs() < 1e-5, "diff: {} for dim: {}", diff, dim);
        }
    }

    #[test]
    fn test_negative_dot_product_distance() {
        let mut rng = thread_rng();

        for dim in [4, 12, 64, 70, 78].into_iter() {
            let lhs = (0..dim).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
            let rhs = (0..dim).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
            let diff = unsafe { neg_dot_product(&lhs, &rhs) } - native_neg_dot_produce(&lhs, &rhs);
            assert!(diff.abs() < 1e-5, "diff: {} for dim: {}", diff, dim);
        }
    }
}
