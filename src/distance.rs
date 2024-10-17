//! Compute the distance between vectors.

/// Distance metrics.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub enum Distance {
    /// L2 distance
    #[default]
    EuclideanSquared,
    /// Dot Product distance
    DotProduct,
}

fn euclidean_squared(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&l, &r)| (l - r) * (l - r))
        .sum()
}

fn neg_dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(&l, &r)| -l * r).sum()
}

/// Assign vectors to centroids.
pub fn assign(vecs: &[f32], centroids: &[f32], dim: usize, distance: Distance, labels: &mut [u32]) {
    let mut distances = vec![f32::MAX; centroids.len()];
    for (i, vec) in vecs.chunks(dim).enumerate() {
        for (j, centroid) in centroids.chunks(dim).enumerate() {
            distances[j] = match distance {
                Distance::EuclideanSquared => euclidean_squared(vec, centroid),
                Distance::DotProduct => neg_dot_product(vec, centroid),
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
    let mut elements = vec![0; centroids.len()];
    for (i, vec) in vecs.chunks(dim).enumerate() {
        let label = labels[i] as usize;
        elements[label] += 1;
        means[label * dim..(label + 1) * dim]
            .iter_mut()
            .zip(vec.iter())
            .for_each(|(m, &v)| *m += v);
    }
    for ((centroid, mean), &element) in centroids
        .chunks_mut(dim)
        .zip(means.chunks(dim))
        .zip(elements.iter())
    {
        if element == 0 {
            continue;
        }
        let divider = (element as f32).recip();
        centroid
            .iter_mut()
            .zip(mean.iter())
            .for_each(|(c, &m)| *c = m * divider);
    }
}
