//! Backend for row-major assignment score tiles. Rayon owns outer parallelism.
#[cfg(test)]
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};

/// Write `scale * vectors * centroids.transpose()` into a row-major score tile.
pub(crate) fn scores(
    vectors: &[f32],
    centroids: &[f32],
    scores: &mut [f32],
    dim: usize,
    scale: f32,
) {
    // Validate the layout before passing slice pointers to BLAS. These checks also
    // apply in release builds; a malformed internal call must never cause UB.
    assert!(dim > 0);
    assert_eq!(vectors.len() % dim, 0);
    assert_eq!(centroids.len() % dim, 0);
    let rows = vectors.len() / dim;
    let cols = centroids.len() / dim;
    assert_eq!(rows.checked_mul(cols), Some(scores.len()));
    if scores.is_empty() {
        return;
    }
    crate::gemm::matmul(
        MatMut::from_row_major_slice_mut(scores, rows, cols),
        Accum::Replace,
        MatRef::from_row_major_slice(vectors, rows, dim),
        MatRef::from_row_major_slice(centroids, cols, dim).transpose(),
        scale,
        Par::Seq,
    );
}

#[cfg(test)]
fn portable(
    vectors: &[f32],
    centroids: &[f32],
    scores: &mut [f32],
    rows: usize,
    cols: usize,
    dim: usize,
    scale: f32,
) {
    matmul(
        MatMut::from_row_major_slice_mut(scores, rows, cols),
        Accum::Replace,
        MatRef::from_row_major_slice(vectors, rows, dim),
        MatRef::from_row_major_slice(centroids, cols, dim).transpose(),
        scale,
        Par::Seq,
    );
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use seed_rand::seeded_rng;

    use super::{portable, scores};

    #[test]
    fn score_backends_match_direct_rectangular_products() {
        let mut rng = seeded_rng();
        // Include single rows, vector tails, and the real-data dimensions.
        for (rows, cols, dim) in [
            (1, 3, 1),
            (3, 5, 33),
            (65, 17, 65),
            (2, 3, 768),
            (3, 2, 960),
        ] {
            let vectors: Vec<f32> = (0..rows * dim)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let centroids: Vec<f32> = (0..cols * dim)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            for scale in [-1.0, -2.0] {
                let mut actual = vec![f32::NAN; rows * cols];
                let mut fallback = actual.clone();
                scores(&vectors, &centroids, &mut actual, dim, scale);
                portable(&vectors, &centroids, &mut fallback, rows, cols, dim, scale);
                for row in 0..rows {
                    for col in 0..cols {
                        let terms = vectors[row * dim..][..dim]
                            .iter()
                            .zip(&centroids[col * dim..][..dim]);
                        let expected: f64 = terms
                            .clone()
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum::<f64>()
                            * f64::from(scale);
                        let magnitude: f64 = terms
                            .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
                            .sum::<f64>()
                            * f64::from(scale.abs());
                        // An absolute-plus-relative reduction bound, including cancellation.
                        let tolerance =
                            1e-6 + 4.0 * dim as f64 * f64::from(f32::EPSILON) * magnitude;
                        for value in [actual[row * cols + col], fallback[row * cols + col]] {
                            assert!(
                                (f64::from(value) - expected).abs() <= tolerance,
                                "shape {rows}x{cols}x{dim}: {value} != {expected}"
                            );
                        }
                    }
                }
            }
        }
        scores(&[], &[1.0; 3], &mut [], 3, -1.0);
        scores(&[1.0; 3], &[], &mut [], 3, -2.0);
    }
}
