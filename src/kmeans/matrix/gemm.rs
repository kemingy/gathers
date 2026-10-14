//! Backend for row-major assignment score tiles. Rayon owns outer parallelism.
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
    #[cfg(target_os = "macos")]
    if let (Ok(rows), Ok(cols), Ok(dim)) =
        (i32::try_from(rows), i32::try_from(cols), i32::try_from(dim))
    {
        accelerate::scores(vectors, centroids, scores, rows, cols, dim, scale);
        return;
    }
    // Other platforms, or shapes outside the legacy BLAS integer range, retain
    // the portable sequential backend. Do not truncate BLAS dimensions.
    portable(vectors, centroids, scores, rows, cols, dim, scale);
}

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

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod accelerate {
    use std::ffi::c_int;

    const ROW_MAJOR: c_int = 101;
    const NO_TRANSPOSE: c_int = 111;
    const TRANSPOSE: c_int = 112;

    // The LP64 symbol is available on older macOS versions as well as current
    // Apple Silicon. The newer LAPACK interface would raise the OS requirement.
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            rows: c_int,
            cols: c_int,
            dim: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }

    pub(crate) fn scores(
        vectors: &[f32],
        centroids: &[f32],
        scores: &mut [f32],
        rows: c_int,
        cols: c_int,
        dim: c_int,
        scale: f32,
    ) {
        // SAFETY: The parent validates complete row-major slices with lengths
        // rows*dim, cols*dim and rows*cols, and positive dimensions fitting c_int.
        // A and B are read-only and may alias; the exclusive output slice cannot
        // alias either input. All memory is initialized and f32-aligned; BLAS
        // requires no stronger alignment. Row-major B is transposed logically,
        // with its physical row stride still dim. beta=0 replaces the output.
        // This synchronous call retains no pointers. Accelerate dispatches CPU
        // features itself. Its thread policy is left to the system/caller; we
        // never mutate process-wide or caller-thread BLAS settings.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANSPOSE,
                TRANSPOSE,
                rows,
                cols,
                dim,
                scale,
                vectors.as_ptr(),
                dim,
                centroids.as_ptr(),
                dim,
                0.0,
                scores.as_mut_ptr(),
                cols,
            );
        }
    }
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
