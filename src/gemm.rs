//! Shared dense f32 matrix products, with checked Accelerate dispatch on macOS.

use faer::{Accum, MatMut, MatRef, Par};

/// Compute `dst = alpha * lhs * rhs`, optionally adding the previous destination.
/// Unsupported BLAS strides or dimensions retain faer's portable implementation.
pub(crate) fn matmul(
    dst: MatMut<'_, f32>,
    accum: Accum,
    lhs: MatRef<'_, f32>,
    rhs: MatRef<'_, f32>,
    alpha: f32,
    parallelism: Par,
) {
    assert_eq!(lhs.ncols(), rhs.nrows());
    assert_eq!(dst.nrows(), lhs.nrows());
    assert_eq!(dst.ncols(), rhs.ncols());
    if dst.nrows() == 0 || dst.ncols() == 0 {
        return;
    }
    #[cfg(target_os = "macos")]
    if accelerate::matmul(&dst, accum, lhs, rhs, alpha).is_some() {
        return;
    }
    faer::linalg::matmul::matmul(dst, accum, lhs, rhs, alpha, parallelism);
}

/// Compute the lower triangle of `dst = alpha * input.transpose() * input`.
/// The upper triangle is unspecified; unsupported BLAS layouts use the full GEMM fallback.
pub(crate) fn symmetric_rank_k_lower(
    mut dst: MatMut<'_, f32>,
    input: MatRef<'_, f32>,
    alpha: f32,
    parallelism: Par,
) {
    assert_eq!(dst.nrows(), input.ncols());
    assert_eq!(dst.ncols(), input.ncols());
    if dst.nrows() == 0 {
        return;
    }
    #[cfg(target_os = "macos")]
    if accelerate::symmetric_rank_k_lower(dst.as_mut(), input, alpha).is_some() {
        return;
    }
    matmul(
        dst.as_mut(),
        Accum::Replace,
        input.transpose(),
        input,
        alpha,
        parallelism,
    );
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod accelerate {
    use std::ffi::c_int;

    use faer::{Accum, MatMut, MatRef};

    const ROW_MAJOR: c_int = 101;
    const NO_TRANSPOSE: c_int = 111;
    const TRANSPOSE: c_int = 112;
    const LOWER: c_int = 122;

    // Use the legacy LP64 symbol: the newer LAPACK interface raises the minimum macOS version.
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_ssyrk(
            order: c_int,
            uplo: c_int,
            trans: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
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

    fn layout(matrix: MatRef<'_, f32>) -> Option<(c_int, c_int)> {
        if matrix.col_stride() == 1 && matrix.row_stride() >= matrix.ncols() as isize {
            Some((NO_TRANSPOSE, c_int::try_from(matrix.row_stride()).ok()?))
        } else if matrix.row_stride() == 1 && matrix.col_stride() >= matrix.nrows() as isize {
            Some((TRANSPOSE, c_int::try_from(matrix.col_stride()).ok()?))
        } else {
            None
        }
    }

    pub(crate) fn symmetric_rank_k_lower(
        dst: MatMut<'_, f32>,
        input: MatRef<'_, f32>,
        alpha: f32,
    ) -> Option<()> {
        let n = c_int::try_from(dst.nrows()).ok()?;
        let k = c_int::try_from(input.nrows()).ok()?;
        if k == 0
            || dst.col_stride() != 1
            || dst.row_stride() < isize::try_from(dst.ncols()).ok()?
        {
            return None;
        }
        let ldc = c_int::try_from(dst.row_stride()).ok()?;
        let (trans, lda) = layout(input.transpose())?;
        // SAFETY: The caller checked the square output shape against input.ncols().
        // MatRef/MatMut establish initialized f32 elements and exclusive output storage,
        // disjoint from input. Checked unit minor strides and positive leading dimensions
        // describe the same physical bounds to BLAS, including transposed input layouts.
        // Dimensions/strides fit c_int; BLAS requires only f32 alignment and retains no
        // pointers. SYRK replaces only the lower triangle, as required by the helper's
        // contract; callers must not read the unspecified upper triangle. Accelerate owns
        // CPU/thread dispatch.
        unsafe {
            cblas_ssyrk(
                ROW_MAJOR,
                LOWER,
                trans,
                n,
                k,
                alpha,
                input.as_ptr(),
                lda,
                0.0,
                dst.as_ptr_mut(),
                ldc,
            );
        }
        Some(())
    }

    pub(crate) fn matmul(
        dst: &MatMut<'_, f32>,
        accum: Accum,
        lhs: MatRef<'_, f32>,
        rhs: MatRef<'_, f32>,
        alpha: f32,
    ) -> Option<()> {
        let rows = c_int::try_from(dst.nrows()).ok()?;
        let cols = c_int::try_from(dst.ncols()).ok()?;
        let dim = c_int::try_from(lhs.ncols()).ok()?;
        if dim == 0
            || dst.col_stride() != 1
            || dst.row_stride() < isize::try_from(dst.ncols()).ok()?
        {
            return None;
        }
        let ldc = c_int::try_from(dst.row_stride()).ok()?;
        let (trans_a, lda) = layout(lhs)?;
        let (trans_b, ldb) = layout(rhs)?;
        let beta = match accum {
            Accum::Replace => 0.0,
            Accum::Add => 1.0,
        };
        // SAFETY: The parent checked product/output shapes. MatRef/MatMut establish valid,
        // initialized f32 memory for every indexed element; unit minor strides and positive
        // leading dimensions checked above express the same bounds to row-major BLAS.
        // Transposed inputs retain their physical leading dimension. All dimensions and
        // strides fit c_int. Inputs may alias each other; MatMut exclusively owns the output,
        // so it cannot alias either input. BLAS needs no alignment beyond f32. The synchronous
        // call retains no pointers. Accelerate handles CPU dispatch and its own thread policy;
        // this wrapper never changes process-wide or caller-thread BLAS settings.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                trans_a,
                trans_b,
                rows,
                cols,
                dim,
                alpha,
                lhs.as_ptr(),
                lda,
                rhs.as_ptr(),
                ldb,
                beta,
                dst.as_ptr_mut(),
                ldc,
            );
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use faer::{Accum, MatMut, MatRef, Par};

    use super::{matmul, symmetric_rank_k_lower};

    #[test]
    fn symmetric_products_cover_layouts_tails_and_empty_reductions() {
        for (rows, dim) in [(3, 0), (0, 3), (1, 1), (7, 3), (33, 5)] {
            let data: Vec<f32> = (0..rows * dim)
                .map(|index| (index as f32 - 7.0) * 0.125)
                .collect();
            for transpose in [false, true] {
                let input = if transpose {
                    MatRef::from_row_major_slice(&data, dim, rows).transpose()
                } else {
                    MatRef::from_row_major_slice(&data, rows, dim)
                };
                let stride = dim + 2;
                let mut actual = vec![f32::NAN; dim * stride];
                symmetric_rank_k_lower(
                    MatMut::from_row_major_slice_mut(&mut actual, dim, stride).subcols_mut(0, dim),
                    input,
                    0.5,
                    Par::Seq,
                );
                for row in 0..dim {
                    for col in 0..=row {
                        let expected = (0..rows)
                            .map(|k| f64::from(input[(k, row)]) * f64::from(input[(k, col)]))
                            .sum::<f64>()
                            * 0.5;
                        assert!(
                            (f64::from(actual[row * stride + col]) - expected).abs()
                                <= 1e-5 + 1e-5 * expected.abs()
                        );
                    }
                    assert!(
                        actual[row * stride + dim..(row + 1) * stride]
                            .iter()
                            .all(|value| value.is_nan())
                    );
                }
            }
        }
        // A column-major destination exercises the portable fallback even on macOS.
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut actual = [f32::NAN; 4];
        symmetric_rank_k_lower(
            MatMut::from_column_major_slice_mut(&mut actual, 2, 2),
            MatRef::from_row_major_slice(&input, 3, 2),
            1.0,
            Par::Seq,
        );
        assert_eq!(actual, [35.0, 44.0, 44.0, 56.0]);
    }

    #[test]
    fn matrix_layouts_and_accumulation_match_direct_products() {
        for (rows, cols, dim) in [(1, 3, 1), (3, 5, 7), (5, 2, 33)] {
            for transpose_a in [false, true] {
                for transpose_b in [false, true] {
                    let a: Vec<f32> = (0..rows * dim).map(|i| (i as f32 - 4.0) * 0.125).collect();
                    let b: Vec<f32> = (0..dim * cols).map(|i| (i as f32 - 2.0) * 0.25).collect();
                    let lhs = if transpose_a {
                        MatRef::from_row_major_slice(&a, dim, rows).transpose()
                    } else {
                        MatRef::from_row_major_slice(&a, rows, dim)
                    };
                    let rhs = if transpose_b {
                        MatRef::from_row_major_slice(&b, cols, dim).transpose()
                    } else {
                        MatRef::from_row_major_slice(&b, dim, cols)
                    };
                    for accum in [Accum::Replace, Accum::Add] {
                        // Padded rows exercise BLAS leading dimensions and preserve the padding.
                        let stride = cols + 2;
                        let mut actual = vec![3.0; rows * stride];
                        matmul(
                            MatMut::from_row_major_slice_mut(&mut actual, rows, stride)
                                .subcols_mut(0, cols),
                            accum,
                            lhs,
                            rhs,
                            -0.5,
                            Par::Seq,
                        );
                        for row in 0..rows {
                            for col in 0..cols {
                                let expected = (0..dim)
                                    .map(|k| f64::from(lhs[(row, k)]) * f64::from(rhs[(k, col)]))
                                    .sum::<f64>()
                                    * -0.5
                                    + if matches!(accum, Accum::Add) {
                                        3.0
                                    } else {
                                        0.0
                                    };
                                assert!(
                                    (f64::from(actual[row * stride + col]) - expected).abs()
                                        < 1e-5 + 1e-5 * expected.abs()
                                );
                            }
                            assert_eq!(&actual[row * stride + cols..(row + 1) * stride], &[3.0; 2]);
                        }
                    }
                }
            }
        }
        // Column-major destination takes the portable fallback on macOS too.
        let a = [1.0, 2.0, 3.0, 4.0];
        let mut out = [f32::NAN; 4];
        matmul(
            MatMut::from_column_major_slice_mut(&mut out, 2, 2),
            Accum::Replace,
            MatRef::from_row_major_slice(&a, 2, 2),
            MatRef::from_row_major_slice(&a, 2, 2),
            1.0,
            Par::Seq,
        );
        assert_eq!(out, [7.0, 15.0, 10.0, 22.0]);
    }
}
