//! Utility functions for manipulating vectors and reading/writing files.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use aligned_vec::{AVec, avec};
use num_traits::{AsPrimitive, Float, FromBytes, FromPrimitive, Num, NumAssign, ToBytes};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

/// Center vectors in-place and return the mean subtracted from each vector.
pub fn centroid_residual<T>(vecs: &mut [T], dim: usize) -> Vec<T>
where
    T: Float + AsPrimitive<f64> + FromPrimitive + NumAssign + Copy,
{
    assert!(!vecs.is_empty());
    let n = vecs.len() / dim;
    let mut mean = vec![0.0f64; dim];

    for vec in vecs.chunks(dim) {
        for (m, v) in mean.iter_mut().zip(vec.iter()) {
            *m += v.as_();
        }
    }
    let mean = mean
        .into_iter()
        .map(|value| T::from_f64(value / n as f64).unwrap())
        .collect::<Vec<_>>();
    for vec in vecs.chunks_mut(dim) {
        for (m, v) in mean.iter().zip(vec.iter_mut()) {
            *v -= *m;
        }
    }
    mean
}

/// Convert a 2-D `Vec<Vec<T>>` to a 1-D continuous aligned vector.
#[inline]
pub fn as_continuous_vec<T>(mat: &[impl AsRef<[T]>]) -> AVec<T>
where
    T: Num + Copy,
{
    let len = mat.iter().map(|v| v.as_ref().len()).sum();
    let mut vec = avec!(T::zero(); len);
    for (i, v) in mat.iter().enumerate() {
        vec[i * v.as_ref().len()..(i + 1) * v.as_ref().len()].copy_from_slice(v.as_ref());
    }
    vec
}

/// Convert a 1-D continuous vector to a 2-D `Vec<Vec<T>>`.
#[inline]
pub fn as_matrix<T>(vecs: &[T], dim: usize) -> Vec<Vec<T>>
where
    T: Num + Copy,
{
    vecs.chunks(dim).map(|chunk| chunk.to_vec()).collect()
}

/// Normalize vectors in-place.
pub fn normalize<T>(vec: &mut [T])
where
    T: Float + Copy,
{
    let norm_squared = vec.iter().fold(T::zero(), |acc, &x| acc + x * x);
    let divider = norm_squared.sqrt().recip();
    for x in vec.iter_mut() {
        *x = *x * divider;
    }
}

/// Error returned by [`try_normalize_rows`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NormalizeRowsError {
    /// The dimension is zero or the flat input does not contain complete rows.
    #[error("dimension must be positive and rows complete")]
    InvalidLayout,
    /// A row has a zero or non-finite L2 norm.
    #[error("row {0} must have a finite nonzero L2 norm")]
    InvalidNorm(usize),
    /// A row has a non-finite L2 norm.
    #[error("row {0} must have a finite L2 norm")]
    NonFiniteNorm(usize),
}

// Return whether the row had a nonzero norm; None means its norm was non-finite.
pub(crate) fn normalize_nonzero_row(vector: &mut [f32]) -> Option<bool> {
    let squared_norm = vector
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    if !squared_norm.is_finite() {
        return None;
    }
    if squared_norm == 0.0 {
        return Some(false);
    }
    let inverse_norm = squared_norm.sqrt().recip();
    let inverse_norm_f32 = inverse_norm as f32;
    if inverse_norm_f32.is_finite() {
        vector
            .iter_mut()
            .for_each(|value| *value *= inverse_norm_f32);
    } else {
        vector
            .iter_mut()
            .for_each(|value| *value = (f64::from(*value) * inverse_norm) as f32);
    }
    Some(true)
}

/// Normalize complete `f32` rows in place, rejecting non-finite norms.
///
/// Zero rows are preserved when `allow_zero` is true and rejected otherwise. Norms are
/// accumulated in `f64` so finite `f32` components do not overflow the sum.
/// Rows are processed in parallel; on error, some other rows may already be normalized.
pub fn try_normalize_rows(
    vecs: &mut [f32],
    dim: usize,
    allow_zero: bool,
) -> Result<(), NormalizeRowsError> {
    if dim == 0 || !vecs.len().is_multiple_of(dim) {
        return Err(NormalizeRowsError::InvalidLayout);
    }
    vecs.par_chunks_exact_mut(dim)
        .enumerate()
        .try_for_each(|(row, vector)| match normalize_nonzero_row(vector) {
            Some(true) => Ok(()),
            Some(false) if allow_zero => Ok(()),
            None if allow_zero => Err(NormalizeRowsError::NonFiniteNorm(row)),
            _ => Err(NormalizeRowsError::InvalidNorm(row)),
        })
}

/// Read the fvces/ivces file.
pub fn read_vecs<T>(path: &Path) -> std::io::Result<Vec<Vec<T>>>
where
    T: Sized + FromBytes<Bytes = [u8; 4]>,
{
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buf = [0u8; 4];
    let mut count: usize;
    let mut vecs = Vec::new();
    loop {
        count = reader.read(&mut buf)?;
        if count == 0 {
            break;
        }
        let dim = u32::from_le_bytes(buf) as usize;
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            reader.read_exact(&mut buf)?;
            vec.push(T::from_le_bytes(&buf));
        }
        vecs.push(vec);
    }
    Ok(vecs)
}

/// Write the fvecs/ivecs file.
pub fn write_vecs<T>(path: &Path, vecs: &[impl AsRef<[T]>]) -> std::io::Result<()>
where
    T: Sized + ToBytes,
{
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for vec in vecs.iter() {
        writer.write_all(&(vec.as_ref().len() as u32).to_le_bytes())?;
        for v in vec.as_ref().iter() {
            writer.write_all(T::to_le_bytes(v).as_ref())?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Write the fvecs/ivecs file from DMatrix.
pub fn write_matrix<T>(path: &Path, matrix: &faer::MatRef<T>) -> std::io::Result<()>
where
    T: Sized + ToBytes,
{
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for vec in matrix.row_iter() {
        writer.write_all(&(vec.ncols() as u32).to_le_bytes())?;
        for i in 0..vec.ncols() {
            writer.write_all(T::to_le_bytes(vec.get(i)).as_ref())?;
        }
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{NormalizeRowsError, try_normalize_rows};

    #[test]
    fn checked_row_normalization_validates_layout_and_norms() {
        let mut vectors = [3.0_f32, 4.0, 5.0, 12.0];
        try_normalize_rows(&mut vectors, 2, false).unwrap();
        for (&actual, expected) in vectors.iter().zip([0.6, 0.8, 5.0 / 13.0, 12.0 / 13.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(
            try_normalize_rows(&mut vectors, 0, false),
            Err(NormalizeRowsError::InvalidLayout)
        );
        assert_eq!(
            try_normalize_rows(&mut vectors, 3, false),
            Err(NormalizeRowsError::InvalidLayout)
        );
        assert_eq!(
            try_normalize_rows(&mut [1.0, 0.0, 0.0, 0.0], 2, false),
            Err(NormalizeRowsError::InvalidNorm(1))
        );
        assert_eq!(
            try_normalize_rows(&mut [f32::NAN, 0.0], 2, false),
            Err(NormalizeRowsError::InvalidNorm(0))
        );
        assert_eq!(
            NormalizeRowsError::InvalidNorm(0).to_string(),
            "row 0 must have a finite nonzero L2 norm"
        );
        let mut tiny = [f32::from_bits(1), 0.0];
        try_normalize_rows(&mut tiny, 2, false).unwrap();
        assert_eq!(tiny, [1.0, 0.0]);
    }

    #[test]
    fn centroid_normalization_preserves_zero_rows() {
        let mut vectors = [0.0, 0.0, 3.0, 4.0];
        try_normalize_rows(&mut vectors, 2, true).unwrap();
        assert_eq!(&vectors[..2], &[0.0, 0.0]);
        assert!((vectors[2] - 0.6).abs() < 1e-6);
        assert!((vectors[3] - 0.8).abs() < 1e-6);
        assert_eq!(
            try_normalize_rows(&mut [f32::NAN, 0.0], 2, true),
            Err(NormalizeRowsError::NonFiniteNorm(0))
        );
    }
}
