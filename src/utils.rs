//! Utility functions for manipulating vectors and reading/writing files.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use num_traits::{AsPrimitive, Float, FromBytes, FromPrimitive, Num, NumAssign, ToBytes};

/// Calculate the centroid of a set of vectors and subtract it from each vector.
pub fn centroid_residual<T>(vecs: &mut [T], dim: usize)
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
    mean.iter_mut().for_each(|m| *m /= n as f64);
    for vec in vecs.chunks_mut(dim) {
        for (m, v) in mean.iter().zip(vec.iter_mut()) {
            *v -= T::from_f64(*m).unwrap();
        }
    }
}

/// Convert a 2-D Vec<Vec<T>> to a 1-D continuous vector.
#[inline]
pub fn as_continuous_vec<T>(mat: &[Vec<T>]) -> Vec<T>
where
    T: Num + Copy,
{
    mat.iter().flat_map(|v| v.iter().cloned()).collect()
}

/// Convert a 1-D continuous vector to a 2-D Vec<Vec<T>>.
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
    T: Sized + ToBytes + faer::Entity,
{
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for vec in matrix.row_iter() {
        writer.write_all(&(vec.ncols() as u32).to_le_bytes())?;
        for i in 0..vec.ncols() {
            writer.write_all(T::to_le_bytes(&vec.read(i)).as_ref())?;
        }
    }
    writer.flush()?;
    Ok(())
}
