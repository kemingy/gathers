//! Validated fvecs I/O into a flat aligned buffer shared by CLI commands.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use aligned_vec::AVec;

use crate::invalid;

pub(crate) struct Vectors {
    pub(crate) data: AVec<f32>,
    pub(crate) dim: usize,
}

impl Vectors {
    pub(crate) fn len(&self) -> usize {
        self.data.len() / self.dim
    }
}

pub(crate) fn read(path: &Path, limit: Option<usize>) -> io::Result<Vectors> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    read_rows(BufReader::new(file), bytes, limit)
}

fn read_rows(mut reader: impl Read, bytes: u64, limit: Option<usize>) -> io::Result<Vectors> {
    let mut header = [0; 4];
    reader.read_exact(&mut header)?;
    let dim = u32::from_le_bytes(header) as usize;
    if dim == 0 {
        return Err(invalid("fvecs dimension must be positive"));
    }
    let row_bytes = (dim as u64 + 1) * 4;
    if !bytes.is_multiple_of(row_bytes) {
        return Err(invalid("fvecs input contains an incomplete row"));
    }
    let available =
        usize::try_from(bytes / row_bytes).map_err(|_| invalid("too many fvecs rows"))?;
    let rows = limit.unwrap_or(available);
    if rows == 0 || rows > available {
        return Err(invalid(
            "requested row count must be positive and fit in the input",
        ));
    }
    let values = rows
        .checked_mul(dim)
        .ok_or_else(|| invalid("input shape overflow"))?;
    let mut data = AVec::new(64);
    data.try_reserve_exact(values)
        .map_err(|error| io::Error::other(format!("cannot allocate vector buffer: {error:?}")))?;
    data.resize(values, 0.0_f32);
    for (index, row) in data.chunks_exact_mut(dim).enumerate() {
        if index > 0 {
            reader.read_exact(&mut header)?;
            if u32::from_le_bytes(header) as usize != dim {
                return Err(invalid("fvecs rows must have the same dimension"));
            }
        }
        reader.read_exact(bytemuck::cast_slice_mut(row))?;
        for value in row {
            *value = f32::from_bits(u32::from_le(value.to_bits()));
            if !value.is_finite() {
                return Err(invalid("fvecs coordinates must be finite"));
            }
        }
    }
    Ok(Vectors { data, dim })
}

pub(crate) fn write(path: &Path, data: &[f32], dim: usize) -> io::Result<()> {
    if dim == 0 || !data.len().is_multiple_of(dim) {
        return Err(invalid("output must contain complete vectors"));
    }
    let header = u32::try_from(dim)
        .map_err(|_| invalid("output dimension exceeds u32"))?
        .to_le_bytes();
    let mut writer = BufWriter::new(File::create(path)?);
    for row in data.chunks_exact(dim) {
        writer.write_all(&header)?;
        for value in row {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{read, read_rows, write};

    fn fixture() -> Vec<u8> {
        let mut bytes = Vec::new();
        for row in [[1.0_f32, -2.0], [3.5, 4.0]] {
            bytes.extend_from_slice(&2_u32.to_le_bytes());
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn reads_prefix_and_round_trips_fvecs() {
        let bytes = fixture();
        let all = read_rows(Cursor::new(&bytes), bytes.len() as u64, None).unwrap();
        assert_eq!(all.dim, 2);
        assert_eq!(&*all.data, &[1.0, -2.0, 3.5, 4.0]);
        let prefix = read_rows(Cursor::new(&bytes), bytes.len() as u64, Some(1)).unwrap();
        assert_eq!(&*prefix.data, &[1.0, -2.0]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.fvecs");
        write(&path, &all.data, all.dim).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(read(&path, None).unwrap().data, all.data);
    }

    #[test]
    fn rejects_malformed_fvecs_and_invalid_prefixes() {
        let bytes = fixture();
        for end in [0, 1, 3, 4, 11, 13, 23] {
            assert!(read_rows(Cursor::new(&bytes[..end]), end as u64, None).is_err());
        }
        for limit in [0, 3] {
            assert!(read_rows(Cursor::new(&bytes), bytes.len() as u64, Some(limit)).is_err());
        }
        for dim in [0_u32, 3] {
            let mut invalid = bytes.clone();
            invalid[12..16].copy_from_slice(&dim.to_le_bytes());
            assert!(read_rows(Cursor::new(&invalid), invalid.len() as u64, None).is_err());
        }
        assert!(read_rows(Cursor::new([0; 4]), 4, None).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid = bytes.clone();
            invalid[4..8].copy_from_slice(&value.to_le_bytes());
            assert!(read_rows(Cursor::new(&invalid), invalid.len() as u64, None).is_err());
        }
    }
}
