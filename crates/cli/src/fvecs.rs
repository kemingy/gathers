//! Validated fvecs I/O into a flat aligned buffer shared by CLI commands.

use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use aligned_vec::AVec;
use anyhow::{Context, Result, anyhow, ensure};

pub(crate) struct Vectors {
    pub(crate) data: AVec<f32>,
    pub(crate) dim: usize,
}

impl Vectors {
    pub(crate) fn len(&self) -> usize {
        self.data.len() / self.dim
    }
}

pub(crate) fn read(path: &Path, limit: Option<usize>) -> Result<Vectors> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let bytes = file
        .metadata()
        .with_context(|| format!("cannot read metadata for {}", path.display()))?
        .len();
    read_rows(BufReader::new(file), bytes, limit)
        .with_context(|| format!("cannot read fvecs from {}", path.display()))
}

fn read_rows(reader: impl Read + Seek, bytes: u64, limit: Option<usize>) -> Result<Vectors> {
    let mut reader = Reader::new(reader, bytes)?;
    let rows = limit.unwrap_or(reader.rows);
    ensure!(
        rows > 0 && rows <= reader.rows,
        "requested row count must be positive and fit in the input"
    );
    let mut data = AVec::new(64);
    reader.read_batch(rows, &mut data)?;
    Ok(Vectors {
        data,
        dim: reader.dim,
    })
}

/// Seekable reader; reopening starts another pass without retaining corpus data.
pub(crate) struct Reader<R> {
    reader: R,
    pub(crate) dim: usize,
    pub(crate) rows: usize,
    position: usize,
}

impl Reader<BufReader<File>> {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        let bytes = file.metadata()?.len();
        Self::new(BufReader::new(file), bytes)
    }
}

impl<R: Read + Seek> Reader<R> {
    fn new(mut reader: R, bytes: u64) -> Result<Self> {
        let mut header = [0; 4];
        reader
            .read_exact(&mut header)
            .context("cannot read first fvecs dimension")?;
        let dim = u32::from_le_bytes(header) as usize;
        ensure!(dim > 0, "fvecs dimension must be positive");
        let row_bytes = (dim as u64 + 1) * 4;
        ensure!(
            bytes.is_multiple_of(row_bytes),
            "fvecs input contains an incomplete row"
        );
        let available = usize::try_from(bytes / row_bytes).context("too many fvecs rows")?;
        ensure!(available > 0, "input must contain at least one vector");
        reader.rewind()?;
        Ok(Self {
            reader,
            dim,
            rows: available,
            position: 0,
        })
    }

    /// Fill a reusable flat buffer. Returns zero and clears its length at EOF.
    pub(crate) fn read_batch(&mut self, max_rows: usize, data: &mut AVec<f32>) -> Result<usize> {
        ensure!(max_rows > 0, "batch size must be positive");
        let rows = max_rows.min(self.rows - self.position);
        let values = rows.checked_mul(self.dim).context("input shape overflow")?;
        data.clear();
        data.try_reserve_exact(values)
            .map_err(|error| anyhow!("cannot allocate vector buffer: {error:?}"))?;
        data.resize(values, 0.0_f32);
        for (offset, row) in data.chunks_exact_mut(self.dim).enumerate() {
            read_row(&mut self.reader, self.position + offset, row)?;
        }
        self.position += rows;
        Ok(rows)
    }

    /// Read sorted indices, coalescing nearby rows into bounded reads. By default only
    /// selected rows are validated; `validate_all` scans and validates the entire source.
    pub(crate) fn read_sample(
        &mut self,
        indices: &[usize],
        batch_rows: usize,
        validate_all: bool,
    ) -> Result<Vectors> {
        ensure!(batch_rows > 0, "batch size must be positive");
        ensure!(self.position == 0, "sampling requires a fresh reader");
        ensure!(!indices.is_empty(), "sample must not be empty");
        ensure!(
            indices.windows(2).all(|pair| pair[0] < pair[1])
                && indices[indices.len() - 1] < self.rows,
            "sample indices must be strictly increasing and within the source"
        );
        let values = indices
            .len()
            .checked_mul(self.dim)
            .context("sample size overflow")?;
        let mut data = AVec::new(64);
        data.try_reserve_exact(values)
            .map_err(|error| anyhow!("cannot allocate sample: {error:?}"))?;
        if validate_all {
            self.scan_sample(indices, batch_rows, &mut data)?;
        } else {
            self.seek_sample(indices, batch_rows, &mut data)?;
        }
        Ok(Vectors {
            data,
            dim: self.dim,
        })
    }

    fn scan_sample(
        &mut self,
        indices: &[usize],
        batch_rows: usize,
        data: &mut AVec<f32>,
    ) -> Result<()> {
        let mut batch = AVec::new(64);
        let mut selected = 0;
        let mut start = 0;
        loop {
            let rows = self.read_batch(batch_rows, &mut batch)?;
            if rows == 0 {
                break;
            }
            while selected < indices.len() && indices[selected] < start + rows {
                let offset = (indices[selected] - start) * self.dim;
                data.extend_from_slice(&batch[offset..offset + self.dim]);
                selected += 1;
            }
            start += rows;
        }
        ensure!(selected == indices.len(), "sample is incomplete");
        Ok(())
    }

    fn seek_sample(
        &mut self,
        indices: &[usize],
        batch_rows: usize,
        data: &mut AVec<f32>,
    ) -> Result<()> {
        let row_bytes = self
            .dim
            .checked_add(1)
            .and_then(|dim| dim.checked_mul(4))
            .context("fvecs row size overflow")?;
        // Read through small gaps to avoid a seek per selected vector. The batch-row
        // limit still bounds scratch, even when every source row is selected.
        let max_gap_rows = (64 * 1024) / row_bytes;
        let mut bytes = Vec::new();
        let mut selected = 0;
        while selected < indices.len() {
            let start = indices[selected];
            let mut end = selected + 1;
            while end < indices.len()
                && indices[end] - start < batch_rows
                && indices[end] - indices[end - 1] - 1 <= max_gap_rows
            {
                end += 1;
            }
            let rows = indices[end - 1] - start + 1;
            let count = rows
                .checked_mul(row_bytes)
                .context("read batch size overflow")?;
            bytes.clear();
            bytes
                .try_reserve_exact(count)
                .context("cannot allocate read batch")?;
            bytes.resize(count, 0);
            if self.position != start {
                let offset = (start as u64)
                    .checked_mul(row_bytes as u64)
                    .context("fvecs seek offset overflow")?;
                self.reader.seek(SeekFrom::Start(offset))?;
            }
            self.reader
                .read_exact(&mut bytes)
                .with_context(|| format!("cannot read batch starting at row {}", start + 1))?;
            self.position = start + rows;
            for &index in &indices[selected..end] {
                let offset = (index - start) * row_bytes;
                let output_start = data.len();
                data.resize(output_start + self.dim, 0.0);
                read_row(
                    &mut Cursor::new(&bytes[offset..offset + row_bytes]),
                    index,
                    &mut data[output_start..],
                )?;
            }
            selected = end;
        }
        Ok(())
    }
}

fn read_row(reader: &mut impl Read, index: usize, row: &mut [f32]) -> Result<()> {
    let mut header = [0; 4];
    reader
        .read_exact(&mut header)
        .with_context(|| format!("cannot read dimension for row {}", index + 1))?;
    ensure!(
        u32::from_le_bytes(header) as usize == row.len(),
        "fvecs row {} has a different dimension (expected {})",
        index + 1,
        row.len()
    );
    reader
        .read_exact(bytemuck::cast_slice_mut(row))
        .with_context(|| format!("cannot read coordinates for row {}", index + 1))?;
    for value in row {
        *value = f32::from_bits(u32::from_le(value.to_bits()));
        ensure!(
            value.is_finite(),
            "fvecs row {} contains a non-finite coordinate",
            index + 1
        );
    }
    Ok(())
}

pub(crate) fn write(path: &Path, data: &[f32], dim: usize) -> Result<()> {
    ensure!(
        dim > 0 && data.len().is_multiple_of(dim),
        "output must contain complete vectors"
    );
    let header = u32::try_from(dim)
        .context("output dimension exceeds u32")?
        .to_le_bytes();
    let mut writer = BufWriter::new(
        File::create(path).with_context(|| format!("cannot create {}", path.display()))?,
    );
    for row in data.chunks_exact(dim) {
        writer
            .write_all(&header)
            .context("cannot write fvecs dimension")?;
        for value in row {
            writer
                .write_all(&value.to_le_bytes())
                .context("cannot write fvecs coordinate")?;
        }
    }
    writer
        .flush()
        .with_context(|| format!("cannot flush {}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::io::{self, Cursor, Read, Seek, SeekFrom};

    use aligned_vec::AVec;

    use super::{Reader, read, read_rows, write};

    struct CountedReader {
        inner: Cursor<Vec<u8>>,
        bytes: usize,
        reads: usize,
        seeks: usize,
    }

    impl Read for CountedReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let count = self.inner.read(buffer)?;
            self.bytes += count;
            self.reads += 1;
            Ok(count)
        }
    }

    impl Seek for CountedReader {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.seeks += 1;
            self.inner.seek(position)
        }
    }

    #[test]
    fn sample_reads_coalesce_nearby_rows_and_skip_large_gaps() {
        let mut bytes = Vec::new();
        for index in 0..600 {
            bytes.extend_from_slice(&64_u32.to_le_bytes());
            for coordinate in 0..64 {
                bytes.extend_from_slice(&((index + coordinate) as f32).to_le_bytes());
            }
        }
        let indices = [1, 2, 10, 500, 511];
        let expected = indices
            .iter()
            .flat_map(|index| (0..64).map(move |coordinate| (index + coordinate) as f32))
            .collect::<Vec<_>>();
        for batch_rows in [1, 16, 4096] {
            for validate_all in [false, true] {
                let counted = CountedReader {
                    inner: Cursor::new(bytes.clone()),
                    bytes: 0,
                    reads: 0,
                    seeks: 0,
                };
                let mut reader = Reader::new(counted, bytes.len() as u64).unwrap();
                reader.reader.bytes = 0;
                reader.reader.reads = 0;
                reader.reader.seeks = 0;
                let sample = reader
                    .read_sample(&indices, batch_rows, validate_all)
                    .unwrap();
                assert_eq!(&*sample.data, expected);
                if !validate_all && batch_rows > 1 {
                    assert_eq!(reader.reader.reads, 2);
                    assert_eq!(reader.reader.seeks, 2);
                    assert_eq!(reader.reader.bytes, (10 + 12) * 260);
                }
            }
        }
        // Row 6 is fetched inside a coalesced span but is not selected: selected-only
        // validation must not depend on the batch size or coalescing policy.
        bytes[5 * 260 + 4..5 * 260 + 8].copy_from_slice(&f32::NAN.to_le_bytes());
        for validate_all in [false, true] {
            let mut reader = Reader::new(Cursor::new(&bytes), bytes.len() as u64).unwrap();
            assert_eq!(
                reader.read_sample(&indices, 4096, validate_all).is_err(),
                validate_all
            );
        }
        let mut reader = Reader::new(Cursor::new(&bytes), bytes.len() as u64).unwrap();
        assert!(reader.read_sample(&[5], 4096, false).is_err());
    }

    #[test]
    fn batches_reuse_storage_and_sampling_is_independent_of_batch_size() {
        let bytes = fixture();
        let mut reader = Reader::new(Cursor::new(&bytes), bytes.len() as u64).unwrap();
        let mut buffer = AVec::new(64);
        assert_eq!(reader.read_batch(1, &mut buffer).unwrap(), 1);
        assert_eq!(&*buffer, &[1.0, -2.0]);
        let address = buffer.as_ptr();
        assert_eq!(reader.read_batch(1, &mut buffer).unwrap(), 1);
        assert_eq!(&*buffer, &[3.5, 4.0]);
        assert_eq!(address, buffer.as_ptr());
        assert_eq!(reader.read_batch(1, &mut buffer).unwrap(), 0);
        assert!(buffer.is_empty());
        assert!(reader.read_batch(0, &mut buffer).is_err());
        for batch_rows in [1, 2, 3] {
            let mut reader = Reader::new(Cursor::new(&bytes), bytes.len() as u64).unwrap();
            assert_eq!(
                &*reader.read_sample(&[1], batch_rows, false).unwrap().data,
                &[3.5, 4.0]
            );
        }
    }

    #[test]
    fn sampling_checks_indices_and_obeys_validation_scope() {
        let bytes = fixture();
        for indices in [vec![], vec![2], vec![0, 0], vec![1, 0]] {
            let mut reader = Reader::new(Cursor::new(&bytes), bytes.len() as u64).unwrap();
            assert!(reader.read_sample(&indices, 1, false).is_err());
        }
        let mut invalid = bytes.clone();
        invalid[12..16].copy_from_slice(&3_u32.to_le_bytes());
        let mut reader = Reader::new(Cursor::new(&invalid), invalid.len() as u64).unwrap();
        assert!(reader.read_sample(&[0], 1, true).is_err());
        let mut reader = Reader::new(Cursor::new(&invalid), invalid.len() as u64).unwrap();
        assert!(reader.read_sample(&[0], 1, false).is_ok());
        let mut reader = Reader::new(Cursor::new(&invalid), invalid.len() as u64).unwrap();
        assert!(reader.read_sample(&[1], 1, false).is_err());
        let mut reader = Reader::new(Cursor::new(&bytes[..16]), bytes.len() as u64).unwrap();
        assert!(reader.read_sample(&[1], 1, false).is_err());
    }

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
