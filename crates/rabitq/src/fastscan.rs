//! Batched binary-code layout for lookup-table scans.
//!
//! This follows upstream RaBitQ's 32-vector FastScan principle, but keeps centroids in
//! natural NEON lane order instead of the AVX-specific permutation used upstream.

pub(crate) const BATCH_SIZE: usize = 32;
const LANES: usize = BATCH_SIZE / 2;
const DIMS_PER_CODE: usize = 4;
const MIN_CENTROIDS: usize = 256;

#[cfg(target_arch = "aarch64")]
mod backend {
    use pulp::aarch64::Neon;

    pub(crate) type Backend = Neon;

    pub(crate) fn detect() -> Option<Backend> {
        Neon::try_new()
    }

    pub(crate) fn accumulate(
        backend: Backend,
        codes: &[u8],
        lut: &[u8],
        result: &mut [u32; crate::fastscan::BATCH_SIZE],
    ) {
        crate::simd::aarch64::fastscan_accumulate(backend, codes, lut, result);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod backend {
    use crate::simd::x86::Avx2;

    pub(crate) type Backend = Avx2;

    pub(crate) fn detect() -> Option<Backend> {
        Avx2::try_new()
    }

    pub(crate) fn accumulate(
        backend: Backend,
        codes: &[u8],
        lut: &[u8],
        result: &mut [u32; crate::fastscan::BATCH_SIZE],
    ) {
        crate::simd::x86::fastscan_accumulate(backend, codes, lut, result);
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
mod backend {
    #[derive(Clone, Copy)]
    pub(crate) struct Backend;

    pub(crate) fn detect() -> Option<Backend> {
        None
    }

    pub(crate) fn accumulate(
        _backend: Backend,
        _codes: &[u8],
        _lut: &[u8],
        _result: &mut [u32; crate::fastscan::BATCH_SIZE],
    ) {
        unreachable!("FastScan has no backend on this architecture");
    }
}

pub(crate) enum BinaryVectors {
    Row(Vec<u64>),
    FastScan(FastScan),
}

impl BinaryVectors {
    pub(crate) fn new(codes: Vec<u64>, num_vectors: usize, dim: usize) -> Self {
        if num_vectors < MIN_CENTROIDS {
            return Self::Row(codes);
        }
        let Some(backend) = backend::detect() else {
            return Self::Row(codes);
        };
        Self::FastScan(FastScan {
            codes: pack_codes(&codes, num_vectors, dim),
            backend,
            bytes_per_batch: dim / DIMS_PER_CODE * LANES,
        })
    }
}

pub(crate) struct FastScan {
    codes: Vec<u8>,
    backend: backend::Backend,
    bytes_per_batch: usize,
}

impl FastScan {
    pub(crate) fn batches(&self) -> impl Iterator<Item = &[u8]> {
        self.codes.chunks_exact(self.bytes_per_batch)
    }

    pub(crate) fn accumulate(&self, codes: &[u8], workspace: &mut Workspace) {
        backend::accumulate(self.backend, codes, &workspace.lut, &mut workspace.scores);
    }
}

pub(crate) struct Workspace {
    lut: Vec<u8>,
    scores: [u32; BATCH_SIZE],
}

impl Workspace {
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            lut: vec![0; dim / DIMS_PER_CODE * 16],
            scores: [0; BATCH_SIZE],
        }
    }

    pub(crate) fn prepare(&mut self, query: &[u8]) {
        build_lut(query, &mut self.lut);
    }

    pub(crate) fn scores(&self) -> &[u32; BATCH_SIZE] {
        &self.scores
    }
}

pub(crate) fn pack_codes(codes: &[u64], num_vectors: usize, dim: usize) -> Vec<u8> {
    assert_eq!(dim % 64, 0);
    let words_per_vector = dim / 64;
    assert_eq!(
        codes.len(),
        num_vectors
            .checked_mul(words_per_vector)
            .expect("binary code length overflowed")
    );

    let groups = dim / DIMS_PER_CODE;
    let mut packed = vec![0; num_vectors.div_ceil(BATCH_SIZE) * groups * LANES];
    for batch in 0..num_vectors.div_ceil(BATCH_SIZE) {
        let batch_start = batch * BATCH_SIZE;
        let packed_batch = &mut packed[batch * groups * LANES..(batch + 1) * groups * LANES];
        for group in 0..groups {
            let bit = group * DIMS_PER_CODE;
            let word = bit / 64;
            let shift = bit % 64;
            for lane in 0..LANES {
                let lower = batch_start + lane;
                let upper = lower + LANES;
                let lower_code = if lower < num_vectors {
                    (codes[lower * words_per_vector + word] >> shift) as u8 & 0x0f
                } else {
                    0
                };
                let upper_code = if upper < num_vectors {
                    (codes[upper * words_per_vector + word] >> shift) as u8 & 0x0f
                } else {
                    0
                };
                packed_batch[group * LANES + lane] = lower_code | (upper_code << 4);
            }
        }
    }
    packed
}

pub(crate) fn build_lut(query: &[u8], lut: &mut [u8]) {
    assert_eq!(query.len() % DIMS_PER_CODE, 0);
    assert_eq!(lut.len(), query.len() / DIMS_PER_CODE * 16);

    let (queries, query_tail) = query.as_chunks::<DIMS_PER_CODE>();
    let (luts, lut_tail) = lut.as_chunks_mut::<16>();
    assert!(query_tail.is_empty());
    assert!(lut_tail.is_empty());
    for (query, lut) in queries.iter().zip(luts) {
        lut[0] = 0;
        for code in 1usize..16 {
            let bit = code.trailing_zeros() as usize;
            lut[code] = lut[code & (code - 1)] + query[bit];
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use seed_rand::seeded_rng;

    use super::{BATCH_SIZE, LANES, backend, build_lut, pack_codes};
    use crate::{THETA_LOG_DIM, simd, vector_binarize_u64};

    fn accumulate_native(codes: &[u8], lut: &[u8], result: &mut [u32; BATCH_SIZE]) {
        assert_eq!(codes.len(), lut.len() / 16 * LANES);
        result.fill(0);
        let (codes, code_tail) = codes.as_chunks::<LANES>();
        let (luts, lut_tail) = lut.as_chunks::<16>();
        assert!(code_tail.is_empty());
        assert!(lut_tail.is_empty());
        for (codes, lut) in codes.iter().zip(luts) {
            for (lane, &code) in codes.iter().enumerate() {
                result[lane] += u32::from(lut[(code & 0x0f) as usize]);
                result[lane + LANES] += u32::from(lut[(code >> 4) as usize]);
            }
        }
    }

    fn accumulate_simd(codes: &[u8], lut: &[u8], result: &mut [u32; BATCH_SIZE]) -> bool {
        let Some(simd) = backend::detect() else {
            return false;
        };
        backend::accumulate(simd, codes, lut, result);
        true
    }

    #[test]
    fn packed_scores_match_asymmetric_binary_dot_products() {
        let mut rng = seeded_rng();
        for (num_vectors, dim) in [
            (1, 64),
            (31, 128),
            (32, 960),
            (37, 1_024),
            (64, 1_088),
            (33, 4_416),
        ] {
            let vectors = (0..num_vectors * dim)
                .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                .collect::<Vec<_>>();
            let codes = vectors
                .chunks_exact(dim)
                .flat_map(vector_binarize_u64)
                .collect::<Vec<_>>();
            let query = (0..dim)
                .map(|_| rng.random_range(0..1 << THETA_LOG_DIM))
                .collect::<Vec<u8>>();
            let mut binary_query = vec![0; dim * THETA_LOG_DIM / 64];
            simd::native::vector_binarize_query(&query, &mut binary_query);
            let packed = pack_codes(&codes, num_vectors, dim);
            let mut lut = vec![0; dim / 4 * 16];
            build_lut(&query, &mut lut);

            for (batch, packed) in packed.chunks_exact(dim / 4 * 16).enumerate() {
                let mut actual = [0; BATCH_SIZE];
                accumulate_native(packed, &lut, &mut actual);
                let mut simd_actual = [0; BATCH_SIZE];
                if accumulate_simd(packed, &lut, &mut simd_actual) {
                    assert_eq!(simd_actual, actual);
                }
                let batch_start = batch * BATCH_SIZE;
                for (lane, &actual) in actual
                    .iter()
                    .take((num_vectors - batch_start).min(BATCH_SIZE))
                    .enumerate()
                {
                    let vector = batch_start + lane;
                    let expected = simd::asymmetric_binary_dot_product(
                        &codes[vector * dim / 64..(vector + 1) * dim / 64],
                        &binary_query,
                    );
                    assert_eq!(actual, expected, "vector {vector}, dimension {dim}");
                }
            }
        }
    }

    #[test]
    fn wide_dimension_accumulation_does_not_overflow() {
        let dim = 4_416;
        let codes = vec![u64::MAX; BATCH_SIZE * dim / 64];
        let query = vec![15; dim];
        let packed = pack_codes(&codes, BATCH_SIZE, dim);
        let mut lut = vec![0; dim / 4 * 16];
        build_lut(&query, &mut lut);

        let mut expected = [0; BATCH_SIZE];
        accumulate_native(&packed, &lut, &mut expected);
        assert_eq!(expected, [dim as u32 * 15; BATCH_SIZE]);

        let mut actual = [0; BATCH_SIZE];
        if accumulate_simd(&packed, &lut, &mut actual) {
            assert_eq!(actual, expected);
        }
    }
}
