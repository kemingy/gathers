//! Native e-means updates following Algorithm 1 and the pinned Faiss CPU recipe.
//!
//! Reference: <https://e-means.github.io/> and Faiss commit
//! `2b648f0605f2ef3d14c4705350f8a0b214463878`, `contrib/e_means` (MIT).
//! Random permutations use gathers' Rust RNG; seeds are not NumPy-compatible.

use aligned_vec::AVec;
use log::debug;
use rand::seq::SliceRandom;
use rand::{Rng, RngExt};
#[cfg(not(feature = "perf"))]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[cfg(not(feature = "perf"))]
use rayon::slice::ParallelSliceMut;

use crate::distance::Distance;
#[cfg(feature = "perf")]
use crate::kmeans::base_assign;
use crate::kmeans::{KMeansError, TrainingFit, TrainingParameters};
#[cfg(not(feature = "perf"))]
use crate::kmeans::{base_assign_parallel, matrix::MatrixAssignmentWorkspace};
use crate::utils::{normalize_nonzero_row, try_normalize_rows};

const INITIAL_MASS: f32 = 1.0 / 65_536.0;
const SPLIT_EPSILON: f32 = 1e-4;
const ASSIGNMENT_CHUNK_BYTES: usize = 32 << 20;

/// Native e-means settings. The pass budget, sampling and projection use `KMeansConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EmeansConfig {
    /// Logical batch rows; `None` uses the upstream throughput law on retained training rows.
    /// Assignment chunks are bounded independently of this statistical batch size.
    pub batch_size: Option<usize>,
    /// Initial EMA step in (0, 1]; `None` uses the upstream step law at the resolved batch size.
    pub alpha0: Option<f64>,
    /// Revival threshold relative to mean mass. Zero disables revival. `None` uses 0.1,
    /// except at training dimensions <= 2 where the upstream default disables it.
    pub revival_threshold: Option<f32>,
}

/// Resolved throughput recipe, derived after training-row selection and projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmeansRecipe {
    /// Number of full passes.
    pub max_iter: u32,
    /// Rows per logical EMA update, including a smaller remainder at each pass's end.
    pub batch_size: usize,
    /// Initial step, annealed by a cosine schedule towards one tenth of this value.
    /// Updates exclude the schedule endpoint, as in the pinned upstream recipe.
    pub alpha0: f64,
    /// Threshold relative to mean mass; checked after every fifth logical update.
    pub revival_threshold: f32,
    /// Number of logical EMA updates, including remainder batches.
    pub total_steps: usize,
}

fn round_power_of_two(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let low = 2.0_f64.powf(value.log2().floor());
    (if value / low < 2.0 * low / value {
        low
    } else {
        2.0 * low
    }) as usize
}

impl EmeansConfig {
    /// Resolve the recipe from retained rows, training dimension, cluster count and pass budget.
    /// The statistical laws were fitted on at least 100,000 rows; smaller workloads are accepted
    /// but need workload-specific quality checks. No vector buffers are allocated here.
    pub fn resolve(
        self,
        rows: usize,
        dim: usize,
        clusters: usize,
        max_iter: u32,
    ) -> Result<EmeansRecipe, KMeansError> {
        if rows == 0 || dim == 0 || clusters == 0 || clusters > rows || max_iter == 0 {
            return Err(KMeansError::InvalidConfig(
                "e-means requires positive sizes and max_iter, with clusters <= rows",
            ));
        }
        if self.batch_size == Some(0)
            || self.alpha0.is_some_and(|value| {
                !value.is_finite() || value <= 0.0 || value > 1.0 || (value * 0.1) as f32 == 0.0
            })
            || self
                .revival_threshold
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(KMeansError::InvalidConfig(
                "e-means requires positive batch size, alpha0 in (0, 1], and finite nonnegative \
                 revival threshold",
            ));
        }
        let n = rows as f64;
        let k = clusters as f64;
        let epochs = f64::from(max_iter);
        let batch_size = self
            .batch_size
            .unwrap_or_else(|| {
                round_power_of_two(
                    0.035134 * k.powf(0.134749) * n.powf(0.876473) * epochs.powf(0.813344),
                )
                .min(round_power_of_two(n / 2.0))
            })
            .clamp(1, rows);
        let alpha0 = self.alpha0.unwrap_or_else(|| {
            (0.8244
                * k.powf(-0.4617)
                * (batch_size as f64).powf(0.9003)
                * n.powf(-0.4498)
                * epochs.powf(-0.1057))
            .clamp(1e-4, 1.0)
        });
        let total_steps = rows
            .div_ceil(batch_size)
            .checked_mul(max_iter as usize)
            .ok_or(KMeansError::InvalidConfig("e-means update count overflow"))?;
        Ok(EmeansRecipe {
            max_iter,
            batch_size,
            alpha0,
            revival_threshold: self
                .revival_threshold
                .unwrap_or(if dim <= 2 { 0.0 } else { 0.1 }),
            total_steps,
        })
    }
}

impl EmeansRecipe {
    fn alpha(self, step: usize) -> f64 {
        let fraction = step.min(self.total_steps) as f64 / self.total_steps.max(1) as f64;
        self.alpha0 * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos()))
    }
}

struct State {
    positions: AVec<f32>,
    masses: Vec<f32>,
    sums: Vec<f32>,
    wide_sums: Vec<f64>,
    counts: Vec<usize>,
    closing_sums: Vec<f64>,
    closing_counts: Vec<usize>,
    representatives: Vec<Option<usize>>,
    unused: Vec<usize>,
    donors: Vec<(f64, usize)>,
    dim: usize,
}

impl State {
    fn new(positions: AVec<f32>, dim: usize) -> Self {
        let clusters = positions.len() / dim;
        Self {
            masses: vec![INITIAL_MASS; clusters],
            sums: vec![0.0; positions.len()],
            wide_sums: Vec::new(),
            counts: vec![0; clusters],
            closing_sums: vec![0.0; positions.len()],
            closing_counts: vec![0; clusters],
            representatives: vec![None; clusters],
            unused: Vec::with_capacity(clusters),
            donors: Vec::with_capacity(clusters),
            positions,
            dim,
        }
    }

    fn observe(&mut self, row: &[f32], source: usize, label: usize, closing: bool) {
        self.counts[label] += 1;
        for (sum, &value) in self.sums[label * self.dim..(label + 1) * self.dim]
            .iter_mut()
            .zip(row)
        {
            *sum += value;
        }
        if closing {
            self.closing_counts[label] += 1;
            for (sum, &value) in self.closing_sums[label * self.dim..(label + 1) * self.dim]
                .iter_mut()
                .zip(row)
            {
                *sum += f64::from(value);
            }
            if self.representatives[label].is_none() && row.iter().any(|&value| value != 0.0) {
                self.representatives[label] = Some(source);
            }
        }
    }

    fn update(
        &mut self,
        vectors: &[f32],
        order: &[usize],
        labels: &[u32],
        alpha: f64,
    ) -> Result<(), KMeansError> {
        // Match ordinary FP32 statistics; retry overflowing or very wide batches in FP64.
        // The fallback storage is allocated once, only if needed, and reused thereafter.
        let wide = order.len() > (1 << 24) || self.sums.iter().any(|value| !value.is_finite());
        if wide {
            self.wide_sums.resize(self.sums.len(), 0.0);
            self.wide_sums.fill(0.0);
            for &source in order {
                let label = labels[source] as usize;
                for (sum, &value) in self.wide_sums[label * self.dim..(label + 1) * self.dim]
                    .iter_mut()
                    .zip(&vectors[source * self.dim..(source + 1) * self.dim])
                {
                    *sum += f64::from(value);
                }
            }
        }
        let step = alpha as f32;
        if alpha != 1.0 {
            let decay = (1.0 - alpha) as f32;
            self.masses.iter_mut().for_each(|mass| *mass *= decay);
        }
        for (cluster, &count) in self.counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let contribution = step * count as f32;
            let mass = if alpha == 1.0 {
                contribution
            } else {
                self.masses[cluster] + contribution
            };
            let weight = contribution / mass;
            self.masses[cluster] = mass;
            for coordinate in 0..self.dim {
                let index = cluster * self.dim + coordinate;
                let mean = if wide {
                    (self.wide_sums[index] / count as f64) as f32
                } else {
                    self.sums[index] / count as f32
                };
                let previous = self.positions[index];
                let mut position = if weight == 1.0 {
                    mean
                } else {
                    previous + weight * (mean - previous)
                };
                if !position.is_finite() {
                    position = (f64::from(previous)
                        + f64::from(weight) * (f64::from(mean) - f64::from(previous)))
                        as f32;
                }
                if !position.is_finite() {
                    return Err(KMeansError::NumericalOverflow);
                }
                self.positions[index] = position;
            }
        }
        Ok(())
    }

    fn revive<R: Rng + ?Sized>(
        &mut self,
        coefficient: f32,
        rng: &mut R,
    ) -> Result<usize, KMeansError> {
        if coefficient == 0.0 {
            return Ok(0);
        }
        let threshold = self.masses.iter().sum::<f32>() * coefficient / self.masses.len() as f32;
        self.unused.clear();
        self.donors.clear();
        for (index, &mass) in self.masses.iter().enumerate() {
            if mass < threshold {
                self.unused.push(index);
            } else if mass > 0.0 {
                // Gumbel top-k samples distinct donors in proportion to their masses.
                let uniform = rng.random::<f64>().max(f64::MIN_POSITIVE);
                self.donors
                    .push((f64::from(mass).ln() - (-uniform.ln()).ln(), index));
            }
        }
        let count = self.unused.len().min(self.donors.len());
        if count == 0 {
            return Ok(0);
        }
        let descending = |left: &(f64, usize), right: &(f64, usize)| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        };
        self.donors.select_nth_unstable_by(count - 1, descending);
        self.donors[..count].sort_unstable_by(descending);
        for (&(_, source), &target) in self.donors[..count].iter().zip(&self.unused) {
            let mut collapsed = true;
            for coordinate in 0..self.dim {
                let value = self.positions[source * self.dim + coordinate];
                let offset = if coordinate % 2 == 0 {
                    SPLIT_EPSILON
                } else {
                    -SPLIT_EPSILON
                };
                let donor = value * (1.0 - offset);
                let child = value * (1.0 + offset);
                if !donor.is_finite() || !child.is_finite() {
                    return Err(KMeansError::NumericalOverflow);
                }
                self.positions[source * self.dim + coordinate] = donor;
                self.positions[target * self.dim + coordinate] = child;
                collapsed &= donor == child;
            }
            if collapsed {
                let value = &mut self.positions[target * self.dim];
                *value = if value.next_up().is_finite() {
                    value.next_up()
                } else {
                    value.next_down()
                };
            }
            let half_mass = self.masses[source] / 2.0;
            self.masses[source] = half_mass;
            self.masses[target] = half_mass;
        }
        Ok(count)
    }

    fn finish(mut self, vectors: &[f32], spherical: bool) -> Result<AVec<f32>, KMeansError> {
        let fallback = vectors
            .chunks_exact(self.dim)
            .position(|row| row.iter().any(|&value| value != 0.0));
        for (cluster, row) in self.positions.chunks_exact_mut(self.dim).enumerate() {
            let count = self.closing_counts[cluster];
            if count > 0 {
                for (value, &sum) in row
                    .iter_mut()
                    .zip(&self.closing_sums[cluster * self.dim..(cluster + 1) * self.dim])
                {
                    *value = (sum / count as f64) as f32;
                }
            }
            if spherical {
                match normalize_nonzero_row(row) {
                    Some(true) => {}
                    Some(false) => {
                        // Preserve gathers' nonzero cosine output contract on cancellation.
                        // An all-zero projected sample stays zero until original-space reconstruction.
                        if let Some(source) = self.representatives[cluster].or(fallback) {
                            row.copy_from_slice(
                                &vectors[source * self.dim..(source + 1) * self.dim],
                            );
                            normalize_nonzero_row(row).ok_or(KMeansError::NumericalOverflow)?;
                        }
                    }
                    None => return Err(KMeansError::NumericalOverflow),
                }
            }
        }
        Ok(self.positions)
    }
}

/// E-means update policy; the recipe is resolved from the prepared training shape.
pub(crate) struct Emeans {
    config: EmeansConfig,
    max_iter: u32,
}

impl Emeans {
    pub(crate) fn new(config: EmeansConfig, max_iter: u32) -> Self {
        Self { config, max_iter }
    }

    pub(crate) fn fit<R: Rng + ?Sized>(
        &self,
        vectors: &[f32],
        dim: usize,
        centroids: AVec<f32>,
        distance: Distance,
        rng: &mut R,
    ) -> Result<TrainingFit, KMeansError> {
        self.fit_chunked(
            vectors,
            dim,
            centroids,
            distance,
            rng,
            (ASSIGNMENT_CHUNK_BYTES / size_of::<f32>() / dim).max(1),
        )
    }

    fn fit_chunked<R: Rng + ?Sized>(
        &self,
        vectors: &[f32],
        dim: usize,
        centroids: AVec<f32>,
        distance: Distance,
        rng: &mut R,
        max_chunk_rows: usize,
    ) -> Result<TrainingFit, KMeansError> {
        let rows = vectors.len() / dim;
        let clusters = centroids.len() / dim;
        let recipe = self.config.resolve(rows, dim, clusters, self.max_iter)?;
        debug!("e-means recipe: {recipe:?}");
        let spherical = distance != Distance::SquaredEuclidean;
        let chunk_rows = recipe.batch_size.min(max_chunk_rows);
        let mut batch = vec![0.0; chunk_rows * dim];
        let mut batch_labels = vec![0; chunk_rows];
        let mut directions = if spherical {
            vec![0.0; centroids.len()]
        } else {
            Vec::new()
        };
        #[cfg(not(feature = "perf"))]
        let mut workspace = MatrixAssignmentWorkspace::try_new(&batch, clusters, dim, distance);
        #[cfg(not(feature = "perf"))]
        let source_norms = workspace.as_ref().map_or_else(Vec::new, |workspace| {
            workspace.cache_vector_norms(vectors, dim)
        });
        let mut state = State::new(centroids, dim);
        let mut labels = vec![0; rows];
        let mut order = (0..rows).collect::<Vec<_>>();
        let mut step = 0;
        let mut revivals = 0;
        for epoch in 0..recipe.max_iter {
            for (index, source) in order.iter_mut().enumerate() {
                *source = index;
            }
            order.shuffle(rng);
            for indices in order.chunks(recipe.batch_size) {
                state.sums.fill(0.0);
                state.counts.fill(0);
                if spherical {
                    directions.copy_from_slice(&state.positions);
                    try_normalize_rows(&mut directions, dim, true)?;
                }
                // Keep one centroid snapshot for the entire statistical batch, even when
                // bounded assignment chunks split it into multiple matrix calls.
                for chunk in indices.chunks(chunk_rows) {
                    let centroids = if spherical {
                        &directions[..]
                    } else {
                        &state.positions[..]
                    };
                    let batch = &mut batch[..chunk.len() * dim];
                    #[cfg(feature = "perf")]
                    for (target, &source) in batch.chunks_exact_mut(dim).zip(chunk) {
                        target.copy_from_slice(&vectors[source * dim..(source + 1) * dim]);
                    }
                    // Copy independent rows in parallel; observation still follows shuffled order
                    // below, so neither statistics nor RNG consumption changes.
                    #[cfg(not(feature = "perf"))]
                    batch
                        .par_chunks_exact_mut(dim)
                        .zip(chunk.par_iter())
                        .with_min_len(64)
                        .for_each(|(target, &source)| {
                            target.copy_from_slice(&vectors[source * dim..(source + 1) * dim]);
                        });
                    let batch_labels = &mut batch_labels[..chunk.len()];
                    #[cfg(feature = "perf")]
                    base_assign(batch, centroids, dim, distance, batch_labels);
                    #[cfg(not(feature = "perf"))]
                    if let Some(workspace) = &mut workspace {
                        workspace.gather_vector_norms(&source_norms, chunk);
                        workspace.assign(batch, centroids, dim, batch_labels);
                    } else {
                        base_assign_parallel(batch, centroids, dim, distance, batch_labels);
                    }
                    // End the centroid borrow before accumulating into the reusable state.
                    for ((row, &source), &label) in
                        batch.chunks_exact(dim).zip(chunk).zip(batch_labels.iter())
                    {
                        labels[source] = label;
                        state.observe(row, source, label as usize, epoch + 1 == recipe.max_iter);
                    }
                }
                state.update(vectors, indices, &labels, recipe.alpha(step))?;
                step += 1;
                if step % 5 == 0 {
                    revivals += state.revive(recipe.revival_threshold, rng)?;
                }
            }
        }
        debug!("e-means completed {step} updates, {revivals} revivals");
        Ok(TrainingFit {
            centroids: state.finish(vectors, spherical)?,
            labels,
            parameters: TrainingParameters::Emeans(recipe),
        })
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use seed_rand::seeded_rng;

    use super::{Emeans, EmeansConfig, INITIAL_MASS, State};
    use crate::distance::Distance;
    use crate::kmeans::{KMeans, KMeansConfig, ReductionConfig, TrainingAlgorithm, base_assign};
    use crate::utils::{as_continuous_vec, try_normalize_rows};

    #[test]
    fn recipe_and_update_match_pinned_upstream_fixtures() {
        for (rows, clusters, epochs, batch, alpha, steps) in [
            (100_000, 512, 3, 4096, 0.41499607256086607, 75),
            (100_000, 2048, 2, 4096, 0.22839454909195195, 50),
            (78, 2, 3, 4, 0.2616321893299979, 60),
        ] {
            let recipe = EmeansConfig::default()
                .resolve(rows, 128, clusters, epochs)
                .unwrap();
            assert_eq!(recipe.batch_size, batch);
            assert!((recipe.alpha0 - alpha).abs() < 1e-12);
            assert_eq!(recipe.total_steps, steps);
            assert_eq!(recipe.alpha(0), recipe.alpha0);
            assert!(recipe.alpha(steps - 1) > alpha * 0.1);
            assert!((recipe.alpha(steps) - alpha * 0.1).abs() < 1e-12);
        }
        // Generated with the pinned NumpyEmeans.begin/accumulate/update/finalize on
        // these explicit batches, no shuffling or revival. Tests the update algebra,
        // decaying untouched masses, the cosine step, and unweighted closing means.
        let vectors = as_continuous_vec(&[
            [-2.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [4.0, 1.0, 2.0],
            [6.0, -1.0, 0.0],
            [5.0, 0.0, 3.0],
        ]);
        let mut state = State::new(as_continuous_vec(&[[-2.0, 1.0, 0.0], [4.0, 1.0, 2.0]]), 3);
        let labels = [0, 0, 0, 1, 1, 1];
        let recipe = EmeansConfig {
            batch_size: Some(3),
            alpha0: Some(0.4),
            ..Default::default()
        }
        .resolve(6, 3, 2, 1)
        .unwrap();
        for (step, indices) in [[0, 1, 2], [3, 4, 5]].iter().enumerate() {
            state.sums.fill(0.0);
            state.counts.fill(0);
            for &source in indices {
                state.observe(
                    &vectors[source * 3..(source + 1) * 3],
                    source,
                    labels[source] as usize,
                    true,
                );
            }
            state
                .update(&vectors, indices, &labels, recipe.alpha(step))
                .unwrap();
        }
        for (&actual, expected) in state.positions.iter().zip([
            -1.0000076,
            1.0,
            0.6666616,
            4.999989,
            0.000010848045,
            1.6666702,
        ]) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert!((state.masses[0] - 0.93600714).abs() < 1e-6);
        assert!((state.masses[1] - 0.6600071).abs() < 1e-6);
        let closing = state.finish(&vectors, false).unwrap();
        assert_eq!(&*closing, &[-1.0, 1.0, 2.0 / 3.0, 5.0, 0.0, 5.0 / 3.0]);
    }

    #[test]
    fn spherical_three_pass_updates_match_pinned_upstream_fixture() {
        // The same explicit batches were run through pinned NumpyEmeans(spherical=True).
        // Fixed order and no revival isolate the implementation from RNG differences.
        let mut vectors = as_continuous_vec(&[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 1.0],
        ]);
        try_normalize_rows(&mut vectors, 3, false).unwrap();
        let initial = as_continuous_vec(&[&vectors[..3], &vectors[9..12]]);
        let mut state = State::new(initial, 3);
        let recipe = EmeansConfig {
            batch_size: Some(3),
            alpha0: Some(0.4),
            revival_threshold: Some(0.0),
        }
        .resolve(6, 3, 2, 3)
        .unwrap();
        let mut labels = [0; 6];
        let mut step = 0;
        for epoch in 0..3 {
            for indices in [[0, 3, 1], [4, 2, 5]] {
                let mut directions = state.positions.clone();
                try_normalize_rows(&mut directions, 3, true).unwrap();
                state.sums.fill(0.0);
                state.counts.fill(0);
                for source in indices {
                    let row = &vectors[source * 3..(source + 1) * 3];
                    let mut label = [0];
                    base_assign(
                        row,
                        &directions,
                        3,
                        Distance::NegativeDotProduct,
                        &mut label,
                    );
                    labels[source] = label[0];
                    state.observe(row, source, label[0] as usize, epoch == 2);
                }
                state
                    .update(&vectors, &indices, &labels, recipe.alpha(step))
                    .unwrap();
                step += 1;
            }
        }
        assert_eq!(labels, [0, 0, 0, 1, 1, 1]);
        for (&actual, expected) in state.positions.iter().zip([
            0.56685364,
            0.5668517,
            0.0,
            -0.24327537,
            -0.5598941,
            0.47519106,
        ]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (&actual, expected) in state.masses.iter().zip([1.2638466, 1.2438664]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (&actual, expected) in state.finish(&vectors, true).unwrap().iter().zip([
            0.70710677,
            0.70710677,
            0.0,
            -0.30389065,
            -0.7336569,
            0.6077813,
        ]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn chunking_keeps_batch_snapshots_and_source_order_labels() {
        let mut rng = seeded_rng();
        let mut vectors = as_continuous_vec(&[(0..137 * 5)
            .map(|_| rng.random::<f32>() + 0.1)
            .collect::<Vec<_>>()]);
        for distance in [Distance::SquaredEuclidean, Distance::NegativeDotProduct] {
            if distance == Distance::NegativeDotProduct {
                try_normalize_rows(&mut vectors, 5, false).unwrap();
            }
            let initial = as_continuous_vec(&[&vectors[..15]]);
            let config = EmeansConfig {
                batch_size: Some(17),
                revival_threshold: Some(0.0),
                ..Default::default()
            };
            let mut expected = None;
            for chunk_rows in [1, 7, 17] {
                let fit = Emeans::new(config, 3)
                    .fit_chunked(
                        &vectors,
                        5,
                        initial.clone(),
                        distance,
                        &mut StdRng::seed_from_u64(11),
                        chunk_rows,
                    )
                    .unwrap();
                let (centroids, labels) = (fit.centroids, fit.labels);
                if let Some((expected_centroids, expected_labels)) = &expected {
                    assert_eq!(&labels, expected_labels);
                    assert_eq!(&centroids, expected_centroids);
                }
                // Every occupied centroid is the mean of source rows selected by the
                // final moving labels, including the remainder batch.
                for (cluster, centroid) in centroids.as_chunks::<5>().0.iter().enumerate() {
                    let mut mean = [0.0; 5];
                    let mut count = 0;
                    for (row, &label) in vectors.as_chunks::<5>().0.iter().zip(&labels) {
                        if label as usize == cluster {
                            for (sum, &value) in mean.iter_mut().zip(row) {
                                *sum += f64::from(value);
                            }
                            count += 1;
                        }
                    }
                    if count > 0 {
                        let mut mean = mean
                            .iter()
                            .map(|sum| (sum / f64::from(count)) as f32)
                            .collect::<Vec<_>>();
                        if distance == Distance::NegativeDotProduct {
                            try_normalize_rows(&mut mean, 5, false).unwrap();
                        }
                        for (&a, &b) in centroid.iter().zip(&mean) {
                            assert!((a - b).abs() < 1e-6);
                        }
                    }
                }
                expected = Some((centroids, labels));
            }
        }
    }

    #[cfg(not(feature = "perf"))]
    #[test]
    fn matrix_and_scalar_chunks_preserve_training_results() {
        use crate::kmeans::matrix::MatrixAssignmentWorkspace;

        // Cross the portable matrix threshold, including a remainder batch. The
        // small chunks use scalar/SIMD assignment and the large chunks use GEMM.
        let (rows, dim, clusters, batch_size) = (2053, 64, 257, 2048);
        let mut rng = seeded_rng();
        let mut vectors = (0..rows * dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect::<Vec<f32>>();
        for distance in [Distance::SquaredEuclidean, Distance::NegativeDotProduct] {
            if distance == Distance::NegativeDotProduct {
                try_normalize_rows(&mut vectors, dim, false).unwrap();
            }
            assert!(
                MatrixAssignmentWorkspace::try_new(
                    &vectors[..batch_size * dim],
                    clusters,
                    dim,
                    distance,
                )
                .is_some()
            );
            assert!(
                MatrixAssignmentWorkspace::try_new(&vectors[..7 * dim], clusters, dim, distance)
                    .is_none()
            );
            let initial = as_continuous_vec(&[&vectors[..clusters * dim]]);
            let trainer = Emeans::new(
                EmeansConfig {
                    batch_size: Some(batch_size),
                    alpha0: Some(0.4),
                    revival_threshold: Some(0.0),
                },
                2,
            );
            let run = |chunk_rows| {
                trainer
                    .fit_chunked(
                        &vectors,
                        dim,
                        initial.clone(),
                        distance,
                        &mut StdRng::seed_from_u64(11),
                        chunk_rows,
                    )
                    .unwrap()
            };
            let expected = run(7);
            let actual = run(batch_size);
            assert_eq!(actual.labels, expected.labels);
            assert_eq!(actual.centroids, expected.centroids);
            assert_eq!(actual.parameters, expected.parameters);
        }
    }

    #[test]
    fn opposite_extreme_positions_retry_the_update_in_f64() {
        let mut state = State::new(as_continuous_vec(&[[f32::MAX, -f32::MAX]]), 2);
        state.masses[0] = 1.0;
        let vectors = [-f32::MAX, f32::MAX];
        state.observe(&vectors, 0, 0, false);
        // Both means are finite, but mean - previous overflows FP32. Equal
        // retained and observed mass should place the centroid at their midpoint.
        state.update(&vectors, &[0], &[0], 0.5).unwrap();
        assert_eq!(&*state.positions, &[0.0, 0.0]);
        assert_eq!(state.masses, vec![1.0]);
    }

    #[test]
    fn revival_splits_distinct_donors_and_handles_zero_positions() {
        let mut rng = seeded_rng();
        let mut state = State::new(as_continuous_vec(&[[2.0, 3.0, 4.0], [0.0; 3]]), 3);
        state.masses.copy_from_slice(&[100.0, 0.0]);
        assert_eq!(state.revive(0.1, &mut rng).unwrap(), 1);
        assert_eq!(state.masses, vec![50.0, 50.0]);
        for coordinate in 0..3 {
            assert_ne!(state.positions[coordinate], state.positions[3 + coordinate]);
        }
        state.positions.fill(0.0);
        state.masses.copy_from_slice(&[INITIAL_MASS, 0.0]);
        assert_eq!(state.revive(0.1, &mut rng).unwrap(), 1);
        assert_eq!(state.positions[3], f32::from_bits(1));
        assert!(state.positions.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn fits_preserve_reduction_cosine_and_extreme_mean_contracts() {
        for distance in [
            Distance::SquaredEuclidean,
            Distance::Cosine,
            Distance::NegativeDotProduct,
        ] {
            for reduction in [
                ReductionConfig::None,
                ReductionConfig::PCA {
                    output_dim: 1,
                    training_samples: None,
                },
                ReductionConfig::SRHT { output_dim: 1 },
            ] {
                let model = KMeans::new(KMeansConfig {
                    n_clusters: Some(1),
                    distance,
                    reduction,
                    seed: Some(11),
                    algorithm: TrainingAlgorithm::Emeans(EmeansConfig::default()),
                    ..Default::default()
                });
                let magnitude = if matches!(reduction, ReductionConfig::SRHT { .. }) {
                    3.0
                } else {
                    f32::MAX
                };
                let rows = as_continuous_vec(&[[magnitude; 2]; 40]);
                let fit = model.fit_sample(rows, 2).unwrap();
                assert_eq!(fit.labels, vec![0; 40]);
                let expected = if distance == Distance::SquaredEuclidean {
                    magnitude
                } else {
                    0.5_f32.sqrt()
                };
                for &value in fit.centroids.iter() {
                    assert!((value - expected).abs() <= expected.abs() * 1e-6);
                }
            }
        }
        // A cancelling cosine mean must still return a finite unit direction.
        let model = KMeans::new(KMeansConfig {
            n_clusters: Some(1),
            distance: Distance::Cosine,
            algorithm: TrainingAlgorithm::Emeans(EmeansConfig::default()),
            ..Default::default()
        });
        let rows = (0..40)
            .map(|row| [if row % 2 == 0 { 1.0 } else { -1.0 }, 0.0])
            .collect::<Vec<_>>();
        let fit = model.fit_sample(as_continuous_vec(&rows), 2).unwrap();
        assert_eq!(fit.centroids[0].abs(), 1.0);
        assert_eq!(fit.centroids[1], 0.0);
    }

    #[test]
    fn invalid_recipes_are_rejected_before_training() {
        for config in [
            EmeansConfig {
                batch_size: Some(0),
                ..Default::default()
            },
            EmeansConfig {
                alpha0: Some(f64::NAN),
                ..Default::default()
            },
            EmeansConfig {
                alpha0: Some(1e-300),
                ..Default::default()
            },
            EmeansConfig {
                alpha0: Some(2.0),
                ..Default::default()
            },
            EmeansConfig {
                revival_threshold: Some(-1.0),
                ..Default::default()
            },
        ] {
            assert!(config.resolve(100, 3, 2, 3).is_err());
        }
        assert_eq!(
            EmeansConfig::default()
                .resolve(100, 2, 2, 3)
                .unwrap()
                .revival_threshold,
            0.0
        );
        assert!(EmeansConfig::default().resolve(100, 3, 2, 0).is_err());
        assert_eq!(KMeansConfig::default().algorithm, TrainingAlgorithm::Lloyd);
    }
}
