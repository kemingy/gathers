//! A centroid hierarchy for fast assignment.
//!
//! Fine centroids are grouped bottom-up into routing layers. A layer of `m` centroids
//! proposes `min(sqrt(m), m / 196)` parents and is created only when the proposal reaches
//! 16, so a hierarchy exists iff the fine level has at least `16 * 196 = 3,136` centroids.
//! The 196 figure only sizes the proposal; actual cells may be uneven and are not
//! rebalanced.
//!
//! Training assignment traverses best-first using a heap over cells ordered by
//! center score, scanning a bounded budget of leaf cells per query. The classic
//! radii-based exact early stop was measured to prune nothing in high dimensions (even at
//! the 95th-percentile radius, lower bounds stay vacuous), so it is omitted; the budget
//! keeps cost predictable for every distance. Routing layers are always trained with
//! squared Euclidean, the standard monotone surrogate for unit-norm dot/cosine routing.
//! See `docs/hierarchical-kmeans.md` for the design rules.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use aligned_vec::AVec;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

#[path = "hierarchy/adaptive.rs"]
mod adaptive;
#[path = "hierarchy/batched.rs"]
mod batched;
pub(crate) use adaptive::{GROUPED_ROWS, ProbeWorkspace};
#[path = "hierarchy/policy.rs"]
pub(crate) mod policy;

use crate::distance::Distance;
use crate::kmeans::{AssignmentPolicy, KMeans, KMeansConfig, ReductionConfig};

/// Expected children per routing centroid, used to size layer proposals; not a minimum.
const EXPECTED_CHILDREN: usize = 196;
/// A routing layer is created only when its proposed size reaches this count.
const MIN_ROUTING_CENTROIDS: usize = 16;
/// Leaf cells scanned per query in best-first search: `sqrt(top-level cells)` capped at
/// 10. No established rule exists for the probe count (Faiss/FLANN tune it empirically);
/// this follows the `nlist ≈ sqrt(n)` precedent applied to the routing layer. The budget
/// trades recall for predictable cost but provides no recall guarantee.
#[cfg(test)]
fn scan_budget(cells: usize) -> usize {
    ((cells as f64).sqrt().ceil() as usize).clamp(1, 10)
}
const RAYON_BLOCK_SIZE: usize = 64;

#[derive(Debug)]
struct Level {
    centroids: AVec<f32>,
    /// Node j owns children `child_offsets[j]..child_offsets[j + 1]` in the next lower
    /// level. Empty for the fine level.
    child_offsets: Vec<u32>,
    /// Node j covers fine centroids `fine_offsets[j]..fine_offsets[j + 1]` in level-0
    /// (permuted) order.
    fine_offsets: Vec<u32>,
    /// L2 norm per node, so cosine scoring can skip normalizing the stored centroids.
    norms: Vec<f32>,
}

impl Level {
    fn len(&self) -> usize {
        self.norms.len()
    }
}

/// A multilayer routing hierarchy over trained centroids.
///
/// Level 0 holds the fine centroids permuted so each routing node's children are
/// contiguous; assignment results are mapped back to the original centroid indices.
#[derive(Debug)]
pub struct Hierarchy {
    dim: usize,
    levels: Vec<Level>,
    /// Fine-level position -> original centroid index.
    leaf_order: Vec<u32>,
}

/// Proposed routing-layer size over `m` centroids; `None` when a layer is not worthwhile.
fn proposed_parents(m: usize) -> Option<usize> {
    let proposal = ((m as f64).sqrt() as usize).min(m / EXPECTED_CHILDREN);
    (proposal >= MIN_ROUTING_CENTROIDS).then_some(proposal)
}

fn l2_norm(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt() as f32
}

fn gather(values: &[f32], order: &[u32], dim: usize) -> AVec<f32> {
    let mut output = AVec::with_capacity(64, values.len());
    for &source in order {
        let start = source as usize * dim;
        output.extend_from_slice(&values[start..start + dim]);
    }
    output
}

fn permute<T: Clone>(values: &[T], order: &[u32]) -> Vec<T> {
    if values.is_empty() {
        return Vec::new();
    }
    order
        .iter()
        .map(|&source| values[source as usize].clone())
        .collect()
}

/// Rebuild CSR offsets for reordered contiguous runs. The referenced lower-level rows
/// must be reordered separately. Empty offsets (fine level) stay empty.
fn permuted_offsets(offsets: &[u32], order: &[u32]) -> Vec<u32> {
    if offsets.is_empty() {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(offsets.len());
    output.push(0);
    for &source in order {
        let size = offsets[source as usize + 1] - offsets[source as usize];
        output.push(output.last().unwrap() + size);
    }
    output
}

/// Move entire descendant runs together when a routing level is regrouped.
/// `order` describes the new order of the highest supplied level.
fn reorder_descendants(
    levels: &mut [Level],
    leaf_order: &mut Vec<u32>,
    mut order: Vec<u32>,
    dim: usize,
) {
    for level in levels.iter_mut().rev() {
        let children = if level.child_offsets.is_empty() {
            Vec::new()
        } else {
            order
                .iter()
                .flat_map(|&node| {
                    level.child_offsets[node as usize]..level.child_offsets[node as usize + 1]
                })
                .collect()
        };
        level.centroids = gather(&level.centroids, &order, dim);
        level.norms = permute(&level.norms, &order);
        level.child_offsets = permuted_offsets(&level.child_offsets, &order);
        level.fine_offsets = permuted_offsets(&level.fine_offsets, &order);
        if level.child_offsets.is_empty() {
            *leaf_order = permute(leaf_order, &order);
        }
        order = children;
    }
}

/// Min-heap entry keyed by center score for every distance. Ties resolve
/// toward deeper nodes so a leaf cell scans before its siblings expand.
struct ScoreEntry {
    score: f32,
    level: u32,
    node: u32,
}

impl PartialEq for ScoreEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for ScoreEntry {}
impl PartialOrd for ScoreEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoreEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(other.level.cmp(&self.level))
            .then(other.node.cmp(&self.node))
    }
}

/// Score a contiguous candidate range against one query. L2 uses squared distances;
/// cosine ranks by `-dot / ||centroid||` (the query norm is constant per query) and
/// zero-norm centroids never win.
#[allow(clippy::too_many_arguments)]
fn score_range<S: pulp::Simd>(
    simd: S,
    centroids: &[f32],
    norms: &[f32],
    query: &[f32],
    dim: usize,
    distance: Distance,
    scores: &mut Vec<f32>,
) {
    scores.clear();
    scores.resize(norms.len(), 0.0);
    for ((score, centroid), &norm) in scores
        .iter_mut()
        .zip(centroids.chunks_exact(dim))
        .zip(norms)
    {
        *score = match distance {
            Distance::SquaredEuclidean => {
                crate::simd::pulp::l2_squared_distance(simd, query, centroid)
            }
            Distance::NegativeDotProduct => -crate::simd::pulp::dot_product(simd, query, centroid),
            Distance::Cosine => {
                if norm > 0.0 {
                    -crate::simd::pulp::dot_product(simd, query, centroid) / norm
                } else {
                    f32::INFINITY
                }
            }
        };
    }
}

#[derive(Default)]
struct SearchScratch {
    heap: BinaryHeap<ScoreEntry>,
    scores: Vec<f32>,
}

impl Hierarchy {
    pub(crate) fn supports(centroids: usize) -> bool {
        proposed_parents(centroids).is_some()
    }

    /// Build a hierarchy over flat row-major `centroids`, or return `None` when the layer
    /// rules do not permit one (fewer than `16 * 196 = 3,136` fine centroids, or the
    /// square-root rule stops earlier at deeper levels), or routing-layer training produces
    /// unrepresentable values.
    ///
    /// The seed controls routing-layer training. Panics if `centroids` is empty, `dim` is
    /// zero, or rows are incomplete.
    pub fn build(centroids: &[f32], dim: usize, seed: Option<u64>) -> Option<Self> {
        assert!(dim > 0, "dimension must be greater than zero");
        assert!(
            !centroids.is_empty() && centroids.len().is_multiple_of(dim),
            "centroids must be complete and non-empty"
        );
        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        Self::build_inner(&mut rng, centroids, dim)
    }

    fn build_inner(rng: &mut StdRng, centroids: &[f32], dim: usize) -> Option<Self> {
        let m = centroids.len() / dim;
        proposed_parents(m)?;

        let mut level = Level {
            centroids: {
                let mut copy = AVec::with_capacity(64, centroids.len());
                copy.extend_from_slice(centroids);
                copy
            },
            child_offsets: Vec::new(),
            fine_offsets: (0..=m as u32).collect(),
            norms: centroids.chunks_exact(dim).map(l2_norm).collect(),
        };
        let mut leaf_order: Option<Vec<u32>> = None;
        let mut levels: Vec<Level> = Vec::new();

        while let Some(parents) = proposed_parents(level.len()) {
            // fit_sample trains every supplied row; the 196-based proposal keeps the
            // expected cell size above its 39-rows-per-cluster minimum.
            let config = KMeansConfig {
                n_clusters: Some(parents as u32),
                max_iter: 25,
                assignment: AssignmentPolicy::Exact,
                distance: Distance::SquaredEuclidean,
                reduction: ReductionConfig::None,
                seed: Some(rng.random()),
                ..Default::default()
            };
            let fit = KMeans::new(config)
                .fit_sample(level.centroids.clone(), dim)
                .ok()?;
            let count = level.len();
            debug_assert_eq!(fit.labels.len(), count);

            // Group the current level by parent so every routing node's children are
            // contiguous, then finalize it.
            let mut order: Vec<u32> = (0..count as u32).collect();
            order.sort_by_key(|&index| fit.labels[index as usize]);
            let mut child_offsets = vec![0_u32; parents + 1];
            for &label in &fit.labels {
                child_offsets[label as usize + 1] += 1;
            }
            for index in 0..parents {
                child_offsets[index + 1] += child_offsets[index];
            }
            if let Some(leaf_order) = &mut leaf_order {
                let descendants = order
                    .iter()
                    .flat_map(|&node| {
                        level.child_offsets[node as usize]..level.child_offsets[node as usize + 1]
                    })
                    .collect();
                reorder_descendants(&mut levels, leaf_order, descendants, dim);
            }
            let grouped_fine_offsets = permuted_offsets(&level.fine_offsets, &order);
            let grouped = Level {
                centroids: gather(&level.centroids, &order, dim),
                child_offsets: permuted_offsets(&level.child_offsets, &order),
                norms: permute(&level.norms, &order),
                fine_offsets: grouped_fine_offsets.clone(),
            };
            if leaf_order.is_none() {
                leaf_order = Some(order);
            }
            levels.push(grouped);

            // Each routing node covers the contiguous union of its children's fine ranges.
            let mut fine_offsets = Vec::with_capacity(parents + 1);
            for node in 0..parents {
                fine_offsets.push(grouped_fine_offsets[child_offsets[node] as usize]);
            }
            fine_offsets.push(grouped_fine_offsets[child_offsets[parents] as usize]);

            level = Level {
                norms: fit.centroids.chunks_exact(dim).map(l2_norm).collect(),
                centroids: fit.centroids,
                child_offsets,
                fine_offsets,
            };
        }

        levels.push(level);
        Some(Self {
            dim,
            levels,
            leaf_order: leaf_order.expect("a routing layer was created"),
        })
    }

    /// Number of levels, including the fine level; at least 2.
    #[cfg(test)]
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    fn validate(&self, vecs: &[f32], dim: usize, labels: &[u32]) {
        assert_eq!(dim, self.dim, "dimension must match the hierarchy");
        assert!(vecs.len().is_multiple_of(dim), "vectors must be complete");
        assert_eq!(
            labels.len(),
            vecs.len() / dim,
            "one label is required per vector"
        );
    }

    /// Approximately assign vectors to original centroid indices with greedy descent.
    ///
    /// Scans the top layer, then one child list per level: approximate, minimal work.
    #[cfg(test)]
    pub fn assign_greedy(&self, vecs: &[f32], dim: usize, distance: Distance, labels: &mut [u32]) {
        self.validate(vecs, dim, labels);
        let arch = pulp::Arch::new();
        let mut scratch = SearchScratch::default();
        for (labels, vecs) in labels
            .chunks_mut(RAYON_BLOCK_SIZE)
            .zip(vecs.chunks(dim * RAYON_BLOCK_SIZE))
        {
            arch.dispatch(GreedyBlock {
                hierarchy: self,
                vecs,
                distance,
                labels,
                scratch: &mut scratch,
            });
        }
    }

    /// Greedy descent in parallel, with per-task reusable scratch.
    #[cfg(test)]
    pub fn assign_greedy_parallel(
        &self,
        vecs: &[f32],
        dim: usize,
        distance: Distance,
        labels: &mut [u32],
    ) {
        self.validate(vecs, dim, labels);
        labels
            .par_chunks_mut(RAYON_BLOCK_SIZE)
            .zip(vecs.par_chunks(dim * RAYON_BLOCK_SIZE))
            .for_each_init(SearchScratch::default, |scratch, (labels, vecs)| {
                pulp::Arch::new().dispatch(GreedyBlock {
                    hierarchy: self,
                    vecs,
                    distance,
                    labels,
                    scratch,
                });
            });
    }

    /// Assign vectors with best-first backtracking.
    ///
    /// Cells are explored in center-score order and the scan is capped at
    /// `sqrt(top-level cells)` leaf cells per query (at most 10), so cost is bounded and
    /// predictable; results are approximate. Cosine ranking needs no query normalization:
    /// the query norm is constant per query, so `-dot / ||centroid||` ranks identically.
    /// Callers needing an exact scan can use [`crate::kmeans::base_assign`].
    #[cfg(test)]
    pub fn assign(&self, vecs: &[f32], dim: usize, distance: Distance, labels: &mut [u32]) {
        self.validate(vecs, dim, labels);
        let arch = pulp::Arch::new();
        let mut scratch = SearchScratch::default();
        for (labels, vecs) in labels
            .chunks_mut(RAYON_BLOCK_SIZE)
            .zip(vecs.chunks(dim * RAYON_BLOCK_SIZE))
        {
            arch.dispatch(BestFirstBlock {
                hierarchy: self,
                vecs,
                distance,
                labels,
                scratch: &mut scratch,
            });
        }
    }

    /// Best-first backtracking in parallel, with per-task reusable scratch.
    #[cfg(test)]
    pub fn assign_parallel(
        &self,
        vecs: &[f32],
        dim: usize,
        distance: Distance,
        labels: &mut [u32],
    ) {
        self.validate(vecs, dim, labels);
        labels
            .par_chunks_mut(RAYON_BLOCK_SIZE)
            .zip(vecs.par_chunks(dim * RAYON_BLOCK_SIZE))
            .for_each_init(SearchScratch::default, |scratch, (labels, vecs)| {
                pulp::Arch::new().dispatch(BestFirstBlock {
                    hierarchy: self,
                    vecs,
                    distance,
                    labels,
                    scratch,
                });
            });
    }

    #[cfg(test)]
    fn greedy_one<S: pulp::Simd>(
        &self,
        simd: S,
        query: &[f32],
        distance: Distance,
        scratch: &mut SearchScratch,
    ) -> u32 {
        let top = self.levels.len() - 1;
        let mut node = self.scan_children(
            simd,
            top,
            0,
            self.levels[top].len(),
            query,
            distance,
            scratch,
            true,
        );
        for level in (1..=top).rev() {
            let start = self.levels[level].child_offsets[node] as usize;
            let end = self.levels[level].child_offsets[node + 1] as usize;
            let lower = level - 1;
            // Skip dead routing nodes (empty child ranges); fine centroids are never dead.
            node = start
                + self.scan_children(simd, lower, start, end, query, distance, scratch, lower > 0);
        }
        self.leaf_order[node]
    }

    /// Score nodes `start..end` of `level` and return the offset of the best live one.
    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    fn scan_children<S: pulp::Simd>(
        &self,
        simd: S,
        level: usize,
        start: usize,
        end: usize,
        query: &[f32],
        distance: Distance,
        scratch: &mut SearchScratch,
        skip_dead: bool,
    ) -> usize {
        let level_ref = &self.levels[level];
        score_range(
            simd,
            &level_ref.centroids[start * self.dim..end * self.dim],
            &level_ref.norms[start..end],
            query,
            self.dim,
            distance,
            &mut scratch.scores,
        );
        scratch
            .scores
            .iter()
            .enumerate()
            .filter(|(offset, _)| {
                !skip_dead
                    || level_ref.fine_offsets[start + offset + 1]
                        > level_ref.fine_offsets[start + offset]
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(offset, _)| offset)
            .expect("at least one live routing node")
    }

    #[cfg(test)]
    fn best_first_one<S: pulp::Simd>(
        &self,
        simd: S,
        query: &[f32],
        distance: Distance,
        scratch: &mut SearchScratch,
    ) -> u32 {
        self.probe_one(
            simd,
            query,
            distance,
            scratch,
            scan_budget(self.levels.last().unwrap().len()),
            None,
        )
        .0
    }

    fn probe_one<S: pulp::Simd>(
        &self,
        simd: S,
        query: &[f32],
        distance: Distance,
        scratch: &mut SearchScratch,
        mut budget: usize,
        incumbent: Option<(f32, u32)>,
    ) -> (u32, usize) {
        scratch.heap.clear();
        let top = self.levels.len() - 1;
        score_range(
            simd,
            &self.levels[top].centroids,
            &self.levels[top].norms,
            query,
            self.dim,
            distance,
            &mut scratch.scores,
        );
        for (node, &score) in scratch.scores.iter().enumerate() {
            scratch.heap.push(ScoreEntry {
                score,
                level: top as u32,
                node: node as u32,
            });
        }
        let (mut best, mut best_index) = incumbent.unwrap_or((f32::INFINITY, u32::MAX));
        let mut probes = 0;
        let mut required = 0;
        while let Some(entry) = scratch.heap.pop() {
            let level = entry.level as usize;
            let node = entry.node as usize;
            let start = self.levels[level].child_offsets[node] as usize;
            let end = self.levels[level].child_offsets[node + 1] as usize;
            if self.levels[level].fine_offsets[node] == self.levels[level].fine_offsets[node + 1] {
                continue;
            }
            if level == 1 {
                if budget == 0 {
                    break;
                }
                budget -= 1;
                probes += 1;
                // Leaf cell: exact fine scan.
                score_range(
                    simd,
                    &self.levels[0].centroids[start * self.dim..end * self.dim],
                    &self.levels[0].norms[start..end],
                    query,
                    self.dim,
                    distance,
                    &mut scratch.scores,
                );
                for (offset, &score) in scratch.scores.iter().enumerate() {
                    let index = self.leaf_order[start + offset];
                    if score < best || (score == best && index < best_index) {
                        best = score;
                        best_index = index;
                        required = probes;
                    }
                }
            } else {
                score_range(
                    simd,
                    &self.levels[level - 1].centroids[start * self.dim..end * self.dim],
                    &self.levels[level - 1].norms[start..end],
                    query,
                    self.dim,
                    distance,
                    &mut scratch.scores,
                );
                for (offset, &score) in scratch.scores.iter().enumerate() {
                    scratch.heap.push(ScoreEntry {
                        score,
                        level: (level - 1) as u32,
                        node: (start + offset) as u32,
                    });
                }
            }
        }
        (best_index, required)
    }

    /// Experiment-only configurable probing. Empty `required` omits pilot tracking.
    pub(crate) fn probe_parallel(
        &self,
        vectors: &[f32],
        labels: &mut [u32],
        required: &mut [usize],
        options: ProbeOptions<'_>,
    ) {
        self.validate(vectors, self.dim, labels);
        assert!(options.budget > 0);
        if required.is_empty() {
            labels
                .par_chunks_mut(RAYON_BLOCK_SIZE)
                .zip(vectors.par_chunks(self.dim * RAYON_BLOCK_SIZE))
                .for_each_init(SearchScratch::default, |scratch, (labels, vectors)| {
                    pulp::Arch::new().dispatch(ProbeBlock {
                        hierarchy: self,
                        vectors,
                        labels,
                        required: &mut [],
                        options,
                        scratch,
                    });
                });
        } else {
            assert_eq!(required.len(), labels.len());
            labels
                .par_chunks_mut(RAYON_BLOCK_SIZE)
                .zip(vectors.par_chunks(self.dim * RAYON_BLOCK_SIZE))
                .zip(required.par_chunks_mut(RAYON_BLOCK_SIZE))
                .for_each_init(
                    SearchScratch::default,
                    |scratch, ((labels, vectors), required)| {
                        pulp::Arch::new().dispatch(ProbeBlock {
                            hierarchy: self,
                            vectors,
                            labels,
                            required,
                            options,
                            scratch,
                        });
                    },
                );
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ProbeOptions<'a> {
    pub(crate) distance: Distance,
    pub(crate) budget: usize,
    /// Original centroid order; previous labels are reevaluated at current coordinates.
    pub(crate) incumbent_centroids: Option<&'a [f32]>,
}

struct ProbeBlock<'a> {
    hierarchy: &'a Hierarchy,
    vectors: &'a [f32],
    labels: &'a mut [u32],
    required: &'a mut [usize],
    options: ProbeOptions<'a>,
    scratch: &'a mut SearchScratch,
}

impl pulp::WithSimd for ProbeBlock<'_> {
    type Output = ();

    fn with_simd<S: pulp::Simd>(self, simd: S) {
        for (row, (label, query)) in self
            .labels
            .iter_mut()
            .zip(self.vectors.chunks_exact(self.hierarchy.dim))
            .enumerate()
        {
            let incumbent = self.options.incumbent_centroids.map(|centroids| {
                let start = *label as usize * self.hierarchy.dim;
                let centroid = &centroids[start..start + self.hierarchy.dim];
                let score = match self.options.distance {
                    Distance::SquaredEuclidean => {
                        crate::simd::pulp::l2_squared_distance(simd, query, centroid)
                    }
                    Distance::NegativeDotProduct => {
                        -crate::simd::pulp::dot_product(simd, query, centroid)
                    }
                    Distance::Cosine => unreachable!("normalize training rows before probing"),
                };
                (score, *label)
            });
            let (winner, required) = self.hierarchy.probe_one(
                simd,
                query,
                self.options.distance,
                self.scratch,
                self.options.budget,
                incumbent,
            );
            *label = winner;
            if !self.required.is_empty() {
                self.required[row] = required;
            }
        }
    }
}

#[cfg(test)]
struct GreedyBlock<'a> {
    hierarchy: &'a Hierarchy,
    vecs: &'a [f32],
    distance: Distance,
    labels: &'a mut [u32],
    scratch: &'a mut SearchScratch,
}

#[cfg(test)]
impl pulp::WithSimd for GreedyBlock<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        for (label, query) in self
            .labels
            .iter_mut()
            .zip(self.vecs.chunks_exact(self.hierarchy.dim))
        {
            *label = self
                .hierarchy
                .greedy_one(simd, query, self.distance, self.scratch);
        }
    }
}

#[cfg(test)]
struct BestFirstBlock<'a> {
    hierarchy: &'a Hierarchy,
    vecs: &'a [f32],
    distance: Distance,
    labels: &'a mut [u32],
    scratch: &'a mut SearchScratch,
}

#[cfg(test)]
impl pulp::WithSimd for BestFirstBlock<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        for (label, query) in self
            .labels
            .iter_mut()
            .zip(self.vecs.chunks_exact(self.hierarchy.dim))
        {
            *label = self
                .hierarchy
                .best_first_one(simd, query, self.distance, self.scratch);
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use seed_rand::seeded_rng;

    use super::{
        EXPECTED_CHILDREN, Hierarchy, Level, MIN_ROUTING_CENTROIDS, ProbeOptions, proposed_parents,
    };
    use crate::distance::Distance;
    use crate::kmeans::base_assign;

    fn random_rows(rows: usize, dim: usize) -> Vec<f32> {
        let mut rng = seeded_rng();
        (0..rows * dim).map(|_| rng.random::<f32>() - 0.5).collect()
    }

    /// 100 fine centroids per blob, blobs far apart along coordinate 0.
    fn blob_centroids(blobs: usize, per_blob: usize, dim: usize) -> Vec<f32> {
        let mut rng = seeded_rng();
        let mut centroids = Vec::new();
        for blob in 0..blobs {
            for _ in 0..per_blob {
                let mut centroid = vec![0.0; dim];
                centroid[0] = blob as f32 * 100.0;
                for value in centroid.iter_mut().skip(1) {
                    *value = rng.random::<f32>() - 0.5;
                }
                centroids.extend_from_slice(&centroid);
            }
        }
        centroids
    }

    fn fixture_level(centroids: &[f32], child_offsets: &[u32], fine_offsets: &[u32]) -> Level {
        Level {
            centroids: crate::utils::as_continuous_vec(&[centroids]),
            norms: centroids.iter().map(|value| value.abs()).collect(),
            child_offsets: child_offsets.to_vec(),
            fine_offsets: fine_offsets.to_vec(),
        }
    }

    #[test]
    fn regrouping_moves_all_descendant_ranges_and_original_labels() {
        // Unequal runs expose offset-only permutations; include two routing levels.
        let fine = [0.0, 1.0, 100.0, 101.0, 102.0, 200.0];
        let mut hierarchy = Hierarchy {
            dim: 1,
            levels: vec![
                fixture_level(&fine, &[], &[0, 1, 2, 3, 4, 5, 6]),
                fixture_level(&[0.5, 101.0, 200.0], &[0, 2, 5, 6], &[0, 2, 5, 6]),
                fixture_level(&[50.75, 200.0], &[0, 2, 3], &[0, 5, 6]),
            ],
            leaf_order: (0..6).collect(),
        };
        super::reorder_descendants(
            &mut hierarchy.levels,
            &mut hierarchy.leaf_order,
            vec![1, 0],
            1,
        );
        assert_eq!(hierarchy.leaf_order, [5, 0, 1, 2, 3, 4]);
        assert_eq!(hierarchy.levels[1].child_offsets, [0, 1, 3, 6]);
        assert_eq!(hierarchy.levels[2].child_offsets, [0, 1, 3]);
        assert_eq!(hierarchy.levels[2].fine_offsets, [0, 1, 6]);
        for (position, &original) in hierarchy.leaf_order.iter().enumerate() {
            assert_eq!(
                hierarchy.levels[0].centroids[position],
                fine[original as usize]
            );
        }
        for assign in [
            Hierarchy::assign,
            Hierarchy::assign_parallel,
            Hierarchy::assign_greedy,
            Hierarchy::assign_greedy_parallel,
        ] {
            let mut labels = [u32::MAX; 6];
            assign(
                &hierarchy,
                &fine,
                1,
                Distance::SquaredEuclidean,
                &mut labels,
            );
            assert_eq!(labels, [0, 1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn empty_cells_do_not_consume_the_scan_budget() {
        let mut offsets = vec![0; 15];
        offsets.extend([1, 2]);
        let mut centers = vec![100.0; 14];
        centers.extend([1.0, 2.0]);
        let mut hierarchy = Hierarchy {
            dim: 1,
            levels: vec![
                fixture_level(&[1.0, 2.0], &[], &[0, 1, 2]),
                fixture_level(&centers, &offsets, &offsets),
            ],
            leaf_order: vec![0, 1],
        };
        for depth in 2..=3 {
            if depth == 3 {
                // The first top node owns routing nodes, but no fine descendants.
                hierarchy
                    .levels
                    .push(fixture_level(&[100.0, 2.0], &[0, 14, 16], &[0, 0, 2]));
            }
            for assign in [
                Hierarchy::assign,
                Hierarchy::assign_parallel,
                Hierarchy::assign_greedy,
                Hierarchy::assign_greedy_parallel,
            ] {
                let mut labels = [u32::MAX];
                assign(
                    &hierarchy,
                    &[1.0],
                    1,
                    Distance::NegativeDotProduct,
                    &mut labels,
                );
                assert_eq!(labels, [1]);
            }
        }
    }

    #[test]
    fn layer_proposal_boundaries() {
        assert_eq!(
            proposed_parents(MIN_ROUTING_CENTROIDS * EXPECTED_CHILDREN - 1),
            None
        );
        assert_eq!(proposed_parents(3_136), Some(16));
        assert_eq!(proposed_parents(3_200), Some(16));
        // sqrt rule dominates once sqrt(m) < m / 196: sqrt(40_000) = 200 > 204.
        assert_eq!(proposed_parents(40_000), Some(200));
        assert_eq!(proposed_parents(1_000_000), Some(1000));
    }

    #[test]
    fn build_groups_children_contiguously() {
        let dim = 4;
        let centroids = random_rows(3_200, dim);
        let hierarchy = Hierarchy::build(&centroids, dim, Some(42)).unwrap();
        assert_eq!(hierarchy.levels(), 2);
        let fine = &hierarchy.levels[0];
        let routing = &hierarchy.levels[1];
        assert_eq!(routing.len(), 16);

        // Child ranges tile the fine level without gaps or overlaps.
        assert_eq!(routing.child_offsets[0], 0);
        assert_eq!(*routing.child_offsets.last().unwrap(), 3_200);
        assert!(routing.child_offsets.windows(2).all(|w| w[0] <= w[1]));

        // leaf_order is a permutation mapping back to the original centroids.
        let mut order = hierarchy.leaf_order.clone();
        order.sort_unstable();
        assert_eq!(order, (0..3_200).collect::<Vec<_>>());
        for (position, &original) in hierarchy.leaf_order.iter().enumerate() {
            for coordinate in 0..dim {
                assert_eq!(
                    fine.centroids[position * dim + coordinate],
                    centroids[original as usize * dim + coordinate]
                );
            }
        }

        // Routing centroids are cell means, and norms match the centroids.
        for node in 0..routing.len() {
            let center = &routing.centroids[node * dim..(node + 1) * dim];
            let start = routing.fine_offsets[node] as usize;
            let end = routing.fine_offsets[node + 1] as usize;
            if start == end {
                continue;
            }
            let mut mean = vec![0.0_f64; dim];
            for member in fine.centroids[start * dim..end * dim].chunks_exact(dim) {
                for (m, &v) in mean.iter_mut().zip(member) {
                    *m += f64::from(v);
                }
            }
            let count = (end - start) as f64;
            for coordinate in 0..dim {
                let expected = (mean[coordinate] / count) as f32;
                assert!(
                    (center[coordinate] - expected).abs() < 1e-4 + 1e-4 * expected.abs(),
                    "node {node} coordinate {coordinate}: {} != {expected}",
                    center[coordinate]
                );
            }
            assert!((routing.norms[node] - super::l2_norm(center)).abs() < 1e-4);
        }

        // One centroids fewer than the build threshold yields no hierarchy.
        assert!(Hierarchy::build(&random_rows(3_135, dim), dim, Some(42)).is_none());
    }

    #[test]
    fn best_first_l2_matches_flat_assignment_at_low_dimension() {
        // With well-separated cells (low dimension), the sqrt budget finds the exact
        // winner; high-dimensional uniform data is near-tied by concentration and is
        // covered by the recall test instead.
        let k = 3_200;
        let mut rng = seeded_rng();
        for dim in [1, 3] {
            let data: Vec<f32> = (0..(k + 200) * dim)
                .map(|_| rng.random::<f32>() - 0.5)
                .collect();
            let (centroids, queries) = data.split_at(k * dim);
            let hierarchy = Hierarchy::build(centroids, dim, Some(42)).unwrap();
            let mut expected = vec![0; 200];
            base_assign(
                queries,
                centroids,
                dim,
                Distance::SquaredEuclidean,
                &mut expected,
            );
            for actual in [
                Hierarchy::assign as fn(&Hierarchy, &[f32], usize, Distance, &mut [u32]),
                Hierarchy::assign_parallel,
            ] {
                let mut labels = vec![u32::MAX; 200];
                actual(
                    &hierarchy,
                    queries,
                    dim,
                    Distance::SquaredEuclidean,
                    &mut labels,
                );
                assert_eq!(labels, expected, "dimension {dim}");
            }
        }
    }

    #[test]
    fn greedy_descent_finds_the_blob_on_separated_data() {
        let dim = 8;
        let centroids = blob_centroids(32, 100, dim);
        let hierarchy = Hierarchy::build(&centroids, dim, Some(42)).unwrap();
        // Queries are fine centroids plus small noise: the source centroid stays nearest.
        let mut queries = Vec::new();
        let mut rng = seeded_rng();
        for query in 0..100 {
            let index = (query * 31) % 3_200;
            let mut vector = centroids[index * dim..index * dim + dim].to_vec();
            vector[0] += rng.random::<f32>() - 0.5;
            queries.extend_from_slice(&vector);
        }
        let mut expected = vec![0; 100];
        base_assign(
            &queries,
            &centroids,
            dim,
            Distance::SquaredEuclidean,
            &mut expected,
        );
        let mut labels = vec![0; 100];
        hierarchy.assign_greedy(&queries, dim, Distance::SquaredEuclidean, &mut labels);
        assert_eq!(labels, expected);
    }

    #[test]
    fn build_returns_none_for_numerical_overflow() {
        let centroids = vec![f32::MAX; 16 * 196];
        assert!(Hierarchy::build(&centroids, 1, Some(42)).is_none());
    }

    #[test]
    fn full_probe_assignment_matches_flat() {
        let dim = 8;
        let mut rng = seeded_rng();
        let queries: Vec<f32> = (0..500 * dim).map(|_| rng.random::<f32>() - 0.5).collect();
        for k in [3_200usize, 16_384] {
            let uniform: Vec<f32> = (0..k * dim).map(|_| rng.random::<f32>() - 0.5).collect();
            // 32 direction blobs with varying magnitudes.
            let mut directional = Vec::new();
            for _ in 0..32 {
                let direction: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
                let norm = direction.iter().map(|v| v * v).sum::<f32>().sqrt();
                for _ in 0..(k / 32) {
                    let scale = 0.5 + rng.random::<f32>();
                    for value in &direction {
                        directional.push(value / norm * scale + (rng.random::<f32>() - 0.5) * 0.1);
                    }
                }
            }
            for (name, centroids) in [("uniform", &uniform), ("directional", &directional)] {
                let hierarchy = Hierarchy::build(centroids, dim, Some(42)).unwrap();
                for distance in [
                    Distance::SquaredEuclidean,
                    Distance::NegativeDotProduct,
                    Distance::Cosine,
                ] {
                    let mut expected = vec![0; 500];
                    base_assign(&queries, centroids, dim, distance, &mut expected);
                    let mut actual = vec![0; 500];
                    hierarchy.probe_parallel(
                        &queries,
                        &mut actual,
                        &mut [],
                        ProbeOptions {
                            distance,
                            budget: usize::MAX,
                            incumbent_centroids: None,
                        },
                    );
                    assert_eq!(actual, expected, "k={k} {name} {distance:?}");
                }
            }
        }
    }

    #[test]
    fn deep_tree_best_first_matches_flat() {
        // Depth 3 needs ~9.8M centroids through the build rules, so hand-build one:
        // 4 top nodes x 8 routing children x 4 fine centroids.
        let dim = 2;
        let mut fine = Vec::new();
        for top in 0..4 {
            for middle in 0..8 {
                for leaf in 0..4 {
                    fine.extend_from_slice(&[
                        top as f32 * 100.0 + middle as f32 * 10.0 + leaf as f32,
                        (top + middle + leaf) as f32 * 0.01,
                    ]);
                }
            }
        }
        let fine_count = fine.len() / dim;
        let level0 = Level {
            norms: fine.chunks_exact(dim).map(super::l2_norm).collect(),
            centroids: {
                let mut copy = aligned_vec::AVec::with_capacity(64, fine.len());
                copy.extend_from_slice(&fine);
                copy
            },
            child_offsets: Vec::new(),
            fine_offsets: (0..=fine_count as u32).collect(),
        };
        let mut levels = vec![level0];
        let mut lower_count = fine_count;
        let mut fan_out = 4;
        for _ in 1..3 {
            let nodes = lower_count / fan_out;
            let mut centroids = aligned_vec::AVec::with_capacity(64, nodes * dim);
            let mut child_offsets = vec![0_u32; nodes + 1];
            let mut fine_offsets = Vec::with_capacity(nodes + 1);
            for node in 0..nodes {
                child_offsets[node + 1] = child_offsets[node] + fan_out as u32;
                fine_offsets.push(node as u32 * fan_out as u32);
            }
            fine_offsets.push(lower_count as u32);
            let fine_level = &levels[0];
            for node in 0..nodes {
                let start = fine_offsets[node] as usize * dim;
                let end = fine_offsets[node + 1] as usize * dim;
                let mut mean = vec![0.0_f32; dim];
                for member in fine_level.centroids[start..end].chunks_exact(dim) {
                    for (value, &member) in mean.iter_mut().zip(member) {
                        *value += member;
                    }
                }
                let count = ((end - start) / dim) as f32;
                mean.iter_mut().for_each(|value| *value *= count.recip());
                centroids.extend_from_slice(&mean);
            }
            levels.push(Level {
                norms: centroids.chunks_exact(dim).map(super::l2_norm).collect(),
                centroids,
                child_offsets,
                fine_offsets,
            });
            lower_count = nodes;
            fan_out = 8;
        }
        let hierarchy = Hierarchy {
            dim,
            levels,
            leaf_order: (0..fine_count as u32).collect(),
        };
        assert_eq!(hierarchy.levels(), 3);
        let queries = random_rows(100, dim);
        let mut expected = vec![0; 100];
        base_assign(
            &queries,
            &fine,
            dim,
            Distance::SquaredEuclidean,
            &mut expected,
        );
        // The budget covers 2 of 4 leaf cells per query on this well-separated grid.
        let mut actual = vec![0; 100];
        hierarchy.assign(&queries, dim, Distance::SquaredEuclidean, &mut actual);
        let recall = actual
            .iter()
            .zip(&expected)
            .filter(|(left, right)| left == right)
            .count();
        assert!(recall >= 90, "deep-tree best-first recall {recall}/100");
        let mut batched = vec![0; 100];
        hierarchy.probe_batched(
            &queries,
            &mut batched,
            super::ProbeOptions {
                distance: Distance::SquaredEuclidean,
                budget: super::scan_budget(hierarchy.levels[2].len()),
                incumbent_centroids: None,
            },
            32,
        );
        assert_eq!(batched, actual, "deeper trees retain the SIMD traversal");
        let mut required = vec![0; queries.len() / dim];
        hierarchy.probe_parallel(
            &queries,
            &mut actual,
            &mut required,
            super::ProbeOptions {
                distance: Distance::SquaredEuclidean,
                budget: usize::MAX,
                incumbent_centroids: None,
            },
        );
        let mut batched_required = vec![0; required.len()];
        hierarchy.calibrate_batched(
            &queries,
            &mut batched,
            &mut batched_required,
            Distance::SquaredEuclidean,
        );
        assert_eq!(batched, actual);
        assert_eq!(
            batched_required, required,
            "deep-tree pilot fallback preserves ranks"
        );
        let mut greedy = vec![0; 100];
        hierarchy.assign_greedy(&queries, dim, Distance::SquaredEuclidean, &mut greedy);
        assert_eq!(greedy, expected);
    }
    fn two_cells(fine: &[f32], routing: &[f32]) -> Hierarchy {
        let aligned = |values: &[f32]| {
            let mut output = aligned_vec::AVec::with_capacity(64, values.len());
            output.extend_from_slice(values);
            output
        };
        Hierarchy {
            dim: 1,
            levels: vec![
                Level {
                    centroids: aligned(fine),
                    norms: fine.iter().map(|v| v.abs()).collect(),
                    child_offsets: vec![],
                    fine_offsets: vec![0, 1, 2],
                },
                Level {
                    centroids: aligned(routing),
                    norms: routing.iter().map(|v| v.abs()).collect(),
                    child_offsets: vec![0, 1, 2],
                    fine_offsets: vec![0, 1, 2],
                },
            ],
            leaf_order: vec![0, 1],
        }
    }

    #[test]
    fn pilot_counts_probes_until_exact_winner_and_reevaluates_incumbent() {
        let h = two_cells(&[10.0, 1.0], &[0.0, 5.0]);
        let mut label = [0];
        let mut required = [0];
        h.probe_parallel(
            &[1.0],
            &mut label,
            &mut required,
            ProbeOptions {
                distance: Distance::SquaredEuclidean,
                budget: usize::MAX,
                incumbent_centroids: None,
            },
        );
        assert_eq!((label, required), ([1], [2]));
        h.probe_parallel(
            &[1.0],
            &mut label,
            &mut [],
            ProbeOptions {
                distance: Distance::SquaredEuclidean,
                budget: 1,
                incumbent_centroids: Some(&[10.0, 1.0]),
            },
        );
        assert_eq!(label, [1]);
        // Same previous label, moved centroid: a stale zero distance must not keep it.
        let moved = two_cells(&[10.0, 100.0], &[0.0, 5.0]);
        moved.probe_parallel(
            &[1.0],
            &mut label,
            &mut [],
            ProbeOptions {
                distance: Distance::SquaredEuclidean,
                budget: 1,
                incumbent_centroids: Some(&[10.0, 100.0]),
            },
        );
        assert_eq!(label, [0]);
    }

    #[test]
    fn dot_incumbent_uses_current_coordinates_and_negative_score() {
        let mut label = [0];
        let h = two_cells(&[10.0, 1.0], &[0.0, 5.0]);
        h.probe_parallel(
            &[1.0],
            &mut label,
            &mut [],
            ProbeOptions {
                distance: Distance::NegativeDotProduct,
                budget: 1,
                incumbent_centroids: Some(&[10.0, 1.0]),
            },
        );
        assert_eq!(label, [0]);
        let moved = two_cells(&[-10.0, 1.0], &[0.0, 5.0]);
        moved.probe_parallel(
            &[1.0],
            &mut label,
            &mut [],
            ProbeOptions {
                distance: Distance::NegativeDotProduct,
                budget: 1,
                incumbent_centroids: Some(&[-10.0, 1.0]),
            },
        );
        assert_eq!(label, [1]);
    }

    #[test]
    fn full_probe_resolves_ties_by_original_index() {
        let h = two_cells(&[1.0, -1.0], &[10.0, 0.0]);
        let mut label = [1];
        let mut required = [0];
        h.probe_parallel(
            &[0.0],
            &mut label,
            &mut required,
            ProbeOptions {
                distance: Distance::SquaredEuclidean,
                budget: usize::MAX,
                incumbent_centroids: None,
            },
        );
        assert_eq!((label, required), ([0], [2]));
    }
}
