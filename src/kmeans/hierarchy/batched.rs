//! Experimental grouped leaf scans using the existing guarded matrix assignment path.
//! Compiled privately by the adaptive-probe benchmark and its integration tests.

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::distance::{Distance, neg_dot_product, squared_euclidean};
use crate::hierarchy::{Hierarchy, ProbeOptions, ScoreEntry, score_range};
use crate::kmeans::base_assign_parallel;
use crate::matrix::MatrixAssignmentWorkspace;

#[derive(Default)]
struct Scratch {
    groups: Vec<Vec<usize>>,
    routing_scores: Vec<f32>,
    order: Vec<usize>,
    gathered: Vec<f32>,
    winners: Vec<u32>,
    best: Vec<f32>,
}

fn score(query: &[f32], centroid: &[f32], distance: Distance) -> f32 {
    match distance {
        Distance::SquaredEuclidean => squared_euclidean(query, centroid),
        Distance::NegativeDotProduct => neg_dot_product(query, centroid),
        Distance::Cosine => unreachable!("normalize training rows before probing"),
    }
}

impl Hierarchy {
    /// Calibrate full-probe pilots with guarded flat assignment, then rank each
    /// winner's live cell in the unchanged SIMD routing order. Deeper trees and
    /// cosine retain the reference traversal. Setup and mapping are per call.
    pub(crate) fn calibrate_batched(
        &self,
        vectors: &[f32],
        labels: &mut [u32],
        required: &mut [usize],
        distance: Distance,
    ) {
        self.validate(vectors, self.dim, labels);
        assert_eq!(labels.len(), required.len());
        let fallback = |labels: &mut [u32], required: &mut [usize]| {
            self.probe_parallel(
                vectors,
                labels,
                required,
                ProbeOptions {
                    distance,
                    budget: usize::MAX,
                    incumbent_centroids: None,
                },
            );
        };
        if self.levels.len() != 2 || distance == Distance::Cosine {
            fallback(labels, required);
            return;
        }
        // Original-ID ordering preserves full-probe ties across different cells.
        let mut centers = vec![0.0; self.levels[0].centroids.len()];
        let mut cells = vec![0; self.leaf_order.len()];
        let routing = &self.levels[1];
        for cell in 0..routing.len() {
            for position in
                routing.child_offsets[cell] as usize..routing.child_offsets[cell + 1] as usize
            {
                let original = self.leaf_order[position] as usize;
                centers[original * self.dim..(original + 1) * self.dim].copy_from_slice(
                    &self.levels[0].centroids[position * self.dim..(position + 1) * self.dim],
                );
                cells[original] = cell;
            }
        }
        // The flat fallback uses a different argmin tie rule when overflow or
        // NaNs invalidate its error bounds. Keep the hierarchy reference for
        // exceptional inputs; this conservative limit also bounds L2 differences.
        let limit = (f32::MAX / (8.0 * self.dim as f32)).sqrt();
        if !vectors.iter().chain(&centers).all(|v| v.abs() <= limit) {
            fallback(labels, required);
            return;
        }
        if let Some(mut workspace) =
            MatrixAssignmentWorkspace::try_new(vectors, cells.len(), self.dim, distance)
        {
            workspace.assign(vectors, &centers, self.dim, labels);
        } else {
            fallback(labels, required);
            return;
        }
        // Match the reference even when no finite candidate can win.
        if labels.iter().any(|&label| label as usize >= cells.len()) {
            fallback(labels, required);
            return;
        }
        required
            .par_chunks_mut(64)
            .zip(labels.par_chunks(64))
            .zip(vectors.par_chunks(64 * self.dim))
            .for_each_init(Vec::new, |scores, ((required, labels), vectors)| {
                pulp::Arch::new().dispatch(RankPilots {
                    hierarchy: self,
                    vectors,
                    labels,
                    required,
                    cells: &cells,
                    distance,
                    scores,
                });
            });
    }

    /// Group a bounded query batch by its selected cells, then assign each group with
    /// the guarded GEMM workspace. Sparse groups retain its ordinary SIMD fallback.
    /// Deeper trees retain the existing traversal; the two real workloads have one
    /// routing layer. Grouping, gathering, and workspace creation are included in timing.
    pub(crate) fn probe_batched(
        &self,
        vectors: &[f32],
        labels: &mut [u32],
        options: ProbeOptions<'_>,
        batch_rows: usize,
    ) {
        self.validate(vectors, self.dim, labels);
        assert!(options.budget > 0 && batch_rows > 0);
        assert!(options.distance != Distance::Cosine);
        if self.levels.len() != 2 {
            self.probe_parallel(vectors, labels, &mut [], options);
            return;
        }
        let routing = &self.levels[1];
        // Local argmin resolves ties by local index. Sorting each cell by original
        // centroid ID makes that tie rule agree with the hierarchy's original-ID rule.
        let mut leaf_order = self.leaf_order.clone();
        for range in routing.child_offsets.windows(2) {
            leaf_order[range[0] as usize..range[1] as usize].sort_unstable();
        }
        let mut centers = vec![0.0; self.levels[0].centroids.len()];
        let mut original_to_position = vec![0; leaf_order.len()];
        for (position, &original) in self.leaf_order.iter().enumerate() {
            original_to_position[original as usize] = position;
        }
        for (out, &original) in centers.chunks_exact_mut(self.dim).zip(&leaf_order) {
            let position = original_to_position[original as usize];
            out.copy_from_slice(
                &self.levels[0].centroids[position * self.dim..(position + 1) * self.dim],
            );
        }
        labels
            .par_chunks_mut(batch_rows)
            .zip(vectors.par_chunks(batch_rows * self.dim))
            .for_each_init(Scratch::default, |scratch, (labels, vectors)| {
                scratch.groups.resize_with(routing.len(), Vec::new);
                for group in &mut scratch.groups {
                    group.clear();
                }
                scratch.best.clear();
                scratch.best.resize(labels.len(), f32::INFINITY);
                // Routing is deliberately identical to the SIMD reference. GEMM only
                // changes how the selected fine centroids are evaluated.
                pulp::Arch::new().dispatch(RouteBatch {
                    hierarchy: self,
                    vectors,
                    labels,
                    options,
                    scratch,
                });
                for (cell, group) in scratch.groups.iter().enumerate() {
                    if group.is_empty() {
                        continue;
                    }
                    scratch.gathered.clear();
                    for &row in group {
                        scratch
                            .gathered
                            .extend_from_slice(&vectors[row * self.dim..(row + 1) * self.dim]);
                    }
                    let start = routing.child_offsets[cell] as usize;
                    let end = routing.child_offsets[cell + 1] as usize;
                    let cell_centers = &centers[start * self.dim..end * self.dim];
                    scratch.winners.resize(group.len(), 0);
                    if let Some(mut workspace) = MatrixAssignmentWorkspace::try_new(
                        &scratch.gathered,
                        end - start,
                        self.dim,
                        options.distance,
                    ) {
                        workspace.assign(
                            &scratch.gathered,
                            cell_centers,
                            self.dim,
                            &mut scratch.winners,
                        );
                    } else {
                        base_assign_parallel(
                            &scratch.gathered,
                            cell_centers,
                            self.dim,
                            options.distance,
                            &mut scratch.winners,
                        );
                    }
                    for (&row, &local) in group.iter().zip(&scratch.winners) {
                        let candidate = start + local as usize;
                        let original = leaf_order[candidate];
                        let value = score(
                            &vectors[row * self.dim..(row + 1) * self.dim],
                            &centers[candidate * self.dim..(candidate + 1) * self.dim],
                            options.distance,
                        );
                        if value < scratch.best[row]
                            || (value == scratch.best[row] && original < labels[row])
                        {
                            scratch.best[row] = value;
                            labels[row] = original;
                        }
                    }
                }
            });
    }
}

struct RankPilots<'a> {
    hierarchy: &'a Hierarchy,
    vectors: &'a [f32],
    labels: &'a [u32],
    required: &'a mut [usize],
    cells: &'a [usize],
    distance: Distance,
    scores: &'a mut Vec<f32>,
}

impl pulp::WithSimd for RankPilots<'_> {
    type Output = ();

    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let h = self.hierarchy;
        let routing = &h.levels[1];
        for ((query, &label), required) in self
            .vectors
            .chunks_exact(h.dim)
            .zip(self.labels)
            .zip(self.required)
        {
            score_range(
                simd,
                &routing.centroids,
                &routing.norms,
                query,
                h.dim,
                self.distance,
                self.scores,
            );
            let winner = self.cells[label as usize];
            let entry = |node: usize| ScoreEntry {
                score: self.scores[node],
                level: 1,
                node: node as u32,
            };
            *required = 1
                + (0..routing.len())
                    .filter(|&cell| {
                        routing.fine_offsets[cell] != routing.fine_offsets[cell + 1]
                            && entry(cell) > entry(winner)
                    })
                    .count();
        }
    }
}

struct RouteBatch<'a> {
    hierarchy: &'a Hierarchy,
    vectors: &'a [f32],
    labels: &'a mut [u32],
    options: ProbeOptions<'a>,
    scratch: &'a mut Scratch,
}

impl pulp::WithSimd for RouteBatch<'_> {
    type Output = ();

    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let h = self.hierarchy;
        let routing = &h.levels[1];
        for (row, (query, label)) in self
            .vectors
            .chunks_exact(h.dim)
            .zip(self.labels)
            .enumerate()
        {
            if let Some(centroids) = self.options.incumbent_centroids {
                self.scratch.best[row] = score(
                    query,
                    &centroids[*label as usize * h.dim..(*label as usize + 1) * h.dim],
                    self.options.distance,
                );
            } else {
                *label = u32::MAX;
            }
            score_range(
                simd,
                &routing.centroids,
                &routing.norms,
                query,
                h.dim,
                self.options.distance,
                &mut self.scratch.routing_scores,
            );
            self.scratch.order.clear();
            self.scratch
                .order
                .extend((0..routing.len()).filter(|&cell| {
                    routing.child_offsets[cell] != routing.child_offsets[cell + 1]
                }));
            let entry = |node: usize| ScoreEntry {
                score: self.scratch.routing_scores[node],
                level: 1,
                node: node as u32,
            };
            let probes = self.options.budget.min(self.scratch.order.len());
            // Same total ordering as the reference heap, including equal scores.
            if probes < self.scratch.order.len() {
                self.scratch
                    .order
                    .select_nth_unstable_by(probes, |&a, &b| entry(b).cmp(&entry(a)));
            }
            for &cell in &self.scratch.order[..probes] {
                self.scratch.groups[cell].push(row);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use aligned_vec::AVec;

    use super::super::{Hierarchy, Level, ProbeOptions, l2_norm};
    use crate::distance::Distance;

    #[test]
    fn grouped_scans_preserve_selected_cells_incumbents_ties_and_batch_tails() {
        for dim in [33, 128, 129] {
            // Reversed original IDs, duplicate fine centers, and an empty routing cell
            // exercise tie mapping and the rule that empty cells consume no probes.
            let original: Vec<f32> = (0..128 * dim)
                .map(|i| ((i / dim % 32) as f32 - 16.0) * 0.125)
                .collect();
            let leaf_order: Vec<u32> = (0..128).rev().collect();
            let mut fine = AVec::new(64);
            for &id in &leaf_order {
                fine.extend_from_slice(&original[id as usize * dim..(id as usize + 1) * dim]);
            }
            let mut routing = AVec::new(64);
            for value in [-1.0, 0.0, 1.0] {
                routing.extend_from_slice(&vec![value; dim]);
            }
            let h = Hierarchy {
                dim,
                levels: vec![
                    Level {
                        norms: fine.chunks_exact(dim).map(l2_norm).collect(),
                        centroids: fine,
                        child_offsets: vec![],
                        fine_offsets: (0..=128).collect(),
                    },
                    Level {
                        norms: routing.chunks_exact(dim).map(l2_norm).collect(),
                        centroids: routing,
                        child_offsets: vec![0, 64, 64, 128],
                        fine_offsets: vec![0, 64, 64, 128],
                    },
                ],
                leaf_order,
            };
            let vectors: Vec<f32> = (0..1025 * dim)
                .map(|i| ((i / dim % 35) as f32 - 17.0) * 0.0625)
                .collect();
            for distance in [Distance::SquaredEuclidean, Distance::NegativeDotProduct] {
                let mut reference = vec![0; 1025];
                let mut expected = vec![0; 1025];
                h.probe_parallel(
                    &vectors,
                    &mut reference,
                    &mut expected,
                    ProbeOptions {
                        distance,
                        budget: usize::MAX,
                        incumbent_centroids: None,
                    },
                );
                let mut labels = vec![0; 1025];
                let mut required = vec![0; 1025];
                h.calibrate_batched(&vectors, &mut labels, &mut required, distance);
                assert_eq!(
                    labels, reference,
                    "pilot winners dim={dim}, distance={distance:?}"
                );
                assert_eq!(
                    required, expected,
                    "pilot ranks dim={dim}, distance={distance:?}"
                );
                h.calibrate_batched(&[], &mut [], &mut [], distance);
                // One exceptional row must preserve the full-probe label and rank,
                // including the no-winner sentinel for an all-NaN score row.
                for value in [f32::NAN, f32::INFINITY, f32::MAX] {
                    let mut exceptional = vectors.clone();
                    exceptional[..dim].fill(value);
                    h.probe_parallel(
                        &exceptional,
                        &mut reference,
                        &mut expected,
                        ProbeOptions {
                            distance,
                            budget: usize::MAX,
                            incumbent_centroids: None,
                        },
                    );
                    h.calibrate_batched(&exceptional, &mut labels, &mut required, distance);
                    assert_eq!(labels, reference);
                    assert_eq!(required, expected);
                }
                for budget in [1, 2, usize::MAX] {
                    for incumbent in [false, true] {
                        let options = ProbeOptions {
                            distance,
                            budget,
                            incumbent_centroids: incumbent.then_some(original.as_slice()),
                        };
                        let mut reference = vec![127; 1025];
                        h.probe_parallel(&vectors, &mut reference, &mut [], options);
                        for batch in [1, 512, 2048] {
                            let mut actual = vec![127; 1025];
                            h.probe_batched(&vectors, &mut actual, options, batch);
                            assert_eq!(
                                actual, reference,
                                "dim={dim}, budget={budget}, incumbent={incumbent}, batch={batch}"
                            );
                        }
                    }
                }
            }
            h.probe_batched(
                &[],
                &mut [],
                ProbeOptions {
                    distance: Distance::SquaredEuclidean,
                    budget: 1,
                    incumbent_centroids: None,
                },
                32,
            );
        }
    }
}
