# Hierarchical assignment: a coarse layer over the centroids

Design note for adding a second level above the normal (fine) centroid layer to
accelerate the assign step. This documents design **A** (two-level IVF-style
layering over the centroids); the k-means-tree alternative (design B) is noted
only where it informs the choice.

## Goal

Flat assignment compares every vector against all `k` centroids. Add a coarse
layer: cluster the `k` fine centroids themselves into `k1` groups. Assignment
then scans `k1` coarse centroids plus only the fine centroids in the chosen
cell(s), instead of all `k`.

This composes with the existing accelerators rather than replacing them: the
coarse scan and the in-cell scan are both ordinary flat scans (SIMD and/or
RaBitQ), only over smaller candidate sets.

## Structure

1. Train fine centroids as today (`kmeans.fit`), producing `k` centroids.
2. Run k-means **over the centroids** (not the data) to produce `k1` coarse
   centroids. Each fine centroid belongs to exactly one coarse cell.
3. Store the fine centroids permuted by cell, with cell offsets — one
   contiguous `&[f32]` plus `dim`, matching the existing flat layout contract.
   Also store a `u32` fine-to-coarse map when the original centroid order must
   be preserved in outputs.
4. Optionally precompute each cell's radius `r_c = max ||fine - coarse_c||`
   over its members; this enables early termination (below).

Assignment cost: `k1 + (fine centroids scanned)` vs `k` flat. With
`k1 ≈ √k` and one cell probed, roughly `2√k` distance computations; with
best-first multi-probe, somewhat more but bounded by the termination rule.

## Assignment algorithms

### Fixed multi-probe (`nprobe`)

Compute distances to all `k1` coarse centroids, take the nearest `nprobe`
cells, scan their fine centroids, keep the global argmin. `nprobe` is a pure
query-time knob: the index never changes. Simple, branch-predictable, and easy
to SIMD; the right default inside `fit` where small boundary errors are
tolerable.

### Priority-queue backtracking (best-cell-first)

The coarse cells form a shallow tree of depth 2, so best-bin-first traversal
applies directly — this is the same idea as FLANN's priority-queue search on a
k-means tree, transferred to the two-level layout:

1. Push all `k1` coarse cells onto a max/min heap keyed by `||q - coarse_c||`
   (cheapest first).
2. Pop the closest unvisited cell, scan its fine centroids, update the best
   candidate.
3. **Early termination:** if `||q - coarse_c|| - r_c ≥ best_dist` for the next
   cell (triangle inequality, using the precomputed radius `r_c`), no fine
   centroid in that cell — or in any later cell — can beat the current best;
   stop. Without radii, fall back to a scan budget (`max_cells`, the analogue
   of FLANN's `checks`).

Fixed `nprobe` is the special case "pop N cells, skip the termination rule."
Best-first ordering means the cells most likely to contain the true nearest
centroid are scanned first, so it reaches equal recall with fewer fine scans,
or better recall at equal scans. Its costs are a heap over `k1` entries and the
radius precomputation at build time.

### Recall failure mode (both variants)

The recall ceiling comes from the coarse Voronoi boundary: a point whose true
nearest fine centroid sits in a coarse cell that was never scanned is lost,
regardless of fine-layer accuracy. Boundary loss grows with coarse-cell
imbalance. K-means over centroids yields much more uniform cells than k-means
over raw data, which favors small `nprobe`; empty cells still need repair
(split or perturb-and-copy) as in the existing training path.

## Training the coarse layer

Follow the Faiss `Clustering` conventions, confirmed against its source:

- **Subsample for training**: cap the training set at `256 * k1` points; warn
  below `39 * k1`. Never train the coarse layer on the full dataset.
- **Fewer iterations than the fine layer**: Faiss uses `niter = 10` for the
  level-1 quantizer vs 25 for ordinary clustering. Rough convergence is fine
  here; FLANN reports ~7 iterations retain >90% of tree quality at <10% of
  build cost.
- **Empty-cluster repair**: split, or copy the largest cluster's centroid and
  apply symmetric `×(1 ± 1/1024)` perturbation (already the established
  pattern in this repository).
- **Choosing `k1`**: between `4√k` and `16√k`. Smaller `k1` means larger cells
  and more fine scans per probe; larger `k1` shifts cost back toward the
  coarse scan. `√k` balances the two. For `k` beyond ~2^18, consider training
  the coarse layer with two-level clustering (k-means to √k1, then within each
  cell), as Faiss does for very large coarse quantizers.

## When it pays

- Worth it when `k ≳ 4k–10k`. Below that, the flat SIMD scan is already cheap
  and the extra indirection, permutation, and memory are not amortized.
- Inside `fit`, every Lloyd iteration re-assigns all points, so the saving
  multiplies by `max_iter * n`; this is the strongest use case. Use small
  `nprobe` (1–2) there.
- For the public assign API, expose the recall/cost knob (`nprobe` or
  `max_cells`) and default to best-first with radii, which is near-exact for a
  small scan overhead.

## Complexity summary

| Method | Distance computations per assignment | Extra state |
| --- | --- | --- |
| Flat SIMD | `k` | centroids |
| Two-level, `nprobe = 1` | `k1 + max_cell_size ≈ 2√k` | + `k1` coarse centroids, permutation, offsets |
| Two-level, best-first | `k1 + scanned_cells * avg_cell_size` (bounded by radii rule) | + heap, cell radii |

Memory overhead is `4 * dim` bytes per coarse centroid plus `4` bytes of
offsets per fine centroid — negligible against the fine centroids themselves.

## References

- Faiss IVF structure, `nprobe`, and training rules:
  [Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes),
  [Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index),
  [Clustering.cpp](https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/Clustering.cpp),
  [IndexIVF.cpp](https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/IndexIVF.cpp).
- Priority-queue / best-bin-first traversal and the visit-budget idea:
  [Muja & Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration", VISAPP 2009](https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_visapp09.pdf)
  (§3.2; branching-factor and iteration experiments in §3.3, §4.2).
- Triangle-inequality branch-and-bound over cluster trees (the radii
  termination rule): [Fukunaga & Narendra, IEEE Trans. Computers 1975](https://ieeexplore.ieee.org/document/1672899).
- Vocabulary tree (k-means tree + inverted files; branching 8–16 at 1M leaves):
  [Nistér & Stewénius, CVPR 2006](https://ieeexplore.ieee.org/document/1641018).
- Two-level clustering of a large coarse quantizer in production Faiss:
  [Faiss library paper](https://arxiv.org/html/2401.08281v4),
  `demo_two_level_clustering.ipynb`.
