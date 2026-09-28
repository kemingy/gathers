# Hierarchical assignment: a centroid hierarchy for fast assign

Design note for replacing flat nearest-centroid assignment with a multilayer
hierarchy built **over the centroids themselves**. Each level clusters the
centroids of the level below, so assignment descends from a small top level to
the fine centroids at the leaves, scanning only a fraction of the `k` centroids.
The two-level IVF-style scheme (one coarse layer over the fine centroids) is
the depth-2 special case; the k-means-tree literature is the source of the
traversal and training rules.

## Goal

Flat assignment compares every vector against all `k` centroids. Instead,
organize the centroids into a tree: level 0 holds `B` top centroids, every
internal node has up to `B` children, and the leaves are the fine centroids
(`B^L ≈ k` for depth `L`). A query scans `B` candidates at each level plus the
fine centroids in the leaf cell(s) it reaches — `O(B·L + B)` distance
computations instead of `O(k)`.

This composes with the existing accelerators rather than replacing them: every
per-node scan is an ordinary flat scan (SIMD and/or RaBitQ), just over a small
candidate set.

## Structure

1. Train fine centroids as today (`kmeans.fit`), producing `k` centroids.
2. Build the hierarchy top-down, one level at a time: run k-means **over the
   centroids of the current level** (not the data) to group them into cells,
   then recurse within each cell until cells are small enough to be leaves.
3. Store each node's children contiguously with per-level offset arrays — one
   contiguous `&[f32]` plus `dim` per level, matching the existing flat layout
   contract. Store a `u32` leaf-to-original-index map when outputs must keep
   the original centroid order.
4. Precompute each node's radius `r_n = max ||member - node_centroid||` over
   the centroids in its subtree; this enables the early-termination rule
   below. Radii only need members one level down if the scan budget, not
   exactness, is the stopping criterion.

Assignment cost: `B·L + (fine centroids scanned)` vs `k` flat. With balanced
levels (`k_i ≈ √k_{i+1}` per level) and greedy descent, that is roughly
`B·log_B(k)` comparisons — e.g. `B = 16, L = 5` for `k ≈ 1M` gives ~85
comparisons vs 1,000,000.

## Assignment algorithms

### Greedy descent

At each level, compute distances to the current node's `B` children and follow
the argmin; at a leaf, scan its fine centroids. Minimal state, fully
branch-predictable, trivially SIMD. This is the right default inside `fit`,
where small boundary errors are tolerable and every Lloyd iteration re-assigns
all points.

### Priority-queue backtracking (best-node-first)

Generalize best-bin-first search (FLANN's priority-queue traversal on k-means
trees) to any depth:

1. Start with the root's children on a min-heap keyed by `||q - node_centroid||`.
2. Pop the closest node. If it is internal, compute distances to its children
   and push them. If it is a leaf, scan its fine centroids and update the best
   candidate.
3. **Early termination (radii rule):** if `||q - node_centroid|| - r_n ≥
   best_dist` for the next node (triangle inequality), no fine centroid in its
   subtree can beat the current best; prune it and every node behind it in the
   heap. Without radii, fall back to a scan budget (`max_nodes`, the analogue
   of FLANN's `checks`).

Greedy descent is the special case "follow one path, budget = 1 leaf."
Best-first ordering opens the subtrees most likely to contain the true nearest
centroid first, so it reaches equal recall with fewer fine scans, or better
recall at equal scans — at the cost of a heap and per-node radii. In the
depth-2 special case this reduces to multi-probe IVF with `nprobe` replaced by
a provable stopping rule.

### Recall failure mode (both variants)

The recall ceiling comes from Voronoi boundaries at every internal node: a
query that takes the wrong child at any level loses every centroid in the
sibling subtrees below it. Errors compound with depth — qualitatively, if each
level independently retains a fraction `ρ` of true matches, the tree retains
roughly `ρ^L`. Consequences:

- Balance matters most at the **top** levels, where a mistake prunes the
  largest subtrees. K-means over centroids yields more uniform cells than
  k-means over raw data, which helps; empty cells still need repair (split or
  perturb-and-copy) as in the existing training path.
- Deeper trees need the backtracking search (or a larger scan budget) to hold
  recall; greedy descent alone favors shallow trees with larger fan-out.
- Boundary loss grows with cell imbalance, so prefer balancing heuristics
  (e.g. `k_i ≈ √k_{i+1}` per level, or FLANN's autotuned branching) over
  squeezing the last bit of k-means objective.

## Training the hierarchy

Follow the Faiss `Clustering` conventions, confirmed against its source, at
every node:

- **Subsample for training**: cap each node's training set at `256 * B` points
  (children centroids of the level below); warn below `39 * B`. Never train on
  more than a sample.
- **Fewer iterations at the top**: Faiss uses `niter = 10` for the level-1
  quantizer vs 25 for ordinary clustering. Rough convergence is fine for
  routing layers; FLANN reports ~7 iterations retain >90% of tree quality at
  <10% of build cost, and 0 iterations degrades performance ~2x.
- **Empty-cluster repair**: split, or copy the largest cluster's centroid and
  apply symmetric `×(1 ± 1/1024)` perturbation (already the established
  pattern in this repository).
- **Fan-out and depth**: FLANN's autotuner searches branching `B ∈ {16, 32,
  64, 128, 256}` (OpenCV default 32); the vocabulary tree used `B = 8–16` at
  1M leaves. A simple default is balanced levels, `k_i ≈ √k_{i+1}`, which keeps
  every per-node scan the same size and SIMD-friendly. Stop deepening when a
  cell has fewer than `B` members or `B^L ≥ k`.

## When it pays, and how deep

- Worth it when `k ≳ 4k–10k`. Below that, the flat SIMD scan is already cheap
  and the hierarchy is not amortized.
- Depth 2 (one routing layer) is enough for `k` up to ~10^5. Go deeper
  (`L = 3–6`, vocabulary-tree style, e.g. `B = 10, L = 6` for 1M centroids)
  when `k` reaches 10^5–10^6, because per-level scans stay small only if the
  fan-out stays moderate.
- Deeper is not automatically better: recall compounds per level, and
  production IVF stacks typically stop at two routing layers plus at most one
  *accelerated* layer (an HNSW or second IVF over the top centroids) rather
  than growing the tree — see the Faiss library paper. Prefer increasing
  fan-out or adding backtracking over adding levels once `L ≥ 3`.
- Inside `fit`, the saving multiplies by `max_iter * n`; this is the strongest
  use case. Use greedy descent there.
- For the public assign API, expose the recall/cost knob (`max_nodes` scan
  budget, or radii-based exact stop) and default to best-first with radii,
  which is near-exact for a small scan overhead.

## Complexity summary

| Method | Distance computations per assignment | Extra state |
| --- | --- | --- |
| Flat SIMD | `k` | centroids |
| Greedy descent | `B·L + B ≈ B·log_B(k)` | hierarchy centroids (`< k·B/(B-1)` floats), offsets |
| Best-first with radii | bounded by `B·L + scanned_leaves * leaf_size`; exact when the radii rule empties the heap | + heap, per-node radii |

Total hierarchy memory is `4 * dim` bytes per node; the node count is
`(k·B/(B-1) - B)/(B - 1) + …`, i.e. the fine centroids plus a `B/(B-1)`
factor — comparable to storing the centroids themselves, plus `4` bytes of
offsets per node.

## References

- Faiss IVF structure, `nprobe`, and training rules (the depth-2 special case):
  [Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes),
  [Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index),
  [Clustering.cpp](https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/Clustering.cpp),
  [IndexIVF.cpp](https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/IndexIVF.cpp).
- Priority-queue / best-bin-first traversal, branching-factor and
  iteration-count experiments, visit-budget (`checks`):
  [Muja & Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration", VISAPP 2009](https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_visapp09.pdf)
  (§3.2–3.3, §4.2).
- Triangle-inequality branch-and-bound over cluster trees (the radii
  termination rule): [Fukunaga & Narendra, IEEE Trans. Computers 1975](https://ieeexplore.ieee.org/document/1672899).
- Vocabulary tree (recursive k-means hierarchy, `B = 8–16` at 1M leaves):
  [Nistér & Stewénius, CVPR 2006](https://ieeexplore.ieee.org/document/1641018).
- Production depth limits (two routing layers plus one accelerated coarse
  quantizer, e.g. `IVF65536_HNSW32,PQ64`):
  [Faiss library paper](https://arxiv.org/html/2401.08281v4).
