# Hierarchical assignment: a centroid hierarchy for fast assign

Design note for replacing flat nearest-centroid assignment with a multilayer
hierarchy built **over the centroids themselves**. Construction groups fine
centroids into a routing layer, then groups that layer into successively
coarser layers. Assignment traverses the finished hierarchy in reverse, from
the small top layer down to the fine centroids, scanning only a fraction of `k`.
The two-level IVF-style scheme (one coarse layer over the fine centroids) is
the depth-2 special case; the k-means-tree literature is the source of the
traversal and training rules.

## Goal

Flat assignment compares every vector against all `k` centroids. Instead,
organize the centroids into a tree: level 0 holds `B` top centroids, every
internal node has up to `B` children, and the leaves are cells of fine
centroids. A query scans `B` candidates at each of `L` routing levels plus
the `m` fine centroids in the leaf cell it reaches — `O(B·L + m)` distance
computations instead of `O(k)` for greedy descent.

This composes with the existing accelerators rather than replacing them: every
per-node scan is an ordinary flat scan (SIMD and/or RaBitQ), just over a small
candidate set.

## Structure

1. Train fine centroids as today (`kmeans.fit`), producing `k` centroids.
2. Build the hierarchy bottom-up: cluster the fine centroids into the first
   routing layer, then cluster that layer's centroids into the next coarser
   layer. Repeat until the top has at most `B` centroids. Record which
   previous-layer centroids belong to each parent. The first routing layer
   requires at least 196 fine centroids per routing centroid.
3. Store each node's children contiguously with per-level offset arrays — one
   contiguous `&[f32]` plus `dim` per level, matching the existing flat layout
   contract. Store a `u32` leaf-to-original-index map when outputs must keep
   the original centroid order.
4. Precompute each node's radius `r_n = max ||member - node_centroid||` over
   all fine centroids in its subtree; this enables the exact early-termination
   rule below. A radius over immediate child centers alone is insufficient
   for that rule.

Assignment cost: `B·L + (fine centroids scanned)` vs `k` flat. Balanced
routing levels grow by a factor of `B` toward the leaves, but the minimum of
196 fine centroids per routing centroid limits depth. For `B = 16` and
`k ≈ 1M`, three routing levels leave about 244 fine centroids per leaf cell:
greedy descent makes about `3·16 + 244 = 292` comparisons vs 1,000,000.
Four routing levels would leave only about 15 fine centroids per lowest cell
and violate the 196-member rule.

## Assignment algorithms

### Greedy descent

At each level, compute distances to the current node's `B` children and follow
the argmin; at a leaf, scan its fine centroids. Minimal state, fully
branch-predictable, trivially SIMD. This is the right default inside `fit`,
where small boundary errors are tolerable and every Lloyd iteration re-assigns
all points.

### Priority-queue backtracking (best-node-first)

Generalize best-bin-first search (FLANN's priority-queue traversal on k-means
trees) to any depth. For exact Euclidean search:

1. Compute each root child's lower bound
   `lb_n = max(0, ||q - node_centroid|| - r_n)` and push it on a min-heap
   keyed by `lb_n`.
2. Pop the node with the smallest bound. If it is internal, compute exact
   distances and bounds for its children and push them. If it is a leaf, scan
   its fine centroids exactly and update the best distance.
3. **Early termination (radii rule):** stop when the heap's smallest bound is
   at least the best distance. By the triangle inequality, no remaining
   subtree can improve the result. Ordering by center distance alone cannot
   justify this stop because a farther center can have a larger radius and a
   smaller bound.

The bound uses Euclidean distances. With `SquaredEuclidean` scores, compare
`lb_n²` with the best squared distance. `NegativeDotProduct` and approximate
RaBitQ scores do not support this exact stopping rule; use a scan budget
(`max_nodes`, the analogue of FLANN's `checks`) for those paths. Greedy descent
is a separate one-path search. Budgeted best-first search can revisit sibling
subtrees to improve recall, at the cost of a heap and extra scans. In the
depth-2 special case, this resembles multi-probe IVF, with a provable stop
only for the exact Euclidean path.

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
  (e.g. `k_{i+1} ≈ B·k_i` for global level sizes, or FLANN's autotuned
  branching) over squeezing the last bit of k-means objective.

## Training the hierarchy

Follow the Faiss `Clustering` conventions, confirmed against its source, when
training each routing layer:

- **Subsample for training**: when creating `k_i` centroids from the previous
  layer, cap its training set at `256 * k_i` previous-layer centroids. Faiss
  warns below `39 * k_i`; it does not require 39 fine descendants per parent.
- **Training-size constraint**: this repository's `KMeans::fit` panics below
  39 input centroids per output centroid. A balanced layer with fan-out 16
  supplies only about 16 previous-layer centroids per parent, even though
  each parent represents far more than 196 fine centroids. Upper layers need
  a dedicated routing-layer trainer that permits this ratio.
- **Fewer iterations at the top**: Faiss uses `niter = 10` for the level-1
  quantizer vs 25 for ordinary clustering. Rough convergence is fine for
  routing layers; FLANN reports ~7 iterations retain >90% of tree quality at
  <10% of build cost, and 0 iterations degrades performance ~2x.
- **Empty-cluster repair**: split, or copy the largest cluster's centroid and
  apply symmetric `×(1 ± 1/1024)` perturbation (already the established
  pattern in this repository).
- **Fan-out and depth**: FLANN's autotuner searches branching `B ∈ {16, 32,
  64, 128, 256}` (OpenCV default 32); the vocabulary tree used `B = 8–16` at
  1M leaves. A simple default is `B = 16`. Create the first routing layer only
  when at least `196 * B` fine centroids are available, and ensure that each
  resulting routing centroid owns at least 196 fine centroids. For `B = 16`,
  the threshold is 3,136. Cluster about `B` previous-layer centroids per
  parent at each subsequent layer; split oversized groups and merge undersized
  groups as needed to enforce the fan-out and descendant minima. For a balanced
  1M-centroid example, the routing layers have about 4,096, 256, and 16
  centroids from bottom to top.

## When it pays, and how deep

- Worth it when `k ≳ 4k–10k`. Below that, the flat SIMD scan is already cheap
  and the hierarchy is not amortized.
- Depth 2 (one routing layer) is enough for `k` up to ~10^5. Go deeper when
  `k` reaches 10^5–10^6 if each added routing centroid still owns at least
  196 fine centroids; `B = 16` permits three routing levels for a balanced
  1M-centroid dataset.
- Deeper is not automatically better: recall compounds per level, and
  production IVF stacks typically stop at two routing layers plus at most one
  *accelerated* layer (an HNSW or second IVF over the top centroids) rather
  than growing the tree — see the Faiss library paper. Prefer increasing
  fan-out or adding backtracking over adding levels once `L ≥ 3`.
- Inside `fit`, the saving multiplies by `max_iter * n`; this is the strongest
  use case. Use greedy descent there.
- For the public assign API, expose the recall/cost knob (`max_nodes` scan
  budget, or radii-based exact stop for Euclidean distance) and default to
  best-first with radii where the exact bound applies.

## Complexity summary

| Method | Distance computations per assignment | Extra state |
| --- | --- | --- |
| Flat SIMD | `k` | centroids |
| Greedy descent | `B·L + leaf_size` | hierarchy centroids, offsets |
| Best-first with radii | up to all routing nodes and all `k` fine centroids; exact when the minimum lower bound cannot improve the result | + heap, per-node radii |

Each stored centroid uses `4 * dim` bytes. A balanced tree with `L` routing
levels stores `k + Σ_{j=1}^{L} B^j` centroid vectors, plus offsets, radii,
and any leaf-to-original-index map.

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
