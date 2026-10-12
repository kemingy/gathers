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

Flat assignment compares every vector against all `k` fine centroids. Instead,
organize them into a tree. Each routing centroid represents at least 196
centroids from the immediately lower layer; a new routing layer must contain
at least 16 centroids. A query scans the top layer, then one child list per
layer down to a cell of fine centroids. Its greedy-descent cost is the top
layer size plus the sizes of those child lists, instead of `k` comparisons.

This composes with the existing accelerators rather than replacing them: every
per-node scan is an ordinary flat scan (SIMD and/or RaBitQ), just over a small
candidate set.

## Structure

1. For `n` input rows, choose about `k = n^0.8 / 16` fine centroids and train
   them with `kmeans.fit`. This is the proposed leaf-count rule; the current
   `KMeans::default()` uses a different automatic count.
2. Build the hierarchy bottom-up: cluster the fine centroids into the first
   routing layer, then cluster that layer's centroids into the next coarser
   layer. Given `m` centroids in the previous layer, propose about `√m` new
   centroids, capped at `⌊m / 196⌋` so each can have at least 196 immediate
   children. Create the layer only if that count is at least 16. Record which
   previous-layer centroids belong to each parent and verify the actual
   assignments meet the 196-child minimum.
3. Store each node's children contiguously with per-level offset arrays — one
   contiguous `&[f32]` plus `dim` per level, matching the existing flat layout
   contract. Store a `u32` leaf-to-original-index map when outputs must keep
   the original centroid order.
4. Precompute each node's radius `r_n = max ||member - node_centroid||` over
   all fine centroids in its subtree; this enables the exact early-termination
   rule below. A radius over immediate child centers alone is insufficient
   for that rule.

For 1,000,000 input rows, the proposed leaf count is about 3,943 fine
centroids. Its square root is about 63, but the 196-child cap permits only
`⌊3,943 / 196⌋ = 20` routing centroids. The next layer is impossible because
20 centroids cannot form 16 groups of 196. With balanced cells, greedy descent
scans about `20 + 3,943 / 20 ≈ 217` candidates versus 3,943 for flat
assignment. These are comparison counts, not measured runtimes.

## Assignment algorithms

### Greedy descent

At each level, compute distances to the current node's children and follow
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
`lb_n²` with the best squared distance. `NegativeDotProduct`, `Cosine`, and approximate
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
- Boundary loss grows with cell imbalance, so balance child-list sizes while
  preserving the minimum of 196 immediate children per routing centroid.

## Training the hierarchy

Follow the Faiss `Clustering` conventions, confirmed against its source, when
training each routing layer:

- **Subsample for training**: when creating `k_i` centroids from the previous
  layer, cap its training set at `256 * k_i` previous-layer centroids. Faiss
  warns below `39 * k_i`.
- **Training-size constraint**: this repository's `KMeans::fit` requires at
  least 39 input centroids per output centroid, so the 196-child rule clears
  its global training-size check at every layer. K-means does not guarantee
  that each resulting cell has 196 members; verify and repair the assignments.
- **Fewer iterations at the top**: Faiss uses `niter = 10` for the level-1
  quantizer vs 25 for ordinary clustering. Rough convergence is fine for
  routing layers; FLANN reports ~7 iterations retain >90% of tree quality at
  <10% of build cost, and 0 iterations degrades performance ~2x.
- **Empty-cluster repair**: split, or copy the largest cluster's centroid and
  apply symmetric `×(1 ± 1/1024)` perturbation (already the established
  pattern in this repository).
- **Fan-out and depth**: from `m` previous-layer centroids, choose roughly
  `min(√m, m / 196)` centroids for the new layer, rounding down and stopping
  if the result is below 16. K-means can produce uneven cells, so rebalance
  or reduce the new-layer count when a cell has fewer than 196 children. For
  1,000,000 input rows, `m ≈ 3,943` gives one 20-centroid routing layer.
  FLANN's smaller branching factors and the vocabulary tree's `B = 8–16`
  are alternative design choices rather than this hierarchy's fan-out.

## When it pays, and how deep

- Worth it when `k ≳ 4k–10k`. Below that, the flat SIMD scan is already cheap
  and the hierarchy is not amortized.
- A routing layer needs at least `16 * 196 = 3,136` centroids below it, and
  the square-root rule can stop construction earlier. Apply the rule at each
  layer rather than inferring depth from the raw row count alone.
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
| Greedy descent | top-layer size + child-list sizes along one path | hierarchy centroids, offsets |
| Best-first with radii | up to all routing nodes and all `k` fine centroids; exact when the minimum lower bound cannot improve the result | + heap, per-node radii |

Each stored centroid uses `4 * dim` bytes. With `k` fine centroids and routing
layer sizes `k_1, …, k_L` from bottom to top, the tree stores
`k + Σ_{i=1}^{L} k_i` centroid vectors, plus offsets, radii, and any
leaf-to-original-index map.

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
