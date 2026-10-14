# Hierarchical assignment: a centroid hierarchy for fast assign

**Status: implemented** in `src/kmeans/hierarchy.rs` (`kmeans::Hierarchy`). Construction
groups fine centroids into a routing layer, then groups that layer into successively
coarser layers. Assignment traverses the finished hierarchy in reverse, from the small top
layer down to the fine centroids, scanning only a fraction of `k`. The two-level IVF-style
scheme (one coarse layer over the fine centroids) is the depth-2 special case; the k-means
tree literature is the source of the traversal and training rules.

`fit` automatically routes exact assignment through greedy descent when the build rules
permit a hierarchy (3,136+ centroids); below that it stays on the flat matrix/SIMD path.
The public `Hierarchy::assign` is budgeted best-first for all distances.

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
4. Per-node radii `r_n = max ||member - node_centroid||` over a subtree were
   dropped after measurement (see below): at high dimension they prune nothing,
   and the 95th-percentile variant stays vacuous while sacrificing recall.
   Nodes store only centroids and L2 norms (norms enable cosine scoring without
   normalizing stored centroids).

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
trees) to any depth. Cells sit in a min-heap ordered by center score; popping a
routing node expands its children, and popping a leaf cell scans its fine
centroids exactly. The scan stops after a bounded budget of leaf cells per
query, the analogue of FLANN's `checks` / IVF's `nprobe`. The budget defaults
to `sqrt(top-level cells)` capped at 10 — no established rule exists for the
probe count — and `Hierarchy::calibrate` replaces it with a measured value:
representative rows are searched with an unlimited budget, and the budget
becomes the 99th percentile of the leaf-scan counts needed to reach each row's
true cell. Calibration adapts the probe count to the data geometry and scoring
function (cosine needs more probes than L2 at the same tree; see below).
In the depth-2 special case this is multi-probe IVF.

The classic exact early-termination rule (`lb_n = max(0, ||q - c_n|| - r_n)`,
stop when the heap minimum reaches the best distance) was implemented and
measured, then removed: in high dimensions radii cover most of a cell's width,
so the bound clamps to zero for nearly every cell — 19.8/20 cells expanded per
query on GIST-960, i.e. flat-scan cost with heap overhead. Tightening radii to
the 95th percentile of member distances does not fix the bound (still 18.4/20)
and drops recall to ~48% when the budget interacts with vacuous bounds. Greedy
descent is a separate one-path search.

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
  its global training-size check at every layer. The 196-child count is an
  expectation for sizing, not an enforced minimum: uneven or smaller cells are
  kept as-is.
- **Iterations at the top**: Faiss uses `niter = 10` for the level-1 quantizer
  vs 25 for ordinary clustering, and FLANN reports ~7 iterations retain >90% of
  tree quality. That guidance assumes budgeted (`nprobe`-style) search, which
  tolerates rough routing. One-path greedy descent does not: under-trained
  routing makes entire cells unreachable (measured: whole clusters misassigned,
  empty-cluster repair cascading). Routing layers here train with 25 iterations.
- **Empty-cluster repair**: split, or copy the largest cluster's centroid and
  apply symmetric `×(1 ± 1/1024)` perturbation (already the established
  pattern in this repository). With a fixed partition, a repaired centroid moved
  next to its donor stays invisible to the donor's rows, so `fit` rebuilds the
  hierarchy after any iteration that repaired empties and otherwise refreshes
  coordinates and norms in place.
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
- For the public assign API (`Hierarchy::assign`), best-first is budgeted for
  all distances: `sqrt(top-level cells)` leaf cells per query by default
  (capped at 10), or a data-adaptive budget after `Hierarchy::calibrate` —
  no public knob; the flat `base_assign` path remains for callers that need a
  guaranteed-exact scan.

## Measured results (Apple M5 Pro)

Assignment over random centroids (2,048 queries, dim 128): greedy descent is
12×/42×/104× faster than flat at k = 4,096/16,384/65,536. Budgeted best-first
(`sqrt(top cells)` ≤ 10 leaf cells) for dot is 2.3×/10×/33× faster; measured
recall (500 queries, dim 8): ~0.93-0.98 at k = 3,200 (16 cells, budget 4) and
≥ 0.97 at k ≥ 16,384.
**Exact best-first L2 does not prune in high dimensions**: radii cover most of
the cell width, so lower bounds rarely exclude a subtree — it matches flat-scan
speed on uniform data (1.0× at k ≤ 16k, 0.69× at 65k from heap overhead) and on
GIST-960 centroids (412 ms vs 397 ms flat for 20k queries). The 95th-percentile
radius variant stays vacuous (18.4/20 cells still unprunable) while cutting
recall to ~48%, so the exact stop was removed entirely in favor of the bounded
budget for all distances. Greedy remains the fit-time path.

End-to-end GIST (1M×960 base, 400k training rows, PCA to 128, k = 3,943, 8
threads): `fit_ms` 6,668 → 1,327 ms (5.0×) with identical query inertia. With
the full 1M-row training sample: `fit_ms` 3,249 ms, zero empty clusters at
reconstruction.

Assignment recall on real data (centroids trained on full GIST, k = 3,943,
dim 960; the 1,000 gist_query vectors as queries, flat assignment as ground
truth), budgeted best-first at 5 cells: L2 99.8% (2.9× faster than flat), dot
100% (5.4×), cosine 93.1% (2.4×). Greedy holds 80.2% for L2 but only 9.8% for
dot — Euclidean routing is a poor surrogate for dot when centroid magnitudes
vary, so greedy-in-fit relies on dot training's unit-norm centroids. Note the
heap must be ordered by center score, not lower bounds: ordering by vacuous
bounds collapses budgeted L2 recall to 27%.

Calibration (`Hierarchy::calibrate` on 500k gist_learn rows, ~18 ms one-time)
corrects the default in both directions on the same setup: L2 drops to 4 cells
at 99.2% recall (6.2 ms vs 7.6 ms), dot to 3 cells at 99.1% (2.2 ms vs
3.5 ms), and cosine rises to 8 cells, lifting recall from 93.1% to 99.5%
(11.0 ms vs 7.2 ms). The sqrt default over-probes L2/dot and under-probes
cosine; the measured budget holds ~99% recall for each scoring function.

## Complexity summary

| Method | Distance computations per assignment | Extra state |
| --- | --- | --- |
| Flat SIMD | `k` | centroids |
| Greedy descent | top-layer size + child-list sizes along one path | hierarchy centroids, offsets |
| Best-first (budgeted) | top layer + expansions + budgeted leaf-cell scans | + heap, per-node norms |

Each stored centroid uses `4 * dim` bytes. With `k` fine centroids and routing
layer sizes `k_1, …, k_L` from bottom to top, the tree stores
`k + Σ_{i=1}^{L} k_i` centroid vectors, plus offsets, norms, and any
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
