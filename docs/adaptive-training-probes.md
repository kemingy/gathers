# Pilot-calibrated probing during Lloyd training

This is an isolated experiment, not a change to default K-means training. The
`adaptive_probe` Cargo benchmark compiles the current PR65 hierarchy privately,
alongside the exact matrix workspace from `2295538`. No benchmark hooks or new
public library APIs are exported. The original checkout and its uncommitted changes
are untouched.

For each approximate Lloyd round:

1. Build a fresh hierarchy from the current fine centroids, using PR65's existing
   routing-layer training and partition rules.
2. Select 256, 512 or 1,024 distinct source rows uniformly without replacement.
   Pilot sets are nested across pilot-count trials at a given iteration, using the
   same seed. A separate 16,384-point validation set is excluded from the pilots.
3. Fully traverse every live leaf cell for each pilot. Record the leaf-cell probe
   number at which its exact winning fine centroid is found. Equal scores select
   the lowest original centroid index, matching the direct reference. Full-probe
   pilot labels are checked against direct SIMD assignment outside training timing.
4. Let `p98` be the nearest-rank percentile, at sorted position
   `ceil(0.98 * pilots)`. Use `min(p100, ceil(1.10 * p98))`, where `p100` is the
   largest required count in that round's pilots. Integer arithmetic implements
   both ceilings. No extra fixed maximum of 10 applies to this policy.
5. Assign **all source rows** with that budget. From the second round onward,
   evaluate the previous assignment's centroid at its **current** coordinates as
   an incumbent. Full-probe calibration does not assume an incumbent, making the
   calibration independent of previous training labels.
6. Update centroids with the existing production update function. For spherical
   training, normalize the updated means. Stop and report any empty training
   cluster before invoking random repair, so a routing failure is visible and
   different repair randomness cannot confound the comparison.

Every method uses the same source rows, PCA projection, seed, initial centroids,
iteration count and distance kernels. Initial centroids use the existing uniform
index sampler; they are shared across experiment arms, not compared against an
unrelated historical training run. Hierarchy seeds depend only on the iteration.

The baseline uses exact Accelerate assignment for all ten rounds. Approximate arms
use eight probing rounds and two exact assignment/update rounds. A fixed-budget
alternative uses PR65's `ceil(sqrt(top cells))`, capped at 10, with the same incumbent
and cleanup policy. An ablation removes the incumbent while retaining calibration
and cleanup. Fresh exact final assignment over every source row measures the
returned centers' objective and empty-cluster count.

The measured objective is in the PCA128 training space: mean squared L2 for GIST,
and mean `1 - dot` for normalized Cohere projections. Original-space reconstructed
centroid quality and a complete disk-to-output workflow are separate questions.
Neither pilot p98 nor pilot p100 provides a recall guarantee for unobserved rows.

## Workloads and timing

Use only the local GIST and Cohere sources: 1,000,000 x 960 and 10,000,000 x 768.
The default cluster-count rule uses the original source shape, yielding 3,943 and
24,881 clusters. PCA128 is fitted on **all** source rows, after original-row cosine
normalization for Cohere; projected Cohere rows are normalized again. Each cache
contains every projected source row. No training or PCA row cap is used.

Apple M3 Max, 16 physical cores, 64 GiB, macOS 26.7.1, aarch64/NEON, rustc 1.99.0.
Bench profile: thin LTO, one codegen unit, 16 Rayon workers, default Accelerate
threading, no native CPU flag and no library `perf` feature. Timed trials run
serially, without concurrent compilation or tests.

Per-round timing includes hierarchy rebuilding, pilot selection/gathering,
full-probe calibration, complete-source assignment, cluster counts and centroid
updates/normalization. Full-source workspace creation is included for exact
assignment. Held-out label/reference checks and objective diagnostics are measured
separately and excluded from training-stage time. Cached PCA preparation and input
loading are shared setup, outside training-stage comparisons. Reported process wall
time additionally includes per-round diagnostics and a full exact final audit;
it excludes initial cache loading and does not represent the complete CLI workflow.

## GIST result

Single complete trials, ten rounds each. The approximate methods include two exact
cleanup rounds. Final objectives use exact assignment of all one million rows.

| Method | Training stage (s) | Final mean L2 error | Difference versus exact | Empty final clusters |
| --- | ---: | ---: | ---: | ---: |
| Exact Accelerate | 5.565 | 0.77019203 | — | 0 |
| Adaptive, 256 pilots | 12.672 | 0.77025309 | +0.0079% | 0 |
| Adaptive, 512 pilots | 14.101 | 0.77023961 | +0.0062% | 0 |
| Adaptive, 1,024 pilots | 13.800 | 0.77023958 | +0.0062% | 0 |
| Fixed budget, same incumbent/cleanup | 12.402 | 0.77017677 | -0.0020% | 0 |
| Adaptive 256, without incumbent | 12.890 | 0.77001774 | -0.0226% | 0 |

For 256 pilots, the eight budgets are `7, 5, 5, 6, 5, 5, 6, 5`, compared with
PR65's fixed budget of 5. Held-out exact-label recall ranges from 99.023% to
99.963%; without the incumbent it ranges from 99.023% to 99.615%. Fixed probing
starts lower at 96.570% and reaches 99.969% later. The methods follow different
training trajectories after their first differing assignment, so later-round
recall differences also reflect changed centroids.

Calibration totals 0.026 s across eight rounds (0.054 s for 1,024 pilots), versus
12.027 s in assignment for adaptive 256. The hierarchy builds total 0.069 s.
The policy works as an inexpensive empirical accuracy calibration, but its current
per-point SIMD traversal is 2.28x slower than exact Accelerate training on GIST.
Larger pilots do not reverse that result. Objective differences in either direction
are outcomes from one initialization, not evidence that approximation improves
clustering quality in general.

## Cohere results

Single complete trials over all ten million source rows, with the same ten-round
and exact-cleanup protocol as GIST. Final mean errors use exact assignment in the
normalized PCA128 space.

| Method | Training stage (s) | Speedup versus exact | Final mean cosine error | Difference versus exact | Empty final clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact Accelerate | 309.955 | 1.00x | 0.09753728 | — | 0 |
| Adaptive, 256 pilots | 234.135 | 1.32x | 0.09756873 | +0.0322% | 0 |
| Adaptive, 512 pilots | 218.800 | 1.42x | 0.09756910 | +0.0326% | 0 |
| Adaptive, 1,024 pilots | 224.181 | 1.38x | 0.09756898 | +0.0325% | 0 |
| Fixed budget 10, same incumbent/cleanup | 238.106 | 1.30x | 0.09756256 | +0.0259% | 0 |
| Adaptive 256, without incumbent | 230.002 | 1.35x | 0.09756960 | +0.0331% | 0 |

The fixed-budget first-round held-out recall is 96.106%, versus 98.151%
for adaptive 256. Final objective differences remain small in both cases; a single
initialization does not establish an accuracy advantage for either design.

The 256-pilot budgets are `15, 11, 8, 10, 9, 11, 6, 8`; observed pilot maxima
range from 12 to 34 cells. Held-out exact-label recall ranges from 98.151% to
99.762%. Calibration totals 0.157 s and hierarchy construction 0.377 s across all
eight probing rounds, versus 227.748 s in assignment across the complete ten rounds.

Without the incumbent, held-out recall ranges from 98.151% to 99.133%, compared
with 98.151% to 99.762% with it. Both runs finish without empty clusters and have
similar final objectives after cleanup. Thus the incumbent improves intermediate
assignment agreement in this run, but these data do not establish an empty-cluster
or final-objective benefit. Different budgets after the second round reflect the
subsequently different training trajectories.

The first 512-pilot round selects 14 cells instead of the 256-pilot budget of 15.
Its held-out recall is 97.888%, demonstrating that the empirical p98 rule plus
10% does not certify 98% population recall. More pilots can decrease an estimated
percentile; they do not monotonically increase the chosen budget. Compare both
held-out quality and full-source final objective rather than assuming a guarantee.

## Ten probing rounds without exact cleanup

This directly tests the 256-pilot policy throughout training. All other choices,
including the current-coordinate incumbent, remain the same. A separate final exact
assignment evaluates every source row; it is an audit, not a training update.

| Dataset | Exact training (s) | Adaptive training (s) | Exact / adaptive | Final objective difference | Empty final clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| GIST | 5.565 | 13.429 | 0.41x | +0.0100% | 0 |
| Cohere | 309.955 | 207.994 | 1.49x | +0.0461% | 0 |

On Cohere, two exact cleanup rounds reduce the observed final objective difference
from +0.0461% to +0.0322%, while raising training time from 207.994 to 234.135 s.
On GIST, exact assignment remains faster with or without cleanup. Calibration and
hierarchy refresh are small costs; the main tradeoff is point-wise SIMD candidate
scanning versus dense Accelerate GEMM.

The measured training-plus-diagnostics-and-final-audit wall times are
341.171 → 241.386 s for Cohere and 6.214 → 14.175 s for GIST. These wall timers
exclude initial loading/projection, as specified above, and are not complete CLI
end-to-end timings.

These are exploratory, single complete runs per arm using seed 42. They establish
that the requested calibration policy can improve full-source Cohere training on
this machine; they do not justify enabling it universally, establish a recall
guarantee, or establish clustering-quality equivalence across initializations.
Keep exact training for GIST on this hardware. A production decision for Cohere
should additionally assess multiple initialization seeds and original-space
reconstructed centroid quality.

## Reproduction

```sh
cargo bench -p gathers-cli --bench adaptive_probe -- \
  /private/tmp/pr65-real-data/gist.fvecs l2 \
  --prepared /private/tmp/gathers-full-gist-pca128.fvecs \
  --method adaptive --pilots 256
```

Repeat with `--pilots 512` and `--pilots 1024`; `--method exact` runs the exact
baseline and `--method fixed` runs the previous fixed-budget alternative.
`--no-incumbent` isolates the incumbent's contribution. Use
`--exact-final-rounds 0` to test calibration without exact cleanup. The prepared
cache must belong to the specified source and metric; use a distinct path for each.
For Cohere, supply the full local `cohere-large-10m.fvecs`, `cos` and a separate cache.

The experiment emits JSON lines for each round and summary. Raw results, commands,
hardware metadata and the serial runner are saved under the local
`/Users/keming.yang/Documents/ChatGPT/Papers/gathers-adaptive-probes` report directory.

## Validation

`make format` and `make lint lint-ast test-ast test` passed. The focused integration
target runs the existing hierarchy correctness tests plus regressions for winning
probe counts, current-coordinate incumbents under both metrics, original-index tie
handling, and percentile/margin/cap arithmetic at 256/512/1,024 pilots: 12 tests in
all. The Python suite passes all 73 tests.

`cargo check --offline -p gathers-cli --bench adaptive_probe --target TARGET` passes
for `x86_64-unknown-linux-gnu` and `x86_64-apple-darwin`. These are compile checks;
all runtime timing and dataset evidence above is from Apple Silicon.

## Grouped leaf GEMM experiment

`--assignment-backend gemm` retains the full-probe pilot calibration and routing
scores from the SIMD implementation. For a hierarchy with one routing layer, each
bounded source batch selects the same nonempty cells, groups query indices by
selected cell, gathers each group's vectors, and calls the existing guarded matrix
assignment workspace against that cell's fine centroids. Sparse groups retain the
workspace's normal SIMD fallback. Each cell's centers are sorted by original ID so
local argmin ties preserve the hierarchy's original-ID tie rule. Winners from all
selected cells are merged with the previous assignment's centroid, reevaluated at
its current coordinates. Deeper trees retain the existing SIMD traversal.

The initial experiment batches the fine scans only. Routing still uses the original
SIMD scores, making the selected cells directly comparable. It does not change the
pilot policy or introduce a new default into the library. `--batch-rows` bounds query
grouping memory and controls the GEMM size/parallelism tradeoff. Gathering repeats
query rows across their selected cells; this additional memory traffic is measured,
as are grouping, workspace creation, and original-ID remapping.

Use `--compare-backends` to verify every source label against SIMD probing at the
same centroids and budget in every round. Reference comparisons and their setup are
excluded from training timing and reported separately; process wall time includes
them. `--audit-final-probes` additionally rebuilds/calibrates on the final updated
centroids and compares adaptive labels with the exact final assignment over every
source row, retaining the current-coordinate incumbent. This final audit is outside
training time. Its agreement is distinct from the held-out, pre-update per-round
agreement and from agreement with a separately trained exact model.

```sh
cargo bench -p gathers-cli --bench adaptive_probe -- \
  /private/tmp/pr65-real-data/gist.fvecs l2 \
  --prepared /private/tmp/gathers-full-gist-pca128.fvecs \
  --method adaptive --pilots 256 --exact-final-rounds 0 \
  --assignment-backend gemm --batch-rows 32768 \
  --compare-backends --audit-final-probes
```

Hardware and projection protocol remain as above. Raw measurements, serial commands,
and source/binary hashes are under the report directory's `batched-gemm/` subdirectory.

### Grouped GEMM results

Full local source rows, PCA128, ten adaptive rounds, 256 fresh pilots per round,
seed 42, and no exact cleanup, on the M3 Max/16-worker setup above. All timed runs
were serial, without concurrent compilation or tests. These are training-stage
measurements including grouping/gathering and workspace setup, excluding shared PCA
preparation/loading and quality diagnostics.

| Dataset | Exact flat training | Adaptive SIMD training | Grouped GEMM training | Exact / grouped GEMM |
| --- | ---: | ---: | ---: | ---: |
| GIST, 1M rows | 5.566 ± 0.181 s (3 runs) | 14.429 s (1 run) | 4.423 ± 0.076 s (3 runs; batch 8,192) | 1.26x |
| Cohere, 10M rows | 320.377 s (1 fresh run) | 207.994 s (earlier run above) | 81.680 ± 2.179 s (2 runs; batch 32,768) | 3.92x |

The ± values are sample standard deviations of complete repeated trials with the
same initialization, not variation across seeds. Cohere's two grouped measurements
are 83.221 and 80.140 s. The first trial includes full-source SIMD comparisons
outside training timing; the second omits them. Interleaved diagnostics can still
influence cache/thermal state. The fresh exact Cohere result is 3.4% slower than the
earlier 309.955 s baseline; either baseline leaves a substantial improvement.

Batch sensitivity, with all other parameters fixed:

| Maximum source batch rows | GIST training (s) | Cohere training (s) |
| --- | ---: | ---: |
| 8,192 | 4.423 ± 0.076 (3 runs) | 86.154 (1 run) |
| 32,768 | 4.798 (1 run) | 81.680 ± 2.179 (2 runs) |
| 131,072 | 5.206 (1 run) | 82.230 (1 run) |

The larger batches do not universally win. GIST favors 8,192 here; Cohere's
32,768/131,072 difference is small relative to the observed repeated-run variation.
The benchmark default remains 32,768; these results do not introduce an automatic
production batch-size policy.

At batch 32,768, comparisons against the current SIMD implementation found **zero
mismatches in all 110 million assignments** (1M + 10M rows, each over ten rounds).
Cohere assignment totals in that paired trial were 76.716 s for grouped GEMM and
195.671 s for SIMD; rebuild/calibration/update cost another 6.505 s for the grouped
training path. The per-round held-out agreements and pilot budgets match the
previous all-adaptive trajectory. Every tested batch finishes without empty clusters
and with the same exact final objective as adaptive SIMD: 0.7702690705478242 for
GIST, 0.09758219875674248 for Cohere. Differences from separately trained exact
centroids remain +0.0100% and +0.0461%, respectively.

After the final centroid update, a new hierarchy and fresh 256-pilot calibration,
retaining the current-coordinate incumbent, give full-source agreement with exact
assignment **against those same final centroids**:

| Dataset | Final calibrated budget | Exact-label matches | Agreement |
| --- | ---: | ---: | ---: |
| GIST | 5 | 999,635 / 1,000,000 | 99.9635% |
| Cohere | 7 | 9,972,716 / 10,000,000 | 99.72716% |

These full-source audits close the earlier missing post-update agreement
measurement. They do not compare label IDs between separately trained models or
establish population guarantees for future datasets. Only one initialization was
tested. Original-space reconstructed-centroid quality remains unmeasured.

For this extension, `make format lint lint-ast test-ast test-rust` passed. The private
integration target now runs 33 tests, including the existing matrix and
linear-algebra coverage compiled alongside the hierarchy, a focused grouped-scan
regression covering both metrics, original-ID ties, incumbents, empty cells, short
batches and tails, and the existing deep-tree test extended to check SIMD fallback.
The benchmark and integration target also pass offline compile checks for Linux
x86-64 and Intel macOS. Python behavior is unchanged; its suite was not rerun for
this private benchmark extension.
