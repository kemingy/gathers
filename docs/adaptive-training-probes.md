# Pilot-calibrated probing during Lloyd training

Automatic Lloyd assignment chooses between approximate hierarchy probing and exact
flat assignment using internal eligibility and calibration heuristics that may change.
`KMeansConfig::assignment` defaults to
`AssignmentPolicy::Auto`; select `AssignmentPolicy::Exact` for guarded flat
assignment in every round. Ineligible workloads and numerical hierarchy failures
use exact flat assignment. The old input-size-based L2 RaBitQ training switch is
removed. Explicit Rust RaBitQ retrieval remains available. Python `batch_assign`
now uses the shared policy for all metrics instead of selecting RaBitQ for L2.

Python `Gathers.fit(..., assignment="auto"|"exact")` and CLI `kmeans --assignment
auto|exact` forward this policy to the same Rust training implementation. Raw, PCA
and SRHT training all use that dispatcher. No hierarchy or backend-selection API is
exported to Python. Public `batch_assign(..., assignment="auto"|"exact", seed=...)`
assigns against supplied centroids in their original space; it never projects queries.
`exact` is available even when those centroids were trained approximately.

## Current implementation details

The values in this section describe the current implementation, not public API
commitments. They may change as workloads and measurements justify tuning.

A hierarchy is eligible at 3,136 centroids. Production rebuilds it each approximate
round, selects 512 fresh source rows without replacement and computes `min(p100, ceil(1.10 * p99))`. Routing-layer
K-means is always exact. Pilot winners and grouped leaf assignments use the guarded
matrix path when eligible; sparse groups, deeper trees and exceptional values retain
SIMD. The grouped batch bound is 32,768 rows. Previous labels provide incumbents only
after reevaluating their centroids at the current coordinates. Normal centroid updates
and empty-cluster repair remain shared with exact training.

A small shift from an approximate update requests an exact confirmation round before
convergence is accepted, within `max_iter`. There are no unconditional exact cleanup
rounds. Returned training labels still precede the final centroid update and any
original-space reconstruction; final memberships require a separate assignment pass.
Probing changes training trajectories and provides no population recall guarantee.
The single-thread profiling `perf` feature continues to use serial exact assignment.

## Public batch assignment

Rust exposes `kmeans::batch_assign` for one-shot calls and `CentroidAssigner` for
fixed-centroid reuse. Both take `AssignmentPolicy::{Auto, Exact}`. A reusable assigner
builds its hierarchy once, validates each incoming batch and recalibrates pilots over
that complete batch. Training uses the same calibration, grouped leaf scans and guarded
exact kernels, while rebuilding from moving centroids every round.

Cosine centroids are normalized once per assigner; query normalization uses bounded
reusable chunks. Adaptive cosine chunks run in parallel with task-local reusable
scratch, bounded to one chunk per Rayon worker. Exact cosine uses one reusable
normalization buffer. Pilot rows are drawn from the full batch before chunking, so
chunking does not restrict calibration to a prefix. Empty query batches return empty labels;
finite coordinates and nonzero cosine rows are validated before writing labels.
Small or numerically failed hierarchy workloads use the exact matrix/SIMD fallback.
Public query batches do not carry previous Lloyd assignments as incumbents, so their
agreement must be evaluated separately from training. Fresh calibration can change the
budget and labels across calls even for fixed centroids. Matrix norms are recomputed
for new query batches. A seed reproduces the same sequence
of hierarchy construction and pilot samples for the same calls, build and worker count.

This changes Python batch defaults: L2 no longer automatically uses RaBitQ, and eligible
cosine/dot batches can now be approximate. Use `assignment="exact"` to search every
centroid. The single-vector API remains direct SIMD.

The existing `gemm_parallelism` driver accepts `--public-api` to compare public exact
and automatic fixed-centroid assignment on complete source files. It records index
build time separately, warms each assigner once and reuses it for timed calls. Validation,
pilot calibration, normalization and query workspace setup are timed. Exact labels at
those same centroids provide the full-source agreement reference; no sampling is used
for the query workload or agreement measurement. Public mode accepts no private backend
or crossover overrides. For example:

```sh
cargo bench -p gathers-cli --bench gemm_parallelism -- \
  /path/to/sift_base.fvecs /path/to/sift_centroids.fvecs l2 \
  --public-api --threads 16 --repeats 3
```

Full-source SIFT, GIST and Cohere runs favored adaptive over exact flat assignment
on the measured machine. Cohere initially lost to exact when cosine chunks were
processed serially; bounded parallel chunks removed that bottleneck. These observations
do not establish a universal crossover or a recall guarantee. Timings and agreement
reports remain private outside this repository.

## Controlled benchmark and historical measurements

The `adaptive_probe` benchmark retains explicit backend, pilot and cleanup ablations.
It compiles the production hierarchy privately in the same module namespace and shares
the production budget policy. Its historical measurement tables below describe the
listed experimental driver and commit, not the new library defaults: initialization,
RNG streams, convergence and repair differ. New M3 Max production run data stay private.

For each approximate Lloyd round, the benchmark:

1. Builds a fresh hierarchy from the current fine centroids using the existing
   routing-layer training and partition rules.
2. Selects distinct source rows uniformly without replacement. The current default is
   **512 fresh pilots** per round. A separate 16,384-point validation set is excluded.
3. Fully traverses every live leaf cell for each pilot and records the probe number at
   which its exact winning fine centroid is found. Equal scores select the lowest
   original centroid index. Full-probe pilot labels are checked against direct SIMD
   outside training timing.
4. Computes nearest-rank `p99` and uses `min(p100, ceil(1.10 * p99))`, where `p100` is
   the largest required count in that round. Integer arithmetic implements the ceiling.
5. Assigns **all source rows** with that budget. From the second round onward, it
   evaluates the previous assignment's centroid at its current coordinates as an
   incumbent. Calibration itself does not assume an incumbent.
6. Updates centroids with the production update function. Empty training clusters stop
   the run before random repair so routing failures cannot be hidden.

Every method uses the same source rows, projection, seed, initial centroids, iteration
count and distance kernels. Initial centroids use the existing uniform index sampler;
they are shared across experiment arms. The exact baseline uses exact assignment for
all ten rounds. The main adaptive comparison uses ten probing rounds without exact
cleanup. A separate run exercises the benchmark default of eight probing rounds plus
two exact cleanup rounds.

A fresh exact final assignment over every source row measures the returned centers'
mean squared-L2 objective and empty-cluster count. A separate final adaptive audit
reports agreement with exact assignment against those same final centroids. Neither
pilot p99 nor pilot p100 provides a recall guarantee for unobserved rows.

## Grouped assignment and calibration backends

The default experimental assignment backend is point-wise SIMD. The grouped GEMM path
retains the same routing scores and full-probe calibration, groups bounded source
batches by selected leaf cell, and calls the existing guarded matrix assignment
workspace against each cell's fine centroids. Sparse or ineligible groups retain the
normal SIMD fallback. Original-centroid ordering is preserved for ties. Grouping,
gathering, workspace creation and original-ID remapping are included in training time.

The GEMM pilot path finds all pilots' exact fine-centroid winners through the same
guarded matrix workspace, then recovers their live-cell visit ranks using unchanged
SIMD routing order. Deeper trees, ineligible matrix workloads and exceptional numeric
inputs retain full-probe SIMD. The calibration default remains SIMD independently of
the assignment backend.

`--compare-backends` checks every source label against SIMD probing at the same
centroids and budget in every round. `--calibration-repeats 7` performs alternating
paired pilot checks at each round and requires identical winners, required probe counts
and budgets. These checks are excluded from training time. Process wall time includes
them and is not an end-to-end CLI measurement.

## Workloads and timing

Only complete local datasets are used:

| Dataset | Source rows | Training dimensions | Clusters | Preparation |
| --- | ---: | ---: | ---: | --- |
| SIFT | 1,000,000 | 128 | 3,943 | Native input; no projection |
| GIST | 1,000,000 | 960 → 128 | 3,943 | PCA128 fitted and applied to every row |

Both use squared L2, seed 42 and ten rounds. GIST PCA preparation took 2.755 s and is
outside training timing. No training or PCA row cap is used. Only pilot selection and
the diagnostic holdout sample rows.

Apple M5 Pro, 18 physical cores, 64 GiB, macOS 27.0.1, aarch64/NEON, rustc 1.99.0.
The bench profile uses thin LTO and one codegen unit, with 16 Rayon workers, default
Accelerate threading, no `target-cpu=native` and no library `perf` feature. Timed
trials ran serially without concurrent compilation or tests. The measured code was
`02958e9`; the benchmark binary SHA-256 was
`af2c70353efed23d697db586cbc7f0b381d6108dd6dbbdaabba121075b90f1fa`.

Training time includes hierarchy rebuilding, pilot selection/gathering, calibration,
complete-source assignment, cluster counts and centroid updates. Loading, PCA
preparation, held-out diagnostics and final audits are excluded. Exact and grouped
SIMD-pilot results are means and sample standard deviations over three complete runs.
GEMM-pilot results use two clean runs. Other rows are single complete trials.

## Training results

All adaptive rows below use 512 pilots/p99, the incumbent, ten probing rounds and no
exact cleanup. Grouped rows use 32,768-row source batches.

| Dataset / method | Training (s) | Final mean L2 | Difference versus exact |
| --- | ---: | ---: | ---: |
| SIFT, exact flat | 5.664 ± 0.016 | 44846.94449 | — |
| SIFT, adaptive SIMD | 8.647 | 44848.34650 | +0.00313% |
| SIFT, grouped GEMM, SIMD pilots | 3.549 ± 0.024 | 44848.34650 | +0.00313% |
| SIFT, grouped GEMM, GEMM pilots | 3.692 ± 0.098 | 44848.34650 | +0.00313% |
| GIST, exact flat | 6.006 ± 0.130 | 0.7701903810 | — |
| GIST, adaptive SIMD | 12.004 | 0.7702500352 | +0.00775% |
| GIST, grouped GEMM, SIMD pilots | 4.096 ± 0.018 | 0.7702500352 | +0.00775% |
| GIST, grouped GEMM, GEMM pilots | 4.122 ± 0.046 | 0.7702500352 | +0.00775% |

Exact flat training is faster than point-wise adaptive SIMD on both datasets. Grouping
the leaf scans reverses that result: exact/grouped-SIMD-pilot speedups are **1.60x for
SIFT** and **1.47x for GIST**. The approximate objectives are higher in these two
single-initialization trajectories; this is not evidence that approximation improves
or preserves clustering quality in general.

GEMM pilots are about 4% slower on SIFT and nearly unchanged on GIST. Two clean runs
cannot establish small effects reliably, so the useful speedup claim is for grouped
assignment, not pilot calibration.

The ten no-cleanup budgets are `7, 6, 6, 6, 5, 6, 5, 7, 6, 7` for SIFT and
`7, 6, 6, 7, 7, 6, 7, 6, 6, 5` for GIST. All assignment and pilot backends produce
the same budgets and final adaptive objective for each dataset.

## Historical benchmark cleanup policy

The benchmark defaults to two exact cleanup rounds and SIMD assignment/calibration.
These single runs omit all pilot, percentile and cleanup overrides:

| Dataset | Training (s) | Final mean L2 | Difference versus exact | Final adaptive agreement |
| --- | ---: | ---: | ---: | ---: |
| SIFT | 8.677 | 44848.02789 | +0.00242% | 99.9806% |
| GIST | 11.614 | 0.7702408113 | +0.00655% | 99.9905% |

The cleanup runs remain slower than exact flat training. Final audit budgets are 5
for SIFT and 6 for GIST. Agreement is measured against exact assignment at the same
final updated centroids, not against labels from the separately trained exact model.

## Batch sensitivity

These are single grouped-GEMM/SIMD-pilot trials with ten probing rounds and no exact
cleanup. The 32,768-row configuration is repeated in the primary table above.

| Maximum source batch rows | SIFT training (s) | GIST training (s) |
| ---: | ---: | ---: |
| 8,192 | 3.496 | 3.934 |
| 32,768 | 3.546 | 4.088 |
| 131,072 | 3.927 | 4.464 |

The current 32,768 default is close to the best measured batch for both datasets;
131,072 rows is slower. One run per non-default batch is sensitivity evidence, not an
automatic batch-selection policy.

## Paired pilot-calibration comparison

Seven alternating paired comparisons per round produce these sums of the ten per-round
medians. Both paths include their own allocation/setup; these are calibration timings,
not complete training timings.

| Dataset | SIMD pilots (ms / ten rounds) | GEMM pilots (ms / ten rounds) | Speedup |
| --- | ---: | ---: | ---: |
| SIFT | 24.572 | 14.929 | 1.65x |
| GIST | 24.569 | 16.900 | 1.45x |

The isolated savings are 9.643 ms for SIFT and 7.669 ms for GIST, approximately 0.27%
and 0.19% of their grouped training means. This explains why the isolated calibration
speedup does not imply a material end-to-end improvement.

## Correctness and final audits

One diagnostic run per dataset used grouped GEMM assignment and SIMD pilots, with seven
paired calibration repeats and full-source backend comparisons:

- Grouped and SIMD assignment had **zero mismatches across all 10,000,000 assignments
  per dataset** (one million rows over ten rounds).
- All 512 pilot winners, required probe counts and resulting budgets matched at every
  round and repeat.
- Every tested run completed all ten rounds with zero empty training/final clusters.
- Final fresh-calibration agreement with exact assignment at the same final centroids
  was **99.9903% for SIFT** and **99.9863% for GIST**, both with audit budget 6.

These checks cover one initialization and the specified cached projections. They do
not compare label IDs between independently trained exact/adaptive models, establish
population recall guarantees or measure original-space reconstructed-centroid quality.

## Reproduction

Build and run through the maintained Cargo benchmark. Parent directories for prepared
caches must exist. A missing GIST cache is fitted and written using all source rows.
Each generated cache has a versioned `<prepared>.json` sidecar recording its source
path, source/prepared file sizes and modification times, and preparation settings.
Every reuse, including `--prepare-only`, validates that metadata; recreate older caches
without the current schema. These are filesystem checks, not content hashes: changes
that preserve both size and modification time require manual cache invalidation. The
native shortcut of passing the same 128D L2 source as `--prepared` needs no sidecar.

```sh
# One-time full-source GIST PCA128 preparation.
cargo bench -p gathers-cli --bench adaptive_probe -- \
  /Users/keming/GitHub/ann_datasets/gist/gist_base.fvecs l2 \
  --prepared /tmp/gathers-m5-stack/gist_base.pca128.fvecs --prepare-only

# Exact baseline. For SIFT, pass sift_base.fvecs as both source and prepared input.
cargo bench -p gathers-cli --bench adaptive_probe -- \
  /Users/keming/GitHub/ann_datasets/gist/gist_base.fvecs l2 \
  --prepared /tmp/gathers-m5-stack/gist_base.pca128.fvecs --method exact

# Grouped assignment with SIMD pilots and final adaptive audit.
cargo bench -p gathers-cli --bench adaptive_probe -- \
  /Users/keming/GitHub/ann_datasets/gist/gist_base.fvecs l2 \
  --prepared /tmp/gathers-m5-stack/gist_base.pca128.fvecs \
  --assignment-backend gemm --calibration-backend simd --batch-rows 32768 \
  --exact-final-rounds 0 --audit-final-probes

# Diagnostic grouped/GEMM-pilot run with all comparisons.
cargo bench -p gathers-cli --bench adaptive_probe -- \
  /Users/keming/GitHub/ann_datasets/gist/gist_base.fvecs l2 \
  --prepared /tmp/gathers-m5-stack/gist_base.pca128.fvecs \
  --assignment-backend gemm --calibration-backend gemm --batch-rows 32768 \
  --calibration-repeats 7 --compare-backends \
  --exact-final-rounds 0 --audit-final-probes
```

For SIFT, substitute `/Users/keming/GitHub/ann_datasets/sift/sift_base.fvecs` for both
the source and `--prepared` paths; this avoids an unnecessary 128→128 PCA fit. Use
`--assignment-backend simd --calibration-backend simd` for point-wise adaptive arms,
`--calibration-backend gemm` for the pilot ablation, and omit `--exact-final-rounds 0`
to exercise the benchmark two-cleanup-round default. Raw JSONL results and preparation
reports are under the local `/tmp/gathers-m5-stack/` measurement directory.

## Result storage

Keep M3 Max raw JSONL, numeric summaries, profiles and preparation reports in the
private sibling `../benchmark` directory of the primary checkout, outside this
repository. Repository updates from those runs should record methods, ideas and
qualitative observations. Label
measurements from other hardware with their actual environment; removing a machine
label does not turn a private measurement into a portable result.

## Validation

For the measured code, `make lint lint-ast test-ast test-rust` passed, including 37
focused adaptive-probe integration tests, and the Python suite passed all 73 tests
through a `uv` environment. All three previously failing CI seeds pass full-probe
correctness. The benchmark driver also accepts native SIFT and rejects an older cache
without provenance metadata. `git diff --check` passes for the documentation update.

`cargo check --offline -p gathers-cli --bench adaptive_probe --target TARGET` passed
for `x86_64-unknown-linux-gnu` and `x86_64-apple-darwin`. These are compile checks;
runtime timing and dataset evidence above are from Apple Silicon.

Cache regressions cover same-size source and prepared-file replacements using
explicit modification times. Hierarchy, pilot-policy and training dispatch regressions
now run as library unit tests; only prepared-cache tests remain in the CLI integration
target. The Python suite exercises eligible default training, the exact option, seeded
reproducibility, validation and input preservation.

Production integration was also exercised through the optimized CLI on every source
row of local SIFT, GIST and Cohere, with no training row caps. SIFT used native
128D L2; GIST used L2 and Cohere cosine, both with PCA128 fitted on all source rows.
The runs validated source/projection counts, finite original-dimensional centroids
and cosine normalization. SIFT and GIST also ran the exact-training control. These
single trials validate integration; their M3 Max timings and reports remain private.
The updated checks pass 110 library unit tests and 79 Python tests, plus workspace,
AST and cross-target compile checks.
