# Pilot-calibrated probing during Lloyd training

This is an isolated experiment, not a change to default K-means training. The
`adaptive_probe` Cargo benchmark compiles the hierarchy and exact matrix workspace
privately. No benchmark hooks or new public library APIs are exported.

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
cleanup. A separate run exercises the actual CLI default of eight probing rounds plus
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

## Actual CLI-default cleanup policy

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
Each generated cache has a `<prepared>.json` sidecar recording its source identity and
preparation settings. Every reuse, including `--prepare-only`, validates that metadata;
recreate older caches without it. The native shortcut of passing the same 128D L2
source as `--prepared` does not require a sidecar.

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
to exercise the actual two-cleanup-round default. Raw JSONL results and preparation
reports are under the local `/tmp/gathers-m5-stack/` measurement directory.

## Validation

For the measured code, `make lint lint-ast test-ast test-rust` passed, including 37
focused adaptive-probe integration tests, and the Python suite passed all 73 tests
through a `uv` environment. All three previously failing CI seeds pass full-probe
correctness. The benchmark driver also accepts native SIFT and rejects an older cache
without provenance metadata. `git diff --check` passes for the documentation update.

`cargo check --offline -p gathers-cli --bench adaptive_probe --target TARGET` passed
for `x86_64-unknown-linux-gnu` and `x86_64-apple-darwin`. These are compile checks;
runtime timing and dataset evidence above are from Apple Silicon.
