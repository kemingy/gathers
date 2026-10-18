# Assignment GEMM backends and parallelism

Matrix assignment uses Apple Accelerate's system `cblas_sgemm` on macOS and
sequential `faer` GEMM elsewhere. Accelerate ships with macOS: no extra package,
Cargo feature, runtime download or system BLAS installation is required. Building
on macOS needs the normal Xcode Command Line Tools. Linux and Windows retain the
existing Rust dependencies and require no BLAS installation.

The shared private backend wrapper accepts row-major matrices and logical input
transposes, checks their shapes and strides before FFI, and transposes the centroid
matrix without copying it. It uses the legacy LP64 symbol for compatibility with older
macOS versions, with checked integer conversion and a portable fallback for shapes
outside its range. A comparison of legacy/new LP64 and new ILP64 interfaces found
no consistent speedup; see [Accelerate refinements](./accelerate-refinements.md#blas-interface-ablation).
The system framework dispatches CPU features. PCA covariance uses checked
SYRK; projection and inverse projection share SGEMM. Its eigendecomposition still uses `faer`.
See [PCA performance](./pca-performance.md) for the bounded-memory pipeline and measurements.

Rayon parallelizes row tiles. The preferred row count is `ceil(rows / workers)`
clamped to 64–512, giving every worker one exact-fit tile; centroid tiles hold at
most 1,024 rows. Exact-fit tiles replace an earlier round-up-to-a-power-of-two
policy: at sixteen workers, mid-size batches of 5,000–6,144 rows measured 22–37%
faster with exact-fit tiles under faer and up to 17% faster under Accelerate,
because rounding up to 512 rows creates fewer tasks than workers. The existing
128 MiB aggregate score-scratch cap can further reduce
row tiles, or select direct assignment if fewer than 64 rows fit. For 1,000
queries, row tiles are 512 / 250 / 125 / 84 / 64 with 1 / 4 / 8 / 12 / 16
workers. Full 65,536-row streaming batches retain 512-row tiles at sixteen workers.

Accelerate owns its internal threading policy. The library does not alter BLAS
settings or environment variables, so a requested Rayon worker count is not a
promise about the total number of native threads. A separate process-startup
`VECLIB_MAXIMUM_THREADS=1` ablation below checks nested-thread sensitivity without
changing a host application's BLAS configuration or requiring newer macOS APIs.

For dimensions 128 and above on Apple Silicon macOS, batches with at least sixteen rows and
4,096 query values use a lower crossover of 32,768 vector-centroid comparisons. Smaller batches,
64–127D inputs, Intel macOS and other platforms retain the previous 524,288-comparison cutoff;
below 64D it remains 4,194,304 comparisons. This keeps single queries on the existing direct path
at the tested cluster counts. See [Accelerate refinements](./accelerate-refinements.md) for the
full-data PCA128 checks, batch sensitivity and rejected GEMV experiment.

## Numerical behavior

Accelerate still computes `f32` products. The L2 norm identity can lose precision
through cancellation, and dot-product rounding can reorder nearly tied scores.
The existing uncertainty bounds, candidate refinement, direct near-tie fallback,
non-finite handling and subnormal margin are retained. No safeguards were removed,
and no public API changed. Tests exercise the selected backend and the portable
backend against a direct `f64` product with an absolute-plus-relative reduction
bound, and assignment tests check direct-reference labels.

## Environment and inputs

Apple M5 Pro, 18 physical cores (12 Performance + 6 Super), 64 GiB, macOS 26.6,
aarch64/NEON; rustc 1.99.0, optimized `bench` profile (release, thin LTO, one
codegen unit), without `target-cpu=native` or the separate library `perf` feature.
Performance workloads ran serially, with no concurrent compilation or tests.

Only the user's complete local datasets were used:

| Dataset | Source rows | Dimensions | Fixed final centroids | Native queries | Metric |
| --- | ---: | ---: | ---: | ---: | --- |
| SIFT | 1,000,000 | 128 | 3,943 | 10,000 | Squared L2 |
| GIST | 1,000,000 | 960 | 3,943 | 1,000 | Squared L2 |

Both come from the Texmex fvecs files (`sift_base.fvecs`/`sift_query.fvecs`,
`gist_base.fvecs`/`gist_query.fvecs`). Fixed final original-space centroids were
trained on all 1,000,000 source rows with the repository CLI (seed 42, ten maximum
iterations, tolerance 0.01, automatic reduction: raw 128D for SIFT, PCA128 for
GIST); 3,943 clusters, no empty clusters. The benchmark below reuses these models;
it does not retrain per configuration.

## Native-query backend and row-policy ablation

This ablation uses every row of the complete native query files
(`gist_query.fvecs` and `sift_query.fvecs`), not the 1,000,000-row source files;
the next section measures those complete source files. Each configuration warms
once and times seven assignments, including centroid norm refresh and score
allocation. Input loading and workspace creation are outside these query timers.
At sixteen workers, values are medians of three run medians in milliseconds:

| Query dataset | faer 512 | faer adaptive | Accelerate 512 | Accelerate adaptive |
| --- | ---: | ---: | ---: | ---: |
| GIST 1,000 × 960 | 37.181 | 6.889 | 7.227 | 4.109 |
| SIFT 10,000 × 128 | 13.335 | 12.918 | 7.279 | 7.238 |

Accelerate alone improves the adaptive implementation by 1.68x / 1.78x.
Adaptive row scheduling helps Accelerate on the 1,000-row GIST batch by 1.76x
over fixed 512 (64-row tiles). The 10,000-row SIFT batch provides 625 rows per
worker, so both policies select identical 512-row tiles and their difference is
run variation, not an adaptive gain. All query labels match the faer baseline
under every tested configuration. An earlier `matrixmultiply` substitution during
the investigation showed faer-level adaptive times; it adds no production
dependency and says nothing about Linux performance.

Worker sensitivity, adaptive tiles (one seven-call run at 1 and 8 workers;
three runs at 16):

| Dataset | Workers | faer ms | Accelerate ms |
| --- | ---: | ---: | ---: |
| GIST | 1 | 73.231 | 11.633 |
| GIST | 8 | 11.525 | 4.994 |
| GIST | 16 | 6.889 | 4.109 |
| SIFT | 1 | 116.243 | 42.987 |
| SIFT | 8 | 19.323 | 9.429 |
| SIFT | 16 | 12.918 | 7.238 |

With `VECLIB_MAXIMUM_THREADS=1`, sixteen-worker Accelerate adaptive medians were
4.155 ms (GIST) and 7.198 ms (SIFT), three runs each. Capping BLAS threads changes
nothing beyond run variation on this machine: the Accelerate gain does not rely on
nested BLAS threading at these tile sizes. Actual BLAS thread counts were not
independently measured. One CPU does not establish optimal constants or Intel Mac
performance.

## Full-corpus final assignment

Fresh full passes assign all 1,000,000 source rows against the fixed centroids in
one workspace. Timed calls include centroid norm refresh and score allocation and
exclude input reading, workspace creation and label output; every warmup and timed
pass verifies all labels against the direct SIMD reference. Values are medians of
three timed passes after one warmup:

| Dataset | faer assign s | Accelerate assign s | Assignment speedup |
| --- | ---: | ---: | ---: |
| GIST 1M | 6.538 | 3.332 | 1.96x |
| SIFT 1M | 1.061 | 0.590 | 1.80x |

These are final-assignment results, not full training speedups or a full-corpus
comparison to Faiss. Unlike the query ablation, full batches give every worker
bulk rows, so fixed512 and adaptive tiles coincide and are not listed separately.

## Lessons from other implementations

[superkmeans-rs](https://github.com/paradedb/superkmeans-rs/blob/66d2e86bce7d4c6e9b0123bbcec0ca476ba36338/README.md#blas-backends)
also provides a portable default plus Accelerate/OpenBLAS backends. Its
[macOS batching](https://github.com/paradedb/superkmeans-rs/blob/66d2e86bce7d4c6e9b0123bbcec0ca476ba36338/src/batch.rs)
uses 256-row tasks, larger centroid tiles, and retained score buffers.
These are alternatives to measure, not constants to copy without evidence.
Its [later training rounds](https://github.com/paradedb/superkmeans-rs/blob/66d2e86bce7d4c6e9b0123bbcec0ca476ba36338/src/superkmeans.rs)
use previous assignments and partial-dimensional ADSampling pruning. Statistical
pruning is a separate accuracy tradeoff from our agreed exact flat training path.
The Rust port's `layout.rs` explicitly retains plain row-major centroids instead
of the original C++ hybrid layout. Its benchmarks use different dimensions
and cluster counts, so their claims do not establish performance on our inputs.

Prior profiling identified GEMM/packing as the main bottleneck. Numerically
ambiguous rows can add expensive direct rescans. Conservative candidate refinement
is a useful next target; replacing GEMM does not justify bypassing the accuracy guard.

## Reproduction and checks

The maintained Cargo benchmark under `benches/gemm_parallelism/` compiles the
private assignment and linear-algebra modules directly from their production paths.
It uses explicit threshold, row-tile, and kernel parameters; production entry points
supply the normal defaults. There are no public benchmark APIs, global overrides,
source-rewriting scripts, Python preparation steps, or temporary checkouts. Cargo
and `make lint` check the harness against the current sources.

```sh
cargo bench -p gathers-cli --bench gemm_parallelism -- \
  QUERY_FVECS CENTROID_FVECS l2 --threads 16
```

With no input arguments, the target prints help, so checking all Cargo targets
requires no local datasets. The default mode compares faer and Accelerate crossed
with fixed512/adaptive scheduling; non-macOS runs only faer. Both row policies
respect the production scratch cap. Each case warms once and reports seven-call
medians as JSON lines. Input loading, normalization, reference assignment, and
workspace creation are outside these timers; centroid norm refresh and score
scratch allocation remain included. Every warmup and measured call checks all
labels against direct SIMD assignment. `--repeats N` changes the sample count;
`median_ms` also handles even sample counts.

Repeat with 1 and 8 workers, and optionally set `VECLIB_MAXIMUM_THREADS=1` before
process startup. Supply full native query sets and final original-space centroids
trained on all source rows. The harness reads every input row and has no dataset
cap; passing a base file as the query file reproduces the full-corpus table.
Broader full-corpus and alternative-backend harnesses remain local investigation
artifacts.

Required checks: `make format lint lint-ast test-ast test`. Existing assignment
coverage includes worker/score budgets, row/centroid/vector tails, workspace reuse,
near ties, common-offset cancellation, overflow and subnormal scores. Shared backend tests
cover input transposes, padded output, replacement/accumulation, empty products, zero reduction
dimensions, symmetric products, and direct-reference tolerance for both backends. At this
revision, the Linux x86_64 library and Intel macOS library/test compile checks pass;
cross-target tests were not executed. The commands are listed in
[PCA performance](./pca-performance.md#validation-and-reproduction).

To compare the previous and current crossover on the complete native query file,
use `--batch-rows`. This mode includes workspace setup and direct SIMD fallback:

```sh
cargo bench -p gathers-cli --bench gemm_parallelism -- \
  gist_query.fvecs gist_centroids.fvecs l2 --threads 16 --batch-rows 64
```

Batch mode compares the historical and current thresholds with adaptive scheduling
and the platform backend. `matrix_batches` reports how many batches used the matrix
workspace; a smaller final batch may use direct assignment. The default mode reports
this count as zero or one. `batch_rows` controls processing granularity, never the
dataset row count. Python's standalone `batch_assign` routing is unchanged; these
results do not claim its runtime.
