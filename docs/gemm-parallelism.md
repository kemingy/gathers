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
outside its range. The system framework dispatches CPU features. PCA covariance uses checked
SYRK; projection and inverse projection share SGEMM. Its eigendecomposition still uses `faer`.
See [PCA performance](./pca-performance.md) for the bounded-memory pipeline and measurements.

Rayon parallelizes row tiles. The preferred row count is `ceil(rows / workers)`,
rounded up to a power of two and clamped to 64–512; centroid tiles hold at most
1,024 rows. The existing 128 MiB aggregate score-scratch cap can further reduce
row tiles, or select direct assignment if fewer than 64 rows fit. For 1,000
queries, row tiles are 512 / 256 / 128 / 128 / 64 with 1 / 4 / 8 / 12 / 16
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

Apple M3 Max, sixteen physical cores, 64 GiB, macOS 26.7.1, aarch64/NEON;
Rust 1.99.0, optimized `perf` profile (release, thin LTO, one codegen unit),
without `target-cpu=native` or the separate library `perf` feature.
Performance workloads ran serially, with no concurrent compilation or tests.

Only the user's complete local datasets were used:

| Dataset | Source rows | Dimensions | Fixed final centroids | Native queries | Metric |
| --- | ---: | ---: | ---: | ---: | --- |
| GIST | 1,000,000 | 960 | 3,943 | 1,000 | Squared L2 |
| Cohere large | 10,000,000 | 768 | 24,881 | 1,000 | Cosine |

GIST comes from `gist-960-euclidean.hdf5`, losslessly converted to fvecs outside
timing. Cohere comes from `cohere-large-10m.fvecs` and its native query file.
Fixed final original-space centroids come from the earlier full-source training
experiment (seed 42, PCA128, ten maximum iterations, tolerance 0.01). This study
reuses those models; it does not retrain for each backend. Cosine vectors and
centroids are normalized, then searched with negative dot product.

## Backend and row-policy ablation

Every native query participates. Each configuration warms once and times seven
assignments, including centroid norm refresh and score allocation. Input loading,
normalization and workspace creation are outside these query timers. At sixteen
workers, values are medians of three run medians in milliseconds:

| Dataset | faer 512 | faer adaptive | matrixmultiply adaptive | Accelerate 512 | Accelerate adaptive |
| --- | ---: | ---: | ---: | ---: | ---: |
| GIST | 40.412 | 10.475 | 10.245 | 6.624 | 4.494 |
| Cohere | 228.410 | 61.299 | 56.954 | 55.796 | 34.332 |

Accelerate alone improves the adaptive implementation by 2.33x / 1.79x.
Adaptive row scheduling still helps Accelerate by 1.47x / 1.63x over fixed 512.
The row optimization therefore remains part of the same change. `matrixmultiply`
0.3.11 was substituted only in the temporary investigation, with its internal
threading disabled because Rayon already distributes tiles; it adds no production
dependency and these results do not establish its performance on Linux.
All query labels match the faer baseline under every tested configuration.

Worker sensitivity, adaptive tiles (one seven-call run at 1 and 8 workers;
three runs at 16):

| Dataset | Workers | faer ms | Accelerate ms |
| --- | ---: | ---: | ---: |
| GIST | 1 | 81.841 | 12.722 |
| GIST | 8 | 11.481 | 4.405 |
| GIST | 16 | 10.475 | 4.494 |
| Cohere | 1 | 466.126 | 122.609 |
| Cohere | 8 | 71.493 | 40.395 |
| Cohere | 16 | 61.299 | 34.332 |

With `VECLIB_MAXIMUM_THREADS=1`, sixteen-worker Accelerate medians were
4.501 ms / 33.146 ms (three runs). The gain persists under this requested cap;
actual BLAS thread counts were not independently measured. Small differences
between configurations are subject to run variation and fixed benchmark order.
One CPU does not establish optimal constants or Intel Mac performance.

## Full-corpus final assignment

Fresh full passes use adaptive tiles and all source rows, streamed in 65,536-row
batches. This batch size bounds memory and never limits dataset size. Assignment
seconds exclude input reading, normalization, workspace creation and label output;
wall seconds include those stages. GIST Accelerate values are medians of three
passes; other cells represent one pass each.

| Dataset | Passes faer / Accelerate | faer assign s | Accelerate assign s | Assignment speedup | faer wall s | Accelerate wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GIST 1M | 1 / 3 | 7.148 | 2.902 | 2.46x | 8.113 | 3.732 |
| Cohere 10M | 1 / 1 | 446.689 | 213.867 | 2.09x | 454.174 | 222.501 |

Every full label file is byte-identical to the existing exact full-corpus oracle,
including row count and source order; neither dataset has an empty cluster under
these fixed models. File cache state is uncontrolled, output uses buffered flush
without fsync, and single-pass cells provide limited statistical evidence.
These are final-assignment results, not full training speedups or a full-corpus
comparison to Faiss.

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
of the original C++ hybrid layout. Its Cohere benchmarks use different dimensions
and cluster counts, so their claims do not establish performance on our inputs.

Prior profiling identified GEMM/packing as the main bottleneck. Cohere also has
expensive direct rescans on ambiguous rows. Conservative candidate refinement is
a useful next target; replacing GEMM does not justify bypassing the accuracy guard.

## Reproduction and checks

The minimal maintained benchmark under `benches/gemm_parallelism/` archives HEAD,
overlays the working matrix/backend sources, and exposes private workspace and
backend/row-policy overrides only in a temporary copy. No benchmark hooks enter
the production API. Preparation requires Python 3.12+ and an empty destination.

```sh
python3.12 benches/gemm_parallelism/prepare.py /tmp/gathers-gemm-bench
cargo build --offline --profile perf -p gathers-cli --example query_compare \
  --manifest-path /tmp/gathers-gemm-bench/Cargo.toml
/tmp/gathers-gemm-bench/target/perf/examples/query_compare \
  QUERY_FVECS CENTROID_FVECS l2_OR_cos 16
```

The harness compares faer/Accelerate crossed with fixed512/adaptive scheduling,
reports seven-call medians, and checks all labels. Repeat with 1 and 8 workers,
and optionally set `VECLIB_MAXIMUM_THREADS=1` before process startup. On non-macOS
both backend choices use faer. Supply full native query sets and final original-space
centroids trained on all source rows. Broader matrixmultiply and full-corpus
harnesses remain local investigation artifacts.

Required checks: `make format lint lint-ast test-ast test`. Existing assignment
coverage includes worker/score budgets, row/centroid/vector tails, workspace reuse,
near ties, common-offset cancellation, overflow and subnormal scores. The one new
product test checks backend layout, output replacement, empty tiles and direct
product tolerance. Linux and Intel macOS compile checks are recorded separately.

To compare the previous and current crossover on the complete native query file, append a
processing batch size to `query_compare`. This mode includes workspace setup and direct SIMD
fallback, and never caps the number of queries:

```console
/tmp/gathers-gemm-bench/target/perf/examples/query_compare /private/tmp/pr65-real-data/gist-queries.fvecs /private/tmp/pr65-exact-data/gist-exact.fvecs l2 16 64
```

This exercises the private matrix assignment implementation through temporary hooks. Python's
standalone `batch_assign` routing is unchanged; these results do not claim its runtime.
