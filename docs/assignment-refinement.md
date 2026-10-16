# Streamed candidate refinement

Dot-product matrix assignment previously rescanned every centroid for a query whose
two best GEMM scores differed by at most the numerical uncertainty margin. The full
Cohere native query set contains 31 such rows out of 1,000 against 24,881 centroids.
After switching GEMM to Accelerate, those full direct scans remained a substantial
cost in native macOS stack samples.

Assignment now uses the same streaming refinement pattern for dot products as L2.
For each GEMM score, it updates the approximate best and directly scores a centroid
when its score is at most `running_best + ambiguity`. The running best can only
decrease, so this cutoff includes every candidate admitted by the final cutoff.
The direct best is retained across centroid tiles. Ambiguous rows use that direct
winner; unambiguous rows retain the existing GEMM winner. Equal direct scores keep
the first centroid because tiles and candidates are visited in index order and the
best changes only on a strict improvement.

The uncertainty formula and all constants are unchanged. This is deterministic
candidate selection under the existing conservative floating-point bound. It uses
the existing direct `f32` distance kernel, including its accumulation order, rather
than redefining the result with a different precision. Non-finite norms, winning scores
or bounds still invoke the full direct scan. Dot products also keep the full scan when
the unscaled magnitude bound plus its uncertainty overflows, even if multiplying
by gamma would produce a finite error estimate. GEMM and direct reductions can
otherwise overflow differently, outside the error model's finite-arithmetic
assumptions. The existing underflow allowance remains.
Degenerate inputs can admit every centroid; the algorithm places no candidate cap
that could discard a winner. It reuses the existing per-row best state and introduces
no allocations, scratch buffers, production configuration or platform-specific code.

Focused regressions cover a late true winner with a first-index tie across a
centroid-tile boundary when GEMM rounding changes the apparent winner, and a finite
input whose error bound overflows and therefore requires a full scan. A separate
injected-overflow case verifies full scanning when GEMM hides a direct winner as NaN
while leaving finite tied scores. Existing near-tie, asymmetric-scale, subnormal,
workspace-reuse and direct-label tests also exercise refinement.

## Measurement protocol

Apple M3 Max, sixteen physical cores, 64 GiB, macOS 26.7.1, aarch64/NEON, rustc
1.99.0. Optimized builds use thin LTO and one codegen unit, with sixteen Rayon workers,
default Accelerate threading, no target-cpu=native and no separate library perf
feature. Timed processes run serially without concurrent compilation or tests.
The baseline is commit `8260a5a` with only benchmark case selection added.

Native-query measurements use every one of the 1,000 queries in each local dataset
and the fixed models trained on all 1,000,000 GIST / 10,000,000 Cohere source rows.
Original dimensions are 960 / 768 and centroid counts are 3,943 / 24,881. PCA128
inputs use the existing full-source-fitted projections of those same queries and
centroids. Cosine inputs are normalized before assignment and use negative dot
product. Each call checks every output label against direct SIMD assignment.

Three paired runs alternate baseline/candidate order. Each run warms once and
measures seven calls; reported values are medians of the three run medians.
Full-batch query timers exclude input loading, normalization and workspace creation.
The mini-batch timers include workspace creation, norms and direct fallback, and
process the final partial batch. Loading and normalization remain excluded.

Full-source assignment reads every row in 65,536-row processing batches. Each timer
includes workspace creation, vector/centroid norms and assignment; file reading,
cosine normalization and label output are outside that timer. Candidate labels are
compared individually with the baseline's saved labels for every source row. A
single complete pass per version measures this stage; it is not a training benchmark
or a repeated-pass median. Stream wall time additionally includes reading,
normalization, label comparison and label output.

## Results

Assignment medians on complete 1,000-row native query sets (milliseconds):

| Workload | Baseline | Streamed refinement | Speedup |
| --- | ---: | ---: | ---: |
| GIST original, reusable workspace | 4.382 | 4.304 | 1.02x |
| Cohere original, reusable workspace | 31.514 | 21.280 | 1.48x |
| GIST PCA128, reusable workspace | 1.120 | 1.117 | 1.00x |
| Cohere PCA128, reusable workspace | 5.182 | 4.867 | 1.06x |
| GIST original, batches of 64, setup included | 24.704 | 24.149 | 1.02x |
| Cohere original, batches of 16, setup included | 419.298 | 374.468 | 1.12x |

Cohere original run medians range from 30.765–32.044 ms before and
20.272–21.457 ms after. GIST differences are within run-to-run variation.
Cohere PCA128 ranges overlap (5.035–5.306 vs. 4.802–5.177 ms), so that small
improvement needs more evidence. Every warmup and timed label matched direct SIMD.

Full-source assignment, one complete pass per version (seconds):

| Dataset | Rows | Baseline | Streamed refinement | Speedup |
| --- | ---: | ---: | ---: | ---: |
| GIST, 960 dimensions | 1,000,000 | 2.890 | 2.856 | 1.01x |
| Cohere, 768 dimensions | 10,000,000 | 207.734 | 131.754 | 1.58x |

All 11 million output labels matched the baseline individually. Stream wall time
was 3.788 → 3.774 s for GIST and 215.303 → 138.339 s for Cohere (1.56x).
These measure assignment with fixed centroids; they do not establish a full
K-means training speedup. Candidate stream time also includes baseline-label checking.

Sensitivity checks use one warmup and seven timed calls per version on all native
queries, rather than the three paired runs above:

| Backend | Rayon workers | GIST baseline → candidate (ms) | Cohere baseline → candidate (ms) |
| --- | ---: | ---: | ---: |
| faer | 16 | 10.716 → 10.895 | 58.062 → 52.932 |
| Accelerate | 1 | 14.431 → 14.078 | 135.376 → 82.203 |
| Accelerate | 8 | 4.604 → 4.391 | 42.033 → 23.878 |

All sensitivity labels matched direct SIMD. The alternative full-rescan design is
the baseline; only dot-product candidate refinement and its overflow guard change.
No production hyperparameters are introduced. Worker count, backend, original/PCA
width and batch size test how the benefit changes with the existing execution
choices. Most benefit comes from original-width Cohere, whose ambiguous rows made
the previous full rescans expensive; L2 already used streamed refinement.

Native macOS `sample` traces (five seconds, 1 ms intervals, after startup/reference
calculation) show 8,687 baseline top-of-stack samples in direct `AssignBlock`.
That symbol is absent from the candidate's collapsed top-of-stack summary; BLAS,
norm calculation and tiled score processing remain hot. Sample counts span worker
threads and are not wall-time percentages. The paired benchmark timings above
provide the speedup evidence.

## Reproduction

```sh
cargo bench -p gathers-cli --bench gemm_parallelism -- \
  /Users/keming.yang/workdir/datasets/cohere-large-10m-queries.fvecs \
  /private/tmp/pr65-exact-data/cohere-exact.fvecs cos \
  --threads 16 --backend accelerate --row-policy adaptive
```

For mini-batches, replace `--row-policy adaptive` with
`--batch-rows 16 --crossover current`. GIST uses its complete native query file,
GIST centroids and `l2`; use 64 rows per call for the mini-batch comparison. Filters
only choose benchmark cases; no query prefix is taken. The baseline and candidate
binaries use the same harness and input files.

Raw timings, native stack samples, commands and the bounded-memory full-source
investigation runner are saved under the local `gathers-assignment-refinement`
report directory. The source runner compiles the private production modules by
path and does not patch source files or add public APIs.

## Validation

`make format` and `make lint lint-ast test-ast test` passed, including all 17 matrix
assignment tests and 73 Python tests. The benchmark rejects unknown backend, row
policy and crossover values, crossover without batch mode, and row policy with
batch mode before opening input files.

`cargo check --offline -p gathers-cli --bench gemm_parallelism --target TARGET`
passed for `x86_64-unknown-linux-gnu` and `x86_64-apple-darwin`. These are compilation
checks; runtime performance and direct-label checks above ran on Apple Silicon.
