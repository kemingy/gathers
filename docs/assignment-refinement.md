# Streamed candidate refinement

Dot-product matrix assignment previously rescanned every centroid for a query whose
two best GEMM scores differed by at most the numerical uncertainty margin. After
switching GEMM to Accelerate, those full direct scans can remain a substantial cost.

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

Apple M5 Pro, 18 physical cores, 64 GiB, macOS 27.0.1, aarch64/NEON, rustc 1.99.0.
Optimized builds use thin LTO and one codegen unit, with sixteen Rayon workers,
default Accelerate threading, no `target-cpu=native` and no separate library `perf`
feature. Timed processes ran serially without concurrent compilation or tests.
The baseline is `2db8029` (current `main`); the candidate is the refinement code at
`2391b7f`. Only benchmark case selection differs between their harness interfaces.

All measurements use the complete local SIFT and GIST files. Fixed original-space
centroids were trained on every source row with the repository CLI (seed 42, ten
iterations, tolerance 0.01, automatic reduction), yielding 3,943 centroids and no
empty clusters for each dataset. SIFT has 1,000,000 × 128 source rows and 10,000
native queries; GIST has 1,000,000 × 960 source rows and 1,000 native queries.

Native-query cosine cases normalize every query and centroid before assignment and
then use negative dot product; this exercises the changed path. The fixed models were
L2-trained, so these are assignment-mechanism measurements, not model-quality claims.
L2 cases provide a control because L2 already used streamed refinement. Full-source
cases use negative dot product without normalization. Cosine is not used there because
it correctly rejects zero-norm source rows; dot product keeps every source row in the
measurement. Every warmup and timed call checks all labels against direct SIMD.

Three paired runs alternate baseline/candidate order. Each run warms once and measures
seven calls; reported query values are medians of the three run medians. Reusable-query
timers exclude input loading, normalization and workspace creation. The 64-row batch
timers include workspace creation, norms and direct fallback, and process the final
partial batch. Full-source assignment uses 65,536-row batches and one complete timed
pass per version, including workspace creation and vector/centroid norms. File reading
and label output remain outside that timer.

## Results

Complete native-query cosine assignment (milliseconds):

| Dataset / workload | Baseline | Streamed refinement | Speedup |
| --- | ---: | ---: | ---: |
| SIFT, reusable workspace | 6.189 | 5.981 | 1.03x |
| GIST, reusable workspace | 7.510 | 4.462 | 1.68x |
| SIFT, 64-row batches, setup included | 82.310 | 82.114 | 1.00x |
| GIST, 64-row batches, setup included | 43.008 | 20.664 | 2.08x |

GIST reusable-workspace run medians range from 7.508–8.679 ms before and
4.416–4.476 ms after. SIFT ranges are 6.187–6.274 and 5.962–6.091 ms, so its small
improvement is less certain. GIST 64-row batch ranges are 42.987–43.304 and
20.365–21.123 ms; SIFT batch differences are within run variation.

Reusable-workspace L2 controls were 7.294 → 6.911 ms for SIFT and 4.246 → 4.269 ms
for GIST. The implementation change is dot-specific, so these controls are not
claimed as refinement gains.

Full-source negative-dot assignment, one complete pass per version:

| Dataset | Rows | Baseline | Streamed refinement | Speedup |
| --- | ---: | ---: | ---: | ---: |
| SIFT, 128 dimensions | 1,000,000 | 0.497 s | 0.502 s | 0.99x |
| GIST, 960 dimensions | 1,000,000 | 3.477 s | 3.306 s | 1.05x |

All two million output labels matched direct SIMD for both versions. These are
fixed-centroid assignment results, not full K-means training speedups or repeated-pass
medians.

Sensitivity checks use one warmup and seven timed calls per version at 1 and 8 workers;
16-worker values are the three-run medians above. All cases use adaptive rows and the
complete native cosine query sets:

| Backend | Workers | SIFT baseline → candidate (ms) | GIST baseline → candidate (ms) |
| --- | ---: | ---: | ---: |
| faer | 16 | 12.860 → 12.924 | 9.226 → 6.959 |
| Accelerate | 1 | 36.530 → 36.256 | 33.106 → 12.685 |
| Accelerate | 8 | 8.140 → 7.827 | 10.088 → 4.924 |
| Accelerate | 16 | 6.189 → 5.981 | 7.510 → 4.462 |

The alternative full-rescan design is the baseline; only dot-product candidate
refinement and its overflow guard change. The benefit is workload-dependent: it is
substantial for native and mini-batch cosine GIST, but small or absent for SIFT and
the full-source dot workloads. No production hyperparameters are introduced.

## Reproduction

Run the same commands from the baseline and candidate checkouts, with matching input
files. The unfiltered forms work with both harness versions; select matching JSON
cases. The candidate additionally accepts the backend, row-policy and crossover
filters described in [GEMM parallelism](./gemm-parallelism.md).

```sh
# Complete native queries; all backend/row-policy cases.
cargo bench -p gathers-cli --bench gemm_parallelism -- \
  /Users/keming/GitHub/ann_datasets/gist/gist_query.fvecs \
  /tmp/gathers-m5-stack/gist_centroids.fvecs cos \
  --threads 16 --repeats 7

# Mini-batches, including workspace setup and direct fallback.
cargo bench -p gathers-cli --bench gemm_parallelism -- \
  /Users/keming/GitHub/ann_datasets/gist/gist_query.fvecs \
  /tmp/gathers-m5-stack/gist_centroids.fvecs cos \
  --threads 16 --repeats 7 --batch-rows 64

# All source rows, using dot to retain zero-norm rows.
cargo bench -p gathers-cli --bench gemm_parallelism -- \
  /Users/keming/GitHub/ann_datasets/gist/gist_base.fvecs \
  /tmp/gathers-m5-stack/gist_centroids.fvecs dot \
  --threads 16 --repeats 1 --batch-rows 65536
```

Substitute the SIFT paths and centroid file for SIFT. The harness reads every input
row and has no dataset cap. Raw JSONL results and training reports are under the local
`/tmp/gathers-m5-stack/` measurement directory.

## Validation

For the measured code, `make lint lint-ast test-ast test-rust` passed, including all
matrix assignment tests, and the Python suite passed all 73 tests through a `uv`
environment. `git diff --check` passes for the documentation update.

`cargo check --offline -p gathers-cli --bench gemm_parallelism --target TARGET`
passed for `x86_64-unknown-linux-gnu` and `x86_64-apple-darwin`. These are compilation
checks; runtime performance and direct-label checks above ran on Apple Silicon.
