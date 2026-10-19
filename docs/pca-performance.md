# Bounded-memory PCA with Accelerate

PCA covariance uses checked Accelerate SYRK on macOS; projection and inverse projection use
the shared checked SGEMM wrapper. Other platforms, unsupported strides, and dimensions outside the legacy BLAS
integer range fall back to `faer`. The eigensolver remains `faer`; no new dependency, Cargo
feature, or system BLAS installation is required. Assignment keeps its existing numerical guards
and outer Rayon scheduling. Neither backend changes global or caller-thread BLAS settings.

PCA centers at most 16 MiB of rows at a time, or one complete row for exceptionally wide inputs.
It reuses the buffer for both covariance formation and transformation, rather than retaining a
full centered corpus. Block covariance contributions are combined in `f64` and then converted to
`f32`, avoiding extra `f32` summation error across thousands of blocks. Only the lower covariance
triangle is formed and accumulated, since the eigensolver explicitly reads `Side::Lower`. The
portable fallback may form both triangles, but the upper entries are not accumulated. The existing `f64` mean,
finite-result checks, and overflow errors remain. This changes accumulation order and can change
principal directions and cluster labels slightly; bitwise reproduction of the previous PCA is
not promised. Original-space centroid reconstruction still uses the existing accumulation code.

The CLI memory preflight accounts for bounded scratch and covariance/eigensolver storage. A
selected fitting sample is retained only when PCA fits fewer than all supplied rows. Full-source
PCA fitting does not reserve another full-source centered copy. The preflight remains a
conservative estimate, not an enforced RSS cap.

## Full-data training

This historical comparison predates automatic adaptive hierarchy training. Use
`--assignment exact` with the current CLI to reproduce its flat-assignment protocol;
see [adaptive training](./adaptive-training-probes.md) for the current defaults.

Re-measured 2026-10-17 on an Apple M5 Pro, 18 physical cores (12 Performance + 6 Super),
64 GiB, macOS 26.6, aarch64/NEON, rustc 1.99.0. Release builds use thin LTO and one codegen
unit; 16 Rayon workers, default Accelerate thread policy, no `target-cpu=native`, and no
library `perf` feature. Benchmark processes ran serially.

Both datasets are the full local real inputs: SIFT 1,000,000 × 128 and GIST 1,000,000 × 960.
K=3,943, seed=42, ten rounds, tolerance=0.01, repository CLI defaults: automatic reduction
selects raw 128D training for SIFT and PCA128 for GIST, with the PCA fitting sample capped at
96,000 rows (100 × input dimension). One run per configuration. "Previous" is the pre-PCA-change
commit's CLI; "current" is this branch. Training totals sum the reported stage timers
(K-means fit, PCA fit, PCA projection, reconstruction); file loading, sampling, centroid
serialization and final quality evaluation are outside them.

| Training total | Previous | Current | Speedup |
| --- | ---: | ---: | ---: |
| Full GIST (PCA128) | 6.768 s | 6.545 s | 1.03× |
| Full SIFT (raw 128D) | 5.410 s | 5.343 s | 1.01× |

| GIST stage | Previous seconds | Current seconds |
| --- | ---: | ---: |
| Fit PCA on the 96,000-row sample | 0.519 | 0.437 |
| Project all 1M training rows | 0.746 | 0.545 |
| Exact flat K-means in 128D | 5.402 | 5.463 |
| Reconstruct original 960D centroids | 0.101 | 0.100 |

Under this default protocol the totals are dominated by exact flat assignment, so the bounded
PCA pipeline changes them only slightly. The stage-level gains are larger: PCA fit 1.19× and
full-row projection 1.37× on the sampled-fit workload, and 2.17× / 1.79× on full-source
fit / fit+projection stage tests (see [Accelerate refinements](./accelerate-refinements.md)).
SIFT trains raw at 128D and exercises no PCA code; its match within run variation shows the
refactor did not regress the raw path. Final assignment uses the same guarded original-space
flat path for both models; neither training had an empty cluster.

The earlier revision of this document used a forced exact-flat, tolerance-1e-12 protocol with
full-source PCA fits on a different machine; those training totals are not comparable to the
CLI-default protocol above and have been replaced.

## Separate backend and memory contributions

To isolate backend and memory effects, cross the portable faer and Accelerate backends with
full-corpus centering and bounded centering. Fit and project every source row, exclude loading, warm
each variant, and repeat complete passes. Keep the eigensolver, output dimensions and numerical
checks fixed. This distinguishes faster matrix products from reduced allocation and memory traffic.

At 960D and 1M rows, an unblocked transform would hold the ~3.84 GB input, a ~3.84 GB centered copy,
and a ~0.51 GB output before other workspaces. The blocked transform needs the input and output plus
at most 16 MiB of centered scratch; the saving grows linearly with corpus size.

Vary centered scratch across 4, 16 and 64 MiB while fitting every source row. The investigation
found a tradeoff between small-block scheduling overhead and larger temporary storage. The current
16 MiB bound balances those costs; it is not a universal optimum. Historical machine-specific
timings are maintained outside the repository.

The [Accelerate refinements](./accelerate-refinements.md) isolate SYRK and lower-triangle
accumulation and report current measurements against the blocked SGEMM implementation.

## Clustering quality

Reassign every original-space source row to the reconstructed final centroids and sum
selected-centroid squared distances in `f64`. Compare that objective, empty-cluster count and PCA
retained variance using the same initialization and training protocol. Small objective changes were
observed after changing covariance accumulation order; faster PCA is not evidence of better
clustering quality. One seed limits cross-library claims.

In the M5 Pro full-source PCA stage measurements linked above, retained variance was 0.8638668 for
the previous implementation and 0.8638666 for the current implementation. Historical objective
tables from other hardware remain outside the repository.

## Validation and reproduction

- `make format lint lint-ast test-ast test-rust`; library tests plus workspace tests/doctests.
- `make test-python` for the Python behavior tests.
- Linux `cargo check --offline --lib --target x86_64-unknown-linux-gnu`; Intel macOS
  `cargo check --offline --lib --tests --target x86_64-apple-darwin`. Cross-target tests were not
  executed. A Linux test-only C dependency requires an unavailable cross compiler, so Linux
  compilation evidence is for the library, not its test binary.
- Focused tests cover NN/NT/TN/TT matrix layouts, padded output strides, accumulation,
  the portable fallback, and covariance/projection tails at the actual scratch boundary.
  Existing large-offset, overflow, numerical assignment guard and scalar-reference tests pass.
- The training rows above come from the repository CLI (`gathers --threads 16 kmeans -i INPUT
  -o OUTPUT` with defaults) built from each compared commit. The full-source PCA stage
  measurements use the public `PCA::fit`/`transform` API; no production hooks are required.
