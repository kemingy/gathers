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

Earlier measurements, not re-run in this revision; machine labels have been removed.
Fit plus projection only, using all GIST rows, with loading outside the timer:

| PCA backend / centered scratch | GIST seconds, median of 3 |
| --- | ---: |
| faer / full centered corpus | 4.259 |
| Accelerate / full centered corpus | 1.452 |
| faer / 16 MiB blocks | 4.131 |
| Accelerate / 16 MiB blocks | 1.618 |

At 960D and 1M rows, an unblocked transform would hold the ~3.84 GB input, a ~3.84 GB centered
copy, and a ~0.51 GB output before other workspaces. The blocked transform needs the input and
output plus at most 16 MiB of centered scratch; the saving grows linearly with corpus size.

Scratch sensitivity on full GIST (fit + projection, medians of three), same earlier measurement:

| Centered scratch | Seconds |
| --- | ---: |
| 4 MiB | 2.615 |
| 16 MiB | 1.618 |
| 64 MiB | 1.453 |

16 MiB avoids the 4 MiB scheduling overhead while using one quarter of the scratch of 64 MiB.
The latter saves about 0.16 s in this stage test; 16 MiB is the current bounded-memory tradeoff,
not a claim of a universal optimal block size. All these configurations fit every source row.
This revision's full-source GIST stage re-measurement of the current implementation
(SYRK / 16 MiB blocks) is 1.683 s, consistent with the table above.

The subsequent [Accelerate refinements](./accelerate-refinements.md) isolate SYRK and
lower-triangle accumulation and report measurements against the blocked SGEMM implementation.

## Clustering quality

Earlier measurement from the previous protocol, not re-run in this revision; the machine label
has been removed.

| Original-space SSE after final reassignment (lower is better) | Previous PCA | Improved PCA |
| --- | ---: | ---: |
| GIST | 1039435.749 | 1039447.091 |

The SSE change is +0.00109% on GIST, measured by summing selected-centroid squared distances
in `f64`. All completed fits and final assignments had zero empty clusters. Preserved variance
on full-source GIST PCA128 is 0.8638668 (previous) versus 0.8638666 (current) in this revision's
runs. The improvement is in time and memory, not evidence of better clustering quality.
Different initial centroids and one seed limit cross-library claims.

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
