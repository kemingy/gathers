# macOS Accelerate refinements

PCA covariance now uses the lower triangle of Accelerate SYRK. The eigensolver already reads `Side::Lower`, so the pipeline avoids mirroring and accumulating the unused triangle. Fitting keeps its 16 MiB centered buffer, f64 means and block sums, finite checks and overflow errors. Other platforms and unsupported BLAS layouts retain the full matrix-product fallback, with only its lower triangle accumulated. Projection/reconstruction still use SGEMM; the eigensolver remains faer.

On Apple Silicon macOS, matrix assignment lowers its comparison-count crossover from 524,288 to 32,768 only for dimensions at least 128, at least sixteen query rows, and at least 4,096 query values. Thus PCA128 needs at least 32 rows, while 768D/960D batches qualify from sixteen rows. Smaller batches and dimensions, Intel macOS and non-macOS targets retain the previous cutoff. The below-64D cutoff remains 4,194,304 comparisons. These conditions avoid the measured single-query and PCA128 tiny-batch regressions without changing the 64–512 internal row tiles, 1,024-centroid tiles or scratch cap.

## Environment and complete inputs

Apple M5 Pro, 18 physical cores (12 Performance + 6 Super), 64 GiB, macOS 26.6, aarch64/NEON, rustc 1.99.0. Release builds use thin LTO and one codegen unit, sixteen Rayon workers and default Accelerate threading, without target-cpu=native or the separate library perf feature. Benchmark processes run serially without concurrent compilation or tests. The covariance, crossover and validation tables below were measured on this machine. Historical results from other hardware are kept outside the repository; the rejected-alternative sections retain their methods and qualitative observations.

PCA fits and projects every row of local GIST (1,000,000 x 960) with output dimension 128; SIFT (1,000,000 x 128) contributes a synthetic PCA128 rotation workload at its native dimension. No source rows are sampled or capped in these stage tests. Native-query tests process the complete query files (GIST 1,000 x 960, SIFT 10,000 x 128), with K=3,943 fixed centroids trained on all source rows (repository CLI, seed 42, ten iterations, tolerance 0.01). The PCA128 query/centroid inputs use projections learned from the full GIST fit; the centroids are projected fixed models, not separately retrained 128D models. Both datasets use squared L2. Loading is outside assignment timers.

## Covariance ablation

Three full fit/project passes per variant, one process per variant and dataset. Entries are medians; fit includes validation, mean, centering, covariance, f64 accumulation and eigendecomposition. Projection processes the complete source and allocates its output. Loading, serialization and native-query checks are outside these timers. Variants are the public `PCA` API built from the pre-PCA-change commit (previous GEMM/full accumulation), the blocked-SGEMM commit (GEMM/lower), and this branch (SYRK/lower).

| Full dataset | Previous GEMM/full accumulation fit s | GEMM/lower accumulation fit s | SYRK/lower accumulation fit s | Fit speedup |
| --- | ---: | ---: | ---: | ---: |
| gist 960→128 | 2.461 | 1.268 | 1.134 | 2.17x |
| sift 128→128 | 0.215 | 0.079 | 0.078 | 2.76x |

| Full dataset | Previous fit + projection s | Current fit + projection s | Speedup |
| --- | ---: | ---: | ---: |
| gist 960→128 | 3.010 | 1.683 | 1.79x |
| sift 128→128 | 0.316 | 0.185 | 1.71x |

The SYRK step alone accounts for 1.12x of the GIST fit gain over GEMM/lower accumulation; at 128D the two are within 2%, so the SIFT gain comes almost entirely from blocked centering and lower-triangle accumulation. Retained variance is 0.8638668 (previous) versus 0.8638666 (current) on GIST, and 1.0 on the synthetic SIFT rotation. These are PCA-stage wall times, not full K-means or train-plus-assignment speedups.

## Crossover sensitivity

Each entry processes all queries using calls of the listed size, including the last partial batch. Workspace construction, vector norms, centroid norm refresh, score allocation and direct fallback are included. Each policy warms once, then times seven calls. Every label matches the direct SIMD reference under both policies.

| Input | Rows per call | Previous ms | Current ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| gist original 960D | 16 | 296.178 | 48.166 | 6.15x |
| gist original 960D | 64 | 296.497 | 19.646 | 15.09x |
| gist original 960D | 128 | 152.968 | 11.750 | 13.02x |
| sift original 128D | 16 | 256.141 | 256.113 | 1.00x |
| sift original 128D | 64 | 256.728 | 83.332 | 3.08x |
| sift original 128D | 128 | 131.634 | 47.237 | 2.79x |
| gist PCA128 | 16 | 26.510 | 25.852 | 1.00x |
| gist PCA128 | 31 | 26.045 | 25.658 | 1.00x |
| gist PCA128 | 32 | 26.267 | 11.848 | 2.22x |
| gist PCA128 | 64 | 25.777 | 8.630 | 2.99x |

The guards behave as designed on both datasets. Sixteen 128D rows carry only 2,048 values, below the 4,096-value requirement, so SIFT 16-row calls stay on direct SIMD under the current policy and tie the previous one. GIST PCA128 needs 32 rows to reach 4,096 values: 31-row calls tie while 32-row calls gain 2.22x. Cases with identical backend selection vary by a few percent, so those differences are not claimed as improvements. Larger available batches still provide the best total throughput.

These are measurements of the private assignment path. At measurement time, standalone Python `batch_assign` used a separate path. Current batch assignment shares the [automatic/exact policy](./adaptive-training-probes.md#public-batch-assignment). These measurements predate the [adaptive training policy](./adaptive-training-probes.md); select exact training explicitly when reproducing the flat baseline. Query timing does not establish a full-training speedup.

## Rejected GEMV substitution

The trial replaced single-row SGEMM score tiles with Accelerate SGEMV while retaining workspace
setup, centroid norms, score scanning and numerical safeguards. It processed the complete native
GIST query set one query at a time in original dimensions and PCA128, with one warmup and three
timed passes per variant. This measures the existing tiled assignment pipeline, not a new cached
centroid-index API.

The trial did not improve that pipeline, so GEMV is not included in production. This observation
does not rule out a future cached fixed-centroid implementation. No LAPACK/vDSP replacement, public
API or dependency is added. Machine-specific run results are maintained outside the repository.

## BLAS interface ablation

The investigation compared legacy LP64, new LP64 (`$NEWLAPACK`) and new ILP64 (`$NEWLAPACK$ILP64`)
in one compiled release binary. Enum arguments remain 32-bit; matrix dimensions/strides widen only
for ILP64. The newer interface requires macOS 13.3+. The eigensolver remains faer: this is a BLAS
interface comparison, not a LAPACK solver comparison.

Keep source rows, fixed models, algorithms, numerical guards and worker counts identical across
interfaces. Time complete PCA fits/projections after a full-source warmup, rotating API order, and
alternate interfaces for complete original-space final assignments. Include assignment workspace
construction and norms; exclude loading. Check native queries in original dimensions and PCA128 at
several call sizes, including small batches and the complete query set.

The observed differences did not establish a consistent gain. Compare all-value projection hashes,
projected queries/centroids and every assignment label across interfaces, alongside kernel
layout/padding/tail/empty/fallback tests. The historical checks agreed across interfaces; their run
data remain outside the repository. No production API, OS minimum or dependency change was
justified.

## Reproduction

The maintained Cargo benchmark supports the crossover comparison:

```console
cargo bench -p gathers-cli --bench gemm_parallelism -- \
  gist_query.fvecs gist_centroids.fvecs l2 --threads 16 --batch-rows 64
```

`--batch-rows` controls processing granularity, never the dataset row count. Its timer includes workspace creation and fallback. The default backend/row-policy comparison retains its original timing boundaries. See [reproduction details](./gemm-parallelism.md#reproduction-and-checks) for the JSON fields and label checks. The covariance ablation above uses the public `PCA::fit`/`transform` API from each measured commit with full-source inputs; no production hooks are required.

Full-data PCA ablations and rejected alternatives stay in an external investigation harness rather than adding production flags or benchmark targets.

## Validation

The production library, without investigation hooks, fit and projected full-source GIST PCA128 three times per variant in this revision. Preserved variance agreed to the printed f64 precision (0.8638666) between the blocked SGEMM and SYRK implementations, and every crossover run's labels matched the direct SIMD reference. Bitwise projection equality was not re-checked in this revision; these observations do not promise bitwise equality on other inputs or hardware.

`make format lint lint-ast test-ast test-rust` passed, along with all Python tests through `make test-python`. Linux x86_64 library and Intel macOS library/test compile checks passed; cross-target tests were not executed. The focused tests cover symmetric-product layouts, padding, tails, empty reductions, portable fallback, crossover boundaries and L2/dot assignment labels.

A seven-sample reproduction run on the complete GIST native queries with 64 rows per call measured 296.497 ms previously and 19.646 ms currently, with identical labels. The corresponding SIFT run with 64 rows per call measured 256.728 ms previously and 83.332 ms currently, also with identical labels.
