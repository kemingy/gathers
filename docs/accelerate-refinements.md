# macOS Accelerate refinements

PCA covariance now uses the lower triangle of Accelerate SYRK. The eigensolver already reads `Side::Lower`, so the pipeline avoids mirroring and accumulating the unused triangle. Fitting keeps its 16 MiB centered buffer, f64 means and block sums, finite checks and overflow errors. Other platforms and unsupported BLAS layouts retain the full matrix-product fallback, with only its lower triangle accumulated. Projection/reconstruction still use SGEMM; the eigensolver remains faer.

On Apple Silicon macOS, matrix assignment lowers its comparison-count crossover from 524,288 to 32,768 only for dimensions at least 128, at least sixteen query rows, and at least 4,096 query values. Thus PCA128 needs at least 32 rows, while 768D/960D batches qualify from sixteen rows. Smaller batches and dimensions, Intel macOS and non-macOS targets retain the previous cutoff. The below-64D cutoff remains 4,194,304 comparisons. These conditions avoid the measured single-query and PCA128 tiny-batch regressions without changing the 64–512 internal row tiles, 1,024-centroid tiles or scratch cap.

## Environment and complete inputs

Apple M3 Max, 16 physical cores, 64 GiB, macOS 26.7.1, aarch64/NEON, rustc 1.99.0. Release builds use thin LTO and one codegen unit, sixteen Rayon workers and default Accelerate threading, without target-cpu=native or the separate library perf feature. Benchmark processes run serially without concurrent compilation or tests.

PCA fits and projects every row of local GIST (1,000,000 x 960) and Cohere (10,000,000 x 768), with output dimension 128. No source rows are sampled or capped. Native-query tests always process the complete 1,000-row query files, with K=3,943 / 24,881 fixed centroids from the earlier full-source experiment. The PCA128 query/centroid inputs use projections learned from these full fits; the centroids are projected fixed models, not separately retrained 128D models. GIST uses squared L2; Cohere normalizes query and centroid rows and uses negative dot product. Loading and normalization are outside assignment timers.

## Covariance ablation

Three full fit/project passes per variant in one process per dataset, with variant order reversed on the middle pass. Entries are medians; fit includes validation, mean, centering, covariance, f64 accumulation and eigendecomposition. Projection processes the complete source and allocates its output. Loading, serialization and native-query checks are outside these timers.

| Full dataset | Previous GEMM/full accumulation fit s | GEMM/lower accumulation fit s | SYRK/lower accumulation fit s | Fit speedup |
| --- | ---: | ---: | ---: | ---: |
| gist | 1.116 | 1.073 | 0.982 | 1.14x |
| cohere | 8.121 | 7.877 | 6.584 | 1.23x |

| Full dataset | Previous fit + projection s | Current fit + projection s | Speedup |
| --- | ---: | ---: | ---: |
| gist | 1.636 | 1.499 | 1.09x |
| cohere | 12.538 | 10.978 | 1.14x |

Cohere's first full baseline fit was 10.565 s, versus 8.035 and 8.121 s on subsequent passes; the median shown is 8.121 s. SYRK/lower fits were 6.555–6.616 s. The initial gap is not counted as a standalone SYRK win. A separate first experiment using SYRK followed by full mirroring/accumulation showed no repeatable benefit: the covariance consumer also needs to avoid processing the unused triangle.

Retained variance is unchanged at 0.8638670712914187 (GIST) and 0.7783174503018796 (Cohere). These are PCA-stage wall times, not full K-means or train-plus-assignment speedups.

## Crossover sensitivity

Each entry processes all 1,000 queries using calls of the listed size, including the last partial batch. Workspace construction, vector norms, centroid norm refresh, score allocation and direct fallback are included. Each policy warms once, then times three complete passes with alternating policy order. Every label matches the direct SIMD reference under both policies.

| Input | Rows per call | Previous ms | Current ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| gist original | 16 | 341.894 | 71.973 | 4.75x |
| gist original | 64 | 344.623 | 31.992 | 10.77x |
| gist original | 128 | 183.338 | 13.841 | 13.25x |
| cohere original | 16 | 1564.128 | 430.168 | 3.64x |
| gist PCA128 | 16 | 31.413 | 31.321 | 1.00x |
| gist PCA128 | 31 | 31.336 | 31.286 | 1.00x |
| gist PCA128 | 32 | 31.281 | 16.699 | 1.87x |
| gist PCA128 | 64 | 31.789 | 10.583 | 3.00x |

Sensitivity includes original dimensions and PCA128, row counts 1/2/4/8/15/16/17/31/32/33/64/128/256/1,000 across the exploratory and final-policy tests. At PCA128, forcing 16-row GIST calls through GEMM is only roughly tied with direct SIMD; the 4,096-value requirement keeps that case on direct assignment. Single queries remain on the same path. Cases with identical backend selection vary by a few percent, so those differences are not claimed as improvements. Larger available batches still provide the best total throughput.

These are measurements of the private assignment path. Standalone Python `batch_assign` routing is unchanged, and the large-input L2 training heuristic is unchanged. Query timing does not establish a full-training speedup.

## Rejected GEMV substitution

The trial replaced single-row SGEMM score tiles with Accelerate SGEMV while retaining workspace setup, centroid norms, score scanning and numerical safeguards. This measures the existing tiled assignment pipeline, not a new cached centroid-index API. All 1,000 native queries are processed one at a time; entries are medians of three passes after one warmup.

| Input | Direct SIMD ms | SGEMM ms | SGEMV ms |
| --- | ---: | ---: | ---: |
| gist original | 341.404 | 555.933 | 600.625 |
| cohere original | 1533.956 | 4114.455 | 4315.612 |
| gist PCA128 | 31.908 | 242.067 | 254.074 |
| cohere PCA128 | 176.862 | 881.479 | 877.647 |

GEMV does not improve this path and is not included in production. Its results do not rule out a future cached fixed-centroid implementation. No LAPACK/vDSP replacement, public API or dependency is added.

## Reproduction

The maintained isolated benchmark supports the crossover comparison:

```console
python3.12 benches/gemm_parallelism/prepare.py /tmp/gathers-gemm-refinements
cargo build --offline --profile perf -p gathers-cli --example query_compare --manifest-path /tmp/gathers-gemm-refinements/Cargo.toml
/tmp/gathers-gemm-refinements/target/perf/examples/query_compare /private/tmp/pr65-real-data/gist-queries.fvecs /private/tmp/pr65-exact-data/gist-exact.fvecs l2 16 64
```

The optional last argument controls processing granularity, never the dataset row count. Its timer includes workspace creation and fallback. The default four-backend/policy comparison retains its original timing boundaries.

Full-data PCA ablations and rejected alternatives stay in an external investigation harness rather than adding production flags or benchmark targets. Commands, sources, temporary hooks, all raw timing vectors and logs are saved in the accompanying local `gathers-gemm-parallelism/accelerate-extra` report directory.

## Validation

The production library, without investigation hooks, repeated full-source PCA fitting and projection three times per dataset. Its PCA128 projections of every native query and every fixed centroid were bit-for-bit identical to the previous blocked SGEMM implementation in all six passes. Warm production fit times corroborated the ablation: 0.974–0.994 s for GIST and 6.604–6.704 s for Cohere (the first Cohere pass took 8.491 s). These observations do not promise bitwise equality on other inputs or hardware.

`make format lint lint-ast test-ast test-rust` passed, along with all 73 Python tests through `make test-python`. Linux x86_64 library and Intel macOS library/test compile checks passed; cross-target tests were not executed. The new focused tests cover symmetric-product layouts, padding, tails, empty reductions, portable fallback, crossover boundaries and L2/dot assignment labels.

The maintained isolated benchmark built successfully and checked both crossover policies against its existing reference labels on all 1,000 native queries. A seven-sample reproduction run on GIST with 64 rows per call measured 346.102 ms previously and 25.319 ms currently. The corresponding Cohere run with sixteen rows per call measured 1,525.143 ms previously and 431.449 ms currently, also with identical labels. Raw vectors are retained separately from the three-sample ablation table above.
