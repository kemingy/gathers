# Adaptive GEMM row blocks

Exact matrix assignment parallelizes query-row tiles through Rayon and calls each
`faer` GEMM with `Par::Seq`. The previous preferred tile was always 512 rows,
subject to the existing scratch cap. All 1,000 native queries therefore produced
only two tasks, even with sixteen workers.

The tile now depends on `ceil(rows / workers)`, rounded **up** to a power of two
and clamped to 64–512 rows. The existing memory cap can further reduce it; a cap
below 64 still selects direct assignment. Centroid tiles remain at most 1,024
rows. This changes scheduling, while retaining exhaustive scoring, ambiguity
refinement, tie handling and non-finite fallbacks. There is no public API change.

For 1,000 queries, the selected row tiles are 512, 256, 128, 128 and 64 for
1, 4, 8, 12 and 16 workers respectively. A 65,536-row streaming batch still
selects 512 with sixteen workers. Rounding up balances independent tasks against
packing overhead: it need not produce a task for every worker. The 64-row minimum
also limits parallelism for very small batches.

## Environment and inputs

Apple M3 Max, 16 physical cores, 64 GiB, macOS 26.7.1, aarch64/NEON;
Rust 1.99.0, optimized `perf` profile (release, thin LTO, one codegen unit),
without the library's separate `perf` feature or `target-cpu=native`.
Worker counts are explicit Rayon pool sizes. Benchmarks execute serially.

Only the user's local datasets were used:

| Dataset | Full source rows | Dimensions | Final centroids | Native queries | Metric |
| --- | ---: | ---: | ---: | ---: | --- |
| GIST | 1,000,000 | 960 | 3,943 | 1,000 | Squared L2 |
| Cohere large | 10,000,000 | 768 | 24,881 | 1,000 | Cosine |

GIST comes from `gist-960-euclidean.hdf5`, losslessly converted to fvecs outside
timing. Cohere comes from `cohere-large-10m.fvecs` and its native
`cohere-large-10m-queries.fvecs`. Final original-space centroid files were retained
from the full-corpus exact-training experiment (seed 42, PCA128, ten maximum
iterations, tolerance 0.01). Every source row participated in that experiment;
this study reuses the same final models instead of retraining for each row policy.
Cosine inputs and centroids are normalized, then searched with negative dot product.

## Row-policy ablation

The native-query ablation uses every native query. It warms each policy once,
then reports the median of seven assignment calls, including centroid norm
refresh and tile allocation. Loading, query normalization and workspace creation
are outside these timers. Each policy's labels equal the baseline labels.

The following table records the experiment used to choose rounding. All values
are milliseconds. The baseline is the previous 512-row preference; every policy
also obeys the same scratch cap.

| Dataset | Workers | 512 | 256 | 128 | 64 | Adaptive down | Adaptive up (selected) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GIST | 1 | 77.892 | 81.098 | 85.538 | 93.318 | 80.294 | 79.457 |
| GIST | 4 | 39.726 | 21.922 | 22.596 | 24.936 | 22.971 | 21.963 |
| GIST | 8 | 39.981 | 21.334 | 11.433 | 12.626 | 12.695 | 11.516 |
| GIST | 12 | 40.413 | 21.113 | 11.345 | 12.525 | 12.578 | 11.494 |
| GIST | 16 | 40.507 | 21.430 | 11.374 | 10.718 | 10.943 | 10.578 |
| Cohere | 1 | 460.684 | 474.532 | 503.154 | 555.180 | 453.102 | 451.646 |
| Cohere | 4 | 237.814 | 129.358 | 130.494 | 144.431 | 133.474 | 129.260 |
| Cohere | 8 | 229.919 | 125.292 | 70.708 | 77.433 | 76.936 | 70.940 |
| Cohere | 12 | 226.574 | 122.694 | 71.146 | 71.213 | 71.692 | 69.218 |
| Cohere | 16 | 228.393 | 124.072 | 70.136 | 58.365 | 60.082 | 60.548 |

Fixed 64-row tiles penalize the one-worker cases through additional packing and
task overhead. Adaptive up keeps 512 there, and performs better than adaptive
down at eight and twelve workers in this experiment. Both rounding rules choose
64 at sixteen workers; differences between those entries measure run variation.
Policies execute in a fixed order, so small differences are not statistically
established gains. This supports adaptive sizing, rather than a universal optimum
for these constants or for other CPUs.

## Final implementation verification

A fresh run of the final implementation produced these query medians (ms):

| Dataset | Workers | Baseline | Current | Speedup |
| --- | ---: | ---: | ---: | ---: |
| GIST | 1 | 77.395 | 79.272 | 0.976× |
| GIST | 4 | 39.568 | 21.801 | 1.815× |
| GIST | 8 | 39.756 | 11.376 | 3.495× |
| GIST | 12 | 40.619 | 11.288 | 3.598× |
| GIST | 16 | 40.841 | 10.705 | 3.815× |
| Cohere | 1 | 460.130 | 453.033 | 1.016× |
| Cohere | 4 | 231.356 | 128.010 | 1.807× |
| Cohere | 8 | 228.190 | 70.598 | 3.232× |
| Cohere | 12 | 225.407 | 69.748 | 3.232× |
| Cohere | 16 | 225.714 | 61.265 | 3.684× |

Full-corpus final assignment, sixteen workers:

| Dataset | Passes per variant | Baseline assign s | Current assign s | Baseline wall s | Current wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| GIST | 3 | 8.224 | 8.039 | 9.053 | 8.880 |
| Cohere | 1 | 443.414 | 439.106 | 451.949 | 446.673 |

The GIST bulk timings overlap run variation. Every full batch uses the same
512-row tile under baseline and current. The Cohere comparison is a single complete pass
per variant, so small differences cannot establish a bulk improvement.

Full assignment reads **all** source vectors, normalizes Cohere batches, creates
workspaces, counts memberships and writes one little-endian u32 label per row.
The 65,536-row batch size limits streaming memory; it does not limit dataset size.
Assignment time excludes these surrounding stages; wall time includes them.
Input cache state is uncontrolled and output flushes are buffered, without fsync.
Every full output is compared byte-for-byte with the existing exact full-corpus
oracle, including label count and source order.

This addresses small-batch exact assignment. It does not establish a full K-means
training speedup, a Faiss win, or a new state-of-the-art clustering method.
Large batches already provide many independent tasks and retain 512-row tiles.
Other CPUs and intermediate batch sizes need their own measurements.

## Profiling the remaining Faiss gap

All measurements below use the same complete 1,000-query sets and full-trained
centroids, sixteen requested workers and the environment above. Native macOS
`sample` captured eight seconds at 1 ms intervals after loading and warm-up,
while each implementation repeatedly assigned the complete query set. Rust
stacks identify `gemm_f32`'s NEON `4x4x4` microkernel and operand packing. Faiss
stacks identify Apple Accelerate `SGEMM`/`libBLAS`; some private BLAS symbols are
stripped, so these traces do not establish which Apple matrix instructions run.

After excluding wait frames, roughly 87% of GIST and 79% of Cohere Rust samples
are in GEMM/packing. These are sample proportions, not wall-time percentages.
Temporary stage timers give the following shares of **summed worker-stage
elapsed time**, excluding the separate centroid norm refresh:

| Stage | GIST | Cohere |
| --- | ---: | ---: |
| GEMM, including packing | 92.93% | 86.43% |
| Score scan and L2 candidate refinement | 6.56% | 3.66% |
| Final label selection / direct fallback | <0.01% | 9.87% |
| Scratch allocation | 0.51% | 0.04% |

Worker timers overlap and cannot be added to assignment wall time. Centroid norm
refresh separately averages 0.47 ms on GIST and 3.28 ms on Cohere. GIST refines
21,210 candidate distances per call and performs no full direct rescans. Cohere's
numerical uncertainty guard rescans all 24,881 centroids for 31 queries (3.1%).

To isolate the backend, a temporary macOS-only experiment replaced only the GEMM
call with Accelerate `cblas_sgemm`, preserving row/centroid tiles, score scanning,
norm refresh and numerical guards. Three runs per variant alternate backend
order; each run reports seven-call medians after warm-up. Every query label
matches between backends. Medians of those run medians:

| Dataset | Current faer | Accelerate experiment | Fresh Faiss exact flat |
| --- | ---: | ---: | ---: |
| GIST | 11.27 ms | 4.50 ms | 6.22 ms |
| Cohere | 62.60 ms | 35.96 ms | 31.51 ms |

The backend is the largest measured cause of the gap. Allocation is a small
share. Faiss also prefers 4,096-query BLAS blocks (database blocks 1,024), whereas
our adaptive sixteen-worker query tile is 64; the observed Rust packing work
makes layout/packing reuse another useful target. The backend experiment retains
our tiling, so its improvement does not depend on adopting Faiss's block policy.
Accelerate's internal threads remain independently uncontrolled.

A separate diagnostic bypass of the Cohere direct-rescan guard reduced faer
57.21 → 50.28 ms and Accelerate 34.30 → 21.44 ms (three alternating runs,
seven calls each). All fixture labels still matched. This identifies fallback
cost; it does **not** justify removing the guard for general inputs. A useful
follow-up is to refine only candidates admitted by conservative score bounds,
rather than rescanning every centroid for an ambiguous query. Existing near-tie,
overflow and subnormal tests must continue to pass.

The backend swap, guard bypass, counters and sampling loop exist only in the
local investigation artifacts. Production behavior remains the adaptive row
selection. These measurements cover query assignment, not full Faiss training
or full-corpus Accelerate assignment, and one CPU does not establish a portable
backend choice.

## Reproduction

The isolated benchmark harness is under `benches/gemm_parallelism/`. Preparation
archives the checkout's HEAD, overlays the working tree's matrix source, and adds
private-workspace exports and row-policy overrides **only to a temporary copy**.
It requires Python 3.12+ and an empty destination. No benchmark hook is compiled
into the production library.

```sh
python3.12 benches/gemm_parallelism/prepare.py /tmp/gathers-gemm-bench
cargo build --offline --profile perf -p gathers-cli --example query_compare \
  --manifest-path /tmp/gathers-gemm-bench/Cargo.toml

/tmp/gathers-gemm-bench/target/perf/examples/query_compare \
  /path/to/gist-queries.fvecs /path/to/gist-exact.fvecs l2 16
/tmp/gathers-gemm-bench/target/perf/examples/query_compare \
  /path/to/cohere-large-10m-queries.fvecs /path/to/cohere-exact.fvecs cos 16
```

Repeat with 1, 4, 8, 12 and 16 workers. Supply final original-space centroids
trained on all source rows. The maintained harness compares the baseline 512-row
preference (policy 512) with production adaptive sizing (policy 1), writes JSON
medians and checks all query labels. The broader tile/rounding ablation and
full-corpus streaming validation were investigation-only harnesses, retained with
the raw local experiment artifacts rather than duplicated in the repository.

Tests cover scratch limits, worker-count sensitivity, one-row tile tails,
65-dimensional SIMD tails, crossing the 1,024-centroid tile boundary, and direct
reference label equality for squared L2 and negative dot product. Existing
near-tie, overflow, underflow and workspace-reuse coverage is retained.
