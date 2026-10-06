# Clustering and assignment profiling

The non-published `gathers-cli` package in `crates/cli` builds the `gathers` executable.
It provides K-means training and RaBitQ assignment profiling.
Inputs and profiling artifacts stay outside the repository; tests use tiny local fixtures.

## Build and run

```sh
cargo build --profile perf -p gathers-cli
target/perf/gathers --threads 16 --seed 42 kmeans \
  -i /path/to/dataset.fvecs -o /path/to/centroids.fvecs -n 4096
target/perf/gathers --threads 16 --seed 42 assign \
  --vectors /path/to/dataset.fvecs --centroids /path/to/centroids.fvecs \
  --warmup 1 --repeats 5
```

Global options (`--threads`, `--seed`, `--cpu`, `--wait-for-profiler`) go **before** the subcommand.
Use `gathers --help`, `gathers kmeans --help`, or `gathers assign --help` for options.
For local installation, use `cargo install --path crates/cli`.

Only **fvecs** is supported: each row starts with a little-endian u32 dimension followed by that
many little-endian float32 coordinates. Dimensions must be positive and consistent, and coordinates
finite. Query and centroid dimensions must match. There is no `--dim` option or raw-f32 mode.
HDF5 datasets must be converted externally first (use `uv` for Python conversion scripts).

`kmeans` reads the source shape before allocating training data. Without `-n`, it chooses
`max(1, floor(rows^0.8 / 16))` clusters from the **full source row count**, the same formula as the
library's automatic configuration. Explicit `-n` overrides it. By default it selects
up to 256 rows per cluster uniformly without replacement. `--samples-per-cluster 128` lowers this
factor without fixing K; it must be at least 39. The total is min(source rows, factor * K).
`--training-samples N` overrides the factor and sets an exact sample size independently of K:
it must fit the source and provide at least 39 rows per cluster. The unused factor is ignored.
Invalid requests are rejected rather than clamped.

Sampling selects indices before reading vectors. Samples up to one third of the source use
rand's adaptive index sampler; denser samples use a reservoir. Sparse sampling avoids one RNG
draw per source row, including for sources larger than u32. Sorted indices drive seekable fvecs
reads: nearby rows are coalesced across gaps of at most 64 KiB, bounded by `--batch-rows`
(default 4096). Only selected rows are decoded and validated. File shape and the first dimension
are always checked; use `--validate-all` to scan and validate every row, including unselected rows.
Training keeps the sample in flat aligned RAM, not the full corpus. Default `auto` reduction
uses PCA to 128 dimensions when source rows >= 1,000,000 and dimension > 196; other inputs train
raw. `--reduction raw` disables projection. `--distance cos` normalizes training rows inside the
library; see the [reduction guide](./reduction.md).

`--memory-limit-gb` sets an optional preflight limit in decimal GB; there is **no limit by
default**. Before selecting indices or allocating the
sample, a conservative estimate includes the sample, batch, row metadata, centroid/index buffers,
worker scratch and 4 GiB of runtime headroom. The separate index-selection phase includes
conservative scratch for the adaptive sampler. An over-limit workload is rejected, not silently
subsampled further. This is an admission check, **not an OS-enforced RSS cap**; measure peak RSS
and leave room for other processes. For 10M rows at dimension 768, the default K is 24,881 and
the default training sample has 6,369,536 rows (19.57 GB of coordinates). A factor of 128 selects
3,184,768 rows (9.78 GB of coordinates). Pass `--memory-limit-gb` to reject workloads before
loading. Projected training temporarily retains both original and
reduced samples to reconstruct full-dimensional centroids. PCA transformation also holds a
centered copy; the estimate includes these buffers.
Sampling alone does not make arbitrary K fit: at 10B source rows the default K is 6.25M, and
even the minimum 39 rows per cluster would require about 749 GB at dimension 768. Use
`--memory-limit-gb` to bound training before it starts; otherwise reduce K or dimension
explicitly.

Library callers preparing data externally first call `KMeansConfig::resolve(original_rows, dim)`
to obtain the cluster count, training sample size, and projection settings. Use
`sampling::sample_indices` to load those rows, then construct `KMeans` from the resolved config
and call `fit_sample`. It returns `KMeansFit` without subsampling again. Passing the resolved
config preserves source-size decisions; an unresolved config otherwise only sees the supplied
sample's size. Residual centering uses the selected sample. File formats stay in the CLI.

`assign` still loads data into RAM. It accepts `--num-vectors` and `--num-centroids` to load
prefixes; omitted limits read all rows. Its per-row headers and values are checked for the loaded
prefix, not unread rows. The training memory limit does not apply to this profiling command.
Streaming full-corpus assignment remains a separate follow-up step. PCA and SRHT here transform
the sampled training rows, not the full corpus.

The index-sampling ablation needs no vector dataset:

```sh
GATHERS_RANDOM_SEED=42 cargo bench --bench sampling
```

It compares reservoir, rand's sampler, and the adaptive public API, including index allocation
and sorting but excluding RNG setup. Source sizes are logical counts, not allocated vectors.
The 10B-row case runs only the bounded samplers. For I/O comparisons, use identical sample size,
seed and batch size with and without `--validate-all`, and compare `sample_read_ms`. That
comparison includes the intentional difference in validation scope; record cache conditions.

Use trained centroids for representative assignment results: sampled rows can have very different
refinement rates. Training retains the library's minimum-samples-per-cluster requirement.

The `perf` **profile** provides release optimization and debug symbols. Do not enable the separate
library `perf` **feature**, which changes K-means assignment paths. Use `--threads 1` for one worker
or `--threads 0` (default) for Rayon's default. Debug builds work for correctness checks, but warn
and report `debug_assertions: true`; do not use their times for performance comparisons.

## Reports and reproducibility

Each command prints one JSON object to stdout; logging and warnings go to stderr.
The PID and attachment prompt are printed to stderr only with `--wait-for-profiler`.
Reports include shape, paths, seed, workers, CPU description, architecture, and OS. Hardware details
are supplied externally, not detected through a backend-name API.

- `kmeans`: `prepare_ms` measures sample selection and loading plus any normalization and
  projection. Separate `index_sample_ms`, `index_sort_ms`, and `sample_read_ms` isolate the
  sampling stages. `fit_ms` measures K-means fitting, including initialization and internal
  allocations, but no source sampling or I/O. Projected runs also report `projection_fit_ms`,
  `projection_transform_ms`, and `reconstruction_ms`; the projection fields are included in
  `prepare_ms`, while reconstruction is separate. Add `prepare_ms + fit_ms + reconstruction_ms`
  for the reported training stages, excluding centroid output.
  With `--wait-for-profiler`, projected runs pause after source loading and before the library's
  normalization/projection/training/reconstruction pipeline. Raw cosine normalization also happens
  once inside library preparation and is not repeated during dot training.
  Reports include actual `training_rows`, requested `samples_per_cluster` (even when overridden
  by an exact total), `batch_rows`, `validate_all`, `memory_limit_gb` and
  `estimated_memory_bytes`.
  Centroids are saved as fvecs.
- `assign`: builds one RaBitQ index, warms it up, then reuses the index and labels. `build_ms` is
  one construction. `query_ms` and `query_median_ms` measure complete public batch calls, including
  workspace allocation and metrics updates. File loading, pool setup, label allocation, hashing,
  and reporting are excluded. This measures assignment, **not end-to-end K-means training**.
- Assignment metrics include **warm-up and timed calls**, as recorded in `metrics_scope`. Repeated
  identical inputs have the same per-assignment refinement rate. The final label hash checks
  reproducibility; it does not prove agreement with exact assignment.

For comparisons, keep inputs (including centroid order), seed, workers, build flags, and machine
the same. Record `git rev-parse HEAD`, `rustc -Vv`, and any local diff beside the report.
For `assign`, `--seed` fixes the rotation within the same build/target, not across dependency
versions or architectures. For `kmeans`, the seed initializes separate RNGs for source sampling
and library training (initialization, RaBitQ rotations and empty-cluster repair). The sample is
retained in source order; changing the read batch size does not change training results. This
sampling/RNG boundary differs from the former load-all CLI, so its seeded results can change.
The existing library `fit` retains its reservoir order and RNG consumption.
Repeated training with identical input order, configuration,
worker count, build, and target reproduces the trained centroids. Results are not guaranteed
to be bitwise identical across dependency versions, architectures, or worker counts.
Without a library seed, training still uses fresh randomness.

Capture hardware information with shell tools:

```sh
uname -m
# macOS
sysctl machdep.cpu.brand_string hw.ncpu hw.memsize
sysctl hw.optional
# Linux
lscpu
```

Capabilities do not prove which kernel an index selected: small centroid sets use the row path,
for example. Inspect sampled stacks and inline frames when investigating actual kernel selection.

## CPU sampling

Profile separately from benchmarking: sampling adds overhead. For an assignment profile:

```sh
target/perf/gathers --threads 16 --seed 42 --wait-for-profiler assign \
  --vectors /path/to/dataset.fvecs --centroids /path/to/centroids.fvecs \
  --warmup 1 --repeats 5 --min-seconds 30
```

After loading, building, and warm-up, the process prints its PID and waits for Enter. Attach the
sampler in another terminal, then press Enter promptly. `assign` runs at least `--repeats` timed
passes and continues until `--min-seconds` has elapsed. Leave the duration at zero for fixed-repeat
benchmarks. `--wait-for-profiler kmeans ...` instead pauses after sampling, just before `fit_sample`.

On macOS:

```sh
dsymutil target/perf/gathers
sample PID 10 1 -file /tmp/gathers-sample.txt
```

When hot addresses resolve only to a Rayon caller, inspect inline frames using the report's
`Load Address` and sampled instruction address:

```sh
atos -o target/perf/gathers.dSYM -l LOAD_ADDRESS -inlineFrames -fullPath INSTRUCTION_ADDRESS
```

On Linux (subject to perf permissions):

```sh
perf record -F 999 -g --call-graph dwarf -p PID -o /tmp/gathers-perf.data -- sleep 10
perf report -i /tmp/gathers-perf.data
```

Attached profiles exclude input loading, but can include a brief wait for Enter. Ignore waiting
stacks when inspecting compute hotspots. Inclusive counts overlap: do not add them or interpret
them as wall-clock stage times. Inlining can merge scan and refinement into a caller; use inline
source/assembly attribution rather than assuming the caller's name identifies the bottleneck.
Correlate hotspots with refinement rates before selecting the next optimization.
