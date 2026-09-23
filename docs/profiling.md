# Clustering and assignment profiling

The non-published `gathers-cli` package in `crates/cli` builds the `gathers` executable.
It provides K-means training and RaBitQ assignment profiling.
Inputs and profiling artifacts stay outside the repository; tests use tiny local fixtures.

## Build and run

```sh
cargo build --profile perf -p gathers-cli
target/perf/gathers --threads 16 --seed 42 kmeans \
  -i /path/to/dataset.fvecs -o /path/to/centroids.fvecs -n 4096 -m 25
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
`max(1, floor(rows^0.8 / 16))` clusters from the **full source row count**. Explicit `-n` overrides
this CLI policy; the library's default cluster-count policy is unchanged. By default it selects
up to 256 rows per cluster uniformly without replacement. `--training-samples N` sets an explicit
sample size independently of K: it must fit the source and provide at least 39 rows per cluster.
Invalid requests are rejected rather than clamped.

Sampling selects indices before reading vectors. Samples up to one third of the source use
rand's adaptive index sampler; denser samples use a reservoir. Sparse sampling avoids one RNG
draw per source row, including for sources larger than u32. Sorted indices drive seekable fvecs
reads: nearby rows are coalesced across gaps of at most 64 KiB, bounded by `--batch-rows`
(default 4096). Only selected rows are decoded and validated. File shape and the first dimension
are always checked; use `--validate-all` to scan and validate every row, including unselected rows.
Training keeps the sample in one flat aligned RAM buffer, not the full corpus. Coordinates are
preserved: this command does not normalize or apply PCA.

`--memory-budget-gb` defaults to **48 decimal GB**. Before selecting indices or allocating the
sample, a conservative estimate includes the sample, batch, row metadata, centroid/index buffers,
worker scratch and 4 GiB of runtime headroom. The separate index-selection phase includes
conservative scratch for the adaptive sampler. An over-budget workload is rejected, not silently
subsampled further. This is an admission check, **not an OS-enforced RSS cap**; measure peak RSS
and leave room for other processes. For 10M rows at dimension 768, the default K is 24,881 and
the training sample has 6,369,536 rows (19.57 GB of coordinates). Original and projected samples
are not needed simultaneously when an external batch projection prepares training input.
Sampling alone does not make arbitrary K fit: at 10B source rows the default K is 6.25M, and
even the minimum 39 rows per cluster would require about 749 GB at dimension 768. Such a
configuration is rejected under the default budget; reduce K or dimension explicitly.

Library callers preparing data externally can use `sampling::sample_indices` and
`KMeans::training_sample_size`, then pass a flat aligned sample to `KMeans::fit_sample`.
`fit_sample` never subsamples again. Configure K from the original dataset size before sampling;
the default library configuration only sees the supplied sample's size. If residual centering is
enabled, its mean is computed from the supplied sample. File formats stay in the CLI.

`assign` still loads data into RAM. It accepts `--num-vectors` and `--num-centroids` to load
prefixes; omitted limits read all rows. Its per-row headers and values are checked for the loaded
prefix, not unread rows. The training memory budget does not apply to this profiling command.
Streaming full-corpus assignment and PCA are separate follow-up steps.

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

- `kmeans`: `prepare_ms` measures index selection, sorting and sample loading. Separate
  `index_sample_ms`, `index_sort_ms`, and `sample_read_ms` isolate those stages.
  `fit_ms` measures `fit_sample`, including initialization and internal allocations, but no
  source sampling or I/O. Their sum includes preparation and training, excluding centroid output.
  Reports include `training_rows`, `batch_rows`, `validate_all`, the budget and `estimated_memory_bytes`.
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
