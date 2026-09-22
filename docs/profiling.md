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

Both commands load data into a flat aligned buffer in RAM. `kmeans` passes ownership to the library
without an intermediate nested-vector copy. `assign` accepts `--num-vectors` and `--num-centroids`
to load prefixes; omitted limits read all rows. File length must describe complete rows; per-row
headers and values are checked for the loaded prefix, not unread rows.

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

- `kmeans`: `fit_ms` measures one full library `fit` call, including its sampling and internal
  allocations. Input loading and centroid output are excluded. Centroids are saved as fvecs.
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
versions or architectures. For `kmeans`, the seed controls sampling, every internal RaBitQ
rotation, and empty-cluster repair. Repeated training with identical input order, configuration,
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
benchmarks. `--wait-for-profiler kmeans ...` instead pauses after loading, just before one `fit`.

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
