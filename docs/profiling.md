# Profiling RaBitQ assignment

The `profile_assignment` example runs the public batched retrieval API against external data.
It builds one index, warms it up, then reuses the index and labels for repeated assignments.
It does not run K-means training, update centroids, or measure file I/O. Large datasets are not
bundled with the repository or required by tests.

## Build and benchmark

```sh
cargo build --profile perf --example profile_assignment
target/perf/examples/profile_assignment \
  --vectors /path/to/gist-vectors.f32 \
  --centroids /path/to/gist-trained-centroids.f32 \
  --dim 960 --num-centroids 4096 --threads 16 \
  --seed 42 --warmup 1 --repeats 5 --cpu 'Apple M3 Max'
```

Both files must be headerless, row-major, little-endian float32 with finite values and complete
rows. Omit `--num-vectors` or `--num-centroids` to read all rows; specify either to read a prefix.
Use centroids trained on the workload when evaluating production performance: sampled input rows
can have substantially different refinement rates. The runner loads the selected inputs into RAM.

The `perf` **profile** is release optimization plus debug symbols. Do not enable the separate
`perf` **feature** for this runner; use `--threads 1` to measure single-worker retrieval, or
`--threads 0` (the default) for Rayon's default worker count.

One JSON object goes to stdout; readiness and the PID go to stderr. Redirect stdout to save a run.
The report contains build time, individual query times and their median, aggregate throughput,
input shape/paths, seed, worker count, CPU description, a label hash, and metrics.

- `build_ms` is one index construction, not a median.
- `query_ms` includes the complete public batch call, including its workspace allocation and
  metrics updates. File loading, pool setup, label allocation, hashing, and reporting are excluded.
- Metrics are cumulative over **warm-up and timed calls**, as indicated by `metrics_scope`.
  With repeated identical inputs, the refinement rate is also the per-assignment rate.
- The label hash is a reproducibility check, not evidence of agreement with exact assignment.

For comparisons, keep inputs (including centroid order), seed, thread count, build flags, and machine
the same. Record `git rev-parse HEAD`, `rustc -Vv`, and any local diff alongside the output. A fixed
seed reproduces rotations within the same build/target; it does not promise bit-identical behavior
across dependency versions or architectures. Leave `--min-seconds` at zero for fixed-repeat timing.

Capture hardware details alongside the report using shell tools:

```sh
uname -m
# macOS
sysctl machdep.cpu.brand_string hw.ncpu hw.memsize
sysctl hw.optional
# Linux
lscpu
```

These describe hardware capabilities, not the backend selected for a particular index. Selection
also depends on the workload (small centroid sets use the row path). Use sampled stacks and inline
frames to inspect the executed kernels; the runner does not duplicate runtime dispatch detection.

## CPU sampling

Run profiling separately from timing: a sampler adds overhead. No per-query clocks, tracing spans,
or new hot-loop counters are needed. The default production constructor still uses a random seed.

Start a longer run with the same inputs and options, adding:

```sh
--min-seconds 30 --wait-for-profiler
```

After build and warm-up, the runner prints its PID and waits for Enter. In another terminal,
attach a sampler, then press Enter in the runner terminal promptly. The timed phase completes at
least `--repeats` assignments and continues until `--min-seconds` has elapsed.

On macOS:

```sh
dsymutil target/perf/examples/profile_assignment
sample PID 10 1 -file /tmp/rabitq-sample.txt
```

If a hot address resolves only to a Rayon caller, inspect its inlined frames. Substitute the
`Load Address` and a sampled instruction address from the report:

```sh
atos -o target/perf/examples/profile_assignment.dSYM \
  -l LOAD_ADDRESS -inlineFrames -fullPath INSTRUCTION_ADDRESS
```

On Linux (subject to the host's perf permissions):

```sh
perf record -F 999 -g --call-graph dwarf -p PID -o /tmp/rabitq-perf.data -- sleep 10
perf report -i /tmp/rabitq-perf.data
```

An attached profile excludes input loading and index build, but may include a short wait for Enter.
Ignore sleeping/waiting stacks when inspecting compute hotspots. Inclusive stack counts overlap;
do not add them together or treat them as wall-clock stage timings. Inlining may merge query prep,
scan, and refinement into one caller: use source/assembly attribution when symbols are insufficient.

Start with the real worker count, then use a separate single-worker run if scheduler stacks obscure
the computation. Correlate hot stacks with the reported refinement rate before deciding whether to
optimize query preparation, FastScan, or exact refinement. Do not infer an optimization gain from
different seeds or compare profiled times against unsampled baseline times.
