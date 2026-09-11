# Repository Guidelines

## Project Structure and Ownership

The root Cargo workspace contains the public `gathers` crate and two internal crates:

- `src/lib.rs`: public Rust library surface. The crate denies missing documentation, so every new
  public item needs rustdoc.
- `src/kmeans.rs` and `src/kmeans/`: K-means configuration, assignment, centroid updates, and
  matrix-based implementation details.
- `src/distance.rs`: scalar reference implementations and runtime-dispatched distance operations.
- `src/simd.rs` and `src/simd/`: SIMD dispatch and portable kernels implemented with `pulp`.
- `src/{sampling,utils}.rs`: shared sampling and vector-layout helpers.
- `src/main.rs`: small executable/example entrypoint, not a second implementation of the library.
- `crates/rabitq/`: internal, non-published RaBitQ implementation. Keep its quantization, retrieval,
  workspace, and specialized SIMD code here; the root crate re-exports its public API.
- `crates/seed_rand/`: internal test/benchmark helper for reproducible randomized workloads. It
  should not become a general production utility crate.
- `python/`: PyO3 extension and Python package. It is intentionally excluded from the root Cargo
  workspace and has its own `Cargo.toml` and lockfile.
- `benches/`: Criterion Rust benchmarks and comparison scripts. Keep input generation and cloning
  outside timed routines.
- `scripts/`: development and data-generation utilities.

Put reusable clustering behavior in the root library, not in `main.rs` or the Python binding. Keep
the Python layer thin: validate/translate Python inputs, call the Rust API, and translate outputs.
Create a workspace crate only when a component has a clear ownership boundary or needs independent
dependencies; do not split code merely to shorten a module.

## Build, Test, and Development Commands

Use the Makefile targets as the normal repository workflow:

- `make format`: format Rust with nightly rustfmt and fix Python Ruff issues.
- `make lint`: check Rust formatting, run Clippy with warnings denied, and check Python with Ruff.
- `make test-rust`: run the Rust workspace tests verbosely.
- `make install-python`: install the Python package in editable mode.
- `make test-python`: install the editable extension and run `python/tests`.
- `make test`: run both Rust and Python test suites; this is the closest local equivalent to CI.

Targeted commands are useful while iterating:

- `cargo test test_name -- --nocapture`: run one Rust test and display captured output.
- `cargo test -p rabitq`: test only the RaBitQ crate.
- `cargo bench --bench bench` or `cargo bench --bench kmeans`: run optimized Criterion benchmarks.
- `cargo build --profile perf --features perf`: build the single-threaded profiling configuration
  with debug symbols.

The repository defaults to stable Rust, but formatting explicitly uses nightly. Install both
toolchains when running the full checks. Do not hand-edit generated lockfile changes; regenerate
them with Cargo in the manifest that owns them. Because `python/` is excluded from the workspace,
run its direct Cargo commands with `--manifest-path python/Cargo.toml` or from that directory.

## Rust Style and API Design

- Follow `rustfmt.toml`; run formatting rather than manually aligning code.
- Use `snake_case` for modules, functions, and variables, `PascalCase` for types and traits, and
  `UPPER_SNAKE_CASE` for constants.
- Keep flat vector layout contracts explicit: a collection of vectors is generally a contiguous
  `&[f32]` plus `dim`. Validate dimensions at public boundaries, then avoid repeated checks in hot
  inner loops.
- Preserve scalar or straightforward implementations as correctness references when adding an
  optimized path.
- Reuse caller-owned workspaces in repeated operations. Avoid allocation, format conversion, and
  cloning in assignment/retrieval loops.
- Keep `unsafe` blocks narrow. Document pointer bounds, alignment, initialized memory, aliasing,
  and required CPU features at the function that establishes each safety contract.
- Do not change floating-point accumulation order casually. SIMD and FMA can produce small
  differences; use a justified absolute-plus-relative tolerance instead of exact equality.

## SIMD and Hot-Loop Optimization

Measure first and optimize the end-to-end workload, not an isolated instruction count.

### Dispatch and portability

- Prefer `pulp::Arch::new().dispatch(...)` and generic `pulp::Simd` kernels for operations that can
  be expressed portably. This keeps scalar and architecture-specific behavior behind one API.
- Use `std::arch` intrinsics only when the generic implementation cannot express a measured,
  important optimization. Specialized x86/x86_64 code belongs behind architecture `cfg`s and
  runtime feature detection, with a tested scalar/portable fallback.
- A `#[target_feature]` function is not feature detection. Its caller must prove that every enabled
  feature is present before entering it. Keep that proof adjacent to the call.
- Do not make the entire crate require `target-cpu=native`; published binaries must remain runnable
  on CPUs other than the build machine.

### Latency, throughput, and dependency chains

Instruction latency is the time until a dependent instruction can consume a result. Reciprocal
throughput is how often independent instances can begin. They answer different questions:

- For reductions such as dot products and squared L2 distance, one accumulator creates a
  loop-carried dependency chain and tends to expose FMA/add latency.
- Use multiple independent accumulators when measurements show a latency-bound reduction. This
  lets the CPU approach instruction throughput; merge accumulators only after the main loop.
- More unrolling is not automatically faster. Extra accumulators increase register pressure,
  instruction count, and final reduction cost, and can hurt short vectors.
- For a chain where each operation depends on the previous result, optimize latency. For many
  independent vectors or centroids, organize work to exploit throughput and instruction-level
  parallelism.
- Consult instruction data for the actual target microarchitecture (for example the Intel
  Intrinsics Guide and uops.info), but treat published cycle counts as hypotheses. Validate on the
  CPUs represented by the workload.

The existing two-accumulator floating-point reductions and four-way `argmin` are patterns to
evaluate, not universal unroll factors.

### Memory and I/O

- Count bytes loaded and stored as well as arithmetic operations. SIMD cannot overcome a
  memory-bandwidth bottleneck.
- Favor contiguous, unit-stride slices and process enough work per dispatch to amortize detection,
  call, and tail-handling overhead.
- Avoid gather/scatter, temporary transposes, and repeated conversion unless the reuse of the new
  layout pays for its cost. Keep centroids and reusable quantized data in cache-friendly layouts.
- Do not add aligned-load requirements without evidence. Unaligned loads are often efficient and
  are safer for arbitrary slices; if alignment is required, encode and validate it at the API
  boundary.
- Reuse output and scratch buffers, as `RaBitQWorkspace` does. Allocation and zeroing are I/O too.
- When combining Rayon and SIMD, choose chunks large enough to amortize task scheduling and avoid
  false sharing. Check memory bandwidth before adding threads; parallelism can make a
  bandwidth-bound kernel slower.
- Keep file parsing, random input generation, logging, allocation, and setup outside benchmark
  timing. For a production optimization, also measure the end-to-end path so setup costs are not
  hidden.

### Correctness requirements

Every optimized kernel must cover empty/short inputs where the API permits them, lengths below one
vector, exact vector multiples, and non-multiple tails. Compare it with the scalar reference over
several representative dimensions. Preserve documented behavior for equal values, NaNs, overflow,
and underflow; if behavior is intentionally changed, document and test it.

Run portable tests on the normal target before architecture-specific tests. Never execute an
intrinsic test solely because it compiled: gate execution with runtime feature detection. Changes
to x86 code must still compile and pass through the fallback on non-x86 CI targets.

## Testing and Performance Evidence

- Add unit tests beside the owning module and Python behavior tests under `python/tests/`.
- Randomized correctness tests use `seed_rand::seeded_rng()` and intentionally choose fresh data.
  On failure, reproduce with the printed seed:

  ```console
  GATHERS_RANDOM_SEED=123456 cargo test test_name -- --nocapture
  ```

  Keep targeted regression fixtures deterministic. Do not replace the shared randomized helper
  with a fixed seed just to make successful runs identical.
- Benchmark optimized code in a release/bench profile, never infer performance from debug builds.
  Compare multiple input dimensions, including tails, and multiple runs. Pin
  `GATHERS_RANDOM_SEED` when comparing two implementations on identical generated data.
- Report the CPU, target features, thread count, dataset shape, command, and before/after result for
  performance claims. Distinguish wall-clock improvements from isolated kernel improvements.
- Run `make lint` and the relevant targeted tests before review; run `make test` when Rust/Python
  integration or public behavior changes.

## Changes and Reviews

Keep changes focused and use Conventional Commit prefixes such as `feat:`, `fix:`, `perf:`,
`refactor:`, `test:`, and `docs:`. A performance change should explain the bottleneck, why the new
layout or instruction sequence addresses it, correctness coverage, and benchmark evidence. Call
out architecture-specific behavior and any numerical tradeoffs explicitly.
