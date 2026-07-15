# macOS K-Means Performance Optimization Summary

Date: July 15, 2026

## Conclusion

The matrix-assignment optimization benefits both macOS architectures tested:

| Target | Baseline mean | Optimized mean | Speedup | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| Apple Silicon arm64 | `27.821 ms` | `12.904 ms` | `2.16x` | `53.6%` |
| Intel macOS x86_64 under Rosetta | `1.0963 s` | `60.882 ms` | `18.0x` | `94.4%` |

The x86_64 result was measured before and after changing only the macOS compile
gate for the matrix path. It shows that the algorithmic changes and Faer's x86
GEMM backend are also effective for Intel macOS. The optimized production path
therefore compiles on both `aarch64-apple-darwin` and
`x86_64-apple-darwin`. Other operating systems retain the original assignment
implementation.

## Measurement Method and Results

Common setup:

- Host: macOS Darwin 25.5.0 on Apple Silicon
- Rust: `rustc 1.93.1 (01f6ddf75 2026-02-11)`
- Benchmark framework: Criterion using the release benchmark profile
- Workload: 10,240 vectors with 128 dimensions, 256 centroids, and 5 iterations
- Random seed: 43, ensuring identical input before and after the optimization
- Apple Silicon baseline revision: `3507c09`
- x86_64 baseline: `c8fa604`, where the matrix path was compile-time disabled
  for x86_64 and the production assignment path still matched `3507c09`

Detailed Criterion results:

| Target and version | Criterion time interval | Mean time | Relative speed |
| --- | ---: | ---: | ---: |
| arm64 baseline | `26.959–28.642 ms` | `27.821 ms` | `1.00x` |
| arm64 optimized | `11.883–14.323 ms` | `12.904 ms` | `2.16x` |
| x86_64 baseline under Rosetta | `1.0854–1.1062 s` | `1.0963 s` | `1.00x` |
| x86_64 optimized under Rosetta | `59.196–63.179 ms` | `60.882 ms` | `18.0x` |

The relative speeds use the central estimates. Criterion intervals vary with
machine temperature and background load. Rosetta results should not be treated
as absolute native Intel timings, but the before/after comparison uses the same
host, x86_64 target, toolchain, input, and benchmark parameters, making it strong
evidence for enabling the path on Intel macOS.

Reproduction commands:

```shell
# Native Apple Silicon
cargo bench --locked --bench kmeans kmeans -- \
  --warm-up-time 3 --measurement-time 10 --noplot

# x86_64 under Rosetta
rustup target add x86_64-apple-darwin --toolchain stable-aarch64-apple-darwin
cargo bench --locked --target x86_64-apple-darwin --bench kmeans kmeans -- \
  --warm-up-time 3 --measurement-time 10 --noplot
```

## Implementation Overview

The main hotspot was assigning every vector to its nearest centroid during each
iteration. On macOS, the optimized path uses the identity

`||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c`

to replace a large number of pairwise distance calculations with blocked matrix
multiplication. The implementation includes the following safeguards and
optimizations:

- Vector norms, centroid norms, and the dot-product workspace are reused to avoid
  reallocating large buffers on every iteration.
- Matrix multiplication is processed in parallel blocks of 256 rows.
- The workspace is limited to 128 MiB. Oversized or small workloads automatically
  fall back to the original implementation.
- Only squared Euclidean distance uses the optimized path. Other distance types
  retain their original implementations.
- Candidates affected by possible floating-point cancellation are verified with
  the original direct distance formula.
- Tests with random inputs and large common offsets verify that assignments match
  direct distance calculation on both macOS architectures.

## Platform Scope and Benchmark Isolation

The optimization uses compile-time isolation rather than adding a runtime
platform branch:

1. The matrix workspace, Faer matrix multiplication, thresholds, and correctness
   tests compile for `target_os = "macos"`, covering arm64 and x86_64.
2. Linux, Windows, and other operating systems continue to call the original
   `base_assign_parallel` implementation.
3. The `faer/std` feature is enabled only for macOS targets; other targets retain
   the original dependency feature set.
4. Existing x86 microbenchmarks keep their original direct SIMD and
   `pulp::x86::V3` calls. The new assignment and K-Means cases live in the
   separate `kmeans` benchmark target so they do not perturb the old benchmark
   binary.

Dependency verification commands:

```shell
cargo tree --locked --target x86_64-apple-darwin -e features -i faer
cargo tree --locked --target aarch64-apple-darwin -e features -i faer
```

## Correctness and Engineering Validation

Native Apple Silicon validation:

```shell
cargo +nightly fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo clippy --locked --manifest-path python/Cargo.toml -- -D warnings
cargo test --all-targets --locked
cargo check --locked --manifest-path python/Cargo.toml
git diff --check
```

x86_64 macOS validation:

```shell
cargo check --all-targets --locked --target x86_64-apple-darwin
cargo clippy --all-targets --locked --target x86_64-apple-darwin -- -D warnings
cargo test --locked --target x86_64-apple-darwin --lib --bins
```

The x86_64 library suite passed all 12 tests under Rosetta, including both
matrix-assignment correctness tests. Running every legacy benchmark as a test
under Rosetta is not supported because the existing binary-dot-product benchmark
assumes AVX availability; native Intel CI remains the authoritative check for
that pre-existing benchmark path.
