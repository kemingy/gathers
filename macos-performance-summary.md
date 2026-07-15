# Apple Silicon K-Means Performance Optimization Summary

Date: July 15, 2026

## Conclusion

This optimization reduced the mean execution time of the fixed Apple Silicon
K-Means benchmark from `27.821 ms` to `12.904 ms`. The optimized implementation
is `2.16x` as fast, representing a `53.6%` reduction in execution time.

The optimized code is isolated at compile time with
`cfg(all(target_os = "macos", target_arch = "aarch64"))`. x86 and x86_64 builds
neither compile nor call the new matrix assignment path and continue to use the
original `base_assign_parallel` implementation. Therefore, this production-code
optimization does not change the x86 execution path.

## Measurement Method and Results

Test environment:

- System: macOS Darwin 25.5.0 on Apple Silicon arm64
- Rust: `rustc 1.93.1 (01f6ddf75 2026-02-11)`
- Benchmark framework: Criterion using the release benchmark profile
- Workload: 10,240 vectors with 128 dimensions, 256 centroids, and 5 iterations
- Random seed: 43, ensuring identical input before and after the optimization
- Baseline revision: `3507c09`

Results:

| Version | Criterion time interval | Mean time | Relative speed |
| --- | ---: | ---: | ---: |
| Baseline | `26.959–28.642 ms` | `27.821 ms` | `1.00x` |
| Optimized | `11.883–14.323 ms` | `12.904 ms` | `2.16x` |

The relative speed is calculated as `27.821 / 12.904 = 2.16`. Criterion intervals
vary with machine temperature and background load, so the reported speedup uses
the central estimates from the two measurements.

Reproduction command:

```shell
cargo bench --locked --bench kmeans kmeans -- \
  --warm-up-time 3 --measurement-time 10 --noplot
```

## Implementation Overview

The main hotspot was assigning every vector to its nearest centroid during each
iteration. On Apple Silicon, the optimized path uses the identity

`||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c`

to replace a large number of pairwise distance calculations with blocked matrix
multiplication. The implementation also includes the following safeguards and
optimizations:

- Vector norms, centroid norms, and the dot-product workspace are reused to avoid
  reallocating large buffers on every iteration.
- Matrix multiplication is processed in parallel blocks of 256 rows to make
  better use of Apple Silicon.
- The workspace is limited to 128 MiB. Oversized or small workloads automatically
  fall back to the original implementation.
- Only squared Euclidean distance uses the optimized path. Other distance types
  retain their original implementations.
- Candidates affected by possible floating-point cancellation are verified with
  the original direct distance formula.
- Tests with random inputs and large common offsets verify that assignments match
  direct distance calculation.

## x86 Performance Isolation

The optimization uses compile-time isolation at both the source and dependency
levels rather than a runtime branch:

1. The matrix workspace, Faer matrix multiplication, thresholds, and new tests
   compile only for `macOS + aarch64`.
2. On x86 and x86_64, `base_assign` is identical to the baseline implementation.
   The original body of `base_assign_parallel` is unchanged, and the new entry
   path is removed during conditional compilation.
3. On targets other than `macOS + aarch64`, the K-Means training loop continues
   to call the original `base_assign_parallel` directly.
4. The `faer/std` feature is enabled only for `aarch64-apple-darwin`. Target-aware
   dependency resolution produced the following feature sets:

```text
x86_64-apple-darwin: faer features = [linalg]
aarch64-apple-darwin: faer features = [linalg, std]
```

The architecture conditions in `benches/bench.rs`, `src/simd.rs`, and
`src/rabitq.rs` only allow benchmarks and tests to compile on ARM. They do not
modify any x86 production function.

The current machine does not have the Rust `x86_64-apple-darwin` standard library
installed, so a dynamic x86 benchmark under Rosetta was not executed. The
no-regression conclusion for x86 is instead based on stronger compile-time path
isolation and target-aware dependency resolution: none of the new optimized code
is included in an x86 binary.

Dependency verification commands:

```shell
cargo tree --locked --target x86_64-apple-darwin -e features -i faer
cargo tree --locked --target aarch64-apple-darwin -e features -i faer
```

## Correctness and Engineering Validation

The final changes passed the following checks:

```shell
cargo +nightly fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo clippy --locked --manifest-path python/Cargo.toml -- -D warnings
cargo test --all-targets --locked
cargo check --locked --manifest-path python/Cargo.toml
git diff --check
```
