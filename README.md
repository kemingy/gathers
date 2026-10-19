# gathers

[![CI](https://github.com/kemingy/gathers/actions/workflows/check.yml/badge.svg)](https://github.com/kemingy/gathers/actions/workflows/check.yml)
[![crates.io](https://img.shields.io/crates/v/gathers.svg)](https://crates.io/crates/gathers)
[![docs.rs](https://docs.rs/gathers/badge.svg)](https://docs.rs/gathers)

Clustering algorithm implementation in Rust and binding to Python.

For Python users, check the [Python README](./python/README.md).

- [x] K-means
- [x] PyO3 binding
- [x] RaBitQ assignment
- [x] Pilot-calibrated hierarchy assignment during K-means training
- [x] Parallel with Rayon
- [x] SIMD acceleration on `x86_64` and `aarch64` via [`pulp`](https://github.com/sarah-quinones/pulp)
- [ ] mini batch K-means
- [ ] Hierarchical K-means

## Installation

```sh
cargo add gathers
```

## Usage

Check the [library docs](https://docs.rs/gathers).

Beyond k-means assignment, the RaBitQ index is exposed as a standalone module, `gathers::rabitq`,
for approximate top-1 vector retrieval.

The non-published CLI in [`crates/cli`](./crates/cli) supports fvecs input and output:

```sh
cargo run --release -p gathers-cli -- --threads 16 --seed 42 kmeans \
  -i vectors.fvecs -o centroids.fvecs -n 4096
cargo run --release -p gathers-cli -- --threads 16 --seed 42 assign \
  --vectors vectors.fvecs --centroids centroids.fvecs
```

### Dimensionality reduction

Training automatically selects dimensionality reduction from the source shape.
These heuristics may change between versions. Use an explicit reduction and projected
dimension to control the choice, or `--reduction raw` to train without reduction.
To lower the training sample budget, use `--samples-per-cluster 128` (default 256), or set an
exact total with `--training-samples`.

The CLI can train K-means in raw, PCA, or SRHT space and writes full-dimensional centroids for
the existing `assign` command. Select `--distance l2|cos|dot` (default `l2`); explicit PCA/SRHT
default to 128 dimensions, or use `--reduced-dim` to override:

```console
gathers kmeans -i vectors.fvecs -o centroids.fvecs -n 4096 \
  --reduction pca --reduced-dim 128
```

Automatic assignment chooses between approximate hierarchy probing and exact flat
assignment using internal heuristics that may change between versions. Use CLI
`--assignment exact`, Python `assignment="exact"`, or Rust `AssignmentPolicy::Exact`
to keep every Lloyd round exact. An eligible hierarchy is rebuilt and recalibrated
each round; probing has no recall guarantee. See [adaptive training](./docs/adaptive-training-probes.md).

Public Python `batch_assign(..., assignment="auto"|"exact", seed=...)` shares this
policy for L2, cosine and dot product. Rust exposes `kmeans::batch_assign` and a reusable
`CentroidAssigner` for fixed centroids. These APIs use guarded GEMM when eligible and
direct SIMD for small groups. Public assignment applies no dimensionality reduction.

See the [reduction guide](./docs/reduction.md) for method trade-offs and implementation details.

To install the binary from a checkout, run `cargo install --path crates/cli`.
For timing scope and CPU sampling, see the [profiling guide](./docs/profiling.md).
