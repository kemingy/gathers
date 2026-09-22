# gathers

[![CI](https://github.com/kemingy/gathers/actions/workflows/check.yml/badge.svg)](https://github.com/kemingy/gathers/actions/workflows/check.yml)
[![crates.io](https://img.shields.io/crates/v/gathers.svg)](https://crates.io/crates/gathers)
[![docs.rs](https://docs.rs/gathers/badge.svg)](https://docs.rs/gathers)

Clustering algorithm implementation in Rust and binding to Python.

For Python users, check the [Python README](./python/README.md).

- [x] K-means
- [x] PyO3 binding
- [x] RaBitQ assignment
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
  -i vectors.fvecs -o centroids.fvecs -n 4096 -m 25
cargo run --release -p gathers-cli -- --threads 16 --seed 42 assign \
  --vectors vectors.fvecs --centroids centroids.fvecs
```

To install the binary from a checkout, run `cargo install --path crates/cli`.
For timing scope and CPU sampling, see the [profiling guide](./docs/profiling.md).
