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
- [x] `x86` & `x86_64` SIMD acceleration
- [ ] mini batch K-means
- [ ] Hierarchical K-means
- [ ] `arm` & `aarch64` SIMD acceleration

## Installation

```sh
cargo add gathers
```

## Usage

Check the [library docs](https://docs.rs/gathers).

The non-published CLI in [`crates/cli`](./crates/cli) supports fvecs input and output:

```sh
cargo run --release -p gathers-cli -- --threads 16 --seed 42 kmeans \
  -i vectors.fvecs -o centroids.fvecs -n 4096 -m 25
cargo run --release -p gathers-cli -- --threads 16 --seed 42 assign \
  --vectors vectors.fvecs --centroids centroids.fvecs
```

This replaces the root binary: use `-p gathers-cli` and the `kmeans` subcommand for the old
training workflow. To install the binary from a checkout, run `cargo install --path crates/cli`.
For timing scope and CPU sampling, see the [profiling guide](./docs/profiling.md).
