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

Check the [docs](https://docs.rs/gathers) and [main.rs](./src/main.rs).
