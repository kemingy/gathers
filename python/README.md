# Gathers Python

[![PyPI version](https://badge.fury.io/py/gathers.svg)](https://badge.fury.io/py/gathers)

## Installation

```bash
pip install gathers
```

## Usage

```python
from gathers import Gathers
import numpy as np

gathers = Gathers()
data = np.random.default_rng().random((1000, 64), dtype=np.float32)
centroids = gathers.fit(data)  # automatic cluster count
labels = gathers.batch_assign(data, centroids)
```

Key `fit` options (all keyword-only except `n_cluster`):

- `distance`: `"l2"` (default), `"cos"`, or `"dot"`. Cosine normalizes training rows;
  dot preserves magnitudes. Batch assignment uses the policy below for all three metrics.
- `assignment`: `"auto"` (default) uses pilot-calibrated hierarchical probing at 3,136+
  clusters, and exact flat assignment below that threshold. Probing is approximate;
  `"exact"` searches every centroid in each Lloyd round. Both call the shared Rust
  trainer, including for PCA/SRHT, cosine and dot workloads.
- `reduction`: `"auto"` (default) uses PCA to 128 dimensions when the source has at least
  1,000,000 rows and dimension above 196, otherwise trains raw — a workload heuristic, not a
  recall guarantee. `"raw"`, `"pca"`, or `"srht"` choose explicitly; centroids always keep
  the original dimension. `reduced_dim` defaults to 128 for pca/srht.
- `n_cluster=None` derives floor(rows^0.8 / 16). Training samples default to
  `256 * n_cluster` rows; `training_samples=N` sets an exact budget. Set `seed` for
  reproducibility.

Inputs must be 2-D `float32` arrays and are never modified. Invalid inputs raise
`ValueError`. See the [reduction guide](../docs/reduction.md) for method trade-offs and the
`fit` docstring for all options.

`batch_assign` accepts the same `assignment="auto"|"exact"` policy and an optional
`seed`. Auto uses the calibrated hierarchy for 3,136+ supplied centroids; otherwise
assignment is exact. GEMM/SIMD selection stays internal. This replaces the former L2
RaBitQ default; eligible cosine/dot assignment can also be approximate now.

```python
labels = gathers.batch_assign(data, centroids, assignment="auto", seed=42)
exact_labels = gathers.batch_assign(data, centroids, assignment="exact")
```

Assignment uses the supplied centroid dimensions, with no PCA/SRHT projection. Cosine
normalization uses bounded scratch and leaves inputs unchanged. Empty query batches
return an empty list. `assign` remains exact direct SIMD for a single vector.
