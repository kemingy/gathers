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
  dot preserves magnitudes. `batch_assign` is approximate (RaBitQ) for l2, exact otherwise.
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
