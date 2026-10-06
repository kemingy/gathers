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


gathers = Gathers(verbose=True)
rng = np.random.default_rng()
data = rng.random((1000, 64), dtype=np.float32)  # only support float32
centroids = gathers.fit(data)  # automatic cluster count and reduction
labels = gathers.batch_assign(data, centroids)
print(labels)
```

PCA and SRHT can train in reduced space while returning centroids in the original dimension:

```python
centroids = gathers.fit(
    data, 10, reduction="pca", reduced_dim=16, distance="cos", seed=42,
    projection_training_samples=1000,
)
labels = gathers.batch_assign(data, centroids, distance="cos")
```

Default `reduction="auto"` selects PCA to 128 dimensions when the original input has at least
1,000,000 rows and dimension above 196; otherwise it trains raw. This is a workload heuristic,
not a recall guarantee. Use `reduction="raw"` to disable projection, or `"srht"` for a seeded
Hadamard projection. Explicit PCA/SRHT default `reduced_dim` to 128 when omitted; it must be
smaller than the input dimension. Projection options require an explicit PCA/SRHT choice.
Distances are `"l2"`, `"cos"`, and `"dot"`. Cosine normalizes training rows before fitting PCA and
normalizes nonzero projections before dot training; dot preserves original row magnitudes.
Inputs are not modified. Strided arrays are made contiguous by the Python wrapper; arrays must
contain `float32` values. Invalid training options and non-finite coordinates raise `ValueError`.

Omit `n_cluster` (or pass `None`) to use floor(rows^0.8 / 16), bounded by at least 39 rows per
cluster. K-means samples `min(source rows, samples_per_cluster * n_cluster)` rows; the factor
defaults to 256 and must be at least 39. Use `gathers.fit(data, samples_per_cluster=128)` for a
smaller budget. An exact `training_samples=N` overrides the factor, which is then ignored.
Explicit totals must not exceed source rows and require at least 39 rows per cluster.
Smaller samples can save memory and training work, but may change recall.

PCA fits at most `100 * input_dimension` of the selected K-means rows by default;
`projection_training_samples` overrides that separate budget and applies only to PCA.
It must be between 2 and the selected K-means sample size. Set `seed` for reproducibility within
the same build, architecture, and worker count. L2 batch assignment uses approximate RaBitQ;
cosine and dot use exact assignment.
