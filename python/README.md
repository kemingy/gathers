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
centroids = gathers.fit(data, 10)
labels = gathers.batch_assign(data, centroids)
print(labels)
```
