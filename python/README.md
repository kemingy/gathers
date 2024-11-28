# Gathers Python

## Installation

```bash
pip install gathers
```

## Usage

```python
from gathers import assign, kmeans_fit
import numpy as np


data = np.random.rand(1000, 8).astype(np.float32)
centroids = kmeans_fit(data, 10, 10)
label = assign(data[0], centroids)
```
